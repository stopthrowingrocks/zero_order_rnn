#!/usr/bin/env python3
"""
Distributed faster grid search using torch.distributed across 8 GPUs.
Each GPU computes 1 perturbation, giving 8 total perturbations per step.

To run: torchrun --nproc_per_node=8 faster_grid_search_dist.py
"""
import os
import time
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
from models.models import LSTM


def generate_reverse_batch(batch_size, min_seq_length, max_seq_length, vocab_size, device='cuda'):
    """Generate a batch of reverse sequences with variable lengths."""
    BOS = vocab_size - 4
    SEP = vocab_size - 3
    EOS = vocab_size - 2
    PAD = vocab_size - 1
    content_vocab_size = vocab_size - 4

    x_batch = []
    y_batch = []

    for _ in range(batch_size):
        seq_length = np.random.randint(min_seq_length, max_seq_length + 1)
        seq = np.random.randint(0, content_vocab_size, size=seq_length)
        full_seq = [BOS] + seq.tolist() + [SEP] + seq[::-1].tolist() + [EOS]

        x_batch.append(torch.tensor(full_seq, dtype=torch.long, device=device))
        y_batch.append(torch.tensor(full_seq, dtype=torch.long, device=device))

    x_ids = nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=PAD)
    y_ids = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=PAD)

    return x_ids, y_ids


def compute_reverse_loss(logits, targets, pad_id):
    """Compute cross-entropy loss, ignoring padding tokens."""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    loss = nn.functional.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=pad_id,
        reduction='mean'
    )
    return loss


def generate_perturbation(ref: torch.Tensor, scale: float, distribution: str, seed: int) -> torch.Tensor:
    """Generate a perturbation with unit magnitude, scaled by scale."""
    g = torch.Generator(device=ref.device).manual_seed(int(seed))
    if distribution == "rad":
        z = torch.zeros_like(ref).bernoulli_(0.5, generator=g).mul_(2).sub_(1).mul_(scale)
    elif distribution == "normal":
        z = torch.randn_like(ref, generator=g) * scale
    else:  # uniform
        z = (torch.rand_like(ref, generator=g) * 2 - 1) * scale
    return z


def apply_probe(params, scale, base_seed, distn, rolling_sum_weighted_probe=None, coeff=None):
    """
    θ ← θ + scale * δ     where δ has unit magnitude (Rademacher or Normal).

    If `rolling_sum_weighted_probe` and `coeff` are provided we accumulate
        rolling_sum_weighted_probe[i] += coeff * δ
    so a one-shot gradient step can be taken later.
    """
    for i, p in enumerate(params):
        delta = generate_perturbation(p, 1.0, distn, base_seed + i)  # unit δ
        p.data.add_(delta, alpha=scale)                              # fused add

        if rolling_sum_weighted_probe is not None and coeff is not None:
            rolling_sum_weighted_probe[i].add_(delta, alpha=coeff)


def distributed_spsa_step(model, embed, x_ids, y_ids, pad_id, learning_rate, epsilon, num_perturbations, rank, world_size, cache_roll=False):
    """
    Perform one distributed SPSA step following distributed_rge.py:dist_cdrge_step exactly.

    Args:
        cache_roll: If False (default), only communicate scalar losses (efficient).
                    If True, use rolling buffer (communicates parameter-sized tensors).
    """
    # 0. setup -----------------------------------------------------------------
    distributed = dist.is_available() and dist.is_initialized()
    param_list = list(model.parameters()) + list(embed.parameters())
    device = param_list[0].device
    per_rank = num_perturbations // world_size
    if per_rank == 0:
        per_rank = 1
    n_total = per_rank * world_size
    distn = "normal"

    # Validate that x_ids and y_ids are present on all ranks
    assert x_ids is not None, f"Rank {rank}: x_ids is None"
    assert y_ids is not None, f"Rank {rank}: y_ids is None"
    assert x_ids.numel() > 0, f"Rank {rank}: x_ids is empty"
    assert y_ids.numel() > 0, f"Rank {rank}: y_ids is empty"
    assert x_ids.shape[0] > 0, f"Rank {rank}: x_ids has zero batch size"
    assert y_ids.shape[0] > 0, f"Rank {rank}: y_ids has zero batch size"
    assert x_ids.device == device, f"Rank {rank}: x_ids on wrong device ({x_ids.device} vs {device})"
    assert y_ids.device == device, f"Rank {rank}: y_ids on wrong device ({y_ids.device} vs {device})"

    # 1. optional rolling buffer -----------------------------------------------
    rolling_sum_weighted_probe = (
        [torch.zeros_like(p.data) for p in param_list] if cache_roll else None
    )

    # 2. broadcast θ ------------------------------------------------------------
    if distributed and world_size > 1:
        for p in param_list:
            dist.broadcast(p.data, src=0)

    # 3. scatter seeds ----------------------------------------------------------
    seeds_local = torch.zeros(per_rank, dtype=torch.int32, device=device)
    if rank == 0:
        full_seeds = torch.randint(0, 2**31 - 1, (n_total,),
                                   dtype=torch.int32, device=device)
        chunks = list(full_seeds.chunk(world_size, dim=0))
    else:
        full_seeds = torch.empty(n_total, dtype=torch.int32, device=device)
        chunks = None
    if distributed and world_size > 1:
        dist.scatter(seeds_local, chunks, src=0)
    else:
        seeds_local.copy_(full_seeds)

    # 4. local ±ε evaluations ---------------------------------------------------
    loss_pairs_local = torch.zeros(per_rank, 2, dtype=torch.float32, device=device)

    for m in range(per_rank):
        seed_m = int(seeds_local[m].item())

        # +ε
        apply_probe(param_list, +epsilon, seed_m, distn)
        x_emb = embed(x_ids)
        logits_plus, _, _ = model(x_emb, require_gradients=False)
        L_plus = compute_reverse_loss(logits_plus, y_ids, pad_id).item()

        # −ε
        apply_probe(param_list, -2.0 * epsilon, seed_m, distn)
        x_emb = embed(x_ids)
        logits_minus, _, _ = model(x_emb, require_gradients=False)
        L_minus = compute_reverse_loss(logits_minus, y_ids, pad_id).item()

        # coef and restore (+ε again)
        coef = (L_plus - L_minus) / (2.0 * n_total)
        restore_coeff = -coef                           # GD direction
        apply_probe(
            param_list, +epsilon, seed_m, distn,
            rolling_sum_weighted_probe=rolling_sum_weighted_probe,
            coeff=restore_coeff,
        )

        loss_pairs_local[m, 0] = L_plus
        loss_pairs_local[m, 1] = L_minus

    # 5. gather losses (logging only) ------------------------------------------
    if distributed and world_size > 1:
        gather_buf = (
            [torch.empty_like(loss_pairs_local) for _ in range(world_size)]
            if rank == 0 else None
        )
        dist.gather(loss_pairs_local, gather_buf, dst=0)
        if rank == 0:
            loss_pairs_full = torch.cat(gather_buf, dim=0)
    else:
        loss_pairs_full = loss_pairs_local

    # 6. parameter update -------------------------------------------------------
    if cache_roll:
        if distributed and world_size > 1:
            for buf in rolling_sum_weighted_probe:
                dist.reduce(buf, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            for p, acc in zip(param_list, rolling_sum_weighted_probe):
                p.data.add_(acc, alpha=learning_rate)      # apply LR here!
    else:  # fallback (slow loop)
        if rank == 0:
            if world_size == 1:
                full_seeds = seeds_local.clone()
            for i in range(n_total):
                coef = (loss_pairs_full[i, 0] - loss_pairs_full[i, 1]) \
                       / (2.0 * n_total)
                seed_i = int(full_seeds[i].item())
                apply_probe(param_list, -learning_rate * coef.item(), seed_i, distn)

    # 7. broadcast updated parameters -------------------------------------------
    if distributed and world_size > 1:
        for p in param_list:
            dist.broadcast(p.data, src=0)

    # 8. barrier ----------------------------------------------------------------
    if distributed and world_size > 1:
        dist.barrier()

    # 9. return loss ------------------------------------------------------------
    mean_loss = float(
        (loss_pairs_full if rank == 0 else loss_pairs_local).mean().item()
    )
    return mean_loss


def train_with_early_stop_distributed(batch_size, perturbations, num_gpus, learning_rate, vocab_size,
                                        min_tokens, max_tokens, max_time, convergence_loss,
                                        rank, world_size, device, seed,
                                        hidden_size=240, num_heads=12, input_size=100):
    """
    Distributed training with early stopping.
    IMPORTANT: epsilon is always set equal to learning_rate.
    """
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    # Always set epsilon = learning_rate
    epsilon = learning_rate

    PAD = vocab_size - 1

    embed = nn.Embedding(vocab_size, input_size, device=device, dtype=torch.bfloat16)
    model = LSTM(
        input_size=input_size,
        output_size=vocab_size,
        hidden_size=hidden_size,
        memory_size=0,
        head_size=hidden_size // num_heads,
        num_heads=num_heads,
        embed=embed,
        device=device,
        dtype=torch.bfloat16
    )

    start_time = time.time()
    step = 0
    losses = []
    initial_loss = None
    terminated_early = False
    converged = False

    while time.time() - start_time < max_time:
        # Generate batch (all ranks use same data for now, can be optimized)
        x_ids, y_ids = generate_reverse_batch(
            batch_size, min_tokens, int(max_tokens), vocab_size, device
        )

        loss = distributed_spsa_step(
            model, embed, x_ids, y_ids, PAD,
            learning_rate, epsilon, perturbations, rank, world_size
        )

        if rank == 0:
            losses.append(loss)
            step += 1

            # Store initial loss
            if initial_loss is None:
                initial_loss = loss

            # Check for convergence
            if loss < convergence_loss:
                converged = True
                break

            # Early termination if loss increases by 3x
            if loss > 3.0 * initial_loss:
                terminated_early = True
                break

            # Log to wandb every 10 steps
            if step % 10 == 0:
                wandb.log({
                    'loss': loss,
                    'step': step,
                    'elapsed_time': time.time() - start_time
                })

        # Synchronize termination across ranks
        if world_size > 1:
            should_stop = torch.tensor([converged or terminated_early], dtype=torch.int32, device=device)
            dist.broadcast(should_stop, src=0)
            if should_stop.item():
                break

    elapsed = time.time() - start_time
    final_loss = losses[-1] if (rank == 0 and losses) else 0.0

    return {
        'final_loss': final_loss,
        'steps': step,
        'elapsed_time': elapsed,
        'terminated_early': terminated_early,
        'converged': converged
    }


def append_to_csv(filename, row_dict):
    """Append a single row to CSV file (creates file with header if doesn't exist)."""
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as f:
        fieldnames = ['batch_size', 'perturbations', 'num_gpus', 'learning_rate',
                     'vocab_size', 'min_tokens', 'max_tokens', 'seed',
                     'final_loss', 'steps', 'elapsed_time', 'converged',
                     'terminated_early', 'max_time_used', 'optalg']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_dict)


def adaptive_lr_search_for_max_tokens(max_tokens, batch_size, perturbations, vocab_size,
                                        min_tokens, num_gpus, rank, world_size, device, base_seed,
                                        hidden_size=240, num_heads=12, input_size=100):
    """
    Adaptive LR search for a specific max_tokens value (distributed version).
    Only rank 0 does I/O and logging.
    """
    convergence_loss = 0.1

    # LR values from high to low
    learning_rates = 10 ** np.arange(-0.6, -2 - 0.6, -0.2)  # 10^-0.6 down to 10^-2.6

    # Start with 5 second max_time
    initial_max_time = 5.0
    max_time = initial_max_time
    HARD_CAP_TIME = 5.0

    results = []
    first_convergence_idx = None
    first_convergence_time = None

    for lr_idx, lr in enumerate(learning_rates):
        # If we've found convergence and are 2+ LRs past it, stop
        if first_convergence_idx is not None and lr_idx >= first_convergence_idx + 2:
            if rank == 0:
                print(f"  Stopping LR search (2 LRs past first convergence)")
            break

        seed = base_seed + lr_idx

        # Synchronize before wandb init
        if world_size > 1:
            dist.barrier()

        # Only rank 0 initializes wandb
        if rank == 0:
            run_name = (f"fast_dist_batch_{batch_size}_pert_{perturbations}_gpus_{num_gpus}_"
                       f"lr_{lr:.6f}_vocab_{vocab_size}_min_{min_tokens}_max_{int(max_tokens)}")

            print(f"    LR={lr:.6f} (max_time={max_time:.1f}s)...", end=" ", flush=True)

            wandb.init(
                project="zero_order_rnn",
                name=run_name,
                config={
                    'batch_size': batch_size,
                    'perturbations': perturbations,
                    'num_gpus': num_gpus,
                    'learning_rate': lr,
                    'vocab_size': vocab_size,
                    'min_tokens': min_tokens,
                    'max_tokens': int(max_tokens),
                    'seed': seed,
                    'epsilon': lr,  # epsilon = learning_rate
                    'hidden_size': hidden_size,
                    'num_heads': num_heads,
                    'max_time': max_time,
                    'optalg': 'SPSA_distributed'
                },
                reinit=True
            )

        # Synchronize after wandb init (rank 0 may have been blocked)
        if world_size > 1:
            dist.barrier()

        try:
            result = train_with_early_stop_distributed(
                batch_size=batch_size,
                perturbations=perturbations,
                num_gpus=num_gpus,
                learning_rate=lr,
                vocab_size=vocab_size,
                min_tokens=min_tokens,
                max_tokens=int(max_tokens),
                max_time=max_time,
                convergence_loss=convergence_loss,
                rank=rank,
                world_size=world_size,
                device=device,
                seed=seed,
                hidden_size=hidden_size,
                num_heads=num_heads,
                input_size=input_size
            )

            if rank == 0:
                # Log final results to wandb
                wandb.log({
                    'final_loss': result['final_loss'],
                    'total_steps': result['steps'],
                    'total_time': result['elapsed_time'],
                    'terminated_early': result['terminated_early'],
                    'converged': result['converged']
                })

                # Store result
                result['lr'] = lr
                result['max_tokens'] = int(max_tokens)
                result['batch_size'] = batch_size
                result['perturbations'] = perturbations
                result['max_time_used'] = max_time
                results.append(result)

                # Check if this is first convergence
                if result['converged'] and first_convergence_idx is None:
                    first_convergence_idx = lr_idx
                    first_convergence_time = result['elapsed_time']
                    max_time = min(first_convergence_time * 2.0, HARD_CAP_TIME)
                    print(f"CONVERGED in {result['elapsed_time']:.2f}s! Setting max_time={max_time:.1f}s")
                elif result['converged']:
                    print(f"Converged in {result['elapsed_time']:.2f}s")
                elif result['terminated_early']:
                    print(f"Terminated early (3x loss increase)")
                else:
                    print(f"Did not converge (loss={result['final_loss']:.4f})")

                # Append to CSV
                csv_row = {
                    'batch_size': batch_size,
                    'perturbations': perturbations,
                    'num_gpus': num_gpus,
                    'learning_rate': lr,
                    'vocab_size': vocab_size,
                    'min_tokens': min_tokens,
                    'max_tokens': int(max_tokens),
                    'seed': seed,
                    'final_loss': result['final_loss'],
                    'steps': result['steps'],
                    'elapsed_time': result['elapsed_time'],
                    'converged': result['converged'],
                    'terminated_early': result['terminated_early'],
                    'max_time_used': max_time,
                    'optalg': 'SPSA_distributed'
                }
                append_to_csv('losses_max_10.csv', csv_row)

        except Exception as e:
            if rank == 0:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

                # Log error to CSV
                csv_row = {
                    'batch_size': batch_size,
                    'perturbations': perturbations,
                    'num_gpus': num_gpus,
                    'learning_rate': lr,
                    'vocab_size': vocab_size,
                    'min_tokens': min_tokens,
                    'max_tokens': int(max_tokens),
                    'seed': seed,
                    'final_loss': float('nan'),
                    'steps': 0,
                    'elapsed_time': 0.0,
                    'converged': False,
                    'terminated_early': False,
                    'max_time_used': max_time,
                    'optalg': 'SPSA_distributed'
                }
                append_to_csv('losses_max_10.csv', csv_row)

        finally:
            if rank == 0:
                wandb.finish()

    return results


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--local_rank", type=int, default=-1)  # torchrun uses this
    args = parser.parse_args()

    try:
        print(args.local_rank)
        torch.cuda.set_device(args.local_rank)
    except RuntimeError as e:
        print(f"Error metadata. Local rank: {args.local_rank}, CUDA Available: {torch.cuda.is_available()}, Device Count: {torch.cuda.device_count()}, GPU Name 0: {torch.cuda.get_device_name(0)}")
        raise e

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device
    local_rank = args.local_rank if args.local_rank != -1 else rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # Grid search parameters
    batch_sizes = [16, 32]
    perturbations_values = [8, 16, 24, 32, 48, 64]
    max_tokens = 10  # Fixed
    num_gpus = world_size
    vocab_size = 64
    min_tokens = 5

    hidden_size = 240
    num_heads = 12
    input_size = 100
    base_seed = 42

    if rank == 0:
        print("="*80)
        print("DISTRIBUTED GRID SEARCH FOR MAX_TOKENS=10")
        print("="*80)
        print(f"World size: {world_size} GPUs")
        print(f"Fixed: max_tokens={max_tokens}")
        print(f"Grid: batch_size={batch_sizes}")
        print(f"Grid: perturbations={perturbations_values}")
        print(f"Total combinations: {len(batch_sizes)} × {len(perturbations_values)} = {len(batch_sizes) * len(perturbations_values)}")
        print(f"Output: losses_max_10.csv")
        print("="*80)
        print()

    combo_idx = 0
    total_combos = len(batch_sizes) * len(perturbations_values)

    for batch_size in batch_sizes:
        for perturbations in perturbations_values:
            combo_idx += 1
            if rank == 0:
                print(f"\n[{combo_idx}/{total_combos}] BATCH_SIZE={batch_size}, PERTURBATIONS={perturbations}")
                print("-"*80)

            results = adaptive_lr_search_for_max_tokens(
                max_tokens=max_tokens,
                batch_size=batch_size,
                perturbations=perturbations,
                vocab_size=vocab_size,
                min_tokens=min_tokens,
                num_gpus=num_gpus,
                rank=rank,
                world_size=world_size,
                device=device,
                base_seed=base_seed + combo_idx * 1000,
                hidden_size=hidden_size,
                num_heads=num_heads,
                input_size=input_size
            )

            if rank == 0:
                # Summary
                converged_results = [r for r in results if r['converged']]
                print(f"\n  Summary for batch_size={batch_size}, perturbations={perturbations}:")
                print(f"    Tested {len(results)} LR values")
                print(f"    Converged: {len(converged_results)}/{len(results)}")
                if converged_results:
                    best = min(converged_results, key=lambda r: r['elapsed_time'])
                    print(f"    Best: LR={best['lr']:.6f} in {best['elapsed_time']:.2f}s")

    if rank == 0:
        print()
        print("="*80)
        print("DISTRIBUTED GRID SEARCH COMPLETE")
        print("="*80)
        print(f"Results saved to: losses_max_10.csv")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
