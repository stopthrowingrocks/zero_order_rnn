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


def apply_probe(param_list, epsilon, seed):
    """Apply a random perturbation to parameters using a specific seed."""
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    for p in param_list:
        probe = torch.randn_like(p)
        p.data.add_(probe, alpha=epsilon)

    torch.set_rng_state(rng_state)


def distributed_spsa_step(model, embed, x_ids, y_ids, pad_id, learning_rate, epsilon, rank, world_size):
    """
    Perform one distributed SPSA step.
    - Each rank computes one perturbation
    - All ranks reduce gradients to rank 0
    - Only rank 0 updates parameters
    """
    param_list = list(model.parameters()) + list(embed.parameters())
    device = param_list[0].device

    # 1. Broadcast parameters from rank 0 to all ranks
    if world_size > 1:
        for p in param_list:
            dist.broadcast(p.data, src=0)

    # 2. Generate and scatter seeds
    if rank == 0:
        seeds = torch.randint(0, 2**31 - 1, (world_size,), dtype=torch.int32, device=device)
    else:
        seeds = torch.zeros(world_size, dtype=torch.int32, device=device)

    if world_size > 1:
        dist.broadcast(seeds, src=0)

    local_seed = int(seeds[rank].item())

    # 3. Each rank computes its perturbation
    # Apply +ε perturbation
    apply_probe(param_list, epsilon, local_seed)

    with torch.no_grad():
        x_emb = embed(x_ids)
        logits_plus, _, _ = model(x_emb, require_gradients=False)
        loss_plus = compute_reverse_loss(logits_plus, y_ids, pad_id).item()

    # Apply -2ε to get -ε
    apply_probe(param_list, -2.0 * epsilon, local_seed)

    with torch.no_grad():
        x_emb = embed(x_ids)
        logits_minus, _, _ = model(x_emb, require_gradients=False)
        loss_minus = compute_reverse_loss(logits_minus, y_ids, pad_id).item()

    # Restore to original (+ε again)
    apply_probe(param_list, epsilon, local_seed)

    # 4. Compute gradient coefficient
    coef = (loss_plus - loss_minus) / (2.0 * world_size)

    # 5. Apply weighted perturbation
    apply_probe(param_list, -coef, local_seed)

    # 6. Reduce all parameters to rank 0
    if world_size > 1:
        for p in param_list:
            dist.reduce(p.data, dst=0, op=dist.ReduceOp.SUM)

    # 7. Return average loss for logging (only rank 0 needs this)
    current_loss = (loss_plus + loss_minus) / 2.0
    return current_loss


def train_with_early_stop_distributed(batch_size, num_gpus, learning_rate, vocab_size,
                                        min_tokens, max_tokens, max_time, convergence_loss,
                                        rank, world_size, device, seed,
                                        hidden_size=128, num_heads=4):
    """
    Distributed training with early stopping.
    IMPORTANT: epsilon is always set equal to learning_rate.
    """
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    # Always set epsilon = learning_rate
    epsilon = learning_rate

    PAD = vocab_size - 1

    embed = nn.Embedding(vocab_size, hidden_size, device=device, dtype=torch.bfloat16)
    model = LSTM(
        input_size=hidden_size,
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
            learning_rate, epsilon, rank, world_size
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
                                        hidden_size=128, num_heads=4):
    """
    Adaptive LR search for a specific max_tokens value (distributed version).
    Only rank 0 does I/O and logging.
    """
    convergence_loss = 0.1

    # LR values from high to low
    learning_rates = 10 ** np.arange(-0.6, -2 - 0.6, -0.2)  # 10^-0.6 down to 10^-2.6

    # Start with generous max_time
    initial_max_time = 60.0
    max_time = initial_max_time
    HARD_CAP_TIME = 60.0

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

        try:
            result = train_with_early_stop_distributed(
                batch_size=batch_size,
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
                num_heads=num_heads
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
                append_to_csv('losses_fast_dist.csv', csv_row)

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
                append_to_csv('losses_fast_dist.csv', csv_row)

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

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device
    local_rank = args.local_rank if args.local_rank != -1 else args.local__rank if hasattr(args, 'local__rank') else rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # Fixed parameters
    batch_size = 16
    perturbations = world_size  # 1 perturbation per GPU
    num_gpus = world_size
    vocab_size = 64
    min_tokens = 5

    # Test all max_tokens values
    max_tokens_values = 10 ** np.arange(1, 3 + 0.2, 0.2)  # 10^1 to 10^3

    hidden_size = 128
    num_heads = 4
    base_seed = 42

    if rank == 0:
        print("="*80)
        print("DISTRIBUTED FASTER GRID SEARCH")
        print("="*80)
        print(f"World size: {world_size} GPUs")
        print(f"Fixed: batch_size={batch_size}, perturbations={perturbations} (1 per GPU)")
        print(f"Testing {len(max_tokens_values)} max_tokens values")
        print(f"Output: losses_fast_dist.csv")
        print("="*80)
        print()

    for mt_idx, max_tok in enumerate(max_tokens_values):
        if rank == 0:
            print(f"\n[{mt_idx+1}/{len(max_tokens_values)}] MAX_TOKENS={int(max_tok)}")
            print("-"*80)

        results = adaptive_lr_search_for_max_tokens(
            max_tokens=max_tok,
            batch_size=batch_size,
            perturbations=perturbations,
            vocab_size=vocab_size,
            min_tokens=min_tokens,
            num_gpus=num_gpus,
            rank=rank,
            world_size=world_size,
            device=device,
            base_seed=base_seed + mt_idx * 1000,
            hidden_size=hidden_size,
            num_heads=num_heads
        )

        if rank == 0:
            # Summary
            converged_results = [r for r in results if r['converged']]
            print(f"\n  Summary for max_tokens={int(max_tok)}:")
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
        print(f"Results saved to: losses_fast_dist.csv")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
