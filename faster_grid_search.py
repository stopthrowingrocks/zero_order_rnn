#!/usr/bin/env python3
"""
Faster grid search with adaptive timing strategy:
- Early stopping at 3x loss increase (instead of 10x)
- Adaptive max_time: starts high, doubles first convergence time
- LR search from high to low, stops 2 LRs before first convergence
- Tests all max_tokens with fixed batch_size=16, perturbations=8
"""
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
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


def zeroth_order_step(model, embed, x_ids, y_ids, pad_id, learning_rate, epsilon, num_perturbations):
    """Perform a single zero-order optimization step using central difference."""
    with torch.no_grad():
        x_emb = embed(x_ids)
        logits, _, _ = model(x_emb, require_gradients=False)
        current_loss = compute_reverse_loss(logits, y_ids, pad_id).item()

    param_list = list(model.parameters()) + list(embed.parameters())
    pseudo_gradient = [torch.zeros_like(p) for p in param_list]

    for _ in range(num_perturbations):
        perturbations = [torch.randn_like(p) for p in param_list]

        with torch.no_grad():
            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=epsilon)

            x_emb = embed(x_ids)
            logits_plus, _, _ = model(x_emb, require_gradients=False)
            loss_plus = compute_reverse_loss(logits_plus, y_ids, pad_id).item()

            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=-2 * epsilon)

            x_emb = embed(x_ids)
            logits_minus, _, _ = model(x_emb, require_gradients=False)
            loss_minus = compute_reverse_loss(logits_minus, y_ids, pad_id).item()

            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=epsilon)

        grad_estimate = (loss_plus - loss_minus) / (2 * epsilon)
        for i, pert in enumerate(perturbations):
            pseudo_gradient[i].add_(pert, alpha=grad_estimate)

    for pg in pseudo_gradient:
        pg.div_(num_perturbations)

    with torch.no_grad():
        for p, pg in zip(param_list, pseudo_gradient):
            p.add_(pg, alpha=-learning_rate)

    return current_loss


def train_with_early_stop(batch_size, perturbations, num_gpus, learning_rate, vocab_size,
                          min_tokens, max_tokens, max_time, convergence_loss, device, seed,
                          hidden_size=128, num_heads=4):
    """
    Train SPSA with early stopping if loss increases by 3x from initial loss.
    IMPORTANT: epsilon is always set equal to learning_rate.
    Returns: dict with final_loss, steps, elapsed_time, terminated_early, converged
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Always set epsilon = learning_rate
    epsilon = learning_rate

    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    PAD = vocab_size - 1

    embed = nn.Embedding(vocab_size, hidden_size, device=device_obj, dtype=torch.bfloat16)
    model = LSTM(
        input_size=hidden_size,
        output_size=vocab_size,
        hidden_size=hidden_size,
        memory_size=0,
        head_size=hidden_size // num_heads,
        num_heads=num_heads,
        embed=embed,
        device=device_obj,
        dtype=torch.bfloat16
    )

    start_time = time.time()
    step = 0
    losses = []
    initial_loss = None
    terminated_early = False
    converged = False

    while time.time() - start_time < max_time:
        x_ids, y_ids = generate_reverse_batch(
            batch_size, min_tokens, max_tokens, vocab_size, device_obj
        )
        loss = zeroth_order_step(
            model, embed, x_ids, y_ids, PAD,
            learning_rate, epsilon, perturbations
        )
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

    elapsed = time.time() - start_time
    final_loss = losses[-1] if losses else float('inf')

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
                                       min_tokens, num_gpus, device, base_seed,
                                       hidden_size=128, num_heads=4):
    """
    Adaptive LR search for a specific max_tokens value:
    1. Start with high LR, search downwards
    2. First convergence: double that time, set as max_time
    3. Stop search 2 LRs before first convergence

    Returns: list of results dicts
    """
    convergence_loss = 0.1

    # LR values from high to low (reversed from original)
    learning_rates = 10 ** np.arange(-0.6, -2 - 0.6, -0.2)  # 10^0 down to 10^-2

    # Start with generous max_time, will be updated after first convergence
    # Hard cap at 60 seconds per run
    initial_max_time = 60.0
    max_time = initial_max_time
    HARD_CAP_TIME = 60.0

    results = []
    first_convergence_idx = None
    first_convergence_time = None

    for lr_idx, lr in enumerate(learning_rates):
        # If we've found convergence and are 2+ LRs past it, stop
        if first_convergence_idx is not None and lr_idx >= first_convergence_idx + 2:
            print(f"  Stopping LR search (2 LRs past first convergence)")
            break

        seed = base_seed + lr_idx

        # Create wandb run name
        run_name = (f"fast_batch_{batch_size}_pert_{perturbations}_gpus_{num_gpus}_"
                   f"lr_{lr:.6f}_vocab_{vocab_size}_min_{min_tokens}_max_{int(max_tokens)}")

        print(f"    LR={lr:.6f} (max_time={max_time:.1f}s)...", end=" ", flush=True)

        # Initialize wandb
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
                'optalg': 'SPSA_adaptive'
            },
            reinit=True
        )

        try:
            result = train_with_early_stop(
                batch_size=batch_size,
                perturbations=perturbations,
                num_gpus=num_gpus,
                learning_rate=lr,
                vocab_size=vocab_size,
                min_tokens=min_tokens,
                max_tokens=int(max_tokens),
                max_time=max_time,
                convergence_loss=convergence_loss,
                device=device,
                seed=seed,
                hidden_size=hidden_size,
                num_heads=num_heads
            )

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
                # Double the time for remaining searches, but hard cap at 30s
                max_time = min(first_convergence_time * 2.0, HARD_CAP_TIME)
                print(f"CONVERGED in {result['elapsed_time']:.2f}s! Setting max_time={max_time:.1f}s")
            elif result['converged']:
                print(f"Converged in {result['elapsed_time']:.2f}s")
            elif result['terminated_early']:
                print(f"Terminated early (3x loss increase)")
            else:
                print(f"Did not converge (loss={result['final_loss']:.4f})")

            # Prepare and append CSV row
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
                'optalg': 'SPSA_adaptive'
            }
            append_to_csv('losses_fast.csv', csv_row)

        except Exception as e:
            print(f"ERROR: {e}")

            # Log error to CSV with NaN values
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
                'optalg': 'SPSA_adaptive'
            }
            append_to_csv('losses_fast.csv', csv_row)

        finally:
            wandb.finish()

    return results


def main():
    # Fixed parameters
    batch_size = 16
    perturbations = 8
    num_gpus = 8
    vocab_size = 64
    min_tokens = 5

    # Test all max_tokens values
    max_tokens_values = 10 ** np.arange(1, 3 + 0.2, 0.2)  # 10^1 to 10^3

    device = 'cuda'
    hidden_size = 128
    num_heads = 4
    base_seed = 42

    print("="*80)
    print("FASTER ADAPTIVE GRID SEARCH")
    print("="*80)
    print(f"Fixed: batch_size={batch_size}, perturbations={perturbations}")
    print(f"Testing {len(max_tokens_values)} max_tokens values: {[int(x) for x in max_tokens_values]}")
    print(f"Strategy:")
    print(f"  - LR search: high to low (10^0 down to 10^-2)")
    print(f"  - Early stop: 3x loss increase from initial")
    print(f"  - Adaptive timing: double first convergence time")
    print(f"  - Stop: 2 LRs after first convergence")
    print(f"Output: losses_fast.csv")
    print("="*80)
    print()

    for mt_idx, max_tok in enumerate(max_tokens_values):
        print(f"\n[{mt_idx+1}/{len(max_tokens_values)}] MAX_TOKENS={int(max_tok)}")
        print("-"*80)

        results = adaptive_lr_search_for_max_tokens(
            max_tokens=max_tok,
            batch_size=batch_size,
            perturbations=perturbations,
            vocab_size=vocab_size,
            min_tokens=min_tokens,
            num_gpus=num_gpus,
            device=device,
            base_seed=base_seed + mt_idx * 1000,
            hidden_size=hidden_size,
            num_heads=num_heads
        )

        # Summary for this max_tokens
        converged_results = [r for r in results if r['converged']]
        print(f"\n  Summary for max_tokens={int(max_tok)}:")
        print(f"    Tested {len(results)} LR values")
        print(f"    Converged: {len(converged_results)}/{len(results)}")
        if converged_results:
            best = min(converged_results, key=lambda r: r['elapsed_time'])
            print(f"    Best: LR={best['lr']:.6f} in {best['elapsed_time']:.2f}s")

    print()
    print("="*80)
    print("FASTER GRID SEARCH COMPLETE")
    print("="*80)
    print(f"Results saved to: losses_fast.csv")


if __name__ == '__main__':
    main()
