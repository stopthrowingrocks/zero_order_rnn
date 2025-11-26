#!/usr/bin/env python3
"""
Faster grid search with parallel GPU utilization:
- Runs 8 learning rate experiments concurrently, one per GPU
- After all 8 complete, moves to next batch of 8 learning rates
- Much faster than sequential processing
"""
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
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


def train_single_lr(gpu_id, lr, max_tokens, batch_size, perturbations, vocab_size,
                     min_tokens, num_gpus, max_time, convergence_loss, seed,
                     hidden_size, num_heads, result_queue):
    """
    Train with a single learning rate on a specific GPU.
    Results are put into result_queue for collection.
    """
    try:
        # Set GPU device
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Always set epsilon = learning_rate
        epsilon = lr

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

        # Initialize wandb
        run_name = (f"fast_batch_{batch_size}_pert_{perturbations}_gpus_{num_gpus}_"
                   f"lr_{lr:.6f}_vocab_{vocab_size}_min_{min_tokens}_max_{int(max_tokens)}_gpu{gpu_id}")

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
                'optalg': 'SPSA_parallel',
                'gpu_id': gpu_id
            },
            reinit=True
        )

        start_time = time.time()
        step = 0
        losses = []
        initial_loss = None
        terminated_early = False
        converged = False

        while time.time() - start_time < max_time:
            x_ids, y_ids = generate_reverse_batch(
                batch_size, min_tokens, int(max_tokens), vocab_size, device
            )
            loss = zeroth_order_step(
                model, embed, x_ids, y_ids, PAD,
                lr, epsilon, perturbations
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

        # Log final results to wandb
        wandb.log({
            'final_loss': final_loss,
            'total_steps': step,
            'total_time': elapsed,
            'terminated_early': terminated_early,
            'converged': converged
        })

        wandb.finish()

        result = {
            'lr': lr,
            'max_tokens': int(max_tokens),
            'batch_size': batch_size,
            'perturbations': perturbations,
            'final_loss': final_loss,
            'steps': step,
            'elapsed_time': elapsed,
            'terminated_early': terminated_early,
            'converged': converged,
            'gpu_id': gpu_id
        }

        result_queue.put(result)

    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put({
            'lr': lr,
            'max_tokens': int(max_tokens),
            'batch_size': batch_size,
            'perturbations': perturbations,
            'final_loss': float('nan'),
            'steps': 0,
            'elapsed_time': 0.0,
            'terminated_early': False,
            'converged': False,
            'gpu_id': gpu_id,
            'error': str(e)
        })


def append_to_csv(filename, row_dict):
    """Append a single row to CSV file (creates file with header if doesn't exist)."""
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as f:
        fieldnames = ['batch_size', 'perturbations', 'num_gpus', 'learning_rate',
                     'vocab_size', 'min_tokens', 'max_tokens', 'seed',
                     'final_loss', 'steps', 'elapsed_time', 'converged',
                     'terminated_early', 'max_time_used', 'optalg', 'gpu_id']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_dict)


def main():
    mp.set_start_method('spawn', force=True)

    # Fixed parameters
    batch_size = 16
    perturbations = 8
    num_gpus = 8
    vocab_size = 64
    min_tokens = 5

    # Test all max_tokens values
    max_tokens_values = 10 ** np.arange(1, 3 + 0.2, 0.2)  # 10^1 to 10^3

    # Learning rates
    learning_rates = 10 ** np.arange(-0.6, -2 - 0.6, -0.2)  # 10^-0.6 down to 10^-2.6

    device = 'cuda'
    hidden_size = 128
    num_heads = 4
    base_seed = 42
    convergence_loss = 0.1
    max_time = 60.0

    print("="*80)
    print("PARALLEL FASTER GRID SEARCH")
    print("="*80)
    print(f"Using {num_gpus} GPUs in parallel")
    print(f"Fixed: batch_size={batch_size}, perturbations={perturbations}")
    print(f"Testing {len(max_tokens_values)} max_tokens values")
    print(f"Testing {len(learning_rates)} learning rates")
    print(f"Running {num_gpus} LRs concurrently per max_tokens")
    print(f"Output: losses_fast_parallel.csv")
    print("="*80)
    print()

    for mt_idx, max_tok in enumerate(max_tokens_values):
        print(f"\n[{mt_idx+1}/{len(max_tokens_values)}] MAX_TOKENS={int(max_tok)}")
        print("-"*80)

        # Process learning rates in batches of num_gpus
        for batch_start in range(0, len(learning_rates), num_gpus):
            batch_end = min(batch_start + num_gpus, len(learning_rates))
            lr_batch = learning_rates[batch_start:batch_end]

            print(f"\n  Processing LRs {batch_start+1}-{batch_end} / {len(learning_rates)}")

            # Create result queue for collecting results
            result_queue = mp.Queue()

            # Launch processes for this batch
            processes = []
            for i, lr in enumerate(lr_batch):
                gpu_id = i
                seed = base_seed + mt_idx * 1000 + batch_start + i

                p = mp.Process(
                    target=train_single_lr,
                    args=(gpu_id, lr, max_tok, batch_size, perturbations, vocab_size,
                          min_tokens, num_gpus, max_time, convergence_loss, seed,
                          hidden_size, num_heads, result_queue)
                )
                p.start()
                processes.append(p)
                print(f"    Launched GPU {gpu_id}: LR={lr:.6f}")

            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Collect results
            results = []
            while not result_queue.empty():
                result = result_queue.get()
                results.append(result)

            # Sort by GPU ID for consistent output
            results.sort(key=lambda x: x['gpu_id'])

            # Write results to CSV
            for result in results:
                csv_row = {
                    'batch_size': batch_size,
                    'perturbations': perturbations,
                    'num_gpus': num_gpus,
                    'learning_rate': result['lr'],
                    'vocab_size': vocab_size,
                    'min_tokens': min_tokens,
                    'max_tokens': result['max_tokens'],
                    'seed': base_seed + mt_idx * 1000 + result['gpu_id'],
                    'final_loss': result['final_loss'],
                    'steps': result['steps'],
                    'elapsed_time': result['elapsed_time'],
                    'converged': result['converged'],
                    'terminated_early': result['terminated_early'],
                    'max_time_used': max_time,
                    'optalg': 'SPSA_parallel',
                    'gpu_id': result['gpu_id']
                }
                append_to_csv('losses_fast_parallel.csv', csv_row)

                # Print status
                if result['converged']:
                    print(f"      GPU {result['gpu_id']} (LR={result['lr']:.6f}): CONVERGED in {result['elapsed_time']:.2f}s")
                elif result['terminated_early']:
                    print(f"      GPU {result['gpu_id']} (LR={result['lr']:.6f}): Terminated early (3x loss)")
                else:
                    print(f"      GPU {result['gpu_id']} (LR={result['lr']:.6f}): Did not converge (loss={result['final_loss']:.4f})")

        print(f"\n  Completed max_tokens={int(max_tok)}")

    print()
    print("="*80)
    print("PARALLEL GRID SEARCH COMPLETE")
    print("="*80)
    print(f"Results saved to: losses_fast_parallel.csv")


if __name__ == '__main__':
    main()
