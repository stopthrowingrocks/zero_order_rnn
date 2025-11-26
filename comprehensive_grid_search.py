#!/usr/bin/env python3
"""
Comprehensive grid search over SPSA hyperparameters with:
- wandb logging for each run
- Streaming CSV output (appends after each run completes)
- Early termination if loss increases by 10x
- 5 second time cap per experiment
"""
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import wandb
from itertools import product
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
                          min_tokens, max_tokens, max_time, device, seed,
                          hidden_size=128, num_heads=4, epsilon=0.1):
    """
    Train SPSA with early stopping if loss increases by 10x from initial loss.
    Returns: dict with final_loss, steps, elapsed_time, terminated_early
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

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

        # Early termination if loss increases by 10x
        if loss > 10.0 * initial_loss:
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
        'terminated_early': terminated_early
    }


def append_to_csv(filename, row_dict):
    """Append a single row to CSV file (creates file with header if doesn't exist)."""
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as f:
        fieldnames = ['batch_size', 'perturbations', 'num_gpus', 'learning_rate',
                     'vocab_size', 'min_tokens', 'max_tokens', 'seed',
                     'final_loss', 'steps', 'elapsed_time', 'optalg']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_dict)


def main():
    # Define parameter grid
    batch_sizes = [16, 32]
    perturbations_values = [8, 16, 24, 32]
    num_gpus = 8  # Fixed
    learning_rates = 10 ** np.arange(-2, 0 + 0.2, 0.2)  # 10^-2 to 10^0 in steps of 0.2
    vocab_size = 64  # Fixed
    min_tokens = 5  # Fixed
    max_tokens_values = 10 ** np.arange(1, 3 + 0.2, 0.2)  # 10^1 to 10^3 in steps of 0.2

    max_time = 5.0  # 5 seconds per experiment
    device = 'cuda'
    epsilon = 0.1
    hidden_size = 128
    num_heads = 4

    # Fixed seed for reproducibility (can be changed or randomized per run)
    base_seed = 42

    csv_filename = 'losses.csv'

    # Generate all combinations
    param_combinations = list(product(
        batch_sizes,
        perturbations_values,
        [num_gpus],
        learning_rates,
        [vocab_size],
        [min_tokens],
        max_tokens_values
    ))

    total_runs = len(param_combinations)
    print(f"="*80)
    print(f"COMPREHENSIVE GRID SEARCH")
    print(f"="*80)
    print(f"Total parameter combinations: {total_runs}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Perturbations: {perturbations_values}")
    print(f"Learning rates: {len(learning_rates)} values from {learning_rates[0]:.4f} to {learning_rates[-1]:.4f}")
    print(f"Max tokens: {len(max_tokens_values)} values from {int(max_tokens_values[0])} to {int(max_tokens_values[-1])}")
    print(f"Max time per run: {max_time}s")
    print(f"Early termination: Loss increases by 10x")
    print(f"Output: {csv_filename}")
    print(f"="*80)
    print()

    for idx, (batch_size, perturbations, gpus, lr, vocab, min_tok, max_tok) in enumerate(param_combinations):
        # Convert max_tok to int
        max_tok_int = int(round(max_tok))

        # Generate unique seed for this run
        seed = base_seed + idx

        # Create wandb run name
        run_name = (f"train_batch_{batch_size}_perturbations_{perturbations}_gpus_{gpus}_"
                   f"lr_{lr:.6f}_vocab_{vocab}_min_{min_tok}_max_{max_tok_int}")

        print(f"[{idx+1}/{total_runs}] {run_name}")

        # Initialize wandb
        wandb.init(
            project="zero-order-rnn-grid-search",
            name=run_name,
            config={
                'batch_size': batch_size,
                'perturbations': perturbations,
                'num_gpus': gpus,
                'learning_rate': lr,
                'vocab_size': vocab,
                'min_tokens': min_tok,
                'max_tokens': max_tok_int,
                'seed': seed,
                'epsilon': epsilon,
                'hidden_size': hidden_size,
                'num_heads': num_heads,
                'optalg': 'SPSA'
            },
            reinit=True
        )

        try:
            # Run training
            result = train_with_early_stop(
                batch_size=batch_size,
                perturbations=perturbations,
                num_gpus=gpus,
                learning_rate=lr,
                vocab_size=vocab,
                min_tokens=min_tok,
                max_tokens=max_tok_int,
                max_time=max_time,
                device=device,
                seed=seed,
                hidden_size=hidden_size,
                num_heads=num_heads,
                epsilon=epsilon
            )

            # Log final results to wandb
            wandb.log({
                'final_loss': result['final_loss'],
                'total_steps': result['steps'],
                'total_time': result['elapsed_time'],
                'terminated_early': result['terminated_early']
            })

            # Prepare CSV row
            csv_row = {
                'batch_size': batch_size,
                'perturbations': perturbations,
                'num_gpus': gpus,
                'learning_rate': lr,
                'vocab_size': vocab,
                'min_tokens': min_tok,
                'max_tokens': max_tok_int,
                'seed': seed,
                'final_loss': result['final_loss'],
                'steps': result['steps'],
                'elapsed_time': result['elapsed_time'],
                'optalg': 'SPSA'
            }

            # Append to CSV immediately
            append_to_csv(csv_filename, csv_row)

            print(f"  → Loss: {result['final_loss']:.4f}, Steps: {result['steps']}, "
                  f"Time: {result['elapsed_time']:.2f}s, Early stop: {result['terminated_early']}")

        except Exception as e:
            print(f"  → ERROR: {e}")

            # Log error to CSV with NaN values
            csv_row = {
                'batch_size': batch_size,
                'perturbations': perturbations,
                'num_gpus': gpus,
                'learning_rate': lr,
                'vocab_size': vocab,
                'min_tokens': min_tok,
                'max_tokens': max_tok_int,
                'seed': seed,
                'final_loss': float('nan'),
                'steps': 0,
                'elapsed_time': 0.0,
                'optalg': 'SPSA'
            }
            append_to_csv(csv_filename, csv_row)

        finally:
            wandb.finish()

    print()
    print(f"="*80)
    print(f"GRID SEARCH COMPLETE")
    print(f"="*80)
    print(f"Results saved to: {csv_filename}")
    print(f"Total runs: {total_runs}")


if __name__ == '__main__':
    main()
