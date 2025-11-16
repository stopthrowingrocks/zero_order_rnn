#!/usr/bin/env python3
"""
Run 20 SPSA trials for exactly 5 seconds each and record final losses.
Uses the best hyperparameters found: LR=epsilon=0.1, perturbations=8, batch=16
"""
import time
import csv
import numpy as np
import torch
import torch.nn as nn
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


def train_spsa_for_5_seconds(vocab_size, min_seq, max_seq, hidden_size, num_heads,
                              learning_rate, epsilon, num_perturbations, batch_size,
                              device, seed):
    """Train SPSA for exactly 5 seconds and return final loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
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
    time_limit = 5.0  # seconds
    final_loss = None
    step = 0

    while time.time() - start_time < time_limit:
        x_ids, y_ids = generate_reverse_batch(batch_size, min_seq, max_seq, vocab_size, device)
        loss = zeroth_order_step(
            model, embed, x_ids, y_ids, PAD,
            learning_rate, epsilon, num_perturbations
        )
        final_loss = loss
        step += 1

    elapsed = time.time() - start_time
    return final_loss, step, elapsed


def main():
    # Fixed hyperparameters
    vocab_size = 20
    min_seq_length = 10
    max_seq_length = 100
    hidden_size = 128
    num_heads = 4
    device = 'cuda'

    # Best SPSA hyperparameters
    learning_rate = 0.1
    epsilon = 0.1
    num_perturbations = 8
    batch_size = 16

    num_trials = 20
    base_seed = 100

    print("=" * 80)
    print("SPSA 5-SECOND TRIALS")
    print("=" * 80)
    print(f"Task: Reverse sequences of length {min_seq_length}-{max_seq_length}")
    print(f"Vocab size: {vocab_size}, Hidden size: {hidden_size}")
    print(f"SPSA: LR={learning_rate}, Epsilon={epsilon}, Perturbations={num_perturbations}, Batch={batch_size}")
    print(f"Running {num_trials} trials, each for exactly 5 seconds")
    print("=" * 80)
    print()

    results = []

    for trial in range(num_trials):
        seed = base_seed + trial
        print(f"Trial {trial + 1}/{num_trials} (seed={seed})...", end=" ", flush=True)

        final_loss, steps, elapsed = train_spsa_for_5_seconds(
            vocab_size, min_seq_length, max_seq_length, hidden_size, num_heads,
            learning_rate, epsilon, num_perturbations, batch_size, device, seed
        )

        results.append({
            'trial': trial + 1,
            'seed': seed,
            'final_loss': final_loss,
            'steps': steps,
            'elapsed_time': elapsed
        })

        print(f"Final loss: {final_loss:.4f}, Steps: {steps}, Time: {elapsed:.2f}s")

    # Save to CSV
    csv_filename = 'losses.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['trial', 'seed', 'final_loss', 'steps', 'elapsed_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    losses = [r['final_loss'] for r in results]
    steps_list = [r['steps'] for r in results]

    print(f"Final losses after 5 seconds:")
    print(f"  Mean: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print(f"  Min:  {np.min(losses):.4f}")
    print(f"  Max:  {np.max(losses):.4f}")
    print(f"  Median: {np.median(losses):.4f}")
    print()
    print(f"Steps completed in 5 seconds:")
    print(f"  Mean: {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")
    print(f"  Min:  {np.min(steps_list)}")
    print(f"  Max:  {np.max(steps_list)}")
    print()
    print(f"Results saved to {csv_filename}")
    print("=" * 80)


if __name__ == '__main__':
    main()
