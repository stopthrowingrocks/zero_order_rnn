#!/usr/bin/env python3
"""
Hyperparameter sweep for Adam optimizer on reverse task.
Outputs results to sweep_hpps_adam_results.csv and hpps_adam.json (best config).
"""
import argparse
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from models.models import LSTM
from shared import generate_reverse_batch, compute_reverse_loss, compute_reverse_accuracy


def teacher_forcing_loss_for_adam(model, x_ids, y_ids_unpadded, criterion, chunk_size=32):
    """
    Teacher forcing loss computation for Adam (requires gradients).
    Similar to teacher_forcing_loss_emb_parallel but keeps gradients.
    """
    x_emb = model.embed(x_ids)

    next_param = next(model.parameters())
    if x_emb.dtype != next_param.dtype:
        x_emb = x_emb.to(dtype=next_param.dtype)
    Lx = x_emb.shape[1]
    Ly = y_ids_unpadded.shape[1]

    hidden = None
    memory = None
    total_loss = 0.0
    total_predicted_tokens = 0

    # Process input sequence first
    pos = 0
    while pos < Lx:
        chunk_end = min(pos + chunk_size, Lx)
        input_chunk = x_emb[:, pos:chunk_end, :]
        out_chunk, mem_new, hidden_new = model(input_chunk, hidden=hidden, memory=memory, require_gradients=True)
        hidden = hidden_new
        memory = mem_new
        pos = chunk_end

    # Now process target sequence chunk by chunk
    pos = 0
    while pos < Ly - 1:  # -1 because we don't embed the last target token
        chunk_end = min(pos + chunk_size, Ly - 1)
        # Only embed the current chunk of target sequence
        y_chunk = y_ids_unpadded[:, pos:chunk_end]
        y_emb_chunk = model.embed(y_chunk)

        out_chunk, mem_new, hidden_new = model(y_emb_chunk, hidden=hidden, memory=memory, require_gradients=True)

        # Update states
        hidden = hidden_new
        memory = mem_new

        # Compute loss for this chunk (keep in float32 for gradients)
        out_chunk = out_chunk.reshape(-1, out_chunk.size(-1))
        targets = y_ids_unpadded[:, pos+1:chunk_end+1].reshape(-1)  # shift by 1 for next-token prediction

        if targets.size(0) > 0:  # ensure we have targets
            chunk_loss = criterion(out_chunk, targets)
            total_loss += chunk_loss * targets.size(0)
            total_predicted_tokens += targets.size(0)

        pos = chunk_end

    if total_predicted_tokens == 0:
        return torch.tensor(0.0, device=x_ids.device, requires_grad=True)

    avg_loss = total_loss / total_predicted_tokens
    return avg_loss


def adam_step(model, x_ids, y_ids, optimizer, criterion):
    """Perform a single Adam step with gradient computation."""
    optimizer.zero_grad()

    loss = teacher_forcing_loss_for_adam(model, x_ids, y_ids, criterion)
    loss.backward()
    optimizer.step()

    return loss.item()


def test_adam_config(
    vocab_size, min_seq_length, max_seq_length, hidden_size, input_size, num_heads,
    learning_rate, batch_size,
    max_time, convergence_loss, device
):
    """Test a single Adam hyperparameter configuration with early stopping."""
    import time

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    SEP = vocab_size - 3
    PAD = vocab_size - 1

    # Create fresh model
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

    # Create Adam optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(embed.parameters()),
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD).to(device)

    # Training loop
    losses = []
    accuracies = []
    start_time = time.time()
    converged = False

    step = 0
    while time.time() - start_time < max_time:
        # Generate batch
        x_ids, y_ids = generate_reverse_batch(batch_size, min_seq_length, max_seq_length, vocab_size, device)

        # Adam optimization step
        loss = adam_step(model, x_ids, y_ids, optimizer, criterion)

        losses.append(loss)
        step += 1

        # Check for convergence: loss below threshold
        if loss < convergence_loss:
            converged = True
            break

        # Compute accuracy on current batch periodically for logging
        if step % 50 == 0:
            with torch.no_grad():
                x_emb = embed(x_ids)
                logits, _, _ = model(x_emb, require_gradients=False)
                accuracy = compute_reverse_accuracy(logits, y_ids, SEP, PAD)
                accuracies.append(accuracy)

    elapsed_time = time.time() - start_time
    return losses, accuracies, converged, elapsed_time, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=64)
    parser.add_argument('--min_seq_length', type=int, default=5)
    parser.add_argument('--max_seq_length', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=240)
    parser.add_argument('--input_size', type=int, default=100)
    parser.add_argument('--num_heads', type=int, default=20)
    parser.add_argument('--max_time', type=float, default=30.0, help='Maximum time per configuration (seconds)')
    parser.add_argument('--convergence_loss', type=float, default=0.1, help='Loss threshold for convergence')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("HYPERPARAMETER SWEEP: ADAM")
    print("=" * 80)
    print(f"Task: Reverse sequences of length {args.min_seq_length}-{args.max_seq_length}")
    print(f"Vocab size: {args.vocab_size}, Hidden size: {args.hidden_size}")
    print(f"Max time per config: {args.max_time}s")
    print(f"Convergence criterion: loss < {args.convergence_loss}")
    print("=" * 80)
    print()

    # Adam hyperparameter grid
    print("\n" + "=" * 80)
    print("ADAM SWEEP")
    print("=" * 80)

    # Generate hyperparameter grid - same learning rates as SPSA
    learning_rates = 0.1 * 0.5 ** (np.arange(0, 7) / 2)

    # Batch sizes: base batch size (32) multiplied by SPSA perturbation values
    batch_sizes = [64]

    # Generate all combinations
    adam_configs = []
    for lr in learning_rates:
        for bs in batch_sizes:
            adam_configs.append((lr, bs))

    print(f"Total Adam configurations: {len(adam_configs)}")
    print(f"  Learning rates: {len(learning_rates)} values from {learning_rates[0]:.6f} to {learning_rates[-1]:.6f}")
    print(f"  Batch sizes: {batch_sizes}")
    print()

    adam_results = []
    csv_filename = 'sweep_hpps_adam_results.csv'

    # Initialize CSV file with header
    with open(csv_filename, 'w', newline='') as f:
        fieldnames = ['lr', 'batch_size', 'converged', 'elapsed_time',
                     'num_steps', 'initial_loss', 'final_loss', 'min_loss', 'improvement', 'final_acc']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for config_idx, (lr, bs) in enumerate(adam_configs, 1):
        print(f"[{config_idx}/{len(adam_configs)}] Testing Adam: LR={lr:.6f}, Batch={bs}")

        losses, accuracies, converged, elapsed_time, num_steps = test_adam_config(
            args.vocab_size, args.min_seq_length, args.max_seq_length, args.hidden_size, args.input_size, args.num_heads,
            lr, bs, args.max_time, args.convergence_loss, device
        )

        # Analyze results
        initial_loss = float(np.mean(losses[:10]) if len(losses) >= 10 else np.mean(losses))
        final_loss = float(losses[-1])
        min_loss = float(min(losses)) if losses else float('inf')
        loss_improvement = initial_loss - final_loss
        final_acc = float(accuracies[-1] if accuracies else 0.0)

        # Determine status
        timed_out = elapsed_time >= args.max_time - 0.01  # Small tolerance for timing
        status = "CONVERGED" if converged else "TIMEOUT (not converged)"

        print(f"  {status} in {elapsed_time:.2f}s ({num_steps} steps) | "
              f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}, Min: {min_loss:.4f}, Acc: {final_acc:.4f}")

        # Write result to CSV incrementally
        with open(csv_filename, 'a', newline='') as f:
            fieldnames = ['lr', 'batch_size', 'converged', 'elapsed_time',
                         'num_steps', 'initial_loss', 'final_loss', 'min_loss', 'improvement', 'final_acc']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            row = {
                'lr': lr,
                'batch_size': bs,
                'converged': converged,
                'elapsed_time': elapsed_time,
                'num_steps': num_steps,
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'min_loss': min_loss,
                'improvement': loss_improvement,
                'final_acc': final_acc
            }
            writer.writerow(row)

        adam_results.append({
            'lr': lr,
            'batch_size': bs,
            'converged': converged,
            'elapsed_time': elapsed_time,
            'num_steps': num_steps,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'min_loss': min_loss,
            'improvement': loss_improvement,
            'final_acc': final_acc,
            'losses': [float(l) for l in losses]
        })

    print(f"\n✓ Adam results saved to {csv_filename}")

    # Find fastest converging configuration
    converged_results = [r for r in adam_results if r['converged']]
    if converged_results:
        best_result = min(converged_results, key=lambda r: r['elapsed_time'])

        # Save best configuration to JSON
        with open('hpps_adam.json', 'w') as f:
            json.dump(best_result, f, indent=2)

        print(f"\n✓ Best Adam config saved to hpps_adam.json")
        print(f"  LR={best_result['lr']:.6f}, Batch={best_result['batch_size']}")
        print(f"  Converged in {best_result['elapsed_time']:.2f}s ({best_result['num_steps']} steps)")
    else:
        print(f"\n⚠ No Adam configurations converged - no hpps_adam.json written")

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
