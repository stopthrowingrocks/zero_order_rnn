#!/usr/bin/env python3
"""
Hyperparameter sweep for SPSA optimizer on reverse task.
Outputs results to sweep_hpps_spsa_results.csv and hpps_spsa.json (best config).
"""
import argparse
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from models.models import LSTM
from shared import generate_reverse_batch, compute_reverse_loss, compute_reverse_accuracy


def teacher_forcing_loss_for_spsa(model, x_ids, y_ids_unpadded, criterion, chunk_size=32):
    """
    Teacher forcing loss computation for SPSA (no gradients needed).
    Processes input and target sequences in chunks to compute loss over all output tokens.
    """
    with torch.no_grad():
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
            out_chunk, mem_new, hidden_new = model(input_chunk, hidden=hidden, memory=memory, require_gradients=False)
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

            out_chunk, mem_new, hidden_new = model(y_emb_chunk, hidden=hidden, memory=memory, require_gradients=False)

            # Update states
            hidden = hidden_new
            memory = mem_new

            # Compute loss for this chunk
            out_chunk = out_chunk.reshape(-1, out_chunk.size(-1))
            targets = y_ids_unpadded[:, pos+1:chunk_end+1].reshape(-1)  # shift by 1 for next-token prediction

            if targets.size(0) > 0:  # ensure we have targets
                chunk_loss = criterion(out_chunk, targets)
                total_loss += chunk_loss * targets.size(0)
                total_predicted_tokens += targets.size(0)

            pos = chunk_end

        if total_predicted_tokens == 0:
            return 0.0

        avg_loss = total_loss / total_predicted_tokens
        return avg_loss.item()


def spsa_step(model, x_ids, y_ids, criterion, learning_rate, epsilon, num_perturbations):
    """Perform a single SPSA step using central difference."""
    # Get current loss
    current_loss = teacher_forcing_loss_for_spsa(model, x_ids, y_ids, criterion)

    # Collect all parameters
    param_list = list(model.parameters()) + list(model.embed.parameters())

    # Estimate gradient using random perturbations
    pseudo_gradient = [torch.zeros_like(p) for p in param_list]

    for _ in range(num_perturbations):
        # Generate random direction
        perturbations = [torch.randn_like(p) for p in param_list]

        # Positive perturbation
        with torch.no_grad():
            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=epsilon)

        loss_plus = teacher_forcing_loss_for_spsa(model, x_ids, y_ids, criterion)

        # Restore and apply negative perturbation
        with torch.no_grad():
            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=-2 * epsilon)

        loss_minus = teacher_forcing_loss_for_spsa(model, x_ids, y_ids, criterion)

        # Restore parameters
        with torch.no_grad():
            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=epsilon)

        # Accumulate gradient estimate
        grad_estimate = (loss_plus - loss_minus) / (2 * epsilon)
        for i, pert in enumerate(perturbations):
            pseudo_gradient[i].add_(pert, alpha=grad_estimate)

    # Average over perturbations
    for pg in pseudo_gradient:
        pg.div_(num_perturbations)

    # Update parameters
    with torch.no_grad():
        for p, pg in zip(param_list, pseudo_gradient):
            p.add_(pg, alpha=-learning_rate)

    return current_loss


def test_spsa_config(
    args, device
):
    """Test a single SPSA hyperparameter configuration with early stopping."""
    import time

    epsilon = args.learning_rate

    SEP = args.vocab_size - 3
    PAD = args.vocab_size - 1

    # Create fresh model
    embed = nn.Embedding(args.vocab_size, args.input_size, device=device, dtype=torch.bfloat16)
    model = LSTM(
        input_size=args.input_size,
        output_size=args.vocab_size,
        hidden_size=args.hidden_size,
        memory_size=0,
        head_size=args.hidden_size // args.num_heads,
        num_heads=args.num_heads,
        embed=embed,
        device=device,
        dtype=torch.bfloat16
    )

    # Training loop
    losses = []
    accuracies = []
    start_time = time.time()
    converged = False

    step = 0
    while time.time() - start_time < args.max_time:
        # Generate batch
        x_ids, y_ids = generate_reverse_batch(args.batch_size, args.min_seq_length, args.max_seq_length, args.vocab_size, device)

        # SPSA optimization step
        loss = spsa_step(
            model, embed, x_ids, y_ids, PAD,
            args.learning_rate, epsilon, args.num_perturbations
        )

        losses.append(loss)
        step += 1

        if loss >= args.max_loss:
            break

        # Check for convergence: loss below threshold
        if loss < args.convergence_loss:
            converged = True
            break

        # Compute accuracy on current batch periodically for logging
        if step % 50 == 0:
            with torch.no_grad():
                x_emb = embed(x_ids)
                logits, _, _ = model(x_emb, require_gradients=False)
                accuracy = compute_reverse_accuracy(logits, y_ids, SEP, PAD)
                accuracies.append(accuracy)

    x_ids, y_ids = generate_reverse_batch(args.batch_size, args.min_seq_length, args.max_seq_length, args.vocab_size, device)
    
    
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
    parser.add_argument('--max_loss', type=float, default=10.0, help='Maximum loss before quitting run')
    parser.add_argument('--convergence_loss', type=int, default=0.1, help='When this loss is reached, training stops')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("HYPERPARAMETER SWEEP: SPSA")
    print("=" * 80)
    print(f"Task: Reverse sequences of length {args.min_seq_length} -> {args.max_seq_length}")
    print(f"Vocab size: {args.vocab_size}, Hidden size: {args.hidden_size}")
    print(f"Max time per config: {args.max_time}s")
    print("=" * 80)
    print()

    # SPSA hyperparameter grid
    print("\n" + "=" * 80)
    print("SPSA SWEEP")
    print("=" * 80)

    # Generate hyperparameter grid
    learning_rates = 0.2 * 0.5 ** (np.arange(0, 7) / 3)
    batch_sizes = [32]
    perturbations_list = [4, 8, 12, 16, 24, 32, 48]

    # Generate all combinations
    spsa_configs = []
    for lr in learning_rates:
        for bs in batch_sizes:
            for n_pert in perturbations_list:
                spsa_configs.append((lr, n_pert, bs))

    print(f"Total SPSA configurations: {len(spsa_configs)}")
    print(f"  Learning rates: {len(learning_rates)} values from {learning_rates[0]:.6f} to {learning_rates[-1]:.6f}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Perturbations: {perturbations_list}")
    print()

    spsa_results = []
    csv_filename = 'sweep_hpps_spsa_results.csv'

    # Initialize CSV file with header
    with open(csv_filename, 'w', newline='') as f:
        fieldnames = ['lr', 'num_perturbations', 'batch_size', 'converged', 'elapsed_time',
                     'num_steps', 'initial_loss', 'final_loss', 'min_loss', 'improvement', 'final_acc']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for config_idx, (lr, n_pert, bs) in enumerate(spsa_configs, 1):
        print(f"[{config_idx}/{len(spsa_configs)}] Testing SPSA: LR={lr:.6f}, Pert={n_pert}, Batch={bs}")

        args.learning_rate = lr
        args.num_perturbations = n_pert
        args.batch_size = bs
        losses, accuracies, converged, elapsed_time, num_steps = test_spsa_config(args, device)

        # Analyze results
        initial_loss = float(np.mean(losses[:10]) if len(losses) >= 10 else np.mean(losses))
        final_loss = float(np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses))
        min_loss = float(min(losses)) if losses else float('inf')
        loss_improvement = initial_loss - final_loss
        final_acc = float(accuracies[-1] if accuracies else 0.0)

        # Determine status
        timed_out = elapsed_time >= args.max_time - 0.01  # Small tolerance for timing
        if converged:
            status = "CONVERGED"
        elif timed_out:
            status = "TIMEOUT (not converged)"
        else:
            status = "STOPPED (loss >= max_loss)"

        print(f"  {status} in {elapsed_time:.2f}s ({num_steps} steps) | "
              f"Final: {final_loss:.4f}, Min: {min_loss:.4f}, Acc: {final_acc:.4f}")

        # Write result to CSV incrementally
        with open(csv_filename, 'a', newline='') as f:
            fieldnames = ['lr', 'num_perturbations', 'batch_size', 'converged', 'elapsed_time',
                         'num_steps', 'initial_loss', 'final_loss', 'min_loss', 'improvement', 'final_acc']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            row = {
                'lr': lr,
                'num_perturbations': n_pert,
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

        spsa_results.append({
            'lr': lr,
            'num_perturbations': n_pert,
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

    print(f"\n✓ SPSA results saved to {csv_filename}")

    # Find fastest converging configuration
    converged_results = [r for r in spsa_results if r['converged']]
    if converged_results:
        best_result = min(converged_results, key=lambda r: r['elapsed_time'])

        # Save best configuration to JSON
        with open('hpps_spsa.json', 'w') as f:
            json.dump(best_result, f, indent=2)

        print(f"\n✓ Best SPSA config saved to hpps_spsa.json")
        print(f"  LR={best_result['lr']:.6f}, Pert={best_result['num_perturbations']}, "
              f"Batch={best_result['batch_size']}")
        print(f"  Converged in {best_result['elapsed_time']:.2f}s ({best_result['num_steps']} steps)")
    else:
        print(f"\n⚠ No SPSA configurations converged - no hpps_spsa.json written")

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
