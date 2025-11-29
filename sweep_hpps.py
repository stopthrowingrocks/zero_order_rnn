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


def generate_reverse_batch(batch_size, seq_length, vocab_size, device='cuda'):
    """Generate a batch of reverse sequences using raw integers."""
    BOS = vocab_size - 4
    SEP = vocab_size - 3
    EOS = vocab_size - 2
    PAD = vocab_size - 1
    content_vocab_size = vocab_size - 4

    x_batch = []
    y_batch = []

    for _ in range(batch_size):
        seq = np.random.randint(0, content_vocab_size, size=seq_length)
        full_seq = [BOS] + seq.tolist() + [SEP] + seq[::-1].tolist() + [EOS]
        x_seq = full_seq
        y_seq = full_seq

        x_batch.append(torch.tensor(x_seq, dtype=torch.long, device=device))
        y_batch.append(torch.tensor(y_seq, dtype=torch.long, device=device))

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


def compute_reverse_accuracy(logits, targets, sep_id, pad_id):
    """Compute accuracy only on the output part (after SEP token)."""
    B, T, V = logits.shape
    preds = logits.argmax(dim=-1)

    total_correct = 0
    total_tokens = 0

    for b in range(B):
        sep_positions = (targets[b] == sep_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) == 0:
            continue
        sep_pos = sep_positions[0].item()

        for t in range(sep_pos + 1, T):
            if targets[b, t] == pad_id:
                break
            if preds[b, t] == targets[b, t]:
                total_correct += 1
            total_tokens += 1

    return total_correct / max(total_tokens, 1)


def spsa_step(model, embed, x_ids, y_ids, pad_id, learning_rate, epsilon, num_perturbations):
    """Perform a single SPSA step using central difference."""
    # Get current loss
    with torch.no_grad():
        x_emb = embed(x_ids)
        logits, _, _ = model(x_emb, require_gradients=False)
        current_loss = compute_reverse_loss(logits, y_ids, pad_id).item()

    # Collect all parameters
    param_list = list(model.parameters()) + list(embed.parameters())

    # Estimate gradient using random perturbations
    pseudo_gradient = [torch.zeros_like(p) for p in param_list]

    for _ in range(num_perturbations):
        # Generate random direction
        perturbations = [torch.randn_like(p) for p in param_list]

        # Positive perturbation
        with torch.no_grad():
            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=epsilon)

            x_emb = embed(x_ids)
            logits_plus, _, _ = model(x_emb, require_gradients=False)
            loss_plus = compute_reverse_loss(logits_plus, y_ids, pad_id).item()

            # Restore and apply negative perturbation
            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=-2 * epsilon)

            x_emb = embed(x_ids)
            logits_minus, _, _ = model(x_emb, require_gradients=False)
            loss_minus = compute_reverse_loss(logits_minus, y_ids, pad_id).item()

            # Restore parameters
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
    vocab_size, seq_length, hidden_size, input_size, num_heads,
    learning_rate, num_perturbations, batch_size,
    max_time, num_accurate_samples, max_loss, device
):
    """Test a single SPSA hyperparameter configuration with early stopping."""
    import time

    epsilon = learning_rate

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

    # Generate test samples once at the beginning
    test_samples = []
    for _ in range(num_accurate_samples):
        test_x, test_y = generate_reverse_batch(1, seq_length, vocab_size, device)
        test_samples.append((test_x, test_y))

    # Training loop
    losses = []
    accuracies = []
    start_time = time.time()
    converged = False

    step = 0
    while time.time() - start_time < max_time:
        # Generate batch
        x_ids, y_ids = generate_reverse_batch(batch_size, seq_length, vocab_size, device)

        # SPSA optimization step
        loss = spsa_step(
            model, embed, x_ids, y_ids, PAD,
            learning_rate, epsilon, num_perturbations
        )

        losses.append(loss)
        step += 1

        if loss >= max_loss:
            break

        # Check for convergence: test on pre-generated test samples
        if step % 10 == 0:  # Check every 10 steps
            all_correct = True
            for test_x, test_y in test_samples:
                with torch.no_grad():
                    x_emb = embed(test_x)
                    logits, _, _ = model(x_emb, require_gradients=False)
                    accuracy = compute_reverse_accuracy(logits, test_y, SEP, PAD)

                if accuracy < 1.0:  # Not 100% accurate
                    all_correct = False
                    break

            if all_correct:
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
    parser.add_argument('--seq_length', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=240)
    parser.add_argument('--input_size', type=int, default=100)
    parser.add_argument('--num_heads', type=int, default=20)
    parser.add_argument('--max_time', type=float, default=30.0, help='Maximum time per configuration (seconds)')
    parser.add_argument('--max_loss', type=float, default=10.0, help='Maximum loss before quitting run')
    parser.add_argument('--num_accurate_samples', type=int, default=2048, help='Number of samples to test for 100%% accuracy')
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
    print(f"Task: Reverse sequences of length {args.seq_length}")
    print(f"Vocab size: {args.vocab_size}, Hidden size: {args.hidden_size}")
    print(f"Max time per config: {args.max_time}s")
    print(f"Convergence criterion: 100% accuracy on {args.num_accurate_samples} samples")
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

        losses, accuracies, converged, elapsed_time, num_steps = test_spsa_config(
            args.vocab_size, args.seq_length, args.hidden_size, args.input_size, args.num_heads,
            lr, n_pert, bs, args.max_time, args.num_accurate_samples, args.max_loss,
            device,
        )

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
