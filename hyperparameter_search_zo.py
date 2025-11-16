#!/usr/bin/env python3
"""
Hyperparameter search for zero-order optimization on reverse task.
Tests different combinations of learning_rate, epsilon, and num_perturbations.
"""
import argparse
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


def zeroth_order_step(model, embed, x_ids, y_ids, pad_id, learning_rate, epsilon, num_perturbations):
    """Perform a single zero-order optimization step using central difference."""
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


def test_hyperparameters(
    vocab_size, seq_length, hidden_size, num_heads,
    learning_rate, epsilon, num_perturbations, batch_size,
    num_steps, device
):
    """Test a single hyperparameter configuration."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    SEP = vocab_size - 3
    PAD = vocab_size - 1

    # Create fresh model
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

    # Training loop
    losses = []
    accuracies = []

    for step in range(num_steps):
        # Generate batch
        x_ids, y_ids = generate_reverse_batch(batch_size, seq_length, vocab_size, device)

        # Zero-order optimization step
        loss = zeroth_order_step(
            model, embed, x_ids, y_ids, PAD,
            learning_rate, epsilon, num_perturbations
        )

        losses.append(loss)

        # Compute accuracy periodically
        if (step + 1) % 50 == 0:
            with torch.no_grad():
                x_emb = embed(x_ids)
                logits, _, _ = model(x_emb, require_gradients=False)
                accuracy = compute_reverse_accuracy(logits, y_ids, SEP, PAD)
                accuracies.append(accuracy)

    return losses, accuracies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=20)
    parser.add_argument('--seq_length', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("ZERO-ORDER OPTIMIZATION HYPERPARAMETER SEARCH")
    print("=" * 80)
    print(f"Task: Reverse sequences of length {args.seq_length}")
    print(f"Vocab size: {args.vocab_size}, Hidden size: {args.hidden_size}")
    print(f"Testing {args.num_steps} steps per configuration\n")

    # Define hyperparameter grid
    configs = [
        # (lr, epsilon, num_pert, batch_size, name)
        (0.01, 0.01, 4, 16, "Low LR, Small Epsilon, Few Pert"),
        (0.05, 0.01, 4, 16, "Medium-Low LR, Small Epsilon"),
        (0.1, 0.01, 8, 16, "Medium LR, Small Epsilon"),
        (0.01, 0.05, 8, 16, "Low LR, Medium Epsilon"),
        (0.05, 0.05, 8, 16, "Medium LR, Medium Epsilon"),
        (0.01, 0.1, 16, 16, "Low LR, Large Epsilon, Many Pert"),
        (0.05, 0.1, 16, 16, "Medium LR, Large Epsilon"),
        (0.001, 0.01, 8, 8, "Very Low LR, Small Batch"),
    ]

    best_config = None
    best_final_loss = float('inf')
    results = []

    for lr, eps, n_pert, bs, name in configs:
        print(f"\n{'=' * 80}")
        print(f"Testing: {name}")
        print(f"  LR={lr}, Epsilon={eps}, Perturbations={n_pert}, Batch={bs}")
        print(f"{'=' * 80}")

        losses, accuracies = test_hyperparameters(
            args.vocab_size, args.seq_length, args.hidden_size, args.num_heads,
            lr, eps, n_pert, bs, args.num_steps, args.device
        )

        # Analyze results
        initial_loss = np.mean(losses[:10])
        final_loss = np.mean(losses[-50:])
        min_loss = min(losses)
        loss_improvement = initial_loss - final_loss
        final_acc = accuracies[-1] if accuracies else 0.0

        print(f"\nResults:")
        print(f"  Initial loss (avg first 10): {initial_loss:.2f}")
        print(f"  Final loss (avg last 50):    {final_loss:.2f}")
        print(f"  Minimum loss:                {min_loss:.2f}")
        print(f"  Loss improvement:            {loss_improvement:.2f}")
        print(f"  Final accuracy:              {final_acc:.4f}")

        # Check if loss is decreasing
        if loss_improvement > 0:
            print(f"  ✓ Loss is DECREASING (converging)")
        else:
            print(f"  ✗ Loss is INCREASING (diverging)")

        results.append({
            'name': name,
            'lr': lr,
            'epsilon': eps,
            'num_pert': n_pert,
            'batch_size': bs,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'min_loss': min_loss,
            'improvement': loss_improvement,
            'final_acc': final_acc
        })

        if final_loss < best_final_loss:
            best_final_loss = final_loss
            best_config = (lr, eps, n_pert, bs, name)

    # Print summary
    print("\n\n" + "=" * 80)
    print("SUMMARY OF ALL CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Config':<40} {'Initial':>10} {'Final':>10} {'Min':>10} {'Improve':>10} {'Acc':>8}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x['improvement'], reverse=True):
        print(f"{r['name']:<40} {r['initial_loss']:>10.2f} {r['final_loss']:>10.2f} "
              f"{r['min_loss']:>10.2f} {r['improvement']:>10.2f} {r['final_acc']:>8.4f}")

    if best_config:
        lr, eps, n_pert, bs, name = best_config
        print("\n" + "=" * 80)
        print(f"BEST CONFIGURATION: {name}")
        print(f"  Learning Rate: {lr}")
        print(f"  Epsilon: {eps}")
        print(f"  Perturbations: {n_pert}")
        print(f"  Batch Size: {bs}")
        print(f"  Final Loss: {best_final_loss:.2f}")
        print("=" * 80)


if __name__ == '__main__':
    main()
