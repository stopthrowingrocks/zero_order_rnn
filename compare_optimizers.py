#!/usr/bin/env python3
"""
Compare Adam (gradient-based) vs SPSA (zero-order) optimization
on reverse task with variable sequence lengths (5-10 tokens).
"""
import argparse
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
        # Random sequence length
        seq_length = np.random.randint(min_seq_length, max_seq_length + 1)

        # Generate random sequence
        seq = np.random.randint(0, content_vocab_size, size=seq_length)

        # Create full sequence: [BOS, seq, SEP, reversed_seq, EOS]
        full_seq = [BOS] + seq.tolist() + [SEP] + seq[::-1].tolist() + [EOS]

        x_seq = full_seq
        y_seq = full_seq

        x_batch.append(torch.tensor(x_seq, dtype=torch.long, device=device))
        y_batch.append(torch.tensor(y_seq, dtype=torch.long, device=device))

    # Pad to same length
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


def train_with_adam(vocab_size, min_seq, max_seq, hidden_size, num_heads,
                    learning_rate, batch_size, num_steps, device):
    """Train with Adam optimizer (gradient-based)."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    SEP = vocab_size - 3
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

    optimizer = torch.optim.Adam(list(model.parameters()) + list(embed.parameters()), lr=learning_rate)

    losses = []
    accuracies = []

    for step in range(num_steps):
        x_ids, y_ids = generate_reverse_batch(batch_size, min_seq, max_seq, vocab_size, device)

        x_emb = embed(x_ids)
        logits, _, _ = model(x_emb, require_gradients=True)
        loss = compute_reverse_loss(logits, y_ids, PAD)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            with torch.no_grad():
                accuracy = compute_reverse_accuracy(logits, y_ids, SEP, PAD)
                accuracies.append(accuracy)

    return losses, accuracies


def train_with_spsa(vocab_size, min_seq, max_seq, hidden_size, num_heads,
                    learning_rate, epsilon, num_perturbations, batch_size, num_steps, device):
    """Train with SPSA (zero-order optimization)."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    SEP = vocab_size - 3
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

    losses = []
    accuracies = []

    for step in range(num_steps):
        x_ids, y_ids = generate_reverse_batch(batch_size, min_seq, max_seq, vocab_size, device)

        loss = zeroth_order_step(
            model, embed, x_ids, y_ids, PAD,
            learning_rate, epsilon, num_perturbations
        )

        losses.append(loss)

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
    parser.add_argument('--min_seq_length', type=int, default=5)
    parser.add_argument('--max_seq_length', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("OPTIMIZER COMPARISON: Adam vs SPSA")
    print("=" * 80)
    print(f"Task: Reverse sequences of length {args.min_seq_length}-{args.max_seq_length}")
    print(f"Vocab size: {args.vocab_size}, Hidden size: {args.hidden_size}")
    print(f"Testing {args.num_steps} steps per configuration\n")

    # Adam hyperparameter grid
    adam_configs = [
        (0.001, 32, "LR=0.001, Batch=32 (baseline)"),
        (0.0005, 32, "LR=0.0005, Batch=32"),
        (0.002, 32, "LR=0.002, Batch=32"),
        (0.001, 16, "LR=0.001, Batch=16"),
        (0.001, 64, "LR=0.001, Batch=64"),
        (0.0001, 32, "LR=0.0001, Batch=32"),
    ]

    # SPSA hyperparameter grid
    spsa_configs = [
        (0.05, 0.05, 8, 16, "LR=0.05, Eps=0.05, Pert=8, Batch=16 (from fixed-length)"),
        (0.03, 0.05, 8, 16, "LR=0.03, Eps=0.05, Pert=8, Batch=16"),
        (0.07, 0.05, 8, 16, "LR=0.07, Eps=0.05, Pert=8, Batch=16"),
        (0.05, 0.03, 8, 16, "LR=0.05, Eps=0.03, Pert=8, Batch=16"),
        (0.05, 0.07, 8, 16, "LR=0.05, Eps=0.07, Pert=8, Batch=16"),
        (0.05, 0.05, 4, 16, "LR=0.05, Eps=0.05, Pert=4, Batch=16"),
        (0.05, 0.05, 16, 16, "LR=0.05, Eps=0.05, Pert=16, Batch=16"),
        (0.05, 0.05, 8, 32, "LR=0.05, Eps=0.05, Pert=8, Batch=32"),
    ]

    print("\n" + "=" * 80)
    print("TESTING ADAM OPTIMIZER (Gradient-Based)")
    print("=" * 80)

    adam_results = []
    for lr, bs, name in adam_configs:
        print(f"\n{name}")
        print("-" * 80)

        losses, accuracies = train_with_adam(
            args.vocab_size, args.min_seq_length, args.max_seq_length,
            args.hidden_size, args.num_heads, lr, bs, args.num_steps, args.device
        )

        initial_loss = np.mean(losses[:10])
        final_loss = np.mean(losses[-50:])
        min_loss = min(losses)
        improvement = initial_loss - final_loss
        final_acc = accuracies[-1] if accuracies else 0.0

        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss:   {final_loss:.4f}")
        print(f"Min loss:     {min_loss:.4f}")
        print(f"Improvement:  {improvement:.4f}")
        print(f"Final acc:    {final_acc:.4f}")

        adam_results.append({
            'name': name,
            'lr': lr,
            'batch_size': bs,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'min_loss': min_loss,
            'improvement': improvement,
            'final_acc': final_acc
        })

    print("\n\n" + "=" * 80)
    print("TESTING SPSA OPTIMIZER (Zero-Order)")
    print("=" * 80)

    spsa_results = []
    for lr, eps, n_pert, bs, name in spsa_configs:
        print(f"\n{name}")
        print("-" * 80)

        losses, accuracies = train_with_spsa(
            args.vocab_size, args.min_seq_length, args.max_seq_length,
            args.hidden_size, args.num_heads, lr, eps, n_pert, bs, args.num_steps, args.device
        )

        initial_loss = np.mean(losses[:10])
        final_loss = np.mean(losses[-50:])
        min_loss = min(losses)
        improvement = initial_loss - final_loss
        final_acc = accuracies[-1] if accuracies else 0.0

        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss:   {final_loss:.4f}")
        print(f"Min loss:     {min_loss:.4f}")
        print(f"Improvement:  {improvement:.4f}")
        print(f"Final acc:    {final_acc:.4f}")

        spsa_results.append({
            'name': name,
            'lr': lr,
            'epsilon': eps,
            'num_pert': n_pert,
            'batch_size': bs,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'min_loss': min_loss,
            'improvement': improvement,
            'final_acc': final_acc
        })

    # Print summaries
    print("\n\n" + "=" * 80)
    print("ADAM OPTIMIZER SUMMARY")
    print("=" * 80)
    print(f"{'Config':<45} {'Initial':>10} {'Final':>10} {'Min':>10} {'Improve':>10} {'Acc':>8}")
    print("-" * 80)
    for r in sorted(adam_results, key=lambda x: x['final_acc'], reverse=True):
        print(f"{r['name']:<45} {r['initial_loss']:>10.4f} {r['final_loss']:>10.4f} "
              f"{r['min_loss']:>10.4f} {r['improvement']:>10.4f} {r['final_acc']:>8.4f}")

    best_adam = max(adam_results, key=lambda x: x['final_acc'])
    print(f"\nBest Adam: {best_adam['name']}")
    print(f"  Final Loss: {best_adam['final_loss']:.4f}, Final Acc: {best_adam['final_acc']:.4f}")

    print("\n\n" + "=" * 80)
    print("SPSA OPTIMIZER SUMMARY")
    print("=" * 80)
    print(f"{'Config':<50} {'Initial':>10} {'Final':>10} {'Min':>10} {'Improve':>10} {'Acc':>8}")
    print("-" * 80)
    for r in sorted(spsa_results, key=lambda x: x['final_acc'], reverse=True):
        print(f"{r['name']:<50} {r['initial_loss']:>10.4f} {r['final_loss']:>10.4f} "
              f"{r['min_loss']:>10.4f} {r['improvement']:>10.4f} {r['final_acc']:>8.4f}")

    best_spsa = max(spsa_results, key=lambda x: x['final_acc'])
    print(f"\nBest SPSA: {best_spsa['name']}")
    print(f"  Final Loss: {best_spsa['final_loss']:.4f}, Final Acc: {best_spsa['final_acc']:.4f}")

    print("\n\n" + "=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)
    print(f"Best Adam:  {best_adam['final_acc']:.4f} accuracy, {best_adam['final_loss']:.4f} loss")
    print(f"Best SPSA:  {best_spsa['final_acc']:.4f} accuracy, {best_spsa['final_loss']:.4f} loss")
    print("=" * 80)


if __name__ == '__main__':
    main()
