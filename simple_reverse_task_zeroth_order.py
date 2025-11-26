#!/usr/bin/env python3
"""
Simplified LSTM reverse task with ZERO-ORDER OPTIMIZATION (Gradient-Free)
Uses random perturbations instead of backpropagation.

Example usage:
    python simple_reverse_task_zeroth_order.py --vocab_size 20 --seq_length 10 --batch_size 32
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from models.models import LSTM


# ============================================================================
# Task Generation - Direct Integer Sequences
# ============================================================================

def generate_reverse_batch(batch_size, seq_length, vocab_size, device='cuda'):
    """
    Generate a batch of reverse sequences using raw integers.
    Format: [BOS, tok1, tok2, ..., tokN, SEP, tokN, tokN-1, ..., tok1, EOS, PAD, ...]
    """
    # Special tokens
    BOS = vocab_size - 4
    SEP = vocab_size - 3
    EOS = vocab_size - 2
    PAD = vocab_size - 1

    content_vocab_size = vocab_size - 4

    x_batch = []
    y_batch = []

    for _ in range(batch_size):
        # Generate random sequence
        seq = np.random.randint(0, content_vocab_size, size=seq_length)

        # Create full sequence: [BOS, seq, SEP, reversed_seq, EOS]
        full_seq = [BOS] + seq.tolist() + [SEP] + seq[::-1].tolist() + [EOS]

        # Input and target are the same sequence (teacher forcing)
        x_seq = full_seq
        y_seq = full_seq

        x_batch.append(torch.tensor(x_seq, dtype=torch.long, device=device))
        y_batch.append(torch.tensor(y_seq, dtype=torch.long, device=device))

    # Pad to same length
    x_ids = nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=PAD)
    y_ids = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=PAD)

    return x_ids, y_ids


# ============================================================================
# Loss Computation
# ============================================================================

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


# ============================================================================
# Zero-Order Optimization (Central Difference SPSA)
# ============================================================================

def zeroth_order_step(model, embed, x_ids, y_ids, pad_id, learning_rate, epsilon=0.01, num_perturbations=2):
    """
    Perform a single zero-order optimization step using central difference.

    Args:
        model: The LSTM model
        embed: Embedding layer
        x_ids: Input token IDs [B, T]
        y_ids: Target token IDs [B, T]
        pad_id: Padding token ID
        learning_rate: Step size
        epsilon: Perturbation size
        num_perturbations: Number of random directions to sample

    Returns:
        loss: Current loss value
    """
    # Get current loss (evaluation)
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

        # Accumulate gradient estimate: g ≈ (f(θ+εd) - f(θ-εd)) / (2ε) * d
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


# ============================================================================
# Training Loop
# ============================================================================

def train_reverse_task_zeroth_order(
    vocab_size=50,
    seq_length=10,
    hidden_size=256,
    num_heads=8,
    batch_size=32,
    num_steps=1000,
    learning_rate=0.1,
    epsilon=0.01,
    num_perturbations=2,
    device='cuda',
    print_every=100,
    eval_every=100
):
    """
    Train an LSTM to reverse sequences using zero-order optimization.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Vocab size: {vocab_size}, Seq length: {seq_length}")
    print(f"Hidden size: {hidden_size}, Num heads: {num_heads}")
    print(f"Zero-Order Optimization: lr={learning_rate}, epsilon={epsilon}, perturbations={num_perturbations}")

    # Special tokens
    SEP = vocab_size - 3
    PAD = vocab_size - 1

    # Create model
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

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    print("\n" + "="*60)
    print("Training with Zero-Order Optimization (Gradient-Free)...")
    print("="*60)

    for step in range(num_steps):
        # Generate batch
        x_ids, y_ids = generate_reverse_batch(batch_size, seq_length, vocab_size, device)

        # Zero-order optimization step
        loss = zeroth_order_step(
            model, embed, x_ids, y_ids, PAD,
            learning_rate, epsilon, num_perturbations
        )

        # Logging
        if (step + 1) % print_every == 0:
            with torch.no_grad():
                x_emb = embed(x_ids)
                logits, _, _ = model(x_emb, require_gradients=False)
                accuracy = compute_reverse_accuracy(logits, y_ids, SEP, PAD)
            print(f"Step {step+1:4d}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")

        # Evaluation
        if (step + 1) % eval_every == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at step {step+1}")
            print(f"{'='*60}")
            evaluate_model(model, embed, vocab_size, seq_length, device, num_samples=5)
            print()


def evaluate_model(model, embed, vocab_size, seq_length, device, num_samples=5):
    """Evaluate the model and print example predictions."""
    BOS = vocab_size - 4
    SEP = vocab_size - 3
    EOS = vocab_size - 2
    PAD = vocab_size - 1

    with torch.no_grad():
        x_ids, y_ids = generate_reverse_batch(num_samples, seq_length, vocab_size, device)
        x_emb = embed(x_ids)
        logits, _, _ = model(x_emb, require_gradients=False)
        preds = logits.argmax(dim=-1)

        for i in range(num_samples):
            sep_positions = (y_ids[i] == SEP).nonzero(as_tuple=True)[0]
            if len(sep_positions) == 0:
                continue
            sep_pos = sep_positions[0].item()

            input_seq = x_ids[i, 1:sep_pos].cpu().numpy()
            target_seq = y_ids[i, sep_pos+1:].cpu().numpy()
            pred_seq = preds[i, sep_pos+1:].cpu().numpy()

            target_seq = target_seq[target_seq != PAD]
            pred_seq = pred_seq[:len(target_seq)]

            print(f"Example {i+1}:")
            print(f"  Input:    {list(input_seq)}")
            print(f"  Target:   {list(target_seq[:-1])}")
            print(f"  Predicted: {list(pred_seq[:-1])}")
            print(f"  Match: {np.array_equal(target_seq[:-1], pred_seq[:-1])}")
            print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train LSTM on reverse task with zero-order optimization')
    parser.add_argument('--vocab_size', type=int, default=50, help='Vocabulary size')
    parser.add_argument('--seq_length', type=int, default=10, help='Sequence length to reverse')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Perturbation size')
    parser.add_argument('--num_perturbations', type=int, default=2, help='Number of random directions')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--print_every', type=int, default=100, help='Print frequency')
    parser.add_argument('--eval_every', type=int, default=100, help='Evaluation frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Train
    train_reverse_task_zeroth_order(
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        num_perturbations=args.num_perturbations,
        device=args.device,
        print_every=args.print_every,
        eval_every=args.eval_every
    )


if __name__ == '__main__':
    main()
