#!/usr/bin/env python3
"""
Simplified LSTM reverse task - NO TOKENIZATION
Works directly with integer sequences.

Example usage:
    python simple_reverse_task.py --vocab_size 20 --seq_length 10 --batch_size 32
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

    Format: [BOS, tok1, tok2, ..., tokN, SEP, tokN, tokN-1, ..., tok1, EOS, PAD, PAD, ...]

    Special tokens:
        - vocab_size - 4: BOS (beginning of sequence)
        - vocab_size - 3: SEP (separator between input and output)
        - vocab_size - 2: EOS (end of sequence)
        - vocab_size - 1: PAD (padding)

    Args:
        batch_size: Number of sequences in batch
        seq_length: Length of the sequence to reverse (not counting special tokens)
        vocab_size: Total vocabulary size (including special tokens)
        device: Device to create tensors on

    Returns:
        x_ids: [batch_size, max_len] - Input sequences
        y_ids: [batch_size, max_len] - Target sequences (shifted by 1 for teacher forcing)
    """
    # Special tokens
    BOS = vocab_size - 4
    SEP = vocab_size - 3
    EOS = vocab_size - 2
    PAD = vocab_size - 1

    # Content vocab is [0, vocab_size - 4)
    content_vocab_size = vocab_size - 4

    # Maximum sequence length: BOS + seq_length + SEP + seq_length + EOS
    max_len = 1 + seq_length + 1 + seq_length + 1

    # Generate random sequences
    x_batch = []
    y_batch = []

    for _ in range(batch_size):
        # Generate random sequence
        seq = np.random.randint(0, content_vocab_size, size=seq_length)

        # Create input: [BOS, seq, SEP]
        x_seq = [BOS] + seq.tolist() + [SEP]

        # Create target: [seq, SEP, reversed_seq, EOS]
        y_seq = seq.tolist() + [SEP] + seq[::-1].tolist() + [EOS]

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
    """
    Compute cross-entropy loss, ignoring padding tokens.

    Args:
        logits: [batch_size, seq_len, vocab_size]
        targets: [batch_size, seq_len]
        pad_id: Token ID to ignore in loss

    Returns:
        loss: Scalar tensor
    """
    B, T, V = logits.shape

    # Reshape for cross entropy
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    # Compute loss, ignoring padding
    loss = nn.functional.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=pad_id,
        reduction='mean'
    )

    return loss


def compute_reverse_accuracy(logits, targets, sep_id, pad_id):
    """
    Compute accuracy only on the output part (after SEP token).

    Args:
        logits: [batch_size, seq_len, vocab_size]
        targets: [batch_size, seq_len]
        sep_id: Separator token ID
        pad_id: Padding token ID

    Returns:
        accuracy: Float between 0 and 1
    """
    B, T, V = logits.shape
    preds = logits.argmax(dim=-1)  # [B, T]

    total_correct = 0
    total_tokens = 0

    for b in range(B):
        # Find SEP position
        sep_positions = (targets[b] == sep_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) == 0:
            continue
        sep_pos = sep_positions[0].item()

        # Check predictions after SEP (the reversed sequence)
        for t in range(sep_pos + 1, T):
            if targets[b, t] == pad_id:
                break
            if preds[b, t] == targets[b, t]:
                total_correct += 1
            total_tokens += 1

    return total_correct / max(total_tokens, 1)


# ============================================================================
# Training Loop
# ============================================================================

def train_reverse_task(
    vocab_size=50,
    seq_length=10,
    hidden_size=256,
    num_heads=8,
    batch_size=32,
    num_steps=1000,
    learning_rate=0.001,
    device='cuda',
    print_every=100,
    eval_every=100
):
    """
    Train an LSTM to reverse sequences.
    """
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Vocab size: {vocab_size}, Seq length: {seq_length}")
    print(f"Hidden size: {hidden_size}, Num heads: {num_heads}")

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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    print("\n" + "="*60)
    print("Training...")
    print("="*60)

    for step in range(num_steps):
        # Generate batch
        x_ids, y_ids = generate_reverse_batch(batch_size, seq_length, vocab_size, device)

        # Forward pass
        x_emb = embed(x_ids)
        logits, _, _ = model(x_emb, require_gradients=True)

        # Compute loss
        loss = compute_reverse_loss(logits, y_ids, PAD)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if (step + 1) % print_every == 0:
            with torch.no_grad():
                accuracy = compute_reverse_accuracy(logits, y_ids, SEP, PAD)
            print(f"Step {step+1:4d}: Loss={loss.item():.4f}, Accuracy={accuracy:.4f}")

        # Evaluation
        if (step + 1) % eval_every == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at step {step+1}")
            print(f"{'='*60}")
            evaluate_model(model, embed, vocab_size, seq_length, device, num_samples=5)
            print()


def evaluate_model(model, embed, vocab_size, seq_length, device, num_samples=5):
    """
    Evaluate the model and print example predictions.
    """
    # Special tokens
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
            # Find SEP position
            sep_positions = (y_ids[i] == SEP).nonzero(as_tuple=True)[0]
            if len(sep_positions) == 0:
                continue
            sep_pos = sep_positions[0].item()

            # Extract sequences
            input_seq = x_ids[i, 1:sep_pos].cpu().numpy()  # Skip BOS
            target_seq = y_ids[i, sep_pos+1:].cpu().numpy()
            pred_seq = preds[i, sep_pos+1:].cpu().numpy()

            # Remove padding
            target_seq = target_seq[target_seq != PAD]
            pred_seq = pred_seq[:len(target_seq)]

            # Print
            print(f"Example {i+1}:")
            print(f"  Input:    {list(input_seq)}")
            print(f"  Target:   {list(target_seq[:-1])}")  # Remove EOS
            print(f"  Predicted: {list(pred_seq[:-1])}")  # Remove EOS
            print(f"  Match: {np.array_equal(target_seq[:-1], pred_seq[:-1])}")
            print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train LSTM on reverse task (no tokenization)')
    parser.add_argument('--vocab_size', type=int, default=50, help='Vocabulary size')
    parser.add_argument('--seq_length', type=int, default=10, help='Sequence length to reverse')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--print_every', type=int, default=100, help='Print frequency')
    parser.add_argument('--eval_every', type=int, default=100, help='Evaluation frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Train
    train_reverse_task(
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        device=args.device,
        print_every=args.print_every,
        eval_every=args.eval_every
    )


if __name__ == '__main__':
    main()
