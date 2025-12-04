#!/usr/bin/env python3
"""
Compute and visualize loss for different sequence lengths.

Example usage:
    python compute_loss_per_len.py spsa_model.pt
    python compute_loss_per_len.py spsa_model.pt --min_seq_length 1 --max_seq_length 20
    python compute_loss_per_len.py adam_model.pt --batch_size 16 --max_seq_length 50
"""
import argparse
import csv
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.models import LSTM
from shared import compute_loss, generate_reverse_batch


def load_model(model_path, device='cuda'):
    """Load a saved model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model configuration
    vocab_size = checkpoint['vocab_size']
    hidden_size = checkpoint['hidden_size']
    input_size = checkpoint['input_size']
    num_heads = checkpoint['num_heads']

    # Create embedding layer
    embed = nn.Embedding(vocab_size, input_size, device=device, dtype=torch.bfloat16)
    embed.load_state_dict(checkpoint['embed_state_dict'])

    # Create model
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
    model.load_state_dict(checkpoint['model_state_dict'])

    # Print model info
    print("=" * 80)
    print("MODEL LOADED")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Input size: {input_size}")
    print(f"Num heads: {num_heads}")
    print(f"Converged: {checkpoint.get('converged', 'N/A')}")
    print(f"Final loss: {checkpoint.get('final_loss', 'N/A'):.6f}" if 'final_loss' in checkpoint else "Final loss: N/A")
    print(f"Training steps: {checkpoint.get('steps', 'N/A')}")
    print("=" * 80)
    print()

    return model, vocab_size


def analyze_loss_by_seq_length(model, vocab_size, batch_size, min_seq_length, max_seq_length, device, model_name):
    """Analyze loss for different sequence lengths and save results."""
    print("=" * 80)
    print("ANALYZING LOSS BY SEQUENCE LENGTH")
    print("=" * 80)
    print(f"Testing sequence lengths from {min_seq_length} to {max_seq_length}")
    print(f"Batch size: {batch_size}")
    print()

    PAD = vocab_size - 1
    criterion = nn.CrossEntropyLoss(ignore_index=PAD).to(device)

    # Collect loss for each sequence length
    seq_lengths = []
    losses = []

    for seq_len in range(min_seq_length, max_seq_length + 1):
        # Generate batch with fixed sequence length
        x_ids, y_ids = generate_reverse_batch(batch_size, seq_len, seq_len, vocab_size, device)

        # Compute loss
        loss = compute_loss(model, x_ids, y_ids, criterion, require_gradients=False).item()

        seq_lengths.append(seq_len)
        losses.append(loss)

        print(f"  Seq length {seq_len:3d}: Loss = {loss:.6f}")

    print()

    # Save to CSV
    csv_filename = f"{model_name}_losses.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seq_length', 'loss'])
        for seq_len, loss in zip(seq_lengths, losses):
            writer.writerow([seq_len, loss])

    print(f"✓ Saved loss data to {csv_filename}")

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(seq_lengths, losses, alpha=0.6, s=50)
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Loss vs Sequence Length - {model_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    png_filename = f"{model_name}_losses.png"
    plt.savefig(png_filename, dpi=150)
    plt.close()

    print(f"✓ Saved plot to {png_filename}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Compute loss for different sequence lengths')
    parser.add_argument('model_path', type=str, help='Path to saved model (.pt file)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--min_seq_length', type=int, default=1, help='Minimum sequence length')
    parser.add_argument('--max_seq_length', type=int, default=20, help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    model, vocab_size = load_model(args.model_path, device)

    # Extract model name from path (without extension)
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    # Analyze loss by sequence length
    analyze_loss_by_seq_length(
        model, vocab_size, args.batch_size,
        args.min_seq_length, args.max_seq_length,
        device, model_name
    )


if __name__ == '__main__':
    main()
