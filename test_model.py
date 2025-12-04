#!/usr/bin/env python3
"""
Test a saved model by loading it and running inference on a generated batch.

Example usage:
    python test_model.py spsa_model.pt
    python test_model.py spsa_model.pt --batch_size 8 --min_seq_length 5 --max_seq_length 10
"""
import argparse
import csv
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.models import LSTM
from shared import compute_accuracy, compute_loss, generate_reverse_batch, generate_predictions


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

    return model, embed, vocab_size


def run_inference(model, embed, vocab_size, batch_size, min_seq_length, max_seq_length, device):
    """Run inference on a generated batch and return logits.

    Note: embed parameter is unused (model.embed is used instead) but kept for backward compatibility.
    """
    # Generate batch
    x_ids, y_ids = generate_reverse_batch(batch_size, min_seq_length, max_seq_length, vocab_size, device)

    print(f"Generated batch:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {x_ids.shape}")
    print(f"  Target shape: {y_ids.shape}")
    print()

    # Run inference with teacher forcing (same as training)
    logits = generate_predictions(model, x_ids, y_ids, require_gradients=False, chunk_size=32)

    print(f"Inference complete:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits dtype: {logits.dtype}")
    print(f"  Logits device: {logits.device}")
    print()

    # Show predictions for all samples
    PAD = vocab_size - 1

    print("Predictions for all samples in batch:")
    print("=" * 80)

    # Get all samples
    x_samples = x_ids.cpu().numpy()
    y_samples = y_ids.cpu().numpy()
    pred_samples = logits.argmax(dim=-1).cpu().numpy()
    print(f"Pred samples shape {pred_samples.shape}")

    total_correct_all = 0
    total_tokens_all = 0

    for batch_idx in range(batch_size):
        print(f"\nSample {batch_idx + 1}/{batch_size}:")
        print("-" * 80)

        x_sample = x_samples[batch_idx]
        y_sample = y_samples[batch_idx]
        pred_sample = pred_samples[batch_idx]

        print(f"Input:  {list(map(int, list(x_sample)))}")

        # Display target and prediction (entire y_ids sequence, excluding PAD)
        # Note: logits[i] predicts y_ids[i] directly now
        target_tokens = []
        pred_tokens = []

        for i in range(len(y_sample)):
            if y_sample[i] == PAD:
                break
            target_tokens.append(int(y_sample[i]))

            # Prediction at position i predicts token at position i
            if i < len(pred_sample):
                pred_tokens.append(int(pred_sample[i]))

        print(f"Target: {target_tokens}")
        print(f"Pred:   {pred_tokens}")

        # Check accuracy
        if len(target_tokens) > 0 and len(pred_tokens) == len(target_tokens):
            correct = sum(1 for t, p in zip(target_tokens, pred_tokens) if t == p)
            accuracy = correct / len(target_tokens)
            print(f"Accuracy: {correct}/{len(target_tokens)} = {accuracy:.2%}")

            total_correct_all += correct
            total_tokens_all += len(target_tokens)

    # Overall accuracy
    print("\n" + "=" * 80)
    if total_tokens_all > 0:
        overall_accuracy = total_correct_all / total_tokens_all
        print(f"Overall Accuracy: {total_correct_all}/{total_tokens_all} = {overall_accuracy:.2%}")
    print("=" * 80)
    other_accuracy = compute_accuracy(model, x_ids, y_ids, PAD, chunk_size=32)
    print(f"Shared compute_accuracy: {other_accuracy}")

    # Compute loss using shared compute_loss function
    criterion = nn.CrossEntropyLoss(ignore_index=PAD).to(device)
    loss_value = compute_loss(model, x_ids, y_ids, criterion, require_gradients=False).item()
    print(f"Shared compute_loss: {loss_value:.6f}")
    print()

    return logits, x_ids, y_ids


def analyze_loss_by_seq_length(model, vocab_size, batch_size, min_seq_length, max_seq_length, device, model_name):
    """Analyze loss for different sequence lengths and save results."""
    print("\n" + "=" * 80)
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
    parser = argparse.ArgumentParser(description='Test a saved model')
    parser.add_argument('model_path', type=str, help='Path to saved model (.pt file)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--min_seq_length', type=int, default=5, help='Minimum sequence length')
    parser.add_argument('--max_seq_length', type=int, default=10, help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    model, embed, vocab_size = load_model(args.model_path, device)

    # Extract model name from path (without extension)
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    # Run inference
    logits, x_ids, y_ids = run_inference(
        model, embed, vocab_size,
        args.batch_size, args.min_seq_length, args.max_seq_length,
        device
    )

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"Logits returned with shape: {logits.shape}")
    print()

    # Analyze loss by sequence length
    analyze_loss_by_seq_length(
        model, vocab_size, args.batch_size,
        args.min_seq_length, args.max_seq_length,
        device, model_name
    )


if __name__ == '__main__':
    main()
