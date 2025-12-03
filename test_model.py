#!/usr/bin/env python3
"""
Test a saved model by loading it and running inference on a generated batch.

Example usage:
    python test_model.py spsa_model.pt
    python test_model.py spsa_model.pt --batch_size 8 --min_seq_length 5 --max_seq_length 10
"""
import argparse
import torch
import torch.nn as nn
from models.models import LSTM
from shared import generate_reverse_batch


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
    """Run inference on a generated batch and return logits."""
    # Generate batch
    x_ids, y_ids = generate_reverse_batch(batch_size, min_seq_length, max_seq_length, vocab_size, device)

    print(f"Generated batch:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {x_ids.shape}")
    print(f"  Target shape: {y_ids.shape}")
    print()

    # Run inference
    with torch.no_grad():
        x_emb = embed(x_ids)

        # Convert to model dtype if needed
        next_param = next(model.parameters())
        if x_emb.dtype != next_param.dtype:
            x_emb = x_emb.to(dtype=next_param.dtype)

        # Forward pass
        logits, memory, hidden = model(x_emb, require_gradients=False)

    print(f"Inference complete:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits dtype: {logits.dtype}")
    print(f"  Logits device: {logits.device}")
    print()

    # Show example predictions
    SEP = vocab_size - 3
    PAD = vocab_size - 1

    print("Sample predictions (first example in batch):")
    print("-" * 80)

    # Get first example
    x_sample = x_ids[0].cpu().numpy()
    y_sample = y_ids[0].cpu().numpy()
    pred_sample = logits[0].argmax(dim=-1).cpu().numpy()

    # Find SEP position in input
    sep_pos_x = None
    for i, token in enumerate(x_sample):
        if token == SEP:
            sep_pos_x = i
            break

    # Display input sequence (before SEP)
    if sep_pos_x is not None:
        input_seq = x_sample[:sep_pos_x]
        print(f"Input:  {list(input_seq)}")
    else:
        print(f"Input:  {list(x_sample)}")

    # Find SEP position in target
    sep_pos_y = None
    for i, token in enumerate(y_sample):
        if token == SEP:
            sep_pos_y = i
            break

    # Display target and prediction (after SEP, before PAD)
    if sep_pos_y is not None:
        target_tokens = []
        pred_tokens = []
        for i in range(sep_pos_y + 1, len(y_sample)):
            if y_sample[i] == PAD:
                break
            target_tokens.append(int(y_sample[i]))
            # Prediction position is offset by sep_pos_y
            if i < len(pred_sample):
                pred_tokens.append(int(pred_sample[i]))

        print(f"Target: {target_tokens}")
        print(f"Pred:   {pred_tokens}")

        # Check accuracy
        if len(target_tokens) > 0 and len(pred_tokens) == len(target_tokens):
            correct = sum(1 for t, p in zip(target_tokens, pred_tokens) if t == p)
            accuracy = correct / len(target_tokens)
            print(f"Accuracy: {correct}/{len(target_tokens)} = {accuracy:.2%}")

    print("-" * 80)
    print()

    return logits, x_ids, y_ids


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


if __name__ == '__main__':
    main()