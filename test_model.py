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

    # Run inference with teacher forcing (same as training)
    with torch.no_grad():
        x_emb = embed(x_ids)

        # Convert to model dtype if needed
        next_param = next(model.parameters())
        if x_emb.dtype != next_param.dtype:
            x_emb = x_emb.to(dtype=next_param.dtype)

        Lx = x_emb.shape[1]
        Ly = y_ids.shape[1]

        hidden = None
        memory = None
        chunk_size = 32

        # Process input sequence first
        pos = 0
        while pos < Lx:
            chunk_end = min(pos + chunk_size, Lx)
            input_chunk = x_emb[:, pos:chunk_end, :]
            out_chunk, mem_new, hidden_new = model(input_chunk, hidden=hidden, memory=memory, require_gradients=False)
            hidden = hidden_new
            memory = mem_new
            pos = chunk_end

        # Now process target sequence chunk by chunk and collect predictions
        all_logits = []
        pos = 0
        while pos < Ly - 1:
            chunk_end = min(pos + chunk_size, Ly - 1)
            y_chunk = y_ids[:, pos:chunk_end]
            y_emb_chunk = embed(y_chunk)

            out_chunk, mem_new, hidden_new = model(y_emb_chunk, hidden=hidden, memory=memory, require_gradients=False)
            hidden = hidden_new
            memory = mem_new

            # Collect logits for this chunk
            all_logits.append(out_chunk)

            pos = chunk_end

        # Concatenate all logits
        if all_logits:
            logits = torch.cat(all_logits, dim=1)  # [B, Ly-1, vocab_size]
        else:
            logits = torch.zeros(batch_size, 0, vocab_size, device=device, dtype=next_param.dtype)

    print(f"Inference complete:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits dtype: {logits.dtype}")
    print(f"  Logits device: {logits.device}")
    print()

    # Show predictions for all samples
    SEP = vocab_size - 3
    PAD = vocab_size - 1

    print("Predictions for all samples in batch:")
    print("=" * 80)

    # Get all samples
    x_samples = x_ids.cpu().numpy()
    y_samples = y_ids.cpu().numpy()
    pred_samples = logits.argmax(dim=-1).cpu().numpy()

    total_correct_all = 0
    total_tokens_all = 0

    for batch_idx in range(batch_size):
        print(f"\nSample {batch_idx + 1}/{batch_size}:")
        print("-" * 80)

        x_sample = x_samples[batch_idx]
        y_sample = y_samples[batch_idx]
        pred_sample = pred_samples[batch_idx]

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
        # Note: predictions are for next tokens, so pred[i] predicts y[i+1]
        if sep_pos_y is not None:
            target_tokens = []
            pred_tokens = []
            for i in range(sep_pos_y + 1, len(y_sample)):
                if y_sample[i] == PAD:
                    break
                target_tokens.append(int(y_sample[i]))
                # Prediction at position (i-1) predicts token at position i
                pred_idx = i - 1
                if pred_idx >= 0 and pred_idx < len(pred_sample):
                    pred_tokens.append(int(pred_sample[pred_idx]))

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