#!/usr/bin/env python3
"""
Train LSTM with Adam optimizer to convergence (loss <= 0.1) with wandb logging.

Example usage:
    python converge_adam.py --learning_rate 0.01 --batch_size 64
    python converge_adam.py --learning_rate 0.01 --batch_size 64 --wandb_project "zero-order-rnn"
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from models.models import LSTM
from shared import generate_reverse_batch, compute_reverse_loss, compute_reverse_accuracy

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠ wandb not available. Install with: pip install wandb")


def teacher_forcing_loss_for_adam(model, x_ids, y_ids_unpadded, criterion, chunk_size=32):
    """
    Teacher forcing loss computation for Adam (requires gradients).
    Similar to teacher_forcing_loss_emb_parallel but keeps gradients.
    """
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
        out_chunk, mem_new, hidden_new = model(input_chunk, hidden=hidden, memory=memory, require_gradients=True)
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

        out_chunk, mem_new, hidden_new = model(y_emb_chunk, hidden=hidden, memory=memory, require_gradients=True)

        # Update states
        hidden = hidden_new
        memory = mem_new

        # Compute loss for this chunk (keep in float32 for gradients)
        out_chunk = out_chunk.reshape(-1, out_chunk.size(-1))
        targets = y_ids_unpadded[:, pos+1:chunk_end+1].reshape(-1)  # shift by 1 for next-token prediction

        if targets.size(0) > 0:  # ensure we have targets
            chunk_loss = criterion(out_chunk, targets)
            total_loss += chunk_loss * targets.size(0)
            total_predicted_tokens += targets.size(0)

        pos = chunk_end

    if total_predicted_tokens == 0:
        return torch.tensor(0.0, device=x_ids.device, requires_grad=True)

    avg_loss = total_loss / total_predicted_tokens
    return avg_loss


def train_to_convergence(
    vocab_size, min_seq_length, max_seq_length, hidden_size, input_size, num_heads,
    learning_rate, batch_size, convergence_loss, max_steps, device, seed
):
    """Train model with Adam to convergence with wandb logging."""

    # Initialize wandb if requested
    if WANDB_AVAILABLE:
        wandb.init(
            project="zero-order-rnn",
            name=f"adam_lr_{learning_rate}_batch_{batch_size}_max_seq_{max_seq_length}",
            config={
                "optimizer": "adam",
                "vocab_size": vocab_size,
                "min_seq_length": min_seq_length,
                "max_seq_length": max_seq_length,
                "hidden_size": hidden_size,
                "input_size": input_size,
                "num_heads": num_heads,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "convergence_loss": convergence_loss,
                "seed": seed,
            }
        )

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    SEP = vocab_size - 3
    PAD = vocab_size - 1

    # Create model
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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in embed.parameters())

    # Create optimizer and loss
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(embed.parameters()),
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD).to(device)

    print("=" * 80)
    print("TRAINING TO CONVERGENCE WITH ADAM")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Task: Reverse sequences of length {min_seq_length}-{max_seq_length}")
    print(f"Vocab size: {vocab_size}, Hidden size: {hidden_size}, Input size: {input_size}")
    print(f"Num heads: {num_heads}, Total params: {total_params:,}")
    print(f"Learning rate: {learning_rate}, Batch size: {batch_size}")
    print(f"Convergence criterion: loss <= {convergence_loss}")
    print(f"Max steps: {max_steps}")
    if WANDB_AVAILABLE:
        print(f"Weights & Biases: Enabled (project: zero-order-rnn)")
    print("=" * 80)
    print()

    # Training loop
    start_time = time.time()
    converged = False
    best_loss = float('inf')
    min_seq_length_t = 1
    max_seq_length_t = 5

    for step in range(max_steps):
        step_start = time.time()

        # Generate batch
        x_ids, y_ids = generate_reverse_batch(batch_size, min_seq_length_t, max_seq_length_t, vocab_size, device)

        # Forward pass and backward pass
        optimizer.zero_grad()
        loss = teacher_forcing_loss_for_adam(model, x_ids, y_ids, criterion)
        loss.backward()

        # Compute gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(embed.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        loss_value = loss.item()
        best_loss = min(best_loss, loss_value)
        step_time = time.time() - step_start
        elapsed_time = time.time() - start_time

        # Compute accuracy periodically
        accuracy = None
        if step % 10 == 0 or loss_value <= convergence_loss:
            with torch.no_grad():
                x_emb = embed(x_ids)
                logits, _, _ = model(x_emb, require_gradients=False)
                accuracy = compute_reverse_accuracy(logits, y_ids, SEP, PAD)

        # Log to wandb
        if WANDB_AVAILABLE:
            log_dict = {
                "step": step,
                "loss": loss_value,
                "best_loss": best_loss,
                "grad_norm": grad_norm.item(),
                "step_time": step_time,
                "elapsed_time": elapsed_time,
            }
            if accuracy is not None:
                log_dict["accuracy"] = accuracy
            wandb.log(log_dict)

        # Print progress
        if step % 10 == 0 or loss_value <= convergence_loss:
            acc_str = f", Acc: {accuracy:.4f}" if accuracy is not None else ""
            print(f"Step {step:5d}: Loss={loss_value:.6f}, Best={best_loss:.6f}, "
                  f"Grad={grad_norm.item():.4f}{acc_str}, Time={step_time:.3f}s, Elapsed={elapsed_time:.3f}")

        # Check convergence
        if loss_value <= convergence_loss:
            final = True
            if min_seq_length_t < min_seq_length:
                new_min_seq_length_t = min_seq_length_t + 1
                print(f"Min seq length {min_seq_length_t} -> {new_min_seq_length_t} / {min_seq_length}")
                min_seq_length_t = new_min_seq_length_t
                final = False
            if max_seq_length_t < max_seq_length:
                new_max_seq_length_t = max_seq_length_t + 1
                print(f"Max seq length {max_seq_length_t} -> {new_max_seq_length_t} / {max_seq_length}")
                max_seq_length_t = new_max_seq_length_t
                final = False

            if final:
                converged = True
                print()
                print("=" * 80)
                print(f"✓ CONVERGED at step {step}!")
                print(f"  Final loss: {loss_value:.6f} <= {convergence_loss}")
                print(f"  Final accuracy: {accuracy:.4f}" if accuracy is not None else "")
                print(f"  Elapsed time: {elapsed_time:.2f}s")
                print(f"  Steps per second: {step / elapsed_time:.2f}")
                print("=" * 80)
                break

    if not converged:
        print()
        print("=" * 80)
        print(f"⚠ DID NOT CONVERGE after {max_steps} steps")
        print(f"  Final loss: {loss_value:.6f} > {convergence_loss}")
        print(f"  Best loss: {best_loss:.6f}")
        print(f"  Elapsed time: {elapsed_time:.2f}s")
        print("=" * 80)

    # Final summary to wandb
    if WANDB_AVAILABLE:
        wandb.log({
            "converged": converged,
            "final_loss": loss_value,
            "final_best_loss": best_loss,
            "total_steps": step + 1,
            "total_time": elapsed_time,
        })
        wandb.finish()

    return {
        'converged': converged,
        'final_loss': loss_value,
        'best_loss': best_loss,
        'steps': step + 1,
        'elapsed_time': elapsed_time,
    }


def main():
    parser = argparse.ArgumentParser(description='Train LSTM with Adam to convergence')

    # Model architecture
    parser.add_argument('--vocab_size', type=int, default=64, help='Vocabulary size')
    parser.add_argument('--min_seq_length', type=int, default=5, help='Minimum sequence length')
    parser.add_argument('--max_seq_length', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--hidden_size', type=int, default=240, help='Hidden size')
    parser.add_argument('--input_size', type=int, default=100, help='Input/embedding size')
    parser.add_argument('--num_heads', type=int, default=20, help='Number of attention heads')

    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.017678, help='Adam learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--convergence_loss', type=float, default=0.1, help='Loss threshold for convergence')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum training steps')

    # Other settings
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Train to convergence
    result = train_to_convergence(
        vocab_size=args.vocab_size,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.max_seq_length,
        hidden_size=args.hidden_size,
        input_size=args.input_size,
        num_heads=args.num_heads,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        convergence_loss=args.convergence_loss,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
    )

    print()
    print("Final Results:")
    print(f"  Converged: {result['converged']}")
    print(f"  Final loss: {result['final_loss']:.6f}")
    print(f"  Best loss: {result['best_loss']:.6f}")
    print(f"  Steps: {result['steps']}")
    print(f"  Time: {result['elapsed_time']:.2f}s")


if __name__ == '__main__':
    main()
