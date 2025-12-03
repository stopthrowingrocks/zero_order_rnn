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
from shared import generate_reverse_batch, compute_reverse_loss, compute_accuracy

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
    args, device
):
    """Train model with Adam to convergence with wandb logging."""

    # Initialize wandb if requested
    if WANDB_AVAILABLE:
        wandb.init(
            project="zero-order-rnn",
            name=f"adam_lr_{args.learning_rate}_batch_{args.batch_size}_max_seq_{args.max_seq_length}",
            config={
                "optimizer": "adam",
                "vocab_size": args.vocab_size,
                "min_seq_length": args.min_seq_length,
                "max_seq_length": args.max_seq_length,
                "hidden_size": args.hidden_size,
                "input_size": args.input_size,
                "num_heads": args.num_heads,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "convergence_loss": args.convergence_loss,
                "seed": args.seed,
            }
        )

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    SEP = args.vocab_size - 3
    PAD = args.vocab_size - 1

    # Load model from checkpoint if resuming
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)

        # Override args with checkpoint config to ensure consistency
        args.vocab_size = checkpoint['vocab_size']
        args.hidden_size = checkpoint['hidden_size']
        args.input_size = checkpoint['input_size']
        args.num_heads = checkpoint['num_heads']

        # Create model with checkpoint config
        embed = nn.Embedding(args.vocab_size, args.input_size, device=device, dtype=torch.bfloat16)
        model = LSTM(
            input_size=args.input_size,
            output_size=args.vocab_size,
            hidden_size=args.hidden_size,
            memory_size=0,
            head_size=args.hidden_size // args.num_heads,
            num_heads=args.num_heads,
            embed=embed,
            device=device,
            dtype=torch.bfloat16
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        embed.load_state_dict(checkpoint['embed_state_dict'])

        print(f"✓ Loaded checkpoint (previous steps: {checkpoint.get('steps', 'unknown')}, " +
              f"previous loss: {checkpoint.get('final_loss', 'unknown'):.6f})")
    else:
        # Create fresh model
        embed = nn.Embedding(args.vocab_size, args.input_size, device=device, dtype=torch.bfloat16)
        model = LSTM(
            input_size=args.input_size,
            output_size=args.vocab_size,
            hidden_size=args.hidden_size,
            memory_size=0,
            head_size=args.hidden_size // args.num_heads,
            num_heads=args.num_heads,
            embed=embed,
            device=device,
            dtype=torch.bfloat16
        )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in embed.parameters())

    # Save initial parameters for drift calculation
    initial_params = []
    for p in list(model.parameters()) + list(embed.parameters()):
        initial_params.append(p.detach().clone())

    # Create optimizer and loss
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(embed.parameters()),
        lr=args.learning_rate
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD).to(device)

    print("=" * 80)
    print("TRAINING TO CONVERGENCE WITH ADAM")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Task: Reverse sequences of length {args.min_seq_length}-{args.max_seq_length}")
    print(f"Vocab size: {args.vocab_size}, Hidden size: {args.hidden_size}, Input size: {args.input_size}")
    print(f"Num heads: {args.num_heads}, Total params: {total_params:,}")
    print(f"Learning rate: {args.learning_rate}, Batch size: {args.batch_size}")
    print(f"Convergence criterion: loss <= {args.convergence_loss}")
    print(f"Max time: {args.max_time}s")
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
    step = 0

    while time.time() - start_time < args.max_time:
        step_start = time.time()

        # Generate batch
        x_ids, y_ids = generate_reverse_batch(args.batch_size, min_seq_length_t, max_seq_length_t, args.vocab_size, device)

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

        # Compute accuracy periodically using iterative teacher forcing
        accuracy = None
        if step % 10 == 0 or loss_value <= args.convergence_loss:
            accuracy = compute_accuracy(model, x_ids, y_ids, SEP, PAD)

        # Log to wandb
        if WANDB_AVAILABLE:
            log_dict = {
                "step": step,
                "loss": loss_value,
                "best_loss": best_loss,
                "grad_norm": grad_norm.item(),
                "step_time": step_time,
                "elapsed_time": elapsed_time,
                "min_seq_length_t": min_seq_length_t,
                "max_seq_length_t": max_seq_length_t,
            }
            if accuracy is not None:
                log_dict["accuracy"] = accuracy
            wandb.log(log_dict)

        # Print progress
        if step % 10 == 0 or loss_value <= args.convergence_loss:
            acc_str = f", Acc: {accuracy:.4f}" if accuracy is not None else ""
            print(f"Step {step:5d}: Loss={loss_value:.6f}, Best={best_loss:.6f}, "
                  f"Grad={grad_norm.item():.4f}{acc_str}, Time={step_time:.3f}s, Elapsed={elapsed_time:.3f}")

        # Check convergence
        if loss_value <= args.convergence_loss:
            final = True
            if min_seq_length_t < args.min_seq_length:
                new_min_seq_length_t = min_seq_length_t + 1
                print(f"Min seq length {min_seq_length_t} -> {new_min_seq_length_t} / {args.min_seq_length}")
                min_seq_length_t = new_min_seq_length_t
                best_loss = float('inf')  # Reset best_loss for new curriculum stage
                final = False
            if max_seq_length_t < args.max_seq_length:
                new_max_seq_length_t = max_seq_length_t + 1
                print(f"Max seq length {max_seq_length_t} -> {new_max_seq_length_t} / {args.max_seq_length}")
                max_seq_length_t = new_max_seq_length_t
                best_loss = float('inf')  # Reset best_loss for new curriculum stage
                final = False

            if final:
                converged = True
                print()
                print("=" * 80)
                print(f"✓ CONVERGED at step {step}!")
                print(f"  Final loss: {loss_value:.6f} <= {args.convergence_loss}")
                print(f"  Final accuracy: {accuracy:.4f}" if accuracy is not None else "")
                print(f"  Elapsed time: {elapsed_time:.2f}s")
                print(f"  Steps per second: {step / elapsed_time:.2f}")
                print("=" * 80)
                break

        step += 1

    if not converged:
        print()
        print("=" * 80)
        print(f"⚠ DID NOT CONVERGE after {elapsed_time:.2f}s ({step} steps)")
        print(f"  Final loss: {loss_value:.6f} > {args.convergence_loss}")
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
        'model': model,
        'embed': embed,
        'initial_params': initial_params,
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
    parser.add_argument('--max_time', type=float, default=600.0, help='Maximum training time in seconds')

    # Other settings
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Path to model checkpoint to resume training from')
    parser.add_argument('--save_to', type=str, default="adam_model.pt", help='Path to save model checkpoint at')

    args = parser.parse_args()

    # Train to convergence
    result = train_to_convergence(args, device=args.device)

    print()
    print("Final Results:")
    print(f"  Converged: {result['converged']}")
    print(f"  Final loss: {result['final_loss']:.6f}")
    print(f"  Best loss: {result['best_loss']:.6f}")
    print(f"  Steps: {result['steps']}")
    print(f"  Time: {result['elapsed_time']:.2f}s")

    # Calculate parameter drift from initial parameters
    print()
    print("=" * 80)
    print("PARAMETER DRIFT ANALYSIS")
    print("=" * 80)

    current_params = list(result['model'].parameters()) + list(result['embed'].parameters())
    initial_params = result['initial_params']

    # Calculate L2 norm of drift for each parameter
    total_drift_l2 = 0.0
    total_drift_linf = 0.0
    total_initial_norm = 0.0

    for p_init, p_curr in zip(initial_params, current_params):
        drift = p_curr - p_init
        drift_l2 = drift.norm(p=2).item()
        drift_linf = drift.abs().max().item()
        initial_norm = p_init.norm(p=2).item()

        total_drift_l2 += drift_l2 ** 2
        total_drift_linf = max(total_drift_linf, drift_linf)
        total_initial_norm += initial_norm ** 2

    total_drift_l2 = total_drift_l2 ** 0.5
    total_initial_norm = total_initial_norm ** 0.5

    print(f"Total parameter L2 norm (initial): {total_initial_norm:.6f}")
    print(f"Total parameter drift L2 norm: {total_drift_l2:.6f}")
    print(f"Total parameter drift L∞ norm: {total_drift_linf:.6f}")
    print(f"Relative drift (L2): {(total_drift_l2 / total_initial_norm):.6f}")
    print("=" * 80)

    # Save model
    torch.save({
        'model_state_dict': result['model'].state_dict(),
        'embed_state_dict': result['embed'].state_dict(),
        'vocab_size': args.vocab_size,
        'hidden_size': args.hidden_size,
        'input_size': args.input_size,
        'num_heads': args.num_heads,
        'converged': result['converged'],
        'final_loss': result['final_loss'],
        'steps': result['steps'],
    }, args.save_to)
    print(f"\n✓ Model saved to {args.save_to}")


if __name__ == '__main__':
    main()
