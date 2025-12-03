#!/usr/bin/env python3
"""
Train LSTM with SPSA optimizer to convergence (loss <= 0.1) with wandb logging.

Example usage:
    python converge_spsa.py --learning_rate 0.1 --num_perturbations 8 --batch_size 32
    python converge_spsa.py --learning_rate 0.1 --num_perturbations 8 --batch_size 32 --wandb_project "zero-order-rnn"
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from models.models import LSTM
from shared import generate_reverse_batch

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠ wandb not available. Install with: pip install wandb")


def teacher_forcing_loss_for_spsa(model, x_ids, y_ids_unpadded, criterion, chunk_size=32):
    """
    Teacher forcing loss computation for SPSA (no gradients needed).
    Processes input and target sequences in chunks to compute loss over all output tokens.
    """
    with torch.no_grad():
        x_emb = model.embed(x_ids)

        next_param = next(model.parameters())
        if x_emb.dtype != next_param.dtype:
            x_emb = x_emb.to(dtype=next_param.dtype)
        Lx = x_emb.shape[1]
        Ly = y_ids_unpadded.shape[1]

        hidden = None
        memory = None
        # Use float32 for loss accumulation to avoid bfloat16 precision issues
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=x_ids.device)
        total_predicted_tokens = 0

        # Process input sequence first
        pos = 0
        while pos < Lx:
            chunk_end = min(pos + chunk_size, Lx)
            input_chunk = x_emb[:, pos:chunk_end, :]
            out_chunk, mem_new, hidden_new = model(input_chunk, hidden=hidden, memory=memory, require_gradients=False)
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

            out_chunk, mem_new, hidden_new = model(y_emb_chunk, hidden=hidden, memory=memory, require_gradients=False)

            # Update states
            hidden = hidden_new
            memory = mem_new

            # Compute loss for this chunk
            out_chunk = out_chunk.reshape(-1, out_chunk.size(-1))
            targets = y_ids_unpadded[:, pos+1:chunk_end+1].reshape(-1)  # shift by 1 for next-token prediction

            if targets.size(0) > 0:  # ensure we have targets
                chunk_loss = criterion(out_chunk, targets)
                # Convert to float32 before accumulation to avoid bfloat16 precision issues
                total_loss += chunk_loss.float() * targets.size(0)
                total_predicted_tokens += targets.size(0)

            pos = chunk_end

        if total_predicted_tokens == 0:
            return 0.0

        avg_loss = total_loss / total_predicted_tokens
        return avg_loss.item()


def compute_accuracy_iterative(model, x_ids, y_ids, sep_id, pad_id, chunk_size=32):
    """
    Compute accuracy using iterative teacher forcing (same as loss computation).
    This ensures accuracy is measured on the full autoregressive generation.
    """
    with torch.no_grad():
        x_emb = model.embed(x_ids)

        next_param = next(model.parameters())
        if x_emb.dtype != next_param.dtype:
            x_emb = x_emb.to(dtype=next_param.dtype)
        Lx = x_emb.shape[1]
        Ly = y_ids.shape[1]

        B = x_ids.shape[0]

        hidden = None
        memory = None

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
        all_predictions = []
        pos = 0
        while pos < Ly - 1:
            chunk_end = min(pos + chunk_size, Ly - 1)
            y_chunk = y_ids[:, pos:chunk_end]
            y_emb_chunk = model.embed(y_chunk)

            out_chunk, mem_new, hidden_new = model(y_emb_chunk, hidden=hidden, memory=memory, require_gradients=False)
            hidden = hidden_new
            memory = mem_new

            # Get predictions for this chunk
            preds_chunk = out_chunk.argmax(dim=-1)  # [B, chunk_len]
            all_predictions.append(preds_chunk)

            pos = chunk_end

        # Concatenate all predictions
        if all_predictions:
            preds = torch.cat(all_predictions, dim=1)  # [B, Ly-1]
        else:
            return 0.0

        # Compute accuracy only on output part (after SEP token)
        total_correct = 0
        total_tokens = 0

        for b in range(B):
            sep_positions = (y_ids[b] == sep_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) == 0:
                continue
            sep_pos = sep_positions[0].item()

            # Check predictions after SEP
            for t in range(sep_pos, min(sep_pos + preds.shape[1], Ly - 1)):
                target_pos = t + 1  # Shift by 1 for next-token prediction
                if target_pos >= Ly or y_ids[b, target_pos] == pad_id:
                    break

                pred_pos = t - sep_pos
                if pred_pos >= 0 and pred_pos < preds.shape[1]:
                    if preds[b, pred_pos] == y_ids[b, target_pos]:
                        total_correct += 1
                    total_tokens += 1

        return total_correct / max(total_tokens, 1)


def spsa_step(model, x_ids, y_ids, criterion, learning_rate, epsilon, num_perturbations):
    """Perform a single SPSA step using central difference."""
    # Get current loss
    current_loss = teacher_forcing_loss_for_spsa(model, x_ids, y_ids, criterion)

    # Collect all parameters
    param_list = list(model.parameters()) + list(model.embed.parameters())

    # Estimate gradient using random perturbations
    pseudo_gradient = [torch.zeros_like(p) for p in param_list]

    for _ in range(num_perturbations):
        # Generate random direction
        perturbations = [torch.randn_like(p) for p in param_list]

        # Positive perturbation
        with torch.no_grad():
            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=epsilon)

        loss_plus = teacher_forcing_loss_for_spsa(model, x_ids, y_ids, criterion)

        # Restore and apply negative perturbation
        with torch.no_grad():
            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=-2 * epsilon)

        loss_minus = teacher_forcing_loss_for_spsa(model, x_ids, y_ids, criterion)

        # Restore parameters
        with torch.no_grad():
            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=epsilon)

        # Accumulate gradient estimate
        grad_estimate = (loss_plus - loss_minus) / (2 * epsilon)
        for i, pert in enumerate(perturbations):
            pseudo_gradient[i].add_(pert, alpha=grad_estimate)

    # Average over perturbations
    for pg in pseudo_gradient:
        pg.div_(num_perturbations)

    # Compute pseudo gradient norm
    grad_norm = sum(pg.norm().item() ** 2 for pg in pseudo_gradient) ** 0.5

    # Update parameters
    with torch.no_grad():
        for p, pg in zip(param_list, pseudo_gradient):
            p.add_(pg, alpha=-learning_rate)

    return current_loss, grad_norm


def train_to_convergence(args, device):
    """Train model with SPSA to convergence with wandb logging."""

    # Initialize wandb if requested
    if WANDB_AVAILABLE:
        wandb.init(
            project="zero-order-rnn",
            name=f"spsa_lr_{args.learning_rate}_pert_{args.num_perturbations}_batch_{args.batch_size}_max_seq_{args.max_seq_length}",
            config={
                "optimizer": "spsa",
                "vocab_size": args.vocab_size,
                "min_seq_length": args.min_seq_length,
                "max_seq_length": args.max_seq_length,
                "min_seq_length_start": args.min_seq_length_start,
                "max_seq_length_start": args.max_seq_length_start,
                "hidden_size": args.hidden_size,
                "input_size": args.input_size,
                "num_heads": args.num_heads,
                "learning_rate": args.learning_rate,
                "num_perturbations": args.num_perturbations,
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

    epsilon = args.learning_rate  # For SPSA, epsilon = learning_rate

    # Create model
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

    # Create loss
    criterion = nn.CrossEntropyLoss(ignore_index=PAD).to(device)

    print("=" * 80)
    print("TRAINING TO CONVERGENCE WITH SPSA")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Task: Reverse sequences of length {args.min_seq_length}-{args.max_seq_length}")
    print(f"Vocab size: {args.vocab_size}, Hidden size: {args.hidden_size}, Input size: {args.input_size}")
    print(f"Num heads: {args.num_heads}, Total params: {total_params:,}")
    print(f"Learning rate: {args.learning_rate}, Num perturbations: {args.num_perturbations}, Batch size: {args.batch_size}")
    print(f"Convergence criterion: loss <= {args.convergence_loss}")
    print(f"Max steps: {args.max_steps}")
    if WANDB_AVAILABLE:
        print(f"Weights & Biases: Enabled (project: zero-order-rnn)")
    print("=" * 80)
    print()

    # Training loop
    start_time = time.time()
    converged = False
    best_loss = float('inf')
    min_seq_length_t = args.min_seq_length_start
    max_seq_length_t = args.max_seq_length_start

    for step in range(args.max_steps):
        step_start = time.time()

        # Generate batch
        x_ids, y_ids = generate_reverse_batch(args.batch_size, min_seq_length_t, max_seq_length_t, args.vocab_size, device)

        # SPSA step
        loss_value, grad_norm = spsa_step(
            model, x_ids, y_ids, criterion,
            args.learning_rate, epsilon, args.num_perturbations
        )

        best_loss = min(best_loss, loss_value)
        step_time = time.time() - step_start
        elapsed_time = time.time() - start_time

        # Compute accuracy periodically using iterative teacher forcing
        accuracy = None
        if step % 10 == 0 or loss_value <= args.convergence_loss:
            accuracy = compute_accuracy_iterative(model, x_ids, y_ids, SEP, PAD)

        # Log to wandb
        if WANDB_AVAILABLE:
            log_dict = {
                "step": step,
                "loss": loss_value,
                "best_loss": best_loss,
                "grad_norm": grad_norm,
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
                  f"Grad={grad_norm:.4f}{acc_str}, Time={step_time:.3f}s, Elapsed={elapsed_time:.3f}")

        # Check convergence
        if loss_value <= args.convergence_loss:
            final = True
            if min_seq_length_t < args.min_seq_length and min_seq_length_t * 2 <= max_seq_length_t:
                new_min_seq_length_t = min_seq_length_t + 1
                print(f"Min seq length {min_seq_length_t} -> {new_min_seq_length_t} / {args.min_seq_length}")
                min_seq_length_t = new_min_seq_length_t
                final = False
            if max_seq_length_t < args.max_seq_length:
                new_max_seq_length_t = max_seq_length_t + 1
                print(f"Max seq length {max_seq_length_t} -> {new_max_seq_length_t} / {args.max_seq_length}")
                max_seq_length_t = new_max_seq_length_t
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

    if not converged:
        print()
        print("=" * 80)
        print(f"⚠ DID NOT CONVERGE after {args.max_steps} steps")
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
    }


def main():
    parser = argparse.ArgumentParser(description='Train LSTM with SPSA to convergence')

    # Model architecture
    parser.add_argument('--vocab_size', type=int, default=64, help='Vocabulary size')
    parser.add_argument('--min_seq_length', type=int, default=5, help='Minimum sequence length')
    parser.add_argument('--max_seq_length', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--min_seq_length_start', type=int, default=1, help='Starting minimum sequence length')
    parser.add_argument('--max_seq_length_start', type=int, default=1, help='Starting maximum sequence length')
    parser.add_argument('--hidden_size', type=int, default=240, help='Hidden size')
    parser.add_argument('--input_size', type=int, default=100, help='Input/embedding size')
    parser.add_argument('--num_heads', type=int, default=20, help='Number of attention heads')

    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.1, help='SPSA learning rate')
    parser.add_argument('--num_perturbations', type=int, default=8, help='Number of SPSA perturbations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--convergence_loss', type=float, default=0.1, help='Loss threshold for convergence')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum training steps')

    # Other settings
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Train to convergence
    result = train_to_convergence(
        args=args,
        device=args.device,
    )

    print()
    print("Final Results:")
    print(f"  Converged: {result['converged']}")
    print(f"  Final loss: {result['final_loss']:.6f}")
    print(f"  Best loss: {result['best_loss']:.6f}")
    print(f"  Steps: {result['steps']}")
    print(f"  Time: {result['elapsed_time']:.2f}s")

    # Save model
    model_path = 'spsa_model.pt'
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
    }, model_path)
    print(f"\n✓ Model saved to {model_path}")


if __name__ == '__main__':
    main()
