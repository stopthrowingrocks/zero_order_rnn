#!/usr/bin/env python3
"""
Quick local test of distributed_spsa_step logic without actual distribution.
Tests convergence on reverse task with world_size=1.
"""
import torch
import torch.nn as nn
import numpy as np
from models.models import LSTM

# Import functions from faster_grid_search_dist
import sys
sys.path.insert(0, '.')
from faster_grid_search_dist import (
    generate_reverse_batch,
    compute_reverse_loss,
    apply_probe,
    generate_perturbation
)

def simple_spsa_step(model, embed, x_ids, y_ids, pad_id, learning_rate, epsilon, num_perturbations):
    """Single-GPU SPSA step for testing."""
    param_list = list(model.parameters()) + list(embed.parameters())
    device = param_list[0].device
    distn = "normal"

    loss_pairs = torch.zeros(num_perturbations, 2, dtype=torch.float32, device=device)

    # Generate seeds
    seeds = torch.randint(0, 2**31 - 1, (num_perturbations,), dtype=torch.int32, device=device)

    # Evaluate all perturbations
    for m in range(num_perturbations):
        seed_m = int(seeds[m].item())

        # +ε
        apply_probe(param_list, +epsilon, seed_m, distn)
        x_emb = embed(x_ids)
        logits_plus, _, _ = model(x_emb, require_gradients=False)
        L_plus = compute_reverse_loss(logits_plus, y_ids, pad_id).item()

        # −ε
        apply_probe(param_list, -2.0 * epsilon, seed_m, distn)
        x_emb = embed(x_ids)
        logits_minus, _, _ = model(x_emb, require_gradients=False)
        L_minus = compute_reverse_loss(logits_minus, y_ids, pad_id).item()

        # Restore
        apply_probe(param_list, +epsilon, seed_m, distn)

        loss_pairs[m, 0] = L_plus
        loss_pairs[m, 1] = L_minus

    # Apply updates
    for i in range(num_perturbations):
        L_plus = loss_pairs[i, 0].item()
        L_minus = loss_pairs[i, 1].item()
        coef = (L_plus - L_minus) / (2.0 * num_perturbations)
        seed_i = int(seeds[i].item())
        apply_probe(param_list, -learning_rate * coef, seed_i, distn)

    return float(loss_pairs.mean().item())


def test_convergence():
    """Test that SPSA converges quickly on reverse task."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Small model for quick testing
    vocab_size = 64
    hidden_size = 240
    num_heads = 12
    input_size = 100
    batch_size = 16
    min_tokens = 5
    max_tokens = 10
    num_perturbations = 8

    # Test different learning rates
    learning_rates = [0.5, 0.2, 0.1, 0.05, 0.02]
    PAD = vocab_size - 1

    print("\nTesting SPSA convergence with different learning rates:")
    print("=" * 80)

    for lr in learning_rates:
        # Create fresh model
        torch.manual_seed(42)
        np.random.seed(42)

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

        epsilon = lr  # epsilon = learning_rate

        # Train for up to 100 steps or until loss < 0.1
        losses = []
        converged = False

        for step in range(100):
            # Generate batch
            x_ids, y_ids = generate_reverse_batch(
                batch_size, min_tokens, max_tokens, vocab_size, device
            )

            # SPSA step
            loss = simple_spsa_step(
                model, embed, x_ids, y_ids, PAD,
                lr, epsilon, num_perturbations
            )

            losses.append(loss)

            # Check convergence
            if loss < 0.1:
                converged = True
                print(f"LR={lr:.4f}: CONVERGED in {step+1} steps (final loss={loss:.6f})")
                break

            # Print progress every 10 steps
            if (step + 1) % 10 == 0:
                print(f"LR={lr:.4f}: Step {step+1}, Loss={loss:.6f}")

        if not converged:
            print(f"LR={lr:.4f}: Did NOT converge (final loss={losses[-1]:.6f})")

        print()


if __name__ == '__main__':
    with torch.no_grad():
        test_convergence()
