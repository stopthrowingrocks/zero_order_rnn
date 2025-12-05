#!/usr/bin/env python3
"""
Diagnose loss and gradient noise in the optimization landscape.

This script measures:
1. Loss noise: variance in loss values across different batches
2. Gradient noise: variance in true gradients across different batches

Example usage:
    python diagnose_true_noise.py spsa_model.pt --n_batches 100
    python diagnose_true_noise.py adam_model.pt --n_batches 200 --batch_size 64
"""
import argparse
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.models import LSTM
from shared import compute_loss, generate_reverse_batch


def load_model(model_path, device='cuda'):
    """Load a saved model from checkpoint."""
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

    return model, vocab_size


def measure_loss_noise(model, fixed_params, n_batches, batch_size, min_seq_length,
                       max_seq_length, vocab_size, device):
    """
    Measure loss noise: variance in loss values across different batches.

    Args:
        model: The LSTM model
        fixed_params: Fixed model parameters (state_dict)
        n_batches: Number of batches to sample
        batch_size: Batch size
        min_seq_length: Minimum sequence length
        max_seq_length: Maximum sequence length
        vocab_size: Vocabulary size
        device: Device (cuda or cpu)

    Returns:
        dict with 'losses', 'mean', 'std', 'cv'
    """
    print(f"Measuring loss noise across {n_batches} batches...")

    # Freeze model at specific parameters
    model.load_state_dict(fixed_params)
    model.eval()

    PAD = vocab_size - 1
    criterion = nn.CrossEntropyLoss(ignore_index=PAD).to(device)

    losses = []
    for i in range(n_batches):
        if (i + 1) % 20 == 0:
            print(f"  Batch {i + 1}/{n_batches}")

        x_ids, y_ids = generate_reverse_batch(
            batch_size, min_seq_length, max_seq_length, vocab_size, device
        )

        with torch.no_grad():
            loss = compute_loss(model, x_ids, y_ids, criterion,
                              require_gradients=False)
        losses.append(loss.item())

    losses = np.array(losses)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    cv_loss = std_loss / mean_loss if mean_loss > 0 else 0.0

    return {
        'losses': losses,
        'mean': mean_loss,
        'std': std_loss,
        'cv': cv_loss
    }


def measure_gradient_noise(model, fixed_params, n_batches, batch_size, min_seq_length,
                           max_seq_length, vocab_size, device):
    """
    Measure gradient noise: variance in true gradients across different batches.

    Args:
        model: The LSTM model
        fixed_params: Fixed model parameters (state_dict)
        n_batches: Number of batches to sample
        batch_size: Batch size
        min_seq_length: Minimum sequence length
        max_seq_length: Maximum sequence length
        vocab_size: Vocabulary size
        device: Device (cuda or cpu)

    Returns:
        dict with statistics about gradient noise
    """
    print(f"Measuring gradient noise across {n_batches} batches...")

    # Freeze model at specific parameters
    model.load_state_dict(fixed_params)
    model.train()  # Need to be in train mode for gradients

    PAD = vocab_size - 1
    criterion = nn.CrossEntropyLoss(ignore_index=PAD).to(device)

    # Collect gradients from n_batches
    all_gradients = []

    for i in range(n_batches):
        if (i + 1) % 20 == 0:
            print(f"  Batch {i + 1}/{n_batches}")

        x_ids, y_ids = generate_reverse_batch(
            batch_size, min_seq_length, max_seq_length, vocab_size, device
        )

        # Zero gradients
        model.zero_grad()
        for p in model.embed.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Compute loss and gradients
        loss = compute_loss(model, x_ids, y_ids, criterion,
                          require_gradients=True)
        loss.backward()

        # Flatten all gradients into one vector
        grad_vec = []
        for p in model.parameters():
            if p.grad is not None:
                grad_vec.append(p.grad.flatten().detach().cpu().float())
        for p in model.embed.parameters():
            if p.grad is not None:
                grad_vec.append(p.grad.flatten().detach().cpu().float())

        grad_vec = torch.cat(grad_vec)
        all_gradients.append(grad_vec)

        # Restore model state (just to be safe)
        model.load_state_dict(fixed_params)

    # Stack into [n_batches, n_params] array
    all_gradients = torch.stack(all_gradients)  # [n_batches, n_params]

    print(f"  Collected gradients: shape {all_gradients.shape}")

    # Compute statistics
    mean_grad = all_gradients.mean(dim=0)  # E[∇L], shape [n_params]
    std_grad = all_gradients.std(dim=0)    # σ_∇ per parameter, shape [n_params]

    mean_grad_norm = mean_grad.norm().item()
    mean_std_grad = std_grad.mean().item()
    median_std_grad = std_grad.median().item()
    max_std_grad = std_grad.max().item()

    # Compute projection and deviation metrics for each gradient
    # For each gradient v_i:
    # - projection: (v_i · u) / ||u||  (how much in mean direction)
    # - deviation: ||v_i - u|| / ||u||  (normalized distance from mean)

    projections = []  # (v_i · u) / ||u||
    deviations = []   # ||v_i - u|| / ||u||

    for i in range(n_batches):
        v_i = all_gradients[i]  # [n_params]

        # Projection: (v_i · u) / ||u||
        projection = torch.dot(v_i, mean_grad).item() / mean_grad_norm
        projections.append(projection)

        # Deviation: ||v_i - u|| / ||u||
        deviation = (v_i - mean_grad).norm().item() / mean_grad_norm
        deviations.append(deviation)

    projections = np.array(projections)
    deviations = np.array(deviations)

    snr = mean_grad_norm / mean_std_grad if mean_std_grad > 0 else float('inf')

    return {
        'mean_grad_norm': mean_grad_norm,
        'mean_std_grad': mean_std_grad,
        'median_std_grad': median_std_grad,
        'max_std_grad': max_std_grad,
        'snr': snr,
        'projections': projections,  # (v_i · u) / ||u||
        'deviations': deviations,    # ||v_i - u|| / ||u||
        'all_gradients': all_gradients,  # Keep for further analysis if needed
        'mean_grad': mean_grad
    }


def plot_loss_noise(loss_stats, output_path):
    """Plot histogram of loss values."""
    losses = loss_stats['losses']
    mean_loss = loss_stats['mean']
    std_loss = loss_stats['std']
    cv_loss = loss_stats['cv']

    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    plt.axvline(mean_loss, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_loss:.6f}')
    plt.axvline(mean_loss + std_loss, color='orange', linestyle='--', linewidth=2,
                label=f'±1σ: {std_loss:.6f}')
    plt.axvline(mean_loss - std_loss, color='orange', linestyle='--', linewidth=2)

    plt.xlabel('Loss', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Loss Distribution Across {len(losses)} Batches\nCV = {cv_loss:.4f}',
              fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"✓ Saved loss histogram to {output_path}")


def plot_gradient_noise(grad_stats, output_path):
    """
    Plot two histograms for gradient noise:
    1. Projection: (v_i · u) / ||u||
    2. Deviation: ||v_i - u|| / ||u||
    """
    projections = grad_stats['projections']
    deviations = grad_stats['deviations']
    mean_grad_norm = grad_stats['mean_grad_norm']
    snr = grad_stats['snr']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Projection (v_i · u) / ||u||
    ax1.hist(projections, bins=30, alpha=0.7, edgecolor='black', color='green')
    ax1.axvline(np.mean(projections), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(projections):.4f}')
    ax1.axvline(np.mean(projections) + np.std(projections), color='orange',
                linestyle='--', linewidth=2,
                label=f'±1σ: {np.std(projections):.4f}')
    ax1.axvline(np.mean(projections) - np.std(projections), color='orange',
                linestyle='--', linewidth=2)
    ax1.set_xlabel('(v · u) / ||u||', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Gradient Projection onto Mean Direction\n||u|| = {mean_grad_norm:.4f}',
                  fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right plot: Deviation ||v_i - u|| / ||u||
    ax2.hist(deviations, bins=30, alpha=0.7, edgecolor='black', color='purple')
    ax2.axvline(np.mean(deviations), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(deviations):.4f}')
    ax2.axvline(np.mean(deviations) + np.std(deviations), color='orange',
                linestyle='--', linewidth=2,
                label=f'±1σ: {np.std(deviations):.4f}')
    ax2.axvline(np.mean(deviations) - np.std(deviations), color='orange',
                linestyle='--', linewidth=2)
    ax2.set_xlabel('||v - u|| / ||u||', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Gradient Deviation from Mean\nSNR = {snr:.4f}',
                  fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"✓ Saved gradient histogram to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose loss and gradient noise in optimization landscape'
    )
    parser.add_argument('model_path', type=str, help='Path to saved model (.pt file)')
    parser.add_argument('--n_batches', type=int, default=100,
                       help='Number of batches to sample (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--min_seq_length', type=int, default=5,
                       help='Minimum sequence length (default: 5)')
    parser.add_argument('--max_seq_length', type=int, default=10,
                       help='Maximum sequence length (default: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output_prefix', type=str, default=None,
                       help='Output file prefix (default: model name)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    model, vocab_size = load_model(args.model_path, device)
    print(f"✓ Loaded model from {args.model_path}")
    print()

    # Save current parameters (this is our fixed θ)
    fixed_params = copy.deepcopy(model.state_dict())

    # Determine output prefix
    if args.output_prefix is None:
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        output_prefix = model_name
    else:
        output_prefix = args.output_prefix

    print("=" * 80)
    print("NOISE ANALYSIS CONFIGURATION")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Analyzing at current model parameters")
    print(f"n_batches: {args.n_batches}")
    print(f"batch_size: {args.batch_size}")
    print(f"seq_length: {args.min_seq_length}-{args.max_seq_length}")
    print(f"vocab_size: {vocab_size}")
    print(f"device: {device}")
    print(f"output_prefix: {output_prefix}")
    print("=" * 80)
    print()

    # 1. Measure loss noise
    print("=" * 80)
    print("1. MEASURING LOSS NOISE")
    print("=" * 80)
    loss_stats = measure_loss_noise(
        model, fixed_params, args.n_batches,
        args.batch_size, args.min_seq_length, args.max_seq_length,
        vocab_size, device
    )
    print()
    print("Loss Statistics:")
    print(f"  Mean loss:       {loss_stats['mean']:.6f}")
    print(f"  Std loss:        {loss_stats['std']:.6f}")
    print(f"  CV (σ/μ):        {loss_stats['cv']:.6f}")
    print(f"  Min loss:        {loss_stats['losses'].min():.6f}")
    print(f"  Max loss:        {loss_stats['losses'].max():.6f}")
    print()

    # Plot loss noise
    loss_plot_path = f"{output_prefix}_loss_noise.png"
    plot_loss_noise(loss_stats, loss_plot_path)
    print()

    # 2. Measure gradient noise
    print("=" * 80)
    print("2. MEASURING GRADIENT NOISE")
    print("=" * 80)
    grad_stats = measure_gradient_noise(
        model, fixed_params, args.n_batches,
        args.batch_size, args.min_seq_length, args.max_seq_length,
        vocab_size, device
    )
    print()
    print("Gradient Statistics:")
    print(f"  Mean gradient norm:      {grad_stats['mean_grad_norm']:.6f}")
    print(f"  Mean std per parameter:  {grad_stats['mean_std_grad']:.6f}")
    print(f"  Median std per parameter: {grad_stats['median_std_grad']:.6f}")
    print(f"  Max std per parameter:   {grad_stats['max_std_grad']:.6f}")
    print(f"  SNR (||u|| / σ_avg):     {grad_stats['snr']:.6f}")
    print()
    print("Projection Statistics (v · u) / ||u||:")
    print(f"  Mean:  {np.mean(grad_stats['projections']):.6f}")
    print(f"  Std:   {np.std(grad_stats['projections']):.6f}")
    print(f"  Min:   {np.min(grad_stats['projections']):.6f}")
    print(f"  Max:   {np.max(grad_stats['projections']):.6f}")
    print()
    print("Deviation Statistics ||v - u|| / ||u||:")
    print(f"  Mean:  {np.mean(grad_stats['deviations']):.6f}")
    print(f"  Std:   {np.std(grad_stats['deviations']):.6f}")
    print(f"  Min:   {np.min(grad_stats['deviations']):.6f}")
    print(f"  Max:   {np.max(grad_stats['deviations']):.6f}")
    print()

    # Plot gradient noise
    grad_plot_path = f"{output_prefix}_gradient_noise.png"
    plot_gradient_noise(grad_stats, grad_plot_path)
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"✓ Loss noise plot:     {loss_plot_path}")
    print(f"✓ Gradient noise plot: {grad_plot_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
