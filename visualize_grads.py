#!/usr/bin/env python3
"""
Visualize gradients as images for multiple batches.

Creates a 4x3 grid showing gradient "heatmaps" where each pixel represents
a parameter value, colored red (negative) -> white (zero) -> blue (positive).

Example usage:
    python visualize_grads.py spsa_model.pt
    python visualize_grads.py adam_model.pt --n_batches 12 --batch_size 32
"""
import argparse
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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


def compute_gradient_for_batch(model, fixed_params, batch_size, min_seq_length,
                               max_seq_length, vocab_size, device):
    """
    Compute gradient for a single batch.

    Returns:
        gradient vector as 1D numpy array
    """
    # Freeze model at specific parameters
    model.load_state_dict(fixed_params)
    model.train()

    PAD = vocab_size - 1
    criterion = nn.CrossEntropyLoss(ignore_index=PAD).to(device)

    # Generate batch
    x_ids, y_ids = generate_reverse_batch(
        batch_size, min_seq_length, max_seq_length, vocab_size, device
    )

    # Zero gradients
    model.zero_grad()
    for p in model.embed.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # Compute loss and gradients
    loss = compute_loss(model, x_ids, y_ids, criterion, require_gradients=True)
    loss.backward()

    # Flatten all gradients into one vector
    grad_vec = []
    for p in model.parameters():
        if p.grad is not None:
            grad_vec.append(p.grad.flatten().detach().cpu().float())
    for p in model.embed.parameters():
        if p.grad is not None:
            grad_vec.append(p.grad.flatten().detach().cpu().float())

    grad_vec = torch.cat(grad_vec).numpy()

    return grad_vec


def gradient_to_image(grad_vec, image_size=None):
    """
    Convert a 1D gradient vector into a 2D image for visualization.

    Args:
        grad_vec: 1D numpy array of gradient values
        image_size: Tuple (height, width). If None, makes it as square as possible.

    Returns:
        2D numpy array representing the image
    """
    n_params = len(grad_vec)

    if image_size is None:
        # Make it as square as possible
        side = int(np.sqrt(n_params))
        height = side
        width = (n_params + side - 1) // side  # Ceiling division
    else:
        height, width = image_size

    # Pad gradient vector if necessary
    total_pixels = height * width
    if n_params < total_pixels:
        grad_vec_padded = np.zeros(total_pixels)
        grad_vec_padded[:n_params] = grad_vec
        grad_vec = grad_vec_padded
    elif n_params > total_pixels:
        # Truncate if too many parameters
        grad_vec = grad_vec[:total_pixels]

    # Reshape into 2D image
    image = grad_vec.reshape(height, width)

    return image


def visualize_gradient_grid(gradients, output_path, n_rows=4, n_cols=3,
                            vmin=None, vmax=None, figsize=(15, 20)):
    """
    Create a grid of gradient visualizations.

    Args:
        gradients: List of 1D gradient vectors
        output_path: Path to save the output image
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        vmin, vmax: Color scale limits. If None, uses percentiles across all gradients.
        figsize: Figure size
    """
    n_total = n_rows * n_cols
    n_gradients = len(gradients)

    if n_gradients < n_total:
        print(f"Warning: Only {n_gradients} gradients provided, but grid has {n_total} slots.")

    # Convert all gradients to images
    images = []
    for grad_vec in gradients[:n_total]:
        img = gradient_to_image(grad_vec)
        images.append(img)

    # Determine color scale limits across all gradients
    if vmin is None or vmax is None:
        all_values = np.concatenate([img.flatten() for img in images])
        if vmin is None:
            vmin = np.percentile(all_values, 1)  # 1st percentile
        if vmax is None:
            vmax = np.percentile(all_values, 99)  # 99th percentile

    # Make sure the color scale is symmetric around zero
    max_abs = max(abs(vmin), abs(vmax))
    vmin = -max_abs
    vmax = max_abs

    # Create normalization that centers white at zero
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Create grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_total):
        ax = axes[i]

        if i < len(images):
            img = images[i]

            # Plot with red-white-blue colormap
            im = ax.imshow(img, cmap='RdBu_r', norm=norm, aspect='auto')
            ax.set_title(f'Batch {i+1}', fontsize=10)
            ax.axis('off')
        else:
            # Empty subplot
            ax.axis('off')

    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='horizontal',
                fraction=0.02, pad=0.02, label='Gradient Value')

    plt.suptitle('Gradient Visualizations Across Different Batches\n'
                f'Red = Negative, White = Zero, Blue = Positive',
                fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved gradient grid to {output_path}")


def compute_gradient_statistics(gradients):
    """
    Compute statistics across multiple gradients.

    Args:
        gradients: List of 1D gradient vectors (numpy arrays)

    Returns:
        dict with statistics
    """
    # Stack into [n_batches, n_params]
    gradients_stacked = np.stack(gradients)

    mean_grad = gradients_stacked.mean(axis=0)
    std_grad = gradients_stacked.std(axis=0)

    mean_grad_norm = np.linalg.norm(mean_grad)
    mean_std = std_grad.mean()

    # Per-batch gradient norms
    grad_norms = [np.linalg.norm(g) for g in gradients]

    return {
        'mean_grad_norm': mean_grad_norm,
        'mean_std': mean_std,
        'grad_norms': grad_norms,
        'mean_grad_norm_across_batches': np.mean(grad_norms),
        'std_grad_norm_across_batches': np.std(grad_norms),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Visualize gradients as a grid of heatmaps'
    )
    parser.add_argument('model_path', type=str, help='Path to saved model (.pt file)')
    parser.add_argument('--n_batches', type=int, default=12,
                       help='Number of batches to visualize (default: 12)')
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
    parser.add_argument('--grid_rows', type=int, default=4,
                       help='Number of rows in grid (default: 4)')
    parser.add_argument('--grid_cols', type=int, default=3,
                       help='Number of columns in grid (default: 3)')

    args = parser.parse_args()

    # Validate grid size
    if args.n_batches > args.grid_rows * args.grid_cols:
        print(f"Warning: n_batches ({args.n_batches}) exceeds grid size "
              f"({args.grid_rows}x{args.grid_cols}). Only first "
              f"{args.grid_rows * args.grid_cols} will be shown.")

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
    print("GRADIENT VISUALIZATION CONFIGURATION")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"n_batches: {args.n_batches}")
    print(f"batch_size: {args.batch_size}")
    print(f"seq_length: {args.min_seq_length}-{args.max_seq_length}")
    print(f"vocab_size: {vocab_size}")
    print(f"device: {device}")
    print(f"grid: {args.grid_rows}x{args.grid_cols}")
    print(f"output_prefix: {output_prefix}")
    print("=" * 80)
    print()

    # Compute gradients for multiple batches
    print("=" * 80)
    print("COMPUTING GRADIENTS")
    print("=" * 80)
    gradients = []
    for i in range(args.n_batches):
        print(f"Computing gradient for batch {i+1}/{args.n_batches}...")
        grad_vec = compute_gradient_for_batch(
            model, fixed_params,
            args.batch_size, args.min_seq_length, args.max_seq_length,
            vocab_size, device
        )
        gradients.append(grad_vec)

    print(f"✓ Computed {len(gradients)} gradients")
    print(f"  Gradient vector size: {len(gradients[0]):,} parameters")
    print()

    # Compute statistics
    print("=" * 80)
    print("GRADIENT STATISTICS")
    print("=" * 80)
    stats = compute_gradient_statistics(gradients)
    print(f"Mean gradient norm:              {stats['mean_grad_norm']:.6f}")
    print(f"Mean std per parameter:          {stats['mean_std']:.6f}")
    print(f"Mean ||∇|| across batches:       {stats['mean_grad_norm_across_batches']:.6f}")
    print(f"Std of ||∇|| across batches:     {stats['std_grad_norm_across_batches']:.6f}")
    print()
    print("Per-batch gradient norms:")
    for i, norm in enumerate(stats['grad_norms']):
        print(f"  Batch {i+1:2d}: ||∇|| = {norm:.6f}")
    print()

    # Create visualization
    print("=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)
    output_path = f"{output_prefix}_gradient_grid.png"
    visualize_gradient_grid(
        gradients, output_path,
        n_rows=args.grid_rows, n_cols=args.grid_cols
    )
    print()

    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"✓ Gradient grid saved to: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()