#!/usr/bin/env python3
"""
Visualize SPSA gradient estimates as images for multiple batches.

Creates a 4x3 grid showing SPSA gradient estimate "heatmaps" where each pixel
represents a parameter value, colored red (negative) -> white (zero) -> blue (positive).

Example usage:
    python visualize_grads_spsa.py spsa_model.pt
    python visualize_grads_spsa.py adam_model.pt --spsa_epsilon 0.1 --spsa_perturbations 8
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


def estimate_spsa_gradient(model, x_ids, y_ids, criterion, epsilon, num_perturbations):
    """
    Estimate gradient using SPSA (same logic as spsa_step in converge_spsa.py).

    Args:
        model: The LSTM model
        x_ids: Input token IDs
        y_ids: Target token IDs
        criterion: Loss criterion
        epsilon: Perturbation scale
        num_perturbations: Number of perturbations to average over

    Returns:
        SPSA gradient estimate as 1D numpy array
    """
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

        loss_plus = compute_loss(model, x_ids, y_ids, criterion, require_gradients=False).item()

        # Restore and apply negative perturbation
        with torch.no_grad():
            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=-2 * epsilon)

        loss_minus = compute_loss(model, x_ids, y_ids, criterion, require_gradients=False).item()

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

    # Flatten into vector
    spsa_grad_vec = torch.cat([pg.flatten().cpu().float() for pg in pseudo_gradient]).numpy()

    return spsa_grad_vec


def compute_spsa_gradient_for_batch(model, fixed_params, batch_size, min_seq_length,
                                    max_seq_length, vocab_size, epsilon, num_perturbations,
                                    device):
    """
    Compute SPSA gradient estimate for a single batch.

    Returns:
        gradient vector as 1D numpy array
    """
    # Freeze model at specific parameters
    model.load_state_dict(fixed_params)
    model.eval()

    PAD = vocab_size - 1
    criterion = nn.CrossEntropyLoss(ignore_index=PAD).to(device)

    # Generate batch
    x_ids, y_ids = generate_reverse_batch(
        batch_size, min_seq_length, max_seq_length, vocab_size, device
    )

    # Estimate gradient using SPSA
    spsa_grad = estimate_spsa_gradient(
        model, x_ids, y_ids, criterion, epsilon, num_perturbations
    )

    # Restore model state (SPSA modifies parameters)
    model.load_state_dict(fixed_params)

    return spsa_grad


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
                            vmin=None, vmax=None, figsize=(15, 20),
                            title_prefix='SPSA Gradient'):
    """
    Create a grid of gradient visualizations.

    Args:
        gradients: List of 1D gradient vectors
        output_path: Path to save the output image
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        vmin, vmax: Color scale limits. If None, uses percentiles across all gradients.
        figsize: Figure size
        title_prefix: Prefix for subplot titles
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
            ax.set_title(f'{title_prefix} {i+1}', fontsize=10)
            ax.axis('off')
        else:
            # Empty subplot
            ax.axis('off')

    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='horizontal',
                fraction=0.02, pad=0.02, label='Gradient Value')

    plt.suptitle('SPSA Gradient Estimates Across Different Batches\n'
                f'Red = Negative, White = Zero, Blue = Positive',
                fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved SPSA gradient grid to {output_path}")


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
        description='Visualize SPSA gradient estimates as a grid of heatmaps'
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
    parser.add_argument('--spsa_epsilon', type=float, default=0.1,
                       help='SPSA perturbation scale (default: 0.1)')
    parser.add_argument('--spsa_perturbations', type=int, default=8,
                       help='Number of SPSA perturbations (default: 8)')
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
    print("SPSA GRADIENT VISUALIZATION CONFIGURATION")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"n_batches: {args.n_batches}")
    print(f"batch_size: {args.batch_size}")
    print(f"seq_length: {args.min_seq_length}-{args.max_seq_length}")
    print(f"vocab_size: {vocab_size}")
    print(f"SPSA epsilon: {args.spsa_epsilon}")
    print(f"SPSA perturbations: {args.spsa_perturbations}")
    print(f"device: {device}")
    print(f"grid: {args.grid_rows}x{args.grid_cols}")
    print(f"output_prefix: {output_prefix}")
    print("=" * 80)
    print()

    # Compute SPSA gradients for multiple batches
    print("=" * 80)
    print("COMPUTING SPSA GRADIENT ESTIMATES")
    print("=" * 80)
    gradients = []
    for i in range(args.n_batches):
        print(f"Computing SPSA gradient for batch {i+1}/{args.n_batches}...")
        grad_vec = compute_spsa_gradient_for_batch(
            model, fixed_params,
            args.batch_size, args.min_seq_length, args.max_seq_length,
            vocab_size, args.spsa_epsilon, args.spsa_perturbations, device
        )
        gradients.append(grad_vec)

    print(f"✓ Computed {len(gradients)} SPSA gradient estimates")
    print(f"  Gradient vector size: {len(gradients[0]):,} parameters")
    print()

    # Compute statistics
    print("=" * 80)
    print("SPSA GRADIENT STATISTICS")
    print("=" * 80)
    stats = compute_gradient_statistics(gradients)
    print(f"Mean SPSA gradient norm:         {stats['mean_grad_norm']:.6f}")
    print(f"Mean std per parameter:          {stats['mean_std']:.6f}")
    print(f"Mean ||∇_SPSA|| across batches:  {stats['mean_grad_norm_across_batches']:.6f}")
    print(f"Std of ||∇_SPSA|| across batches: {stats['std_grad_norm_across_batches']:.6f}")
    print()
    print("Per-batch SPSA gradient norms:")
    for i, norm in enumerate(stats['grad_norms']):
        print(f"  Batch {i+1:2d}: ||∇_SPSA|| = {norm:.6f}")
    print()

    # Create visualization
    print("=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)
    output_path = f"{output_prefix}_spsa_gradient_grid.png"
    visualize_gradient_grid(
        gradients, output_path,
        n_rows=args.grid_rows, n_cols=args.grid_cols,
        title_prefix='SPSA Batch'
    )
    print()

    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"✓ SPSA gradient grid saved to: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()