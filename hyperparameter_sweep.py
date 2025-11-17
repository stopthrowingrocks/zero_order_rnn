#!/usr/bin/env python3
"""
Directional hyperparameter sweeps for SPSA optimization.
Varies one hyperparameter at a time while keeping others fixed.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.models import LSTM

# Import telegram notifications (will fail silently if not available)
try:
    from telegram_notify import send_message, send_photo
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    def send_message(msg): pass
    def send_photo(path, caption=None): pass


def generate_reverse_batch(batch_size, min_seq_length, max_seq_length, vocab_size, device='cuda'):
    """Generate a batch of reverse sequences with variable lengths."""
    BOS = vocab_size - 4
    SEP = vocab_size - 3
    EOS = vocab_size - 2
    PAD = vocab_size - 1
    content_vocab_size = vocab_size - 4

    x_batch = []
    y_batch = []

    for _ in range(batch_size):
        seq_length = np.random.randint(min_seq_length, max_seq_length + 1)
        seq = np.random.randint(0, content_vocab_size, size=seq_length)
        full_seq = [BOS] + seq.tolist() + [SEP] + seq[::-1].tolist() + [EOS]

        x_batch.append(torch.tensor(full_seq, dtype=torch.long, device=device))
        y_batch.append(torch.tensor(full_seq, dtype=torch.long, device=device))

    x_ids = nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=PAD)
    y_ids = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=PAD)

    return x_ids, y_ids


def compute_reverse_loss(logits, targets, pad_id):
    """Compute cross-entropy loss, ignoring padding tokens."""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    loss = nn.functional.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=pad_id,
        reduction='mean'
    )
    return loss


def zeroth_order_step(model, embed, x_ids, y_ids, pad_id, learning_rate, epsilon, num_perturbations):
    """Perform a single zero-order optimization step using central difference."""
    with torch.no_grad():
        x_emb = embed(x_ids)
        logits, _, _ = model(x_emb, require_gradients=False)
        current_loss = compute_reverse_loss(logits, y_ids, pad_id).item()

    param_list = list(model.parameters()) + list(embed.parameters())
    pseudo_gradient = [torch.zeros_like(p) for p in param_list]

    for _ in range(num_perturbations):
        perturbations = [torch.randn_like(p) for p in param_list]

        with torch.no_grad():
            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=epsilon)

            x_emb = embed(x_ids)
            logits_plus, _, _ = model(x_emb, require_gradients=False)
            loss_plus = compute_reverse_loss(logits_plus, y_ids, pad_id).item()

            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=-2 * epsilon)

            x_emb = embed(x_ids)
            logits_minus, _, _ = model(x_emb, require_gradients=False)
            loss_minus = compute_reverse_loss(logits_minus, y_ids, pad_id).item()

            for p, pert in zip(param_list, perturbations):
                p.add_(pert, alpha=epsilon)

        grad_estimate = (loss_plus - loss_minus) / (2 * epsilon)
        for i, pert in enumerate(perturbations):
            pseudo_gradient[i].add_(pert, alpha=grad_estimate)

    for pg in pseudo_gradient:
        pg.div_(num_perturbations)

    with torch.no_grad():
        for p, pg in zip(param_list, pseudo_gradient):
            p.add_(pg, alpha=-learning_rate)

    return current_loss


def train_until_convergence(vocab_size, min_seq, max_seq, hidden_size, num_heads,
                             learning_rate, epsilon, num_perturbations, batch_size,
                             max_time, convergence_loss, device, seed):
    """Train SPSA until convergence or max_time (in seconds)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    PAD = vocab_size - 1

    embed = nn.Embedding(vocab_size, hidden_size, device=device, dtype=torch.bfloat16)
    model = LSTM(
        input_size=hidden_size,
        output_size=vocab_size,
        hidden_size=hidden_size,
        memory_size=0,
        head_size=hidden_size // num_heads,
        num_heads=num_heads,
        embed=embed,
        device=device,
        dtype=torch.bfloat16
    )

    start_time = time.time()
    converged = False
    step = 0
    losses = []

    while time.time() - start_time < max_time:
        x_ids, y_ids = generate_reverse_batch(batch_size, min_seq, max_seq, vocab_size, device)
        loss = zeroth_order_step(
            model, embed, x_ids, y_ids, PAD,
            learning_rate, epsilon, num_perturbations
        )
        losses.append(loss)
        step += 1

        if loss < convergence_loss:
            converged = True
            break

    elapsed = time.time() - start_time

    return {
        'converged': converged,
        'steps': step,
        'time': elapsed,
        'final_loss': losses[-1] if losses else float('inf'),
        'losses': losses
    }


def sweep_learning_rate(vocab_size, min_seq, max_seq, hidden_size, num_heads,
                         epsilon, num_perturbations, batch_size, max_time,
                         convergence_loss, device, seed):
    """Sweep learning_rate values."""
    # Learning rates to test (matching epsilon constraint: lr = epsilon)
    lr_values = np.logspace(-2, 0, 8)  # 0.01 to 1.0

    results = []

    print(f"\n{'='*80}")
    print(f"SWEEPING LEARNING RATE (epsilon={epsilon}, pert={num_perturbations}, batch={batch_size})")
    print(f"{'='*80}")

    for lr in lr_values:
        print(f"\nTesting LR={lr:.4f}...", end=" ", flush=True)

        result = train_until_convergence(
            vocab_size, min_seq, max_seq, hidden_size, num_heads,
            lr, epsilon, num_perturbations, batch_size,
            max_time, convergence_loss, device, seed
        )

        result['lr'] = lr
        results.append(result)

        if result['converged']:
            print(f"Converged in {result['steps']} steps ({result['time']:.2f}s)")
        else:
            print(f"Did not converge (final loss: {result['final_loss']:.4f}, steps: {result['steps']})")

    return results


def sweep_perturbations(vocab_size, min_seq, max_seq, hidden_size, num_heads,
                        learning_rate, epsilon, batch_size, max_time,
                        convergence_loss, device, seed):
    """Sweep num_perturbations values."""
    pert_values = [1, 2, 4, 8, 16, 32, 64, 128]

    results = []

    print(f"\n{'='*80}")
    print(f"SWEEPING PERTURBATIONS (lr={learning_rate}, epsilon={epsilon}, batch={batch_size})")
    print(f"{'='*80}")

    for pert in pert_values:
        print(f"\nTesting Pert={pert}...", end=" ", flush=True)

        result = train_until_convergence(
            vocab_size, min_seq, max_seq, hidden_size, num_heads,
            learning_rate, epsilon, pert, batch_size,
            max_time, convergence_loss, device, seed
        )

        result['pert'] = pert
        results.append(result)

        if result['converged']:
            print(f"Converged in {result['steps']} steps ({result['time']:.2f}s)")
        else:
            print(f"Did not converge (final loss: {result['final_loss']:.4f}, steps: {result['steps']})")

    return results


def sweep_batch_size(vocab_size, min_seq, max_seq, hidden_size, num_heads,
                     learning_rate, epsilon, num_perturbations, max_time,
                     convergence_loss, device, seed):
    """Sweep batch_size values."""
    batch_values = [4, 8, 16, 32, 64, 128, 256, 512]

    results = []

    print(f"\n{'='*80}")
    print(f"SWEEPING BATCH SIZE (lr={learning_rate}, epsilon={epsilon}, pert={num_perturbations})")
    print(f"{'='*80}")

    for batch in batch_values:
        print(f"\nTesting Batch={batch}...", end=" ", flush=True)

        result = train_until_convergence(
            vocab_size, min_seq, max_seq, hidden_size, num_heads,
            learning_rate, epsilon, num_perturbations, batch,
            max_time, convergence_loss, device, seed
        )

        result['batch'] = batch
        results.append(result)

        if result['converged']:
            print(f"Converged in {result['steps']} steps ({result['time']:.2f}s)")
        else:
            print(f"Did not converge (final loss: {result['final_loss']:.4f}, steps: {result['steps']})")

    return results


def plot_sweep_results(results, param_name, param_key, vocab_size, min_seq, max_seq,
                        fixed_params, output_dir):
    """Plot sweep results."""
    os.makedirs(output_dir, exist_ok=True)

    param_values = [r[param_key] for r in results]
    steps = [r['steps'] for r in results]
    times = [r['time'] for r in results]
    final_losses = [r['final_loss'] for r in results]
    converged = [r['converged'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'SPSA {param_name} Sweep (vocab={vocab_size}, seq={min_seq}-{max_seq})\n{fixed_params}',
                 fontsize=14, fontweight='bold')

    # Steps to convergence
    ax = axes[0, 0]
    colors = ['green' if c else 'red' for c in converged]
    ax.scatter(param_values, steps, c=colors, s=100, alpha=0.6)
    ax.plot(param_values, steps, 'b--', alpha=0.3)
    if param_name == 'Learning Rate' or param_name == 'Batch Size' or param_name == 'Perturbations':
        ax.set_xscale('log')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Steps to Convergence', fontsize=12)
    ax.set_title('Steps to Convergence (green=converged, red=not)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Time to convergence
    ax = axes[0, 1]
    ax.scatter(param_values, times, c=colors, s=100, alpha=0.6)
    ax.plot(param_values, times, 'b--', alpha=0.3)
    if param_name == 'Learning Rate' or param_name == 'Batch Size' or param_name == 'Perturbations':
        ax.set_xscale('log')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Wall-Clock Time to Convergence', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Final loss
    ax = axes[1, 0]
    ax.scatter(param_values, final_losses, c=colors, s=100, alpha=0.6)
    ax.plot(param_values, final_losses, 'b--', alpha=0.3)
    ax.axhline(y=0.05, color='r', linestyle='--', label='Convergence threshold')
    if param_name == 'Learning Rate' or param_name == 'Batch Size' or param_name == 'Perturbations':
        ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title('Final Loss Achieved', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss curves for converged runs
    ax = axes[1, 1]
    for i, r in enumerate(results):
        if r['converged']:
            ax.plot(r['losses'], label=f"{param_name}={r[param_key]:.4g}", alpha=0.7)
    ax.axhline(y=0.05, color='r', linestyle='--', label='Convergence threshold')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Curves (Converged Only)', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Generate filename
    param_abbrev = {
        'Learning Rate': 'lr',
        'Perturbations': 'pert',
        'Batch Size': 'batch'
    }[param_name]

    fixed_str = fixed_params.replace(' ', '_').replace('=', '').replace(',', '_')
    filename = f"sweep_{param_abbrev}_vocab{vocab_size}_seq{min_seq}-{max_seq}_{fixed_str}.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {filepath}")
    plt.close()


def main():
    # Fixed task parameters
    vocab_size = 20
    min_seq_length = 10
    max_seq_length = 100
    hidden_size = 128
    num_heads = 4
    max_time = 30.0  # seconds per experiment
    convergence_loss = 0.05
    device = 'cuda'
    seed = 42

    # Baseline SPSA hyperparameters (from previous best)
    baseline_lr = 0.1
    baseline_epsilon = 0.1
    baseline_pert = 8
    baseline_batch = 16

    output_dir = 'hpp-opt'

    print("="*80)
    print("SPSA HYPERPARAMETER DIRECTIONAL SWEEPS (Time-Based)")
    print("="*80)
    print(f"Task: Reverse sequences of length {min_seq_length}-{max_seq_length}")
    print(f"Vocab size: {vocab_size}, Hidden size: {hidden_size}")
    print(f"Max time per experiment: {max_time}s, Convergence criterion: loss < {convergence_loss}")
    print(f"Baseline: LR={baseline_lr}, Epsilon={baseline_epsilon}, Pert={baseline_pert}, Batch={baseline_batch}")
    print("="*80)

    # Sweep 1: Learning Rate (epsilon = lr)
    lr_results = sweep_learning_rate(
        vocab_size, min_seq_length, max_seq_length, hidden_size, num_heads,
        baseline_epsilon, baseline_pert, baseline_batch, max_time,
        convergence_loss, device, seed
    )
    plot_sweep_results(
        lr_results, 'Learning Rate', 'lr', vocab_size, min_seq_length, max_seq_length,
        f"epsilon={baseline_epsilon}, pert={baseline_pert}, batch={baseline_batch}",
        output_dir
    )

    # Sweep 2: Perturbations
    pert_results = sweep_perturbations(
        vocab_size, min_seq_length, max_seq_length, hidden_size, num_heads,
        baseline_lr, baseline_epsilon, baseline_batch, max_time,
        convergence_loss, device, seed
    )
    plot_sweep_results(
        pert_results, 'Perturbations', 'pert', vocab_size, min_seq_length, max_seq_length,
        f"lr={baseline_lr}, epsilon={baseline_epsilon}, batch={baseline_batch}",
        output_dir
    )

    # Sweep 3: Batch Size
    batch_results = sweep_batch_size(
        vocab_size, min_seq_length, max_seq_length, hidden_size, num_heads,
        baseline_lr, baseline_epsilon, baseline_pert, max_time,
        convergence_loss, device, seed
    )
    plot_sweep_results(
        batch_results, 'Batch Size', 'batch', vocab_size, min_seq_length, max_seq_length,
        f"lr={baseline_lr}, epsilon={baseline_epsilon}, pert={baseline_pert}",
        output_dir
    )

    print("\n" + "="*80)
    print("ALL SWEEPS COMPLETE")
    print("="*80)
    print(f"Plots saved to {output_dir}/")

    # Send Telegram notification
    if TELEGRAM_AVAILABLE:
        # Count converged runs
        lr_converged = sum(1 for r in lr_results if r['converged'])
        pert_converged = sum(1 for r in pert_results if r['converged'])
        batch_converged = sum(1 for r in batch_results if r['converged'])

        # Send summary message
        message = f"""**SPSA Hyperparameter Sweeps Complete!**

Task: Reverse (seq={min_seq_length}-{max_seq_length}, vocab={vocab_size})

**Results:**
- Learning Rate: {lr_converged}/8 converged
- Perturbations: {pert_converged}/8 converged
- Batch Size: {batch_converged}/8 converged

Plots saved to {output_dir}/"""

        send_message(message)

        # Send plots
        import glob
        plots = glob.glob(f"{output_dir}/*.png")
        for plot in plots:
            send_photo(plot, caption=os.path.basename(plot))


if __name__ == '__main__':
    main()
