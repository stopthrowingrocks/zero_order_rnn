#!/usr/bin/env python3
"""
Learning rate sweep for perturbations=2 to see if any LR can make it converge.
Tests a wide range of learning rates from 0.001 to 1.0 with 20 values.
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
    """Train SPSA until convergence or max_time (in seconds). Returns losses with timestamps."""
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
    timestamps = []

    while time.time() - start_time < max_time:
        x_ids, y_ids = generate_reverse_batch(batch_size, min_seq, max_seq, vocab_size, device)
        loss = zeroth_order_step(
            model, embed, x_ids, y_ids, PAD,
            learning_rate, epsilon, num_perturbations
        )
        losses.append(loss)
        timestamps.append(time.time() - start_time)
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
        'losses': losses,
        'timestamps': timestamps
    }


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

    # Fixed SPSA hyperparameters
    num_perturbations = 2  # Testing pert=2 specifically
    epsilon = 0.1
    batch_size = 16

    # Wide learning rate sweep: 20 values from 0.001 to 1.0
    lr_min = 0.001
    lr_max = 1.0
    lr_values = np.geomspace(lr_min, lr_max, 20)

    output_dir = 'hpp-opt'
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("LEARNING RATE SWEEP FOR PERTURBATIONS=2 (20 values)")
    print("="*80)
    print(f"Task: Reverse sequences of length {min_seq_length}-{max_seq_length}")
    print(f"Vocab size: {vocab_size}, Hidden size: {hidden_size}")
    print(f"LR range: {lr_min:.4f} to {lr_max:.4f} (geometric spacing)")
    print(f"Fixed: perturbations={num_perturbations}, epsilon={epsilon}, batch={batch_size}")
    print(f"Max time per experiment: {max_time}s, Convergence criterion: loss < {convergence_loss}")
    print("="*80)
    print()

    results = []

    for i, lr in enumerate(lr_values):
        print(f"[{i+1}/20] Testing LR={lr:.4f}...", end=" ", flush=True)

        result = train_until_convergence(
            vocab_size, min_seq_length, max_seq_length, hidden_size, num_heads,
            lr, epsilon, num_perturbations, batch_size,
            max_time, convergence_loss, device, seed
        )

        result['lr'] = lr
        results.append(result)

        if result['converged']:
            print(f"Converged in {result['steps']} steps ({result['time']:.2f}s)")
        else:
            print(f"Did not converge (final loss: {result['final_loss']:.4f}, steps: {result['steps']})")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Learning Rate Sweep for Perturbations=2 (20 values)\n'
                 f'Task: Reverse (seq={min_seq_length}-{max_seq_length}, vocab={vocab_size})\n'
                 f'Fixed: pert={num_perturbations}, epsilon={epsilon}, batch={batch_size}', fontsize=12)

    # Extract data
    lrs = [r['lr'] for r in results]
    converged = [r['converged'] for r in results]
    times = [r['time'] if r['converged'] else max_time for r in results]
    steps = [r['steps'] for r in results]
    final_losses = [r['final_loss'] for r in results]

    # Colors: green for converged, red for not converged
    colors = ['green' if c else 'red' for c in converged]

    # Plot 1: Convergence status
    ax = axes[0, 0]
    ax.scatter(lrs, [1 if c else 0 for c in converged], c=colors, s=100, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate', fontsize=10)
    ax.set_ylabel('Converged (1=Yes, 0=No)', fontsize=10)
    ax.set_title('Convergence Status vs Learning Rate', fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.grid(True, alpha=0.3)

    # Plot 2: Time to convergence (only converged runs)
    ax = axes[0, 1]
    converged_lrs = [r['lr'] for r in results if r['converged']]
    converged_times = [r['time'] for r in results if r['converged']]
    if converged_lrs:
        ax.scatter(converged_lrs, converged_times, c='green', s=100, alpha=0.6)
        ax.plot(converged_lrs, converged_times, 'g--', alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate', fontsize=10)
    ax.set_ylabel('Time to Convergence (s)', fontsize=10)
    ax.set_title('Time to Convergence (Converged Runs Only)', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: Loss vs Time curves for ALL runs
    ax = axes[1, 0]
    for r in results:
        if len(r['timestamps']) > 0:
            color = 'green' if r['converged'] else 'red'
            linestyle = '-' if r['converged'] else '--'
            ax.plot(r['timestamps'], r['losses'], color=color, alpha=0.5, linewidth=1.5,
                   linestyle=linestyle, label=f"LR={r['lr']:.4f}")
    ax.axhline(y=convergence_loss, color='blue', linestyle='--', linewidth=2,
              label=f'Convergence threshold ({convergence_loss})')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title('Loss vs Time (All Runs: Green=Converged, Red=Failed)', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    # Don't show legend - too many lines

    # Plot 4: Final loss (all runs)
    ax = axes[1, 1]
    ax.scatter(lrs, final_losses, c=colors, s=100, alpha=0.6)
    ax.axhline(y=convergence_loss, color='blue', linestyle='--', linewidth=2,
              label=f'Convergence threshold ({convergence_loss})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Learning Rate', fontsize=10)
    ax.set_ylabel('Final Loss', fontsize=10)
    ax.set_title('Final Loss vs Learning Rate', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = f'{output_dir}/lr_sweep_pert2_20values_vocab{vocab_size}_seq{min_seq_length}-{max_seq_length}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {plot_filename}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    num_converged = sum(converged)
    print(f"Converged: {num_converged}/20 ({100*num_converged/20:.1f}%)")

    if num_converged > 0:
        print(f"\nConverged runs:")
        print(f"  LR range: {min(converged_lrs):.4f} to {max(converged_lrs):.4f}")
        print(f"  Best time: {min(converged_times):.2f}s at LR={converged_lrs[converged_times.index(min(converged_times))]:.4f}")
        best_lr_idx = converged_times.index(min(converged_times))
        best_lr = converged_lrs[best_lr_idx]
        best_steps = [r['steps'] for r in results if r['converged'] and r['lr'] == best_lr][0]
        print(f"  Best steps: {best_steps} at LR={best_lr:.4f}")
    else:
        print(f"\nNo runs converged! Perturbations=2 may be fundamentally too noisy.")

    print("="*80)

    # Send Telegram notification
    if TELEGRAM_AVAILABLE:
        message = f"""**LR Sweep for Pert=2 Complete!**

Task: Reverse (seq={min_seq_length}-{max_seq_length}, vocab={vocab_size})

**Results:**
- Tested 20 LR values from {lr_min:.4f} to {lr_max:.4f}
- Converged: {num_converged}/20 ({100*num_converged/20:.1f}%)

Fixed: pert={num_perturbations}, epsilon={epsilon}, batch={batch_size}"""

        if num_converged > 0:
            best_lr = converged_lrs[converged_times.index(min(converged_times))]
            message += f"\n\nBest: LR={best_lr:.4f} in {min(converged_times):.2f}s"
        else:
            message += f"\n\n**No convergence!** Pert=2 may be too noisy."

        send_message(message)
        send_photo(plot_filename, caption=os.path.basename(plot_filename))


if __name__ == '__main__':
    main()
