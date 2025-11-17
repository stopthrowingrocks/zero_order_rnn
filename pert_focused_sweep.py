#!/usr/bin/env python3
"""
Focused perturbation sweep with 16 integer values between 2 and 32.
Uses best learning rate from previous sweep: LR=0.1808
Plots loss vs time instead of loss vs step.
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

    # Fixed SPSA hyperparameters (using best LR from previous sweep)
    learning_rate = 0.1808  # Best from LR sweep
    epsilon = 0.1
    batch_size = 16

    # Focused perturbation sweep: 16 integer values between 2 and 32
    # From original sweep: pert=2 failed (diverged), pert=4,8,16 converged, pert=32 failed
    pert_min = 2
    pert_max = 32
    # Generate 16 linearly spaced values and round to integers, ensuring uniqueness
    pert_values = np.unique(np.round(np.linspace(pert_min, pert_max, 16)).astype(int))
    pert_values = sorted(pert_values)  # Sort to ensure monotonic order

    output_dir = 'hpp-opt'
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("FOCUSED PERTURBATION SWEEP (16 integer values)")
    print("="*80)
    print(f"Task: Reverse sequences of length {min_seq_length}-{max_seq_length}")
    print(f"Vocab size: {vocab_size}, Hidden size: {hidden_size}")
    print(f"Perturbation range: {pert_min} to {pert_max} (integer values)")
    print(f"Fixed: LR={learning_rate:.4f}, epsilon={epsilon}, batch={batch_size}")
    print(f"Max time per experiment: {max_time}s, Convergence criterion: loss < {convergence_loss}")
    print(f"Testing {len(pert_values)} values: {pert_values}")
    print("="*80)
    print()

    results = []

    for i, pert in enumerate(pert_values):
        print(f"[{i+1}/{len(pert_values)}] Testing Pert={pert}...", end=" ", flush=True)

        result = train_until_convergence(
            vocab_size, min_seq_length, max_seq_length, hidden_size, num_heads,
            learning_rate, epsilon, pert, batch_size,
            max_time, convergence_loss, device, seed
        )

        result['pert'] = pert
        results.append(result)

        if result['converged']:
            print(f"Converged in {result['steps']} steps ({result['time']:.2f}s)")
        else:
            print(f"Did not converge (final loss: {result['final_loss']:.4f}, steps: {result['steps']})")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Focused Perturbation Sweep ({len(pert_values)} values)\n'
                 f'Task: Reverse (seq={min_seq_length}-{max_seq_length}, vocab={vocab_size})\n'
                 f'Fixed: LR={learning_rate:.4f}, epsilon={epsilon}, batch={batch_size}', fontsize=12)

    # Extract data
    perts = [r['pert'] for r in results]
    converged = [r['converged'] for r in results]
    times = [r['time'] if r['converged'] else max_time for r in results]
    steps = [r['steps'] for r in results]
    final_losses = [r['final_loss'] for r in results]

    # Colors: green for converged, red for not converged
    colors = ['green' if c else 'red' for c in converged]

    # Plot 1: Convergence status
    ax = axes[0, 0]
    ax.scatter(perts, [1 if c else 0 for c in converged], c=colors, s=100, alpha=0.6)
    ax.set_xlabel('Number of Perturbations', fontsize=10)
    ax.set_ylabel('Converged (1=Yes, 0=No)', fontsize=10)
    ax.set_title('Convergence Status vs Perturbations', fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.grid(True, alpha=0.3)

    # Plot 2: Time to convergence (only converged runs)
    ax = axes[0, 1]
    converged_perts = [r['pert'] for r in results if r['converged']]
    converged_times = [r['time'] for r in results if r['converged']]
    if converged_perts:
        ax.scatter(converged_perts, converged_times, c='green', s=100, alpha=0.6)
        ax.plot(converged_perts, converged_times, 'g--', alpha=0.3)
    ax.set_xlabel('Number of Perturbations', fontsize=10)
    ax.set_ylabel('Time to Convergence (s)', fontsize=10)
    ax.set_title('Time to Convergence (Converged Runs Only)', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: Loss vs Time curves for converged runs
    ax = axes[1, 0]
    for r in results:
        if r['converged'] and len(r['timestamps']) > 0:
            ax.plot(r['timestamps'], r['losses'], alpha=0.6, linewidth=1.5,
                   label=f"Pert={r['pert']}")
    ax.axhline(y=convergence_loss, color='blue', linestyle='--', linewidth=2,
              label=f'Convergence threshold ({convergence_loss})')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title('Loss vs Time (Converged Runs)', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    if len([r for r in results if r['converged']]) <= 10:
        ax.legend(fontsize=8, loc='best')

    # Plot 4: Final loss (all runs)
    ax = axes[1, 1]
    ax.scatter(perts, final_losses, c=colors, s=100, alpha=0.6)
    ax.axhline(y=convergence_loss, color='blue', linestyle='--', linewidth=2,
              label=f'Convergence threshold ({convergence_loss})')
    ax.set_xlabel('Number of Perturbations', fontsize=10)
    ax.set_ylabel('Final Loss', fontsize=10)
    ax.set_title('Final Loss vs Perturbations', fontsize=11)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = f'{output_dir}/focused_pert_sweep_{len(pert_values)}values_vocab{vocab_size}_seq{min_seq_length}-{max_seq_length}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {plot_filename}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    num_converged = sum(converged)
    print(f"Converged: {num_converged}/{len(pert_values)} ({100*num_converged/len(pert_values):.1f}%)")

    if num_converged > 0:
        print(f"\nConverged runs:")
        print(f"  Perturbation range: {min(converged_perts)} to {max(converged_perts)}")
        print(f"  Best time: {min(converged_times):.2f}s at Pert={converged_perts[converged_times.index(min(converged_times))]}")
        best_pert_idx = converged_times.index(min(converged_times))
        best_pert = converged_perts[best_pert_idx]
        best_steps = [r['steps'] for r in results if r['converged'] and r['pert'] == best_pert][0]
        print(f"  Best steps: {best_steps} at Pert={best_pert}")

    print("="*80)

    # Send Telegram notification
    if TELEGRAM_AVAILABLE:
        message = f"""**Focused Perturbation Sweep Complete!**

Task: Reverse (seq={min_seq_length}-{max_seq_length}, vocab={vocab_size})

**Results:**
- Tested {len(pert_values)} perturbation values from {pert_min} to {pert_max}
- Converged: {num_converged}/{len(pert_values)} ({100*num_converged/len(pert_values):.1f}%)

Fixed: LR={learning_rate:.4f}, epsilon={epsilon}, batch={batch_size}"""

        if num_converged > 0:
            best_pert = converged_perts[converged_times.index(min(converged_times))]
            message += f"\n\nBest: Pert={best_pert} in {min(converged_times):.2f}s"

        send_message(message)
        send_photo(plot_filename, caption=os.path.basename(plot_filename))


if __name__ == '__main__':
    main()
