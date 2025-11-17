#!/usr/bin/env python3
"""
Adaptive hyperparameter search for SPSA with vocab_size=64 and batch_size=32.

Features:
- Adaptive learning rate: If loss increases by 3x, halve LR and retry
- Iterative optimization: Sweep LR, perturbations, batch_size
- Update to argmin after each sweep until convergence
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


def train_with_adaptive_lr(vocab_size, min_seq, max_seq, hidden_size, num_heads,
                            learning_rate, epsilon, num_perturbations, batch_size,
                            max_time, convergence_loss, device, seed, max_lr_reductions=5):
    """
    Train SPSA with adaptive LR: if loss increases by 3x, halve LR and restart.
    Returns losses with timestamps.
    """
    current_lr = learning_rate
    attempt = 0

    while attempt <= max_lr_reductions:
        torch.manual_seed(seed)
        np.random.seed(seed)

        device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
        PAD = vocab_size - 1

        embed = nn.Embedding(vocab_size, hidden_size, device=device_obj, dtype=torch.bfloat16)
        model = LSTM(
            input_size=hidden_size,
            output_size=vocab_size,
            hidden_size=hidden_size,
            memory_size=0,
            head_size=hidden_size // num_heads,
            num_heads=num_heads,
            embed=embed,
            device=device_obj,
            dtype=torch.bfloat16
        )

        start_time = time.time()
        converged = False
        step = 0
        losses = []
        timestamps = []
        min_loss_seen = float('inf')

        while time.time() - start_time < max_time:
            x_ids, y_ids = generate_reverse_batch(batch_size, min_seq, max_seq, vocab_size, device_obj)
            loss = zeroth_order_step(
                model, embed, x_ids, y_ids, PAD,
                current_lr, epsilon, num_perturbations
            )
            losses.append(loss)
            timestamps.append(time.time() - start_time)
            step += 1

            # Track minimum loss
            if loss < min_loss_seen:
                min_loss_seen = loss

            # Check for convergence
            if loss < convergence_loss:
                converged = True
                break

            # Check for divergence (loss increased by 3x from minimum)
            if len(losses) > 10 and loss > 3.0 * min_loss_seen:
                print(f"Divergence detected (loss {loss:.4f} > 3*{min_loss_seen:.4f}), halving LR from {current_lr:.4f} to {current_lr/2:.4f}")
                current_lr = current_lr / 2
                attempt += 1
                break  # Restart with new LR

        elapsed = time.time() - start_time

        # If converged or time ran out, return results
        if converged or time.time() - start_time >= max_time:
            return {
                'converged': converged,
                'steps': step,
                'time': elapsed,
                'final_loss': losses[-1] if losses else float('inf'),
                'losses': losses,
                'timestamps': timestamps,
                'final_lr': current_lr,
                'lr_reductions': attempt
            }

    # Max reductions reached
    return {
        'converged': False,
        'steps': step,
        'time': elapsed,
        'final_loss': losses[-1] if losses else float('inf'),
        'losses': losses,
        'timestamps': timestamps,
        'final_lr': current_lr,
        'lr_reductions': attempt
    }


def sweep_parameter(param_name, param_values, fixed_params, vocab_size, min_seq, max_seq,
                     hidden_size, num_heads, max_time, convergence_loss, device, seed):
    """
    Sweep a single parameter while keeping others fixed.
    param_name: 'lr', 'pert', or 'batch'
    """
    results = []

    print(f"\n{'='*80}")
    print(f"SWEEPING {param_name.upper()} (fixed: {fixed_params})")
    print(f"{'='*80}")

    for i, val in enumerate(param_values):
        print(f"[{i+1}/{len(param_values)}] Testing {param_name}={val}...", end=" ", flush=True)

        # Set up parameters
        if param_name == 'lr':
            lr, pert, batch = val, fixed_params['pert'], fixed_params['batch']
        elif param_name == 'pert':
            lr, pert, batch = fixed_params['lr'], val, fixed_params['batch']
        elif param_name == 'batch':
            lr, pert, batch = fixed_params['lr'], fixed_params['pert'], val

        result = train_with_adaptive_lr(
            vocab_size, min_seq, max_seq, hidden_size, num_heads,
            lr, fixed_params['epsilon'], pert, batch,
            max_time, convergence_loss, device, seed
        )

        result[param_name] = val
        results.append(result)

        if result['converged']:
            print(f"Converged in {result['steps']} steps ({result['time']:.2f}s, LR reductions: {result['lr_reductions']})")
        else:
            print(f"Did not converge (final loss: {result['final_loss']:.4f}, LR reductions: {result['lr_reductions']})")

    return results


def find_best_params(results, param_name):
    """Find the parameter value that achieved best convergence time (or best loss if none converged)."""
    converged_results = [r for r in results if r['converged']]

    if converged_results:
        # Find fastest convergence
        best = min(converged_results, key=lambda r: r['time'])
        return best[param_name], best['time'], True
    else:
        # Find lowest final loss
        best = min(results, key=lambda r: r['final_loss'])
        return best[param_name], best['final_loss'], False


def main():
    # Fixed task parameters
    vocab_size = 64  # Content vocab = 60
    min_seq_length = 10
    max_seq_length = 100
    hidden_size = 128
    num_heads = 4
    max_time = 30.0  # seconds per experiment
    convergence_loss = 0.05
    device = 'cuda'
    seed = 42

    # Initial hyperparameters (doubled batch size from 16 to 32)
    current_params = {
        'lr': 0.18,  # Starting with good LR from previous experiments
        'epsilon': 0.1,
        'pert': 8,
        'batch': 32
    }

    # Sweep ranges
    lr_values = np.geomspace(0.01, 0.5, 8)
    pert_values = [2, 4, 6, 8, 10, 12, 16, 20]
    batch_values = [16, 24, 32, 48, 64, 96, 128, 192]

    output_dir = 'hpp-opt'
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("ADAPTIVE HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Task: Reverse sequences of length {min_seq_length}-{max_seq_length}")
    print(f"Vocab size: {vocab_size} (content vocab: {vocab_size-4})")
    print(f"Adaptive LR: Halves when loss increases by 3x")
    print(f"Initial params: LR={current_params['lr']}, Pert={current_params['pert']}, Batch={current_params['batch']}")
    print(f"Max time per experiment: {max_time}s, Convergence criterion: loss < {convergence_loss}")
    print("="*80)
    print()

    iteration = 0
    max_iterations = 10
    converged_iteration = False

    while iteration < max_iterations and not converged_iteration:
        iteration += 1
        print(f"\n{'#'*80}")
        print(f"ITERATION {iteration}")
        print(f"Current params: LR={current_params['lr']:.4f}, Pert={current_params['pert']}, Batch={current_params['batch']}")
        print(f"{'#'*80}")

        # Sweep 1: Learning Rate
        lr_results = sweep_parameter('lr', lr_values, current_params, vocab_size, min_seq_length,
                                      max_seq_length, hidden_size, num_heads, max_time,
                                      convergence_loss, device, seed + iteration * 1000)
        best_lr, best_lr_metric, lr_converged = find_best_params(lr_results, 'lr')

        # Sweep 2: Perturbations
        current_params['lr'] = best_lr
        pert_results = sweep_parameter('pert', pert_values, current_params, vocab_size, min_seq_length,
                                        max_seq_length, hidden_size, num_heads, max_time,
                                        convergence_loss, device, seed + iteration * 1000 + 100)
        best_pert, best_pert_metric, pert_converged = find_best_params(pert_results, 'pert')

        # Sweep 3: Batch Size
        current_params['pert'] = best_pert
        batch_results = sweep_parameter('batch', batch_values, current_params, vocab_size, min_seq_length,
                                         max_seq_length, hidden_size, num_heads, max_time,
                                         convergence_loss, device, seed + iteration * 1000 + 200)
        best_batch, best_batch_metric, batch_converged = find_best_params(batch_results, 'batch')

        # Update parameters
        old_params = current_params.copy()
        current_params['batch'] = best_batch

        print(f"\n{'='*80}")
        print(f"ITERATION {iteration} SUMMARY")
        print(f"{'='*80}")
        print(f"Best LR: {best_lr:.4f} ({'converged in ' + f'{best_lr_metric:.2f}s' if lr_converged else f'loss {best_lr_metric:.4f}'})")
        print(f"Best Pert: {best_pert} ({'converged in ' + f'{best_pert_metric:.2f}s' if pert_converged else f'loss {best_pert_metric:.4f}'})")
        print(f"Best Batch: {best_batch} ({'converged in ' + f'{best_batch_metric:.2f}s' if batch_converged else f'loss {best_batch_metric:.4f}'})")
        print(f"Updated params: LR={current_params['lr']:.4f}, Pert={current_params['pert']}, Batch={current_params['batch']}")
        print(f"{'='*80}")

        # Check for convergence (no change in parameters)
        if (old_params['lr'] == current_params['lr'] and
            old_params['pert'] == current_params['pert'] and
            old_params['batch'] == current_params['batch']):
            print("\nCONVERGED: No parameter changes in this iteration!")
            converged_iteration = True

    print(f"\n{'#'*80}")
    print("FINAL RESULTS")
    print(f"{'#'*80}")
    print(f"Final hyperparameters after {iteration} iterations:")
    print(f"  LR: {current_params['lr']:.4f}")
    print(f"  Perturbations: {current_params['pert']}")
    print(f"  Batch Size: {current_params['batch']}")
    print(f"  Epsilon: {current_params['epsilon']}")
    print(f"{'#'*80}")

    # Send Telegram notification
    if TELEGRAM_AVAILABLE:
        message = f"""**Adaptive HPP Search Complete!**

Task: Reverse (seq={min_seq_length}-{max_seq_length}, vocab={vocab_size})

**Final Hyperparameters** (after {iteration} iterations):
- LR: {current_params['lr']:.4f}
- Perturbations: {current_params['pert']}
- Batch Size: {current_params['batch']}
- Epsilon: {current_params['epsilon']}

Adaptive LR: Halves when loss increases by 3x"""

        send_message(message)


if __name__ == '__main__':
    main()
