#!/usr/bin/env python3
"""
Convergence Speed Comparison: Adam vs SPSA

Compares time-to-convergence using loss < 0.05 as criterion.
Phase 1: Hyperparameter tuning with single seed
Phase 2: Statistical validation with multiple seeds
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from models.models import LSTM


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
    """Perform a single zero-order optimization step (SPSA with central difference)."""
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


def train_until_convergence_adam(vocab_size, min_seq, max_seq, hidden_size, num_heads,
                                  learning_rate, batch_size, max_steps, convergence_loss,
                                  device, seed):
    """Train with Adam until convergence or max_steps."""
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

    optimizer = torch.optim.Adam(list(model.parameters()) + list(embed.parameters()), lr=learning_rate)

    start_time = time.time()
    converged_step = None
    converged_time = None
    final_loss = None

    for step in range(max_steps):
        x_ids, y_ids = generate_reverse_batch(batch_size, min_seq, max_seq, vocab_size, device)

        x_emb = embed(x_ids)
        logits, _, _ = model(x_emb, require_gradients=True)
        loss = compute_reverse_loss(logits, y_ids, PAD)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()

        # Check convergence
        if converged_step is None and loss_val < convergence_loss:
            converged_step = step + 1
            converged_time = time.time() - start_time
            final_loss = loss_val

        # Early stopping after convergence (give it a few more steps to stabilize)
        if converged_step is not None and step > converged_step + 10:
            break

    if converged_step is None:
        final_loss = loss_val

    return {
        'converged': converged_step is not None,
        'steps': converged_step if converged_step is not None else max_steps,
        'time': converged_time if converged_time is not None else time.time() - start_time,
        'final_loss': final_loss
    }


def train_until_convergence_spsa(vocab_size, min_seq, max_seq, hidden_size, num_heads,
                                  learning_rate, epsilon, num_perturbations, batch_size,
                                  max_steps, convergence_loss, device, seed):
    """Train with SPSA until convergence or max_steps."""
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
    converged_step = None
    converged_time = None
    final_loss = None

    for step in range(max_steps):
        x_ids, y_ids = generate_reverse_batch(batch_size, min_seq, max_seq, vocab_size, device)

        loss = zeroth_order_step(
            model, embed, x_ids, y_ids, PAD,
            learning_rate, epsilon, num_perturbations
        )

        # Check convergence
        if converged_step is None and loss < convergence_loss:
            converged_step = step + 1
            converged_time = time.time() - start_time
            final_loss = loss

        # Early stopping after convergence
        if converged_step is not None and step > converged_step + 10:
            break

    if converged_step is None:
        final_loss = loss

    return {
        'converged': converged_step is not None,
        'steps': converged_step if converged_step is not None else max_steps,
        'time': converged_time if converged_time is not None else time.time() - start_time,
        'final_loss': final_loss
    }


def phase1_hyperparameter_tuning(args):
    """Phase 1: Find optimal hyperparameters for each optimizer."""
    print("=" * 80)
    print("PHASE 1: HYPERPARAMETER TUNING (Single Seed)")
    print("=" * 80)
    print(f"Convergence criterion: Loss < {args.convergence_loss}")
    print(f"Max steps: {args.max_steps}")
    print(f"Seed: {args.seed}")
    print()

    # Adam configurations
    adam_lrs = [0.0001, 0.0005, 0.001, 0.002, 0.005]
    adam_batch_size = 32

    # SPSA configurations (LR = epsilon)
    spsa_lr_eps = [0.01, 0.03, 0.05, 0.07, 0.1]
    spsa_perturbations = 8
    spsa_batch_size = 16

    print("=" * 80)
    print("TESTING ADAM OPTIMIZER")
    print("=" * 80)

    adam_results = []
    for lr in adam_lrs:
        print(f"\nAdam LR={lr}, Batch={adam_batch_size}")
        print("-" * 80)

        result = train_until_convergence_adam(
            args.vocab_size, args.min_seq_length, args.max_seq_length,
            args.hidden_size, args.num_heads, lr, adam_batch_size,
            args.max_steps, args.convergence_loss, args.device, args.seed
        )

        print(f"  Converged: {result['converged']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Time: {result['time']:.2f}s")
        print(f"  Final loss: {result['final_loss']:.4f}")

        adam_results.append({
            'lr': lr,
            'batch_size': adam_batch_size,
            **result
        })

    print("\n\n" + "=" * 80)
    print("TESTING SPSA OPTIMIZER (LR = Epsilon)")
    print("=" * 80)

    spsa_results = []
    for lr_eps in spsa_lr_eps:
        print(f"\nSPSA LR=Epsilon={lr_eps}, Pert={spsa_perturbations}, Batch={spsa_batch_size}")
        print("-" * 80)

        result = train_until_convergence_spsa(
            args.vocab_size, args.min_seq_length, args.max_seq_length,
            args.hidden_size, args.num_heads, lr_eps, lr_eps, spsa_perturbations,
            spsa_batch_size, args.max_steps, args.convergence_loss, args.device, args.seed
        )

        print(f"  Converged: {result['converged']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Time: {result['time']:.2f}s")
        print(f"  Final loss: {result['final_loss']:.4f}")

        spsa_results.append({
            'lr_epsilon': lr_eps,
            'perturbations': spsa_perturbations,
            'batch_size': spsa_batch_size,
            **result
        })

    # Print summaries
    print("\n\n" + "=" * 80)
    print("ADAM RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'LR':<10} {'Converged':<12} {'Steps':<10} {'Time (s)':<12} {'Final Loss':<12}")
    print("-" * 80)
    for r in adam_results:
        conv_str = "Yes" if r['converged'] else "No"
        print(f"{r['lr']:<10.4f} {conv_str:<12} {r['steps']:<10} {r['time']:<12.2f} {r['final_loss']:<12.4f}")

    converged_adam = [r for r in adam_results if r['converged']]
    if converged_adam:
        best_adam = min(converged_adam, key=lambda x: x['steps'])
        print(f"\nBest Adam: LR={best_adam['lr']}, Steps={best_adam['steps']}, Time={best_adam['time']:.2f}s")
    else:
        print("\nNo Adam configurations converged within step budget")

    print("\n\n" + "=" * 80)
    print("SPSA RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'LR=Eps':<10} {'Converged':<12} {'Steps':<10} {'Time (s)':<12} {'Final Loss':<12}")
    print("-" * 80)
    for r in spsa_results:
        conv_str = "Yes" if r['converged'] else "No"
        print(f"{r['lr_epsilon']:<10.4f} {conv_str:<12} {r['steps']:<10} {r['time']:<12.2f} {r['final_loss']:<12.4f}")

    converged_spsa = [r for r in spsa_results if r['converged']]
    if converged_spsa:
        best_spsa = min(converged_spsa, key=lambda x: x['steps'])
        print(f"\nBest SPSA: LR=Eps={best_spsa['lr_epsilon']}, Steps={best_spsa['steps']}, Time={best_spsa['time']:.2f}s")
    else:
        print("\nNo SPSA configurations converged within step budget")

    # Overall comparison
    if converged_adam and converged_spsa:
        print("\n\n" + "=" * 80)
        print("OVERALL COMPARISON (Best Configurations)")
        print("=" * 80)
        print(f"Adam:  {best_adam['steps']} steps, {best_adam['time']:.2f}s")
        print(f"SPSA:  {best_spsa['steps']} steps, {best_spsa['time']:.2f}s")

        step_ratio = best_spsa['steps'] / best_adam['steps']
        time_ratio = best_spsa['time'] / best_adam['time']

        print(f"\nSPSA takes {step_ratio:.2f}x steps compared to Adam")
        print(f"SPSA takes {time_ratio:.2f}x time compared to Adam")
        print("=" * 80)

        return best_adam, best_spsa
    else:
        print("\nCannot compare - not all optimizers converged")
        return None, None


def phase2_statistical_validation(args, best_adam, best_spsa):
    """Phase 2: Run multiple seeds with best hyperparameters."""
    print("\n\n" + "=" * 80)
    print("PHASE 2: STATISTICAL VALIDATION (Multiple Seeds)")
    print("=" * 80)
    print(f"Number of seeds: {args.num_seeds}")
    print(f"Seeds: {list(range(args.seed, args.seed + args.num_seeds))}")
    print()

    # Run Adam with multiple seeds
    print("Running Adam with multiple seeds...")
    adam_seed_results = []
    for seed in range(args.seed, args.seed + args.num_seeds):
        result = train_until_convergence_adam(
            args.vocab_size, args.min_seq_length, args.max_seq_length,
            args.hidden_size, args.num_heads, best_adam['lr'], best_adam['batch_size'],
            args.max_steps, args.convergence_loss, args.device, seed
        )
        adam_seed_results.append(result)
        print(f"  Seed {seed}: Steps={result['steps']}, Time={result['time']:.2f}s, Converged={result['converged']}")

    # Run SPSA with multiple seeds
    print("\nRunning SPSA with multiple seeds...")
    spsa_seed_results = []
    for seed in range(args.seed, args.seed + args.num_seeds):
        result = train_until_convergence_spsa(
            args.vocab_size, args.min_seq_length, args.max_seq_length,
            args.hidden_size, args.num_heads, best_spsa['lr_epsilon'], best_spsa['lr_epsilon'],
            best_spsa['perturbations'], best_spsa['batch_size'],
            args.max_steps, args.convergence_loss, args.device, seed
        )
        spsa_seed_results.append(result)
        print(f"  Seed {seed}: Steps={result['steps']}, Time={result['time']:.2f}s, Converged={result['converged']}")

    # Compute statistics (only for converged runs)
    adam_converged = [r for r in adam_seed_results if r['converged']]
    spsa_converged = [r for r in spsa_seed_results if r['converged']]

    print("\n\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)

    if adam_converged:
        adam_steps = [r['steps'] for r in adam_converged]
        adam_times = [r['time'] for r in adam_converged]
        print(f"\nAdam (LR={best_adam['lr']}, {len(adam_converged)}/{args.num_seeds} converged):")
        print(f"  Steps: {np.mean(adam_steps):.1f} ± {np.std(adam_steps):.1f}")
        print(f"  Time:  {np.mean(adam_times):.2f} ± {np.std(adam_times):.2f}s")
    else:
        print("\nAdam: No runs converged")

    if spsa_converged:
        spsa_steps = [r['steps'] for r in spsa_converged]
        spsa_times = [r['time'] for r in spsa_converged]
        print(f"\nSPSA (LR=Eps={best_spsa['lr_epsilon']}, {len(spsa_converged)}/{args.num_seeds} converged):")
        print(f"  Steps: {np.mean(spsa_steps):.1f} ± {np.std(spsa_steps):.1f}")
        print(f"  Time:  {np.mean(spsa_times):.2f} ± {np.std(spsa_times):.2f}s")
    else:
        print("\nSPSA: No runs converged")

    if adam_converged and spsa_converged:
        step_ratio = np.mean(spsa_steps) / np.mean(adam_steps)
        time_ratio = np.mean(spsa_times) / np.mean(adam_times)
        print(f"\nComparison:")
        print(f"  SPSA takes {step_ratio:.2f}x steps compared to Adam")
        print(f"  SPSA takes {time_ratio:.2f}x time compared to Adam")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Convergence comparison: Adam vs SPSA')
    parser.add_argument('--vocab_size', type=int, default=20)
    parser.add_argument('--min_seq_length', type=int, default=5)
    parser.add_argument('--max_seq_length', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=2000, help='Max steps for convergence')
    parser.add_argument('--convergence_loss', type=float, default=0.05, help='Loss threshold for convergence')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42, help='Base seed for experiments')

    parser.add_argument('--phase', type=int, default=1, choices=[1, 2], help='1=hyperparameter tuning, 2=statistical validation')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of seeds for Phase 2')

    # For Phase 2, can manually specify best configs
    parser.add_argument('--best_adam_lr', type=float, default=None)
    parser.add_argument('--best_spsa_lr_eps', type=float, default=None)

    args = parser.parse_args()

    if args.phase == 1:
        best_adam, best_spsa = phase1_hyperparameter_tuning(args)

        if best_adam and best_spsa:
            print(f"\n\nTo run Phase 2 with these hyperparameters:")
            print(f"python convergence_comparison.py --phase 2 --best_adam_lr {best_adam['lr']} --best_spsa_lr_eps {best_spsa['lr_epsilon']} --num_seeds 5")

    elif args.phase == 2:
        if args.best_adam_lr is None or args.best_spsa_lr_eps is None:
            print("Error: For Phase 2, must specify --best_adam_lr and --best_spsa_lr_eps")
            print("Run Phase 1 first to determine optimal hyperparameters")
            return

        best_adam = {'lr': args.best_adam_lr, 'batch_size': 32}
        best_spsa = {'lr_epsilon': args.best_spsa_lr_eps, 'perturbations': 8, 'batch_size': 16}

        phase2_statistical_validation(args, best_adam, best_spsa)


if __name__ == '__main__':
    main()
