# Noise Analysis Plan: Diagnosing SPSA vs Adam Performance

## Goal
To diagnose why SPSA performs worse than Adam by quantifying three types of noise at a fixed set of model parameters:
1. **Loss noise** (stochasticity from different batches)
2. **Gradient noise** (stochasticity in true gradients from different batches)
3. **SPSA gradient estimation noise** (error in SPSA's gradient approximation)

## Theoretical Background

### SPSA as a Random Walk
The original SPSA work (Spall, 1992) describes SPSA as acting like a "random walk that prefers the gradient direction." The gradient estimate has two sources of noise:
1. **Approximation error**: Using finite differences with random perturbations
2. **Batch stochasticity**: Different batches give different loss values

When the loss surface is noisy, SPSA's zero-order approximation may struggle to distinguish signal from noise, leading to poor gradient estimates.

### Key Quantities to Measure

#### 1. Loss Noise (σ_L)
For a fixed parameter vector θ, measure variance of loss across different batches:
```
L₁ = loss(θ, batch₁)
L₂ = loss(θ, batch₂)
...
Lₙ = loss(θ, batchₙ)

σ²_L = Var(L₁, L₂, ..., Lₙ)
```

#### 2. Gradient Noise (σ_∇)
For a fixed parameter vector θ, measure variance of true gradients across different batches:
```
∇L₁ = ∇loss(θ, batch₁)  [computed via backprop]
∇L₂ = ∇loss(θ, batch₂)
...
∇Lₙ = ∇loss(θ, batchₙ)

σ²_∇ = average variance per parameter across batches
```

#### 3. SPSA Gradient Estimation Error (ε_SPSA)
For a fixed parameter vector θ and a fixed batch, compare SPSA gradient estimate to true gradient:
```
∇L_true = ∇loss(θ, batch)  [via backprop]
∇L_SPSA = SPSA_estimate(θ, batch, ε, K)  [via perturbations]

ε_SPSA = ||∇L_SPSA - ∇L_true||₂
```

Also measure this across different batches to get:
- Mean SPSA error: E[||∇L_SPSA - ∇L_true||]
- Variance of SPSA error

#### 4. Signal-to-Noise Ratios (SNR)
```
SNR_gradient = ||E[∇L]||₂ / σ_∇
SNR_SPSA = ||E[∇L]||₂ / (σ_∇ + ε_SPSA)
```

## Implementation Plan

### Script: `diagnose_noise.py`

#### Command-line Interface
```bash
python diagnose_noise.py MODEL_PATH \
    --n_batches 100 \
    --batch_size 32 \
    --min_seq_length 5 \
    --max_seq_length 10 \
    --spsa_epsilon 0.1 \
    --spsa_perturbations 8 \
    --output noise_analysis.json \
    --device cuda
```

#### Core Functions

##### 1. `measure_loss_noise(model, params, n_batches, batch_config, device)`
**Input**: Fixed model parameters
**Output**:
- `losses`: Array of loss values [n_batches]
- `mean_loss`: E[L]
- `std_loss`: σ_L
- `cv_loss`: Coefficient of variation (σ_L / E[L])

**Method**:
```python
# Freeze model at specific parameters
model.load_state_dict(params)

losses = []
for i in range(n_batches):
    x_ids, y_ids = generate_reverse_batch(...)
    loss = compute_loss(model, x_ids, y_ids, criterion,
                       require_gradients=False)
    losses.append(loss.item())

return {
    'losses': losses,
    'mean': np.mean(losses),
    'std': np.std(losses),
    'cv': np.std(losses) / np.mean(losses)
}
```

##### 2. `measure_gradient_noise(model, params, n_batches, batch_config, device)`
**Input**: Fixed model parameters
**Output**:
- `gradients`: List of gradient dictionaries [n_batches]
- `mean_grad`: E[∇L] per parameter
- `std_grad`: σ_∇ per parameter
- `mean_grad_norm`: ||E[∇L]||₂
- `mean_std_grad`: Average of σ_∇ across parameters
- `snr_gradient`: ||E[∇L]||₂ / mean(σ_∇)

**Method**:
```python
# Freeze model at specific parameters
model.load_state_dict(params)

# Collect gradients from n_batches
all_gradients = []  # List of flat gradient vectors
for i in range(n_batches):
    x_ids, y_ids = generate_reverse_batch(...)

    model.zero_grad()
    loss = compute_loss(model, x_ids, y_ids, criterion,
                       require_gradients=True)
    loss.backward()

    # Flatten all gradients into one vector
    grad_vec = []
    for p in model.parameters():
        if p.grad is not None:
            grad_vec.append(p.grad.flatten().detach().cpu())
    for p in model.embed.parameters():
        if p.grad is not None:
            grad_vec.append(p.grad.flatten().detach().cpu())

    grad_vec = torch.cat(grad_vec)
    all_gradients.append(grad_vec)

# Stack into [n_batches, n_params] array
all_gradients = torch.stack(all_gradients)

# Compute statistics
mean_grad = all_gradients.mean(dim=0)  # E[∇L]
std_grad = all_gradients.std(dim=0)    # σ_∇ per parameter

return {
    'mean_grad_norm': mean_grad.norm().item(),
    'mean_std_grad': std_grad.mean().item(),
    'median_std_grad': std_grad.median().item(),
    'max_std_grad': std_grad.max().item(),
    'snr': mean_grad.norm().item() / std_grad.mean().item(),
    'mean_grad': mean_grad,  # Keep for comparison with SPSA
    'std_grad': std_grad
}
```

##### 3. `measure_spsa_error(model, params, n_trials, batch_config, epsilon, n_perturbations, device)`
**Input**: Fixed model parameters
**Output**:
- `spsa_errors`: Array of ||∇L_SPSA - ∇L_true||₂ values [n_trials]
- `mean_error`: E[||∇L_SPSA - ∇L_true||]
- `std_error`: Variance of SPSA error
- `relative_error`: mean_error / ||∇L_true||
- `cosine_similarities`: Array of cos(∇L_SPSA, ∇L_true) [n_trials]

**Method**:
```python
# Freeze model at specific parameters
model.load_state_dict(params)

errors = []
cosine_sims = []
relative_errors = []

for trial in range(n_trials):
    # Generate one batch
    x_ids, y_ids = generate_reverse_batch(...)

    # Compute TRUE gradient via backprop
    model.zero_grad()
    loss_true = compute_loss(model, x_ids, y_ids, criterion,
                            require_gradients=True)
    loss_true.backward()

    # Collect true gradient
    true_grad = []
    for p in model.parameters():
        if p.grad is not None:
            true_grad.append(p.grad.flatten().detach().cpu().clone())
    for p in model.embed.parameters():
        if p.grad is not None:
            true_grad.append(p.grad.flatten().detach().cpu().clone())
    true_grad = torch.cat(true_grad)

    # Reset gradients
    model.zero_grad()

    # Compute SPSA gradient estimate
    # (Use similar logic to spsa_step but return gradient instead)
    spsa_grad = estimate_spsa_gradient(
        model, x_ids, y_ids, criterion,
        epsilon, n_perturbations
    )

    # Compare
    error = (spsa_grad - true_grad).norm().item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        spsa_grad.unsqueeze(0),
        true_grad.unsqueeze(0)
    ).item()
    relative_error = error / true_grad.norm().item()

    errors.append(error)
    cosine_sims.append(cosine_sim)
    relative_errors.append(relative_error)

    # Restore model state (SPSA modifies params)
    model.load_state_dict(params)

return {
    'mean_error': np.mean(errors),
    'std_error': np.std(errors),
    'mean_relative_error': np.mean(relative_errors),
    'mean_cosine_similarity': np.mean(cosine_sims),
    'std_cosine_similarity': np.std(cosine_sims),
    'errors': errors,
    'cosine_similarities': cosine_sims
}
```

##### 4. `estimate_spsa_gradient(model, x_ids, y_ids, criterion, epsilon, n_perturbations)`
Helper function that returns SPSA gradient estimate as a flat vector.

**Method**:
```python
param_list = list(model.parameters()) + list(model.embed.parameters())

# Initialize pseudo gradient accumulator
pseudo_gradient = [torch.zeros_like(p) for p in param_list]

for _ in range(n_perturbations):
    # Generate random perturbation direction
    perturbations = [torch.randn_like(p) for p in param_list]

    # +ε perturbation
    with torch.no_grad():
        for p, pert in zip(param_list, perturbations):
            p.add_(pert, alpha=epsilon)

    loss_plus = compute_loss(model, x_ids, y_ids, criterion,
                            require_gradients=False).item()

    # -ε perturbation (from +ε, so shift by -2ε)
    with torch.no_grad():
        for p, pert in zip(param_list, perturbations):
            p.add_(pert, alpha=-2 * epsilon)

    loss_minus = compute_loss(model, x_ids, y_ids, criterion,
                             require_gradients=False).item()

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
    pg.div_(n_perturbations)

# Flatten into vector
spsa_grad_vec = torch.cat([pg.flatten().cpu() for pg in pseudo_gradient])

return spsa_grad_vec
```

##### 5. `compare_optimizers_one_step(model, params, batch, config)`
**Input**: Fixed starting parameters, one specific batch
**Output**: Comparison of Adam vs SPSA updates

**Method**:
```python
# Start from same parameters
original_params = copy.deepcopy(params)

# 1. Adam update
model.load_state_dict(original_params)
optimizer = torch.optim.Adam(model.parameters(), lr=config['adam_lr'])
optimizer.zero_grad()
loss = compute_loss(model, x_ids, y_ids, criterion, True)
loss.backward()
optimizer.step()

adam_params_after = copy.deepcopy(model.state_dict())
adam_update = compute_param_diff(original_params, adam_params_after)

# 2. SPSA update
model.load_state_dict(original_params)
loss_before, grad_norm = spsa_step(
    model, x_ids, y_ids, criterion,
    config['spsa_lr'], config['spsa_epsilon'],
    config['spsa_n_perturbations']
)

spsa_params_after = copy.deepcopy(model.state_dict())
spsa_update = compute_param_diff(original_params, spsa_params_after)

# Compare update magnitudes and directions
return {
    'adam_update_norm': compute_norm(adam_update),
    'spsa_update_norm': compute_norm(spsa_update),
    'update_cosine_similarity': cosine_similarity(adam_update, spsa_update),
    'adam_loss_improvement': ...,
    'spsa_loss_improvement': ...
}
```

#### Main Analysis Function

```python
def analyze_noise(model_path, args):
    """
    Main analysis function that orchestrates all measurements.
    """
    # Load model
    model, embed, vocab_size = load_model(model_path, args.device)

    # Save current parameters (this is our fixed θ)
    fixed_params = copy.deepcopy(model.state_dict())

    print("=" * 80)
    print("NOISE ANALYSIS")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Analyzing at current model parameters")
    print(f"n_batches: {args.n_batches}")
    print(f"batch_size: {args.batch_size}")
    print(f"seq_length: {args.min_seq_length}-{args.max_seq_length}")
    print(f"SPSA config: epsilon={args.spsa_epsilon}, K={args.spsa_perturbations}")
    print("=" * 80)
    print()

    # 1. Measure loss noise
    print("1. Measuring loss noise...")
    loss_stats = measure_loss_noise(
        model, fixed_params, args.n_batches,
        batch_config, args.device
    )
    print(f"   Mean loss: {loss_stats['mean']:.6f}")
    print(f"   Std loss: {loss_stats['std']:.6f}")
    print(f"   CV loss: {loss_stats['cv']:.4f}")
    print()

    # 2. Measure gradient noise
    print("2. Measuring gradient noise...")
    grad_stats = measure_gradient_noise(
        model, fixed_params, args.n_batches,
        batch_config, args.device
    )
    print(f"   Mean gradient norm: {grad_stats['mean_grad_norm']:.6f}")
    print(f"   Mean gradient std: {grad_stats['mean_std_grad']:.6f}")
    print(f"   SNR (gradient): {grad_stats['snr']:.4f}")
    print()

    # 3. Measure SPSA error
    print("3. Measuring SPSA gradient estimation error...")
    spsa_stats = measure_spsa_error(
        model, fixed_params, args.n_batches,
        batch_config, args.spsa_epsilon,
        args.spsa_perturbations, args.device
    )
    print(f"   Mean SPSA error: {spsa_stats['mean_error']:.6f}")
    print(f"   Mean relative error: {spsa_stats['mean_relative_error']:.4f}")
    print(f"   Mean cosine similarity: {spsa_stats['mean_cosine_similarity']:.4f}")
    print()

    # 4. Compute composite metrics
    print("4. Computing signal-to-noise ratios...")
    results = {
        'loss_noise': loss_stats,
        'gradient_noise': grad_stats,
        'spsa_error': spsa_stats,
        'snr_gradient': grad_stats['snr'],
        'snr_spsa': grad_stats['mean_grad_norm'] / (
            grad_stats['mean_std_grad'] + spsa_stats['mean_error']
        ),
        'noise_ratio': spsa_stats['mean_error'] / grad_stats['mean_std_grad'],
        'config': vars(args)
    }

    print(f"   SNR (gradient only): {results['snr_gradient']:.4f}")
    print(f"   SNR (SPSA): {results['snr_spsa']:.4f}")
    print(f"   SPSA error / Gradient noise: {results['noise_ratio']:.4f}")
    print()

    # Save results
    with open(args.output, 'w') as f:
        # Remove tensors before saving
        saveable_results = {
            k: v for k, v in results.items()
            if not isinstance(v, torch.Tensor)
        }
        json.dump(saveable_results, f, indent=2)

    print(f"✓ Results saved to {args.output}")

    return results
```

#### Visualization Functions

##### 1. Plot loss distribution
```python
def plot_loss_distribution(losses, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(losses), color='red', linestyle='--',
                label=f'Mean: {np.mean(losses):.4f}')
    plt.axvline(np.mean(losses) + np.std(losses), color='orange',
                linestyle='--', label=f'±1σ: {np.std(losses):.4f}')
    plt.axvline(np.mean(losses) - np.std(losses), color='orange',
                linestyle='--')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.title('Loss Distribution Across Different Batches')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150)
    plt.close()
```

##### 2. Plot SPSA error vs gradient noise
```python
def plot_noise_comparison(grad_stats, spsa_stats, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Error magnitudes
    categories = ['Gradient\nNoise', 'SPSA\nError']
    values = [grad_stats['mean_std_grad'], spsa_stats['mean_error']]
    ax1.bar(categories, values, color=['blue', 'red'], alpha=0.7)
    ax1.set_ylabel('Magnitude')
    ax1.set_title('Gradient Noise vs SPSA Approximation Error')
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: SPSA cosine similarity distribution
    ax2.hist(spsa_stats['cosine_similarities'], bins=30,
             alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(spsa_stats['cosine_similarities']),
                color='red', linestyle='--',
                label=f"Mean: {np.mean(spsa_stats['cosine_similarities']):.4f}")
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Frequency')
    ax2.set_title('SPSA Gradient Direction Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

##### 3. Plot SNR comparison
```python
def plot_snr_comparison(results, output_path):
    snrs = {
        'True Gradient\n(Adam)': results['snr_gradient'],
        'SPSA Gradient': results['snr_spsa']
    }

    plt.figure(figsize=(10, 6))
    plt.bar(snrs.keys(), snrs.values(), color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Signal-to-Noise Ratio')
    plt.title('Optimizer SNR Comparison')
    plt.grid(True, alpha=0.3, axis='y')

    # Add text annotations
    for i, (k, v) in enumerate(snrs.items()):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

## Expected Outputs

### 1. Console Output
```
================================================================================
NOISE ANALYSIS
================================================================================
Model: spsa_model.pt
Analyzing at current model parameters
n_batches: 100
batch_size: 32
seq_length: 5-10
SPSA config: epsilon=0.1, K=8
================================================================================

1. Measuring loss noise...
   Mean loss: 0.523456
   Std loss: 0.042123
   CV loss: 0.0805

2. Measuring gradient noise...
   Mean gradient norm: 12.345678
   Mean gradient std: 0.234567
   SNR (gradient): 52.6234

3. Measuring SPSA gradient estimation error...
   Mean SPSA error: 3.456789
   Mean relative error: 0.2801
   Mean cosine similarity: 0.8234

4. Computing signal-to-noise ratios...
   SNR (gradient only): 52.6234
   SNR (SPSA): 3.3245
   SPSA error / Gradient noise: 14.7321

✓ Results saved to noise_analysis.json
```

### 2. JSON Output (`noise_analysis.json`)
```json
{
  "loss_noise": {
    "mean": 0.523456,
    "std": 0.042123,
    "cv": 0.0805
  },
  "gradient_noise": {
    "mean_grad_norm": 12.345678,
    "mean_std_grad": 0.234567,
    "median_std_grad": 0.189234,
    "max_std_grad": 1.234567,
    "snr": 52.6234
  },
  "spsa_error": {
    "mean_error": 3.456789,
    "std_error": 0.456123,
    "mean_relative_error": 0.2801,
    "mean_cosine_similarity": 0.8234,
    "std_cosine_similarity": 0.0456
  },
  "snr_gradient": 52.6234,
  "snr_spsa": 3.3245,
  "noise_ratio": 14.7321,
  "config": {...}
}
```

### 3. Plots
- `loss_distribution.png`: Histogram of loss values across batches
- `noise_comparison.png`: Bar chart + histogram showing gradient noise vs SPSA error
- `snr_comparison.png`: Bar chart comparing SNR of Adam vs SPSA

## Key Metrics for Diagnosis

### Primary Diagnostic Metrics

1. **Noise Ratio** = SPSA_error / Gradient_noise
   - If >> 1: SPSA approximation error dominates gradient stochasticity
   - If ≈ 1: Both sources of noise are comparable
   - If << 1: Gradient stochasticity dominates

2. **SNR Degradation** = SNR_gradient / SNR_SPSA
   - Shows how much signal quality degrades with SPSA
   - Higher values indicate SPSA is struggling

3. **Cosine Similarity**
   - Mean ≈ 1: SPSA points in right direction but wrong magnitude
   - Mean < 0.9: SPSA has significant directional error
   - Mean < 0.5: SPSA is essentially noise

## Interpretation Guide

### Scenario A: High SPSA Error, Low Gradient Noise
**Observation**:
- `noise_ratio > 10`
- `mean_cosine_similarity < 0.8`

**Interpretation**:
The loss surface is relatively smooth (low gradient noise), but SPSA's finite-difference approximation with K perturbations is insufficient. The gradient estimates are inaccurate.

**Recommendation**:
- Increase K (number of perturbations)
- Tune epsilon (perturbation scale)
- Consider SPSA may need exponentially more samples

### Scenario B: High SPSA Error, High Gradient Noise
**Observation**:
- `noise_ratio > 5`
- `cv_loss > 0.1` (high loss variance)
- `mean_cosine_similarity` varies widely

**Interpretation**:
The loss surface is noisy from batch stochasticity. SPSA's error is amplified because each loss evaluation (L_plus, L_minus) comes from a different implicit batch (due to sampling randomness in the task).

**Recommendation**:
- Increase batch size to reduce gradient noise
- Use variance reduction techniques
- Consider SPSA fundamentally struggles with noisy objectives

### Scenario C: Low SPSA Error, High SNR Degradation
**Observation**:
- `noise_ratio < 2`
- `snr_spsa << snr_gradient`
- `mean_cosine_similarity > 0.9`

**Interpretation**:
SPSA estimates are directionally correct but the SNR is poor. The gradient signal exists but is overwhelmed by noise.

**Recommendation**:
- This is the "random walk with preference" regime
- May need different optimization (momentum, adaptive LR)

## Usage Examples

```bash
# Basic usage with a trained model
python diagnose_noise.py spsa_model.pt --n_batches 100

# Compare SPSA at different epsilon values
python diagnose_noise.py spsa_model.pt --spsa_epsilon 0.01 --output noise_eps001.json
python diagnose_noise.py spsa_model.pt --spsa_epsilon 0.1 --output noise_eps01.json
python diagnose_noise.py spsa_model.pt --spsa_epsilon 1.0 --output noise_eps1.json

# Compare SPSA with different perturbation counts
python diagnose_noise.py spsa_model.pt --spsa_perturbations 1 --output noise_k1.json
python diagnose_noise.py spsa_model.pt --spsa_perturbations 8 --output noise_k8.json
python diagnose_noise.py spsa_model.pt --spsa_perturbations 32 --output noise_k32.json

# Analyze on different sequence lengths
python diagnose_noise.py spsa_model.pt --min_seq_length 1 --max_seq_length 3
python diagnose_noise.py spsa_model.pt --min_seq_length 10 --max_seq_length 15
```

## Follow-up Experiments

Once we have the noise diagnostics, we can test specific hypotheses:

### Experiment 1: Does increasing K help?
Run noise analysis with K = 1, 2, 4, 8, 16, 32, 64
Plot: SPSA error vs K (should decrease, but at what rate?)

### Experiment 2: Does increasing batch size help?
Run noise analysis with batch_size = 8, 16, 32, 64, 128
Plot: All noise metrics vs batch size

### Experiment 3: Loss surface smoothness by sequence length
Run noise analysis for seq_len = 1, 3, 5, 10, 15, 20
Hypothesis: Longer sequences → noisier surface → worse SPSA performance

### Experiment 4: Analyze at different training stages
Run noise analysis on:
- Random initialization
- Early training checkpoint (10% through)
- Mid training checkpoint (50% through)
- Converged model

Hypothesis: Loss surface becomes smoother during training, SPSA performance should improve

## Extensions

### Real-time Monitoring During Training
Modify training scripts to periodically measure these metrics:
```python
if step % 100 == 0:
    quick_noise_analysis = measure_loss_noise(model, fixed_params,
                                             n_batches=20, ...)
    wandb.log({
        'loss_cv': quick_noise_analysis['cv'],
        'snr_estimate': ...
    })
```

### Compare Multiple Models
```bash
python diagnose_noise.py spsa_model.pt --output spsa_noise.json
python diagnose_noise.py adam_model.pt --output adam_noise.json
python compare_noise_profiles.py spsa_noise.json adam_noise.json
```

This would reveal if Adam finds "quieter" regions of parameter space.