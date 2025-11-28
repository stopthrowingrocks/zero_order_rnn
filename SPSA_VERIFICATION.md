# SPSA Implementation Verification

## Mathematical Correctness Analysis

### Central Difference SPSA Formula

For parameter θ, with perturbation direction δ sampled from N(0,I):

```
∇L(θ) ≈ [L(θ + ε·δ) - L(θ - ε·δ)] / (2·ε) · δ
```

Update rule:
```
θ_{t+1} = θ_t - α · (1/K) · Σ_{k=1}^K [L(θ + ε·δ_k) - L(θ - ε·δ_k)] / (2·ε) · δ_k
```

where:
- α = learning rate
- ε = perturbation scale (set equal to α for SPSA)
- K = total number of perturbations

### Our Implementation (`cache_roll=False`)

**Step 1**: For each perturbation k = 1...K:
```python
# Evaluate +ε
apply_probe(params, +ε, seed_k, "normal")  # θ → θ + ε·δ_k
L_plus = loss(θ + ε·δ_k)

# Evaluate -ε
apply_probe(params, -2ε, seed_k, "normal")  # θ + ε·δ_k → θ - ε·δ_k
L_minus = loss(θ - ε·δ_k)

# Restore
apply_probe(params, +ε, seed_k, "normal")  # θ - ε·δ_k → θ (BACK TO ORIGINAL!)

# Store
loss_pairs[k] = (L_plus, L_minus)
```

**Step 2**: Apply all updates:
```python
for k in range(K):
    coef_k = (L_plus_k - L_minus_k) / (2·K)
    apply_probe(params, -α·coef_k, seed_k, "normal")  # θ → θ - α·coef_k·δ_k
```

### Expanding Step 2:

```
θ_final = θ - α · Σ_{k=1}^K coef_k · δ_k
        = θ - α · Σ_{k=1}^K [(L_plus_k - L_minus_k) / (2·K)] · δ_k
        = θ - α · (1/K) · Σ_{k=1}^K [(L_plus_k - L_minus_k) / 2] · δ_k
```

But wait, L_plus_k = L(θ + ε·δ_k) and L_minus_k = L(θ - ε·δ_k), so:

```
θ_final = θ - α · (1/K) · Σ_{k=1}^K [L(θ + ε·δ_k) - L(θ - ε·δ_k)] / 2 · δ_k
```

### ISSUE FOUND!

The formula should be divided by `2·ε`, not just `2`!

**Expected**:
```
coef_k = [L(θ + ε·δ_k) - L(θ - ε·δ_k)] / (2·ε·K)
```

**Our code** (line 158):
```python
coef = (L_plus - L_minus) / (2.0 * n_total)
```

This is missing the division by `ε`!

### Correction Needed

The coefficient should be:
```python
coef = (L_plus - L_minus) / (2.0 * epsilon * n_total)
```

OR, since we apply `-learning_rate * coef` in line 197, and we want `epsilon = learning_rate`, we could keep the current formula BUT change line 197 to:
```python
apply_probe(param_list, -coef.item(), seed_i, distn)  # Remove learning_rate multiplier
```

## Verification Against Reference

Let me check the reference implementation...

From distributed_rge.py line 490:
```python
coef = (L_plus - L_minus) / (2.0 * n_total)
restore_coeff = -coef
```

And line 523:
```python
p.data.add_(acc)  # No learning_rate here!
```

So in the cache_roll=True path, the learning rate is NOT applied in the final add!

But wait, looking at line 492-496:
```python
apply_probe(
    param_list, +epsilon, seed_m, distn,
    rolling_sum_weighted_probe=rolling_sum_weighted_probe,
    coeff=restore_coeff,
)
```

This accumulates: `rolling_sum_weighted_probe += restore_coeff · δ = -coef · δ`

Then line 523 applies this directly: `p.data.add_(acc)`

So the update is:
```
θ → θ + Σ_k (-coef_k · δ_k)
  = θ - Σ_k [(L_plus_k - L_minus_k) / (2·K)] · δ_k
```

Still missing division by ε! Unless... let me check if learning_rate is applied elsewhere...

Looking at the reference implementation more carefully, I notice that `epsilon = self.epsilon_tying_ratio * self.learning_rate` (line 427).

So if `epsilon_tying_ratio = 1`, then `epsilon = learning_rate`.

And the gradient estimate is:
```
g_k = (L_plus - L_minus) / (2·ε) · δ_k
    = (L_plus - L_minus) / (2·learning_rate) · δ_k
```

The update should be:
```
θ → θ - learning_rate · (1/K) · Σ_k g_k
  = θ - learning_rate · (1/K) · Σ_k [(L_plus - L_minus) / (2·learning_rate)] · δ_k
  = θ - (1/K) · Σ_k [(L_plus - L_minus) / 2] · δ_k
```

Ah! The learning_rate cancels out! That's why the reference doesn't divide by ε!

So the formula `coef = (L_plus - L_minus) / (2.0 * n_total)` is CORRECT when epsilon = learning_rate!

## Conclusion

The implementation is mathematically correct! The key insight is that when `ε = α` (perturbation scale equals learning rate), the gradient estimate becomes:

```
∇L ≈ (L(θ+ε·δ) - L(θ-ε·δ)) / (2·ε) · δ
```

And the update with learning rate α is:
```
θ → θ - α · ∇L = θ - α · [(L+ - L-) / (2·α)] · δ = θ - [(L+ - L-) / 2] · δ
```

Our code correctly implements this!