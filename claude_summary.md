# Project Summary: Zero-Order RNN Training

**Last Updated:** 2025-11-26

## Project Overview
Training billion-parameter RNNs (LSTMs and DNCs) using Zero-Order Optimization methods as part of a Stanford Capstone project with Professor Mykel Kochenderfer.

**Paper:** "Scaling Recurrent Neural Networks to a Billion Parameters with Zero-Order Optimization"
**Repository:** https://github.com/stopthrowingrocks/zero_order_rnn

---

## Current Status

### Recent Progress

#### 1. Successfully Set Up Development Environment
- ✅ Provisioned 8×A40 GPUs on RunPod ($3.20/hr on-demand instance)
- ✅ SSH access configured (69.30.85.75:22073)
- ✅ Fixed indentation error in distributed_rge.py:720
- ✅ Resolved initial setup issues (CUDA availability, InfiniBand warnings)

#### 2. Implemented Clean Zero-Order Optimization Code
**New file:** [simple_reverse_task_zeroth_order.py](simple_reverse_task_zeroth_order.py)
- Implements SPSA (Simultaneous Perturbation Stochastic Approximation)
- Uses central difference gradient estimation
- Gradient-free optimization (no backpropagation)
- Successfully training on reverse sequence task

#### 3. Ran Initial Experiments
**Results:** [losses.csv](losses.csv) - 20 trials with different random seeds
- **Task:** Reverse sequences (vocab_size=20, seq_length=10-100)
- **Average final loss:** ~0.21 (range: 0.158-0.322)
- **Steps to convergence:** 91-118 steps
- **Time per trial:** ~5 seconds
- **Consistency:** Good performance across seeds

#### 4. Hyperparameter Exploration
Generated plots for:
- Learning rate sweeps
- Perturbation count sweeps (pert=2, 8, 16)
- Batch size sweeps (16, 32)
- Adaptive learning rate search ([adaptive_hpp_search.py](adaptive_hpp_search.py))

#### 5. Met with Advisor (Mykel Kochenderfer)
**Meeting notes:** [mykel-meeting-notes.md](mykel-meeting-notes.md)

---

## Action Items from Meeting

### High Priority
1. **Email François (paper author)** with experimental results
   - Share hyperparameter sweep findings
   - Ask about network configuration recommendations:
     - Optimal hidden_size / num_heads ratios
     - head_size guidelines
     - num_heads scaling rules

2. **Expand Task Coverage**
   - Currently only testing on reverse task
   - Need more diverse tasks to validate optimization method

3. **Interpretability Analysis**
   - Investigate what hidden states are encoding
   - Analyze neuron activations
   - Understand what the model learns with zero-order vs. first-order methods

4. **Written Report**
   - Document experimental methodology
   - Present findings and results
   - Compare zero-order vs. gradient-based approaches

---

## Technical Details

### Current Branch
Working on `claude` branch (not `main`)

### Key Files
- **Training code:**
  - `distributed_rge.py` - Distributed training with CD-RGE
  - `simple_reverse_task.py` - Basic reverse task with backprop
  - `simple_reverse_task_zeroth_order.py` - Zero-order optimization version

- **Hyperparameter search:**
  - `adaptive_hpp_search.py` - Adaptive LR search (halves LR on divergence)
  - `lr_sweep_pert2.py` - LR sweep for 2 perturbations
  - `pert_focused_sweep.py` - Perturbation count sweep

- **Models:**
  - `models/models.py` - LSTM and DNC implementations

- **Utilities:**
  - `telegram_notify.py` - Experiment notifications
  - `tasks/tasks.py` - Task definitions

### Hardware Setup
- **GPU:** 8× A40 (46GB each) on RunPod
- **Cost:** ~$3-4/hr for 8-10× A40 instance
- **Networking:** Standard datacenter interconnect (no InfiniBand on basic pods)
- **Workaround:** Set `NCCL_P2P_DISABLE=1` for systems without InfiniBand

### Zero-Order Optimization Parameters
Current best hyperparameters (from experiments):
- **Learning rate (lr):** 0.18
- **Perturbation size (epsilon):** 0.1
- **Number of perturbations:** 8
- **Batch size:** 32

---

## Known Issues & Solutions

### Issue 1: HuggingFace Rate Limiting
**Error:** HTTP 429 when downloading OpenWebText dataset
**Impact:** Delays initial dataset download
**Solution:** Wait for retry mechanism, or download dataset manually

### Issue 2: Collective Operation Timeout
**Error:** "Watchdog caught collective operation timeout"
**Cause:** No InfiniBand on basic RunPod instances
**Solution:** Set environment variable: `export NCCL_P2P_DISABLE=1`

### Issue 3: Large Smoke Test
**Issue:** Default smoke test trains 7B parameter model
**Impact:** Not beginner-friendly, slow to verify setup
**Solution:** Use smaller model parameters for initial testing

---

## Experimental Insights

### Zero-Order vs. Gradient-Based Optimization

**Zero-Order (SPSA) Advantages:**
- No backpropagation through time required
- Constant memory usage (doesn't scale with sequence length)
- Can train billion-parameter models on small GPUs
- Works with non-differentiable operations

**Trade-offs:**
- Requires more forward passes (2× perturbations per step)
- Slower convergence in steps (but similar wall-clock time for large models)
- Learning rate typically 10-100× higher than SGD

### Adaptive Learning Rate Strategy
From [adaptive_hpp_search.py](adaptive_hpp_search.py):
- Monitor loss trajectory
- If loss increases by 3× from minimum → halve learning rate
- Allows automatic recovery from divergence
- Up to 5 learning rate reductions per experiment

---

## Next Session Planning

### Immediate Tasks
1. Draft email to François with results
2. Analyze loss curves and convergence patterns
3. Set up additional tasks (beyond reverse)
4. Begin interpretability exploration

### Medium-Term Goals
1. Scale up to larger models (100M+ parameters)
2. Compare convergence on multiple tasks
3. Analyze parameter efficiency of zero-order methods
4. Document findings in written report

### Questions to Resolve
1. What's the optimal hidden_size/num_heads ratio?
2. How does performance scale with model size?
3. What tasks are most suitable for zero-order optimization?
4. How do learned representations differ from gradient-based training?

---

## Budget Notes
- Initial budget: $120 until advisor meeting
- Current spend: ~$8-12 on RunPod (left instance running overnight)
- Trade-off: Administrative overhead vs. monetary cost
- Consider using instance auto-shutdown features

---

## References
- **Paper:** http://arxiv.org/abs/2505.17852
- **FlashRNN:** https://github.com/NX-AI/flashrnn
- **RunPod:** https://www.runpod.io/
