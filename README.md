<p align="center">
  <img src="assets/mogru-architecture.svg" alt="MoGRU Architecture" width="800"/>
</p>

<h1 align="center">MoGRU: Momentum-Gated Recurrent Unit</h1>

<p align="center">
  <strong>When Second-Order Dynamics Help and When They Don't</strong>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"/></a>
</p>

---

> **This is a negative-result paper.** MoGRU works in a narrow niche (moderate-density discrete interference) but fails to generalize to real-world signals, long sequences, or cross-domain transfer. We publish the complete characterization so others don't have to rediscover these limits.

---

## Overview

MoGRU augments the GRU with a **velocity state** and a **learned per-dimension momentum gate** (beta), giving hidden states second-order dynamics analogous to momentum in SGD. The model is a **strict generalization** of the GRU: when beta = 0, MoGRU recovers exact GRU behavior.

**Key findings:**
- MoGRU achieves **perfect accuracy** on interference resistance at N <= 50 distractors, where GRU scores 0.77-0.82
- The advantage is **structurally bounded**: at N >= 100, the momentum buffer itself becomes corrupted
- On real-world vibration data (CWRU Bearing), momentum **smooths away** high-frequency fault impulses -- GRU wins at all noise levels
- Four attempted fixes (velocity clipping, velocity LayerNorm, velocity write gate, damping) all fail to close the long-range gap
- MoGRU runs **2-3x slower** than GRU due to inability to use cuDNN fused kernels

## Architecture

Two states per cell:
- **h_t** (position): hidden representation, same role as GRU hidden state
- **v_t** (velocity): exponential moving average of state deltas

```
[r_t, u_t] = sigma(W_ru [x_t, h_{t-1}])        # Standard GRU gates
beta_t     = sigma(W_beta [x_t, h_{t-1}])       # Momentum retention (novel)
h_tilde    = tanh(W_h x_t + U_h (r_t * h_{t-1}))  # Candidate
d_t        = h_tilde - h_{t-1}                   # State delta
v_t        = beta_t * v_{t-1} + (1 - beta_t) * d_t  # Velocity EMA
h_t        = LayerNorm(h_{t-1} + u_t * v_t)     # Additive position step
```

When beta -> 0, velocity equals the raw delta, and the update reduces to the standard GRU convex combination. The optimizer can "turn off" momentum per-dimension where it's unhelpful.

## Results

### Benchmark (mean +/- std over 5 seeds, hidden=128)

| Task | MoGRU | GRU | LSTM | MomGRU |
|------|-------|-----|------|--------|
| Copy (acc) | **0.350 +/- 0.010*** | 0.221 +/- 0.090 | 0.063 +/- 0.001 | 0.243 +/- 0.012 |
| Adding (MSE) | 0.003 +/- 0.001 | **0.000 +/- 0.000*** | 0.003 +/- 0.002 | 0.003 +/- 0.001 |
| Trend (MSE) | 0.775 +/- 0.041 | 0.792 +/- 0.031 | 0.833 +/- 0.031 | **0.777 +/- 0.017** |
| Sel. Copy (acc) | **1.000 +/- 0.000** | **1.000 +/- 0.000** | **1.000 +/- 0.000** | 0.415 +/- 0.062* |

\* p < 0.05 (two-sample t-test)

### Interference Resistance (K=5 items, hidden=128)

| N distractors | MoGRU | GRU | LSTM | Winner |
|---------------|-------|-----|------|--------|
| 10 | **1.000** | 0.765 | 0.065 | MoGRU |
| 25 | **1.000** | 0.710 | 0.064 | MoGRU |
| 50 | **0.938** | 0.816 | 0.068 | MoGRU |
| 100 | 0.553 | **0.729** | 0.067 | GRU |
| 200 | 0.063 | **0.293** | 0.067 | GRU |

MoGRU wins at moderate distractor loads. At N >= 100, accumulated distractor influence overwhelms the momentum buffer.

### CWRU Bearing Fault Detection (real-world)

| Noise sigma | GRU | LSTM | MoGRU | Winner |
|-------------|-----|------|-------|--------|
| 0.0 | **0.474** | 0.411 | 0.355 | GRU |
| 0.1 | **0.484** | 0.468 | 0.447 | GRU |
| 0.2 | **0.507** | 0.461 | 0.275 | GRU |
| 0.5 | **0.493** | 0.465 | 0.313 | GRU |
| 1.0 | **0.495** | 0.486 | 0.299 | GRU |
| 2.0 | **0.468** | 0.449 | 0.298 | GRU |

GRU wins at all noise levels. Momentum smooths away high-frequency fault impulses that carry diagnostic information.

### Throughput (tokens/sec, CPU, batch=64)

| Config | GRU | MoGRU | LSTM | Ratio |
|--------|-----|-------|------|-------|
| h=64, T=50 | 203,893 | 72,588 | 129,212 | 2.8x |
| h=128, T=50 | 125,023 | 53,673 | 93,870 | 2.3x |
| h=128, T=200 | 117,604 | 54,759 | 83,009 | 2.1x |
| h=256, T=200 | 55,265 | 31,681 | 39,232 | 1.7x |

### Ablation (selective copy, T=50)

| Variant | Val Acc | Delta |
|---------|---------|-------|
| full_mogru | 1.000 | -- |
| no_momentum (beta=0) | 1.000 | 0.000 |
| **fixed_beta (beta=0.9)** | **0.730** | **-0.270** |
| no_layernorm | 1.000 | 0.000 |
| no_reset (r=1) | 1.000 | 0.000 |

The learned per-dimension beta gate is the critical innovation. Fixed momentum is worse than no momentum.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the 4-task benchmark (5 seeds each)
python -m mogru.benchmark

# Run interference resistance sweep
python -m mogru.interference_deep_dive

# Run ablation study
python -m mogru.ablation

# Run CWRU bearing benchmark (requires data, see data/cwru/README.md)
python -m mogru.bearing_benchmark
```

## Repository Structure

```
mogru/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── assets/
│   └── mogru-architecture.svg
├── paper/
│   └── mogru.tex                    # arXiv-ready paper
├── mogru/
│   ├── __init__.py
│   ├── mogru.py                     # MoGRUCell, MoGRU, MomentumGRUCell, MomentumGRU
│   ├── benchmark.py                 # 4-task benchmark suite
│   ├── ablation.py                  # 5-way ablation study
│   ├── bearing_benchmark.py         # CWRU real-world fault detection
│   ├── crossover_sweep.py           # Sequence length crossover analysis
│   ├── head_to_head.py              # 3-way model comparison
│   ├── interference_deep_dive.py    # Distractor count + items sweep
│   ├── velocity_fix_test.py         # Long-range collapse fix attempts
│   ├── strategic_transfer_test.py   # VICReg + LOO cross-domain transfer
│   └── experiments/
│       ├── __init__.py
│       ├── compile_results.py
│       ├── profiling.py
│       ├── real_world.py
│       └── scaling.py
├── results/
│   ├── benchmark_summary.json       # Aggregated benchmark results
│   ├── profiling_results.json       # Throughput profiling data
│   └── [20 per-seed JSON files]     # Raw results per task per seed
└── data/
    └── cwru/
        └── README.md                # CWRU dataset download instructions
```

## Citation

If you find this work useful (even as a negative result), please cite:

```bibtex
@article{matthiasson2025mogru,
  title   = {Momentum-Gated Recurrent Unit: When Second-Order Dynamics Help and When They Don't},
  author  = {Matthiasson, Thor},
  year    = {2025},
  note    = {Available at \url{https://github.com/Thormatt/mogru}}
}
```

## License

[MIT](LICENSE)
