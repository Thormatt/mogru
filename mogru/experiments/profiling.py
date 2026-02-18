"""
MoGRU Throughput Profiling

Measures wall-clock throughput (tokens/sec) and peak memory for all models
across different hidden sizes and sequence lengths.

Usage:
  python -m mogru.experiments.profiling
"""

import os
import json
import time
import argparse
import tracemalloc

import torch
import torch.nn as nn

from mogru.mogru import MoGRU, MomentumGRU, count_parameters
from mogru.head_to_head import SSMModel


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def build_model(model_type, input_size, hidden_size):
    """Build a bare RNN model for throughput testing."""
    if model_type == "MoGRU":
        return MoGRU(input_size, hidden_size)
    elif model_type == "GRU":
        return nn.GRU(input_size, hidden_size, batch_first=True)
    elif model_type == "LSTM":
        return nn.LSTM(input_size, hidden_size, batch_first=True)
    elif model_type == "MomGRU":
        return MomentumGRU(input_size, hidden_size)
    elif model_type == "SSM":
        return SSMModel(input_size, hidden_size, hidden_size, num_layers=2)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def measure_throughput(model, model_type, batch_size, seq_len, input_size, n_warmup=3, n_runs=10):
    """Measure forward+backward throughput in tokens/sec."""
    device = torch.device("cpu")
    model = model.to(device)

    x = torch.randn(batch_size, seq_len, input_size, device=device)

    def run_forward_backward():
        if model_type in ("MoGRU", "MomGRU"):
            out, _, _ = model(x)
        elif model_type == "SSM":
            out = model(x)
        else:
            out, _ = model(x)

        if out.dim() == 3:
            loss = out.sum()
        else:
            loss = out.sum()
        loss.backward()

    # Warmup
    for _ in range(n_warmup):
        model.zero_grad()
        run_forward_backward()

    # Timed runs
    times = []
    for _ in range(n_runs):
        model.zero_grad()
        t0 = time.perf_counter()
        run_forward_backward()
        times.append(time.perf_counter() - t0)

    total_tokens = batch_size * seq_len
    avg_time = sum(times) / len(times)
    tokens_per_sec = total_tokens / avg_time

    return {
        "tokens_per_sec": round(tokens_per_sec, 1),
        "avg_time_ms": round(avg_time * 1000, 2),
        "min_time_ms": round(min(times) * 1000, 2),
        "max_time_ms": round(max(times) * 1000, 2),
    }


def measure_memory(model, model_type, batch_size, seq_len, input_size):
    """Measure peak memory usage during forward+backward."""
    device = torch.device("cpu")
    model = model.to(device)

    x = torch.randn(batch_size, seq_len, input_size, device=device)

    tracemalloc.start()

    model.zero_grad()
    if model_type in ("MoGRU", "MomGRU"):
        out, _, _ = model(x)
    elif model_type == "SSM":
        out = model(x)
    else:
        out, _ = model(x)

    loss = out.sum()
    loss.backward()

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return round(peak / (1024 * 1024), 2)  # MB


def run_profiling(
    hidden_sizes=None,
    seq_lens=None,
    batch_size=64,
    input_size=1,
):
    """Run full profiling suite."""
    ensure_dirs()

    if hidden_sizes is None:
        hidden_sizes = [64, 128, 256]
    if seq_lens is None:
        seq_lens = [50, 100, 200]

    model_types = ["MoGRU", "GRU", "LSTM", "MomGRU", "SSM"]

    print("\n" + "=" * 70)
    print("THROUGHPUT PROFILING")
    print(f"Hidden: {hidden_sizes}, Seq: {seq_lens}, Batch: {batch_size}")
    print("=" * 70)

    all_results = {}

    for hidden in hidden_sizes:
        for seq_len in seq_lens:
            config_key = f"h{hidden}_s{seq_len}"
            print(f"\n--- hidden={hidden}, seq_len={seq_len} ---")
            config_results = {}

            for model_type in model_types:
                model = build_model(model_type, input_size, hidden)
                params = count_parameters(model)

                throughput = measure_throughput(
                    model, model_type, batch_size, seq_len, input_size,
                )

                # Fresh model for memory measurement
                model = build_model(model_type, input_size, hidden)
                peak_mb = measure_memory(
                    model, model_type, batch_size, seq_len, input_size,
                )

                config_results[model_type] = {
                    "params": params,
                    "tokens_per_sec": throughput["tokens_per_sec"],
                    "avg_time_ms": throughput["avg_time_ms"],
                    "peak_memory_mb": peak_mb,
                }

                print(f"  {model_type:6s}: {throughput['tokens_per_sec']:>10.1f} tok/s | {throughput['avg_time_ms']:>8.2f} ms | {peak_mb:>6.1f} MB | {params:,} params")

            all_results[config_key] = {
                "hidden_size": hidden,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "models": config_results,
            }

    # Relative throughput (normalized to GRU=1.0)
    print(f"\n{'─'*70}")
    print("RELATIVE THROUGHPUT (vs GRU=1.0):")
    print(f"{'Config':<15} {'MoGRU':>8} {'GRU':>8} {'LSTM':>8} {'MomGRU':>8} {'SSM':>8}")
    print("-" * 63)
    for config_key, config in all_results.items():
        gru_tps = config["models"]["GRU"]["tokens_per_sec"]
        vals = []
        for mt in model_types:
            ratio = config["models"][mt]["tokens_per_sec"] / gru_tps
            vals.append(f"{ratio:.2f}")
        print(f"  {config_key:<13} {'  '.join(f'{v:>6}' for v in vals)}")

    result = {
        "experiment": "profiling",
        "batch_size": batch_size,
        "input_size": input_size,
        "configs": all_results,
    }

    path = os.path.join(RESULTS_DIR, "profiling_results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nProfiling results saved to {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoGRU Throughput Profiling")
    parser.add_argument("--hidden-sizes", type=str, default="64,128,256")
    parser.add_argument("--seq-lens", type=str, default="50,100,200")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=1)
    args = parser.parse_args()

    hidden_sizes = [int(h.strip()) for h in args.hidden_sizes.split(",")]
    seq_lens = [int(s.strip()) for s in args.seq_lens.split(",")]

    run_profiling(
        hidden_sizes=hidden_sizes,
        seq_lens=seq_lens,
        batch_size=args.batch,
        input_size=args.input_size,
    )
