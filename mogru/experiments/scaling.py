"""
MoGRU Scaling Experiments

Two sweeps:
  1. Hidden size sweep — Copy task at hidden=[32, 64, 128, 256]
  2. Sequence length sweep — Copy task at delay=[20, 50, 100, 200]

Tests whether MoGRU advantage holds, grows, or shrinks at scale.

Usage:
  python -m mogru.experiments.scaling --sweep hidden
  python -m mogru.experiments.scaling --sweep seqlen
  python -m mogru.experiments.scaling --sweep all
"""

import os
import json
import time
import argparse

import torch
import numpy as np

from mogru.mogru import count_parameters
from mogru.benchmark import (
    build_models, build_loaders, train_epoch, evaluate, set_seed,
)


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def run_single(task, seq_len, hidden_size, model_names, epochs, batch_size, lr, seed, device):
    """Run a single experiment config, return final metrics for each model."""
    set_seed(seed)
    task_type = task
    metric_name = "acc" if task in ("copy", "selective_copy") else "mse"

    train_loader, val_loader = build_loaders(
        task, seq_len, batch_size, vocab_size=16, num_markers=5,
    )
    all_models = build_models(task, 16, hidden_size)

    results = {}
    for name in model_names:
        if name not in all_models:
            continue
        set_seed(seed)
        model = all_models[name].to(device)
        params = count_parameters(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val = 0.0 if metric_name == "acc" else float("inf")
        for epoch in range(1, epochs + 1):
            train_epoch(model, train_loader, optimizer, device, task_type)
            val_metric, _ = evaluate(model, val_loader, device, task_type)

            if metric_name == "acc":
                best_val = max(best_val, val_metric)
            else:
                best_val = min(best_val, val_metric)

        results[name] = {"val": best_val, "params": params}

    return results


def run_hidden_sweep(
    seeds, hidden_sizes, task="copy", seq_len=50, epochs=30,
    batch_size=64, lr=1e-3, device_str="cpu",
):
    """Sweep hidden sizes on copy task."""
    ensure_dirs()
    device = torch.device(device_str)
    model_names = ["MoGRU", "GRU", "LSTM"]
    metric_name = "acc" if task in ("copy", "selective_copy") else "mse"

    print("\n" + "=" * 70)
    print(f"HIDDEN SIZE SWEEP: {task} (seq_len={seq_len})")
    print(f"Sizes: {hidden_sizes}, Seeds: {seeds}, Epochs: {epochs}")
    print("=" * 70)

    all_results = {}

    for hidden in hidden_sizes:
        print(f"\n--- Hidden={hidden} ---")
        seed_data = {name: [] for name in model_names}

        for seed in seeds:
            results = run_single(
                task, seq_len, hidden, model_names, epochs, batch_size, lr, seed, device,
            )
            for name in model_names:
                seed_data[name].append(results[name])
                print(f"  seed={seed} {name:6s}: val_{metric_name}={results[name]['val']:.4f} ({results[name]['params']:,} params)")

        agg = {}
        for name in model_names:
            vals = [r["val"] for r in seed_data[name]]
            agg[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": vals,
                "params": seed_data[name][0]["params"],
            }
        all_results[hidden] = agg

        # Summary line
        for name in model_names:
            a = agg[name]
            print(f"  {name:6s}: {a['mean']:.4f} ± {a['std']:.4f}")

    # Compute advantage trends
    print(f"\n{'─'*70}")
    print("ADVANTAGE TREND (MoGRU - GRU):")
    for hidden in hidden_sizes:
        mogru_mean = all_results[hidden]["MoGRU"]["mean"]
        gru_mean = all_results[hidden]["GRU"]["mean"]
        delta = mogru_mean - gru_mean
        print(f"  hidden={hidden:4d}: delta={delta:+.4f}")

    result = {
        "experiment": "hidden_sweep",
        "task": task,
        "seq_len": seq_len,
        "hidden_sizes": hidden_sizes,
        "seeds": seeds,
        "epochs": epochs,
        "metric": metric_name,
        "results": {str(k): v for k, v in all_results.items()},
    }

    path = os.path.join(RESULTS_DIR, "hidden_sweep_results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nHidden sweep results saved to {path}")
    return result


def run_seqlen_sweep(
    seeds, delays, task="copy", hidden_size=128, epochs=30,
    batch_size=64, lr=1e-3, device_str="cpu",
):
    """Sweep sequence lengths (delay) on copy task."""
    ensure_dirs()
    device = torch.device(device_str)
    model_names = ["MoGRU", "GRU", "LSTM"]
    metric_name = "acc" if task in ("copy", "selective_copy") else "mse"

    print("\n" + "=" * 70)
    print(f"SEQUENCE LENGTH SWEEP: {task} (hidden={hidden_size})")
    print(f"Delays: {delays}, Seeds: {seeds}, Epochs: {epochs}")
    print("=" * 70)

    all_results = {}

    for delay in delays:
        print(f"\n--- Delay={delay} (total_len={10+delay+1+10}) ---")
        seed_data = {name: [] for name in model_names}

        for seed in seeds:
            results = run_single(
                task, delay, hidden_size, model_names, epochs, batch_size, lr, seed, device,
            )
            for name in model_names:
                seed_data[name].append(results[name])
                print(f"  seed={seed} {name:6s}: val_{metric_name}={results[name]['val']:.4f}")

        agg = {}
        for name in model_names:
            vals = [r["val"] for r in seed_data[name]]
            agg[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": vals,
                "params": seed_data[name][0]["params"],
            }
        all_results[delay] = agg

        for name in model_names:
            a = agg[name]
            print(f"  {name:6s}: {a['mean']:.4f} ± {a['std']:.4f}")

    # Compute advantage trends
    print(f"\n{'─'*70}")
    print("ADVANTAGE TREND (MoGRU - GRU) by delay:")
    for delay in delays:
        mogru_mean = all_results[delay]["MoGRU"]["mean"]
        gru_mean = all_results[delay]["GRU"]["mean"]
        delta = mogru_mean - gru_mean
        print(f"  delay={delay:4d}: delta={delta:+.4f}")

    result = {
        "experiment": "seqlen_sweep",
        "task": task,
        "hidden_size": hidden_size,
        "delays": delays,
        "seeds": seeds,
        "epochs": epochs,
        "metric": metric_name,
        "results": {str(k): v for k, v in all_results.items()},
    }

    path = os.path.join(RESULTS_DIR, "seqlen_sweep_results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSeq-len sweep results saved to {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoGRU Scaling Experiments")
    parser.add_argument("--sweep", type=str, default="all",
                        choices=["hidden", "seqlen", "all"])
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated seeds")
    parser.add_argument("--hidden-sizes", type=str, default="32,64,128,256",
                        help="Comma-separated hidden sizes for hidden sweep")
    parser.add_argument("--delays", type=str, default="20,50,100,200",
                        help="Comma-separated delays for seq-len sweep")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    hidden_sizes = [int(h.strip()) for h in args.hidden_sizes.split(",")]
    delays = [int(d.strip()) for d in args.delays.split(",")]

    if args.sweep in ("hidden", "all"):
        run_hidden_sweep(
            seeds=seeds, hidden_sizes=hidden_sizes,
            epochs=args.epochs, batch_size=args.batch,
            lr=args.lr, device_str=args.device,
        )

    if args.sweep in ("seqlen", "all"):
        run_seqlen_sweep(
            seeds=seeds, delays=delays,
            epochs=args.epochs, batch_size=args.batch,
            lr=args.lr, device_str=args.device,
        )
