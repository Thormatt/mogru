"""
MoGRU Crossover Sweep

Finds the sequence length where MoGRU starts beating GRU on copy and
selective copy tasks. Sweeps seq_len and reports the crossover point.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time

from mogru.mogru import MoGRU, count_parameters
from mogru.benchmark import (
    build_models, build_loaders, train_epoch, evaluate, set_seed,
)


def sweep_task(task, seq_lengths, hidden_size=128, batch_size=64,
               epochs=20, lr=1e-3, seed=42):
    """Run MoGRU vs GRU across multiple sequence lengths."""
    set_seed(seed)
    device = torch.device("cpu")
    metric_name = "acc" if task in ("copy", "selective_copy") else "mse"
    higher_better = metric_name == "acc"

    print(f"\n{'='*70}")
    print(f"SWEEP: {task} | seq_lengths={seq_lengths}")
    print(f"{'='*70}")

    results = []

    for seq_len in seq_lengths:
        set_seed(seed)
        print(f"\n--- seq_len={seq_len} ---")

        train_loader, val_loader = build_loaders(
            task, seq_len, batch_size,
            train_samples=5000, val_samples=1000,
        )

        mogru_model, gru_model, _ = build_models(task, 16, hidden_size)
        mogru_model, gru_model = mogru_model.to(device), gru_model.to(device)

        # Train MoGRU
        opt_m = torch.optim.Adam(mogru_model.parameters(), lr=lr)
        best_mogru = 0.0 if higher_better else float("inf")
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            train_epoch(mogru_model, train_loader, opt_m, device, task)
            val_metric, beta_info = evaluate(mogru_model, val_loader, device, task)
            if higher_better:
                best_mogru = max(best_mogru, val_metric)
            else:
                best_mogru = min(best_mogru, val_metric)
        mogru_time = time.time() - t0

        # Train GRU
        opt_g = torch.optim.Adam(gru_model.parameters(), lr=lr)
        best_gru = 0.0 if higher_better else float("inf")
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            train_epoch(gru_model, train_loader, opt_g, device, task)
            val_metric, _ = evaluate(gru_model, val_loader, device, task)
            if higher_better:
                best_gru = max(best_gru, val_metric)
            else:
                best_gru = min(best_gru, val_metric)
        gru_time = time.time() - t0

        if higher_better:
            delta = best_mogru - best_gru
            winner = "MoGRU" if delta > 0 else "GRU"
        else:
            delta = best_gru - best_mogru
            winner = "MoGRU" if delta > 0 else "GRU"

        print(f"  MoGRU: {metric_name}={best_mogru:.4f} ({mogru_time:.0f}s)")
        print(f"  GRU:   {metric_name}={best_gru:.4f} ({gru_time:.0f}s)")
        print(f"  Winner: {winner} (delta={delta:+.4f})")

        results.append({
            "seq_len": seq_len,
            "mogru": best_mogru,
            "gru": best_gru,
            "delta": delta,
            "winner": winner,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"CROSSOVER SUMMARY: {task}")
    print(f"{'='*70}")
    print(f"{'seq_len':>8} {'MoGRU':>10} {'GRU':>10} {'Delta':>10} {'Winner':>8}")
    print("-" * 50)

    crossover = None
    for r in results:
        print(f"{r['seq_len']:>8} {r['mogru']:>10.4f} {r['gru']:>10.4f} "
              f"{r['delta']:>+10.4f} {r['winner']:>8}")
        if r["winner"] == "MoGRU" and crossover is None:
            crossover = r["seq_len"]

    if crossover:
        print(f"\nCrossover point: seq_len ~{crossover} (MoGRU starts winning)")
    else:
        print(f"\nNo crossover found in tested range — GRU leads throughout")

    return results


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "copy"

    if task == "copy":
        seq_lengths = [5, 10, 15, 20, 25, 30, 40, 50]
    elif task == "selective_copy":
        seq_lengths = [30, 50, 75, 100, 150, 200]
    elif task == "adding":
        seq_lengths = [50, 100, 200, 300, 500]
    elif task == "trend":
        seq_lengths = [50, 100, 200, 300, 500]
    else:
        raise ValueError(f"Unknown task: {task}")

    sweep_task(task, seq_lengths)
