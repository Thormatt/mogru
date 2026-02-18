"""
MoGRU Ablation Study

Tests four ablations against full MoGRU:
  1. No momentum    (beta=0 always → falls toward vanilla GRU)
  2. Fixed beta     (beta=0.9 constant → not learned)
  3. No LayerNorm   (LayerNorm removed → is it critical?)
  4. No reset gate  (r=1 always → does reset matter with momentum?)

This isolates the contribution of each component.

Supports multiple tasks (copy, trend, selective_copy) and multi-seed runs.

Usage:
  python -m mogru.ablation --tasks copy,trend,selective_copy --seeds 42,123,456
"""

import sys
import os
import argparse
import json
import time
import random

import torch
import torch.nn as nn
import numpy as np

from mogru.mogru import MoGRUCell, MoGRU, count_parameters
from mogru.benchmark import (
    MoGRUCopyModel, MoGRUTrendModel,
    build_loaders, train_epoch, evaluate, set_seed,
)


# ===========================================================================
# Ablation cell variants
# ===========================================================================

class MoGRUCell_NoMomentum(MoGRUCell):
    """Ablation: beta=0 always (no momentum, falls toward GRU)."""

    def forward(self, x_t, h_prev, v_prev):
        cat_xh = torch.cat([x_t, h_prev], dim=-1)
        ru = torch.sigmoid(self.W_ru(cat_xh))
        r, u = ru.chunk(2, dim=-1)

        h_tilde = torch.tanh(self.W_h(x_t) + self.U_h(r * h_prev))
        d = h_tilde - h_prev

        v_t = d
        h_t = h_prev + u * v_t

        if self.use_layernorm:
            h_t = self.ln(h_t)

        beta_t = torch.zeros_like(v_t)
        return h_t, v_t, beta_t


class MoGRUCell_FixedBeta(MoGRUCell):
    """Ablation: beta=0.9 constant (not learned)."""

    FIXED_BETA = 0.9

    def forward(self, x_t, h_prev, v_prev):
        cat_xh = torch.cat([x_t, h_prev], dim=-1)
        ru = torch.sigmoid(self.W_ru(cat_xh))
        r, u = ru.chunk(2, dim=-1)

        h_tilde = torch.tanh(self.W_h(x_t) + self.U_h(r * h_prev))
        d = h_tilde - h_prev

        beta_t = torch.full_like(v_prev, self.FIXED_BETA)
        v_t = beta_t * v_prev + (1 - beta_t) * d
        h_t = h_prev + u * v_t

        if self.use_layernorm:
            h_t = self.ln(h_t)

        return h_t, v_t, beta_t


class MoGRUCell_NoLayerNorm(MoGRUCell):
    """Ablation: LayerNorm removed."""

    def __init__(self, input_size, hidden_size, use_layernorm=True, use_damping=False):
        super().__init__(input_size, hidden_size, use_layernorm=False, use_damping=use_damping)


class MoGRUCell_NoReset(MoGRUCell):
    """Ablation: r=1 always (no reset gate)."""

    def forward(self, x_t, h_prev, v_prev):
        cat_xh = torch.cat([x_t, h_prev], dim=-1)
        ru = torch.sigmoid(self.W_ru(cat_xh))
        _, u = ru.chunk(2, dim=-1)

        beta_t = torch.sigmoid(self.W_beta(cat_xh))

        h_tilde = torch.tanh(self.W_h(x_t) + self.U_h(h_prev))
        d = h_tilde - h_prev
        v_t = beta_t * v_prev + (1 - beta_t) * d
        h_t = h_prev + u * v_t

        if self.use_layernorm:
            h_t = self.ln(h_t)

        return h_t, v_t, beta_t


ABLATION_CONFIGS = {
    "full_mogru":    MoGRUCell,
    "no_momentum":   MoGRUCell_NoMomentum,
    "fixed_beta":    MoGRUCell_FixedBeta,
    "no_layernorm":  MoGRUCell_NoLayerNorm,
    "no_reset":      MoGRUCell_NoReset,
}


# ===========================================================================
# Model builders for different tasks
# ===========================================================================

def build_ablation_model(cell_class, task, vocab_size, hidden_size):
    """Build a MoGRU model with a swapped cell class for ablation."""
    if task in ("copy", "selective_copy"):
        model = MoGRUCopyModel(vocab_size, hidden_size, task=task)
    elif task == "trend":
        model = MoGRUTrendModel(hidden_size)
    else:
        raise ValueError(f"Unsupported ablation task: {task}")

    old_cell = model.rnn.cells[0]
    new_cell = cell_class(old_cell.input_size, hidden_size)
    model.rnn.cells[0] = new_cell
    return model


# ===========================================================================
# Ablation runner
# ===========================================================================

def run_ablation(
    tasks,
    seeds,
    hidden_size=128,
    vocab_size=16,
    num_markers=5,
    batch_size=64,
    num_epochs=30,
    lr=1e-3,
    device_str="cpu",
    train_samples=10_000,
    val_samples=2_000,
    results_dir=None,
):
    """Run ablation study across tasks and seeds."""
    from scipy import stats as scipy_stats

    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device(device_str)

    DEFAULT_SEQ_LENS = {"copy": 50, "trend": 80, "selective_copy": 50}

    all_results = {}

    for task in tasks:
        seq_len = DEFAULT_SEQ_LENS.get(task, 50)
        task_type = task
        metric_name = "acc" if task in ("copy", "selective_copy") else "mse"
        higher_better = task in ("copy", "selective_copy")

        print(f"\n{'='*70}")
        print(f"ABLATION TASK: {task} (seq_len={seq_len})")
        print(f"{'='*70}")

        variant_seed_results = {name: [] for name in ABLATION_CONFIGS}

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            set_seed(seed)

            train_loader, val_loader = build_loaders(
                task, seq_len, batch_size, vocab_size, num_markers,
                train_samples, val_samples,
            )

            for variant_name, cell_class in ABLATION_CONFIGS.items():
                set_seed(seed)
                model = build_ablation_model(cell_class, task, vocab_size, hidden_size)
                model = model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                best_val = 0.0 if higher_better else float("inf")
                best_beta_info = None

                for epoch in range(1, num_epochs + 1):
                    train_loss, train_metric = train_epoch(
                        model, train_loader, optimizer, device, task_type,
                    )
                    val_metric, beta_info = evaluate(model, val_loader, device, task_type)

                    if higher_better:
                        if val_metric > best_val:
                            best_val = val_metric
                            best_beta_info = beta_info
                    else:
                        if val_metric < best_val:
                            best_val = val_metric
                            best_beta_info = beta_info

                variant_seed_results[variant_name].append({
                    "seed": seed,
                    "best_val": best_val,
                    "beta_info": best_beta_info,
                    "params": count_parameters(model),
                })

                print(f"  {variant_name:15s} seed={seed} best_val_{metric_name}={best_val:.4f}")

        # Aggregate
        task_summary = {"task": task, "seq_len": seq_len, "metric": metric_name, "seeds": seeds}
        aggregated = {}
        for variant_name, seed_runs in variant_seed_results.items():
            vals = [r["best_val"] for r in seed_runs]
            aggregated[variant_name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": vals,
                "params": seed_runs[0]["params"],
            }

        # Stats: full_mogru vs each ablation
        stat_tests = {}
        full_vals = [r["best_val"] for r in variant_seed_results["full_mogru"]]
        for variant_name in ABLATION_CONFIGS:
            if variant_name == "full_mogru":
                continue
            other_vals = [r["best_val"] for r in variant_seed_results[variant_name]]
            if len(seeds) >= 3:
                t_stat, p_val = scipy_stats.ttest_rel(full_vals, other_vals)
                stat_tests[f"full_vs_{variant_name}"] = {
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                    "significant_005": bool(p_val < 0.05),
                }

        task_summary["results"] = aggregated
        task_summary["stat_tests"] = stat_tests
        all_results[task] = task_summary

        # Print summary
        print(f"\n{'─'*70}")
        print(f"ABLATION SUMMARY: {task}")
        print(f"{'─'*70}")
        full_mean = aggregated["full_mogru"]["mean"]
        for variant_name, agg in aggregated.items():
            delta = agg["mean"] - full_mean
            sign = "+" if delta >= 0 else ""
            print(f"  {variant_name:15s}  val_{metric_name}={agg['mean']:.4f} ± {agg['std']:.4f}  ({sign}{delta:.4f})")
        for test_name, test_result in stat_tests.items():
            sig = "**" if test_result["significant_005"] else ""
            print(f"  {test_name}: t={test_result['t_stat']:.3f}, p={test_result['p_value']:.4f} {sig}")

    # Save
    summary_path = os.path.join(results_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAblation results saved to {summary_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoGRU Ablation Study")
    parser.add_argument("--tasks", type=str, default="copy,trend,selective_copy",
                        help="Comma-separated tasks")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated seeds")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    run_ablation(
        tasks=tasks,
        seeds=seeds,
        hidden_size=args.hidden,
        num_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device_str=args.device,
        results_dir=args.results_dir,
    )
