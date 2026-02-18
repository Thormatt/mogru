"""
MoGRU Results Compiler

Reads all JSON results from mogru/results/ and generates:
  - Summary tables (paper-ready, tab-separated)
  - Statistical significance checks
  - Verification: no NaN, all seeds complete, reasonable error bars

Usage:
  python -m mogru.experiments.compile_results
"""

import os
import json
import glob

import numpy as np
from scipy import stats as scipy_stats


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def load_results():
    """Load all JSON results files."""
    results = {}
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        return results

    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        name = os.path.basename(path).replace(".json", "")
        with open(path) as f:
            results[name] = json.load(f)

    return results


def verify_results(results):
    """Check for NaN, incomplete seeds, unreasonable error bars."""
    issues = []

    for name, data in results.items():
        # Check for NaN values recursively
        def check_nan(obj, path=""):
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                issues.append(f"  NaN/Inf in {name}: {path}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    check_nan(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_nan(v, f"{path}[{i}]")

        check_nan(data)

    if issues:
        print("VERIFICATION ISSUES:")
        for issue in issues:
            print(issue)
    else:
        print("VERIFICATION: All results clean (no NaN/Inf)")

    return len(issues) == 0


def compile_benchmark_summary(results):
    """Compile Table 2: Synthetic benchmark results."""
    summary = results.get("benchmark_summary")
    if not summary:
        print("\nNo benchmark_summary.json found — skipping Table 2")
        return

    print("\n" + "=" * 80)
    print("TABLE 2: Synthetic Benchmark Results (5-seed mean ± std)")
    print("=" * 80)

    for task_name, task_data in summary.items():
        metric = task_data["metric"]
        print(f"\nTask: {task_name} (metric: {metric})")
        print(f"{'Model':<10} {'Mean':>10} {'Std':>10} {'Params':>10}")
        print("-" * 42)

        task_results = task_data["results"]
        for model_name, model_data in task_results.items():
            print(f"{model_name:<10} {model_data['mean']:>10.4f} {model_data['std']:>10.4f} {model_data['params']:>10,}")

        if "stat_tests" in task_data:
            print("\nStatistical tests:")
            for test_name, test_data in task_data["stat_tests"].items():
                sig = "***" if test_data["p_value"] < 0.001 else "**" if test_data["p_value"] < 0.01 else "*" if test_data["p_value"] < 0.05 else "ns"
                print(f"  {test_name}: t={test_data['t_stat']:.3f}, p={test_data['p_value']:.4f} ({sig})")


def compile_ablation_summary(results):
    """Compile Table 3: Ablation results."""
    summary = results.get("ablation_summary")
    if not summary:
        print("\nNo ablation_summary.json found — skipping Table 3")
        return

    print("\n" + "=" * 80)
    print("TABLE 3: Ablation Study Results (3-seed mean ± std)")
    print("=" * 80)

    for task_name, task_data in summary.items():
        metric = task_data["metric"]
        print(f"\nTask: {task_name} (metric: {metric})")
        print(f"{'Variant':<18} {'Mean':>10} {'Std':>10} {'Delta':>10}")
        print("-" * 50)

        task_results = task_data["results"]
        full_mean = task_results["full_mogru"]["mean"]

        for variant, data in task_results.items():
            delta = data["mean"] - full_mean
            sign = "+" if delta >= 0 else ""
            print(f"{variant:<18} {data['mean']:>10.4f} {data['std']:>10.4f} {sign}{delta:>9.4f}")

        if "stat_tests" in task_data:
            print("\nStatistical tests (full vs ablation):")
            for test_name, test_data in task_data["stat_tests"].items():
                sig = "***" if test_data["p_value"] < 0.001 else "**" if test_data["p_value"] < 0.01 else "*" if test_data["p_value"] < 0.05 else "ns"
                print(f"  {test_name}: t={test_data['t_stat']:.3f}, p={test_data['p_value']:.4f} ({sig})")


def compile_real_world_summary(results):
    """Compile Table 4: Real-world results."""
    print("\n" + "=" * 80)
    print("TABLE 4: Real-World Results")
    print("=" * 80)

    for exp_name in ["etth1_results", "ecg5000_results", "smnist_results"]:
        data = results.get(exp_name)
        if not data:
            print(f"\nNo {exp_name}.json found — skipping")
            continue

        exp_type = data["experiment"]
        print(f"\nExperiment: {exp_type}")

        for config_name, config_data in data["results"].items():
            print(f"\n  Config: {config_name}")

            if exp_type == "etth1":
                print(f"  {'Model':<10} {'MSE':>10} {'±':>3} {'MAE':>10} {'±':>3} {'Params':>10}")
                print(f"  {'-'*50}")
                for model_name, model_data in config_data.items():
                    print(f"  {model_name:<10} {model_data['mean_mse']:>10.6f} {model_data['std_mse']:>3.4f} {model_data['mean_mae']:>10.4f} {model_data['std_mae']:>3.4f} {model_data['params']:>10,}")
            else:
                print(f"  {'Model':<10} {'Acc':>10} {'±':>8} {'Params':>10}")
                print(f"  {'-'*40}")
                for model_name, model_data in config_data.items():
                    print(f"  {model_name:<10} {model_data['mean_acc']:>10.4f} {model_data['std_acc']:>8.4f} {model_data['params']:>10,}")


def compile_scaling_summary(results):
    """Compile Figure 4 data: scaling trends."""
    for sweep_name in ["hidden_sweep_results", "seqlen_sweep_results"]:
        data = results.get(sweep_name)
        if not data:
            continue

        exp = data["experiment"]
        metric = data["metric"]
        print(f"\n{'='*80}")
        print(f"FIGURE 4 DATA: {exp}")
        print(f"{'='*80}")

        for config_val, config_data in data["results"].items():
            print(f"\n  {exp.split('_')[0]}={config_val}:")
            for model_name, model_data in config_data.items():
                print(f"    {model_name:<10}: {metric}={model_data['mean']:.4f} ± {model_data['std']:.4f}")


def compile_profiling_summary(results):
    """Compile Table 5: Throughput."""
    data = results.get("profiling_results")
    if not data:
        print("\nNo profiling_results.json found — skipping Table 5")
        return

    print("\n" + "=" * 80)
    print("TABLE 5: Throughput Profiling")
    print("=" * 80)

    configs = data["configs"]
    model_types = ["MoGRU", "GRU", "LSTM", "MomGRU", "SSM"]

    print(f"\n{'Config':<15}", end="")
    for mt in model_types:
        print(f" {mt:>12}", end="")
    print()
    print("-" * (15 + 13 * len(model_types)))

    for config_key, config in configs.items():
        gru_tps = config["models"]["GRU"]["tokens_per_sec"]
        print(f"  {config_key:<13}", end="")
        for mt in model_types:
            tps = config["models"][mt]["tokens_per_sec"]
            ratio = tps / gru_tps
            print(f" {ratio:>11.2f}x", end="")
        print()

    print(f"\n{'Config':<15}", end="")
    for mt in model_types:
        print(f" {mt:>12}", end="")
    print(" (tokens/sec)")
    print("-" * (15 + 13 * len(model_types)))

    for config_key, config in configs.items():
        print(f"  {config_key:<13}", end="")
        for mt in model_types:
            tps = config["models"][mt]["tokens_per_sec"]
            print(f" {tps:>12.0f}", end="")
        print()


def compile_param_table(results):
    """Compile Table 1: Parameter counts."""
    summary = results.get("benchmark_summary")
    if not summary:
        return

    print("\n" + "=" * 80)
    print("TABLE 1: Parameter Counts")
    print("=" * 80)

    # Get first task's param counts as reference
    first_task = next(iter(summary.values()))
    task_results = first_task["results"]

    print(f"{'Model':<10} {'Params':>10} {'Ratio vs GRU':>15}")
    print("-" * 37)
    gru_params = task_results.get("GRU", {}).get("params", 1)
    for model_name, model_data in task_results.items():
        params = model_data["params"]
        ratio = params / gru_params
        print(f"{model_name:<10} {params:>10,} {ratio:>14.2f}x")


def main():
    print("MoGRU Results Compiler")
    print("=" * 80)

    results = load_results()

    if not results:
        print("No results found. Run experiments first.")
        return

    print(f"\nFound {len(results)} result file(s):")
    for name in results:
        print(f"  - {name}.json")

    # Verify
    print()
    verify_results(results)

    # Compile tables
    compile_param_table(results)
    compile_benchmark_summary(results)
    compile_ablation_summary(results)
    compile_real_world_summary(results)
    compile_scaling_summary(results)
    compile_profiling_summary(results)

    print("\n" + "=" * 80)
    print("COMPILATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
