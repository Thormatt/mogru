"""
MoGRU Benchmark Suite

Four synthetic tasks testing different aspects of momentum dynamics:

1. Copy Task — Store & recall after a delay.
   Momentum preserves representations through the blank gap.

2. Adding Problem — Two marked numbers in a long noisy sequence; output their sum.
   Velocity tracks cumulative effects across long distances.

3. Noisy Trend — Track a signal composed of trend + oscillation + noise.
   Momentum's core strength: smoothing and extrapolation.

4. Selective Copy — Recall only marked tokens after a delay.
   Tests selective memory under distraction.
"""

import sys
import os

# Support running from inside mogru/ and from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import json
import random
from datetime import datetime, timezone

from mogru.mogru import MoGRU, MomentumGRU, count_parameters


# ===========================================================================
# Datasets
# ===========================================================================

class CopyDataset(Dataset):
    """Copy task: memorize a sequence, output it after a blank delay."""

    def __init__(self, num_samples=10_000, seq_len=10, delay=50, vocab_size=8):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.delay = delay
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T = self.seq_len
        # Content: random tokens 1..vocab_size
        content = torch.randint(1, self.vocab_size + 1, (T,))
        # Input: [content | blank(delay) | delimiter(0) | blank(T)]
        total_len = T + self.delay + 1 + T
        input_seq = torch.zeros(total_len, dtype=torch.long)
        input_seq[:T] = content
        # Delimiter token = vocab_size + 1 (special)
        input_seq[T + self.delay] = self.vocab_size + 1
        # Target: predict content at the end
        target = torch.zeros(total_len, dtype=torch.long)
        target[T + self.delay + 1:] = content
        # Loss mask: only on recall positions
        loss_mask = torch.zeros(total_len)
        loss_mask[T + self.delay + 1:] = 1.0
        return input_seq, target, loss_mask


class AddingDataset(Dataset):
    """Adding problem: sum two marked numbers in a long sequence."""

    def __init__(self, num_samples=10_000, seq_len=100):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T = self.seq_len
        # Random numbers in [0, 1]
        numbers = torch.rand(T)
        # Marker channel: two random positions
        markers = torch.zeros(T)
        positions = torch.randperm(T)[:2]
        markers[positions] = 1.0
        # Input: (numbers, markers) stacked -> (T, 2)
        input_seq = torch.stack([numbers, markers], dim=-1)
        # Target: sum of the two marked numbers
        target = numbers[positions].sum().unsqueeze(0)
        return input_seq, target


class NoisyTrendDataset(Dataset):
    """Noisy trend: track signal = trend + oscillation + noise."""

    def __init__(self, num_samples=10_000, seq_len=100):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T = self.seq_len
        t = torch.linspace(0, 1, T)

        # Random parameters for diversity
        slope = (torch.rand(1) - 0.5) * 4       # [-2, 2]
        freq = torch.rand(1) * 6 + 2            # [2, 8]
        amp = torch.rand(1) * 0.5 + 0.1         # [0.1, 0.6]
        noise_std = torch.rand(1) * 0.3 + 0.05  # [0.05, 0.35]

        # Clean signal
        clean = slope * t + amp * torch.sin(2 * 3.14159 * freq * t)
        # Noisy observation
        noisy = clean + noise_std * torch.randn(T)

        # Input: noisy signal (T, 1)
        input_seq = noisy.unsqueeze(-1)
        # Target: clean signal (T, 1) — model should denoise/track
        target = clean.unsqueeze(-1)
        return input_seq, target


class SelectiveCopyDataset(Dataset):
    """Selective copy: recall only marked tokens after a delay."""

    def __init__(self, num_samples=10_000, seq_len=50, num_markers=5, vocab_size=16):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_markers = num_markers
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T = self.seq_len
        K = self.num_markers

        content = torch.randint(1, self.vocab_size + 1, (T,))
        mark_positions = torch.randperm(T)[:K].sort().values
        markers = torch.zeros(T, dtype=torch.long)
        markers[mark_positions] = 1

        input_tokens = torch.zeros(T + K, dtype=torch.long)
        input_markers = torch.zeros(T + K, dtype=torch.long)
        input_tokens[:T] = content
        input_markers[:T] = markers

        target = torch.zeros(T + K, dtype=torch.long)
        target[T:] = content[mark_positions]

        loss_mask = torch.zeros(T + K)
        loss_mask[T:] = 1.0

        return input_tokens, input_markers, target, loss_mask


# ===========================================================================
# Model wrappers
# ===========================================================================

class MoGRUCopyModel(nn.Module):
    """MoGRU for copy / selective-copy classification tasks."""

    def __init__(self, vocab_size, hidden_size, task="copy"):
        super().__init__()
        self.task = task
        if task == "selective_copy":
            self.embed_token = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=0)
            self.embed_marker = nn.Embedding(2, hidden_size)
            input_size = hidden_size
        else:
            # Copy task: vocab_size + 2 (0=blank, 1..V=content, V+1=delimiter)
            self.embed = nn.Embedding(vocab_size + 2, hidden_size, padding_idx=0)
            input_size = hidden_size

        self.rnn = MoGRU(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.head = nn.Linear(hidden_size, vocab_size + 1)

    def forward(self, *args):
        if self.task == "selective_copy":
            tokens, markers = args
            x = self.embed_token(tokens) + self.embed_marker(markers)
        else:
            (tokens,) = args
            x = self.embed(tokens)
        output, _, betas = self.rnn(x)
        logits = self.head(output)
        return logits, betas


class GRUCopyBaseline(nn.Module):
    """Vanilla GRU for copy / selective-copy tasks."""

    def __init__(self, vocab_size, hidden_size, task="copy"):
        super().__init__()
        self.task = task
        if task == "selective_copy":
            self.embed_token = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=0)
            self.embed_marker = nn.Embedding(2, hidden_size)
            input_size = hidden_size
        else:
            self.embed = nn.Embedding(vocab_size + 2, hidden_size, padding_idx=0)
            input_size = hidden_size

        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size + 1)

    def forward(self, *args):
        if self.task == "selective_copy":
            tokens, markers = args
            x = self.embed_token(tokens) + self.embed_marker(markers)
        else:
            (tokens,) = args
            x = self.embed(tokens)
        output, _ = self.rnn(x)
        logits = self.head(output)
        return logits, None


class MomGRUCopyBaseline(nn.Module):
    """MomentumRNN-style GRU (Nguyen 2020) for copy / selective-copy tasks."""

    def __init__(self, vocab_size, hidden_size, task="copy"):
        super().__init__()
        self.task = task
        if task == "selective_copy":
            self.embed_token = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=0)
            self.embed_marker = nn.Embedding(2, hidden_size)
            input_size = hidden_size
        else:
            self.embed = nn.Embedding(vocab_size + 2, hidden_size, padding_idx=0)
            input_size = hidden_size

        self.rnn = MomentumGRU(input_size, hidden_size, num_layers=1)
        self.head = nn.Linear(hidden_size, vocab_size + 1)

    def forward(self, *args):
        if self.task == "selective_copy":
            tokens, markers = args
            x = self.embed_token(tokens) + self.embed_marker(markers)
        else:
            (tokens,) = args
            x = self.embed(tokens)
        output, _, betas = self.rnn(x)
        logits = self.head(output)
        return logits, betas


class MomGRUAddingBaseline(nn.Module):
    """MomentumRNN-style GRU for the adding problem."""

    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = MomentumGRU(2, hidden_size, num_layers=1)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _, betas = self.rnn(x)
        out = self.head(output[:, -1])
        return out, betas


class MomGRUTrendBaseline(nn.Module):
    """MomentumRNN-style GRU for noisy trend tracking."""

    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = MomentumGRU(1, hidden_size, num_layers=1)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _, betas = self.rnn(x)
        out = self.head(output)
        return out, betas


# --- LSTM baselines ---

class LSTMCopyBaseline(nn.Module):
    """Vanilla LSTM for copy / selective-copy tasks."""

    def __init__(self, vocab_size, hidden_size, task="copy"):
        super().__init__()
        self.task = task
        if task == "selective_copy":
            self.embed_token = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=0)
            self.embed_marker = nn.Embedding(2, hidden_size)
            input_size = hidden_size
        else:
            self.embed = nn.Embedding(vocab_size + 2, hidden_size, padding_idx=0)
            input_size = hidden_size

        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size + 1)

    def forward(self, *args):
        if self.task == "selective_copy":
            tokens, markers = args
            x = self.embed_token(tokens) + self.embed_marker(markers)
        else:
            (tokens,) = args
            x = self.embed(tokens)
        output, _ = self.rnn(x)
        logits = self.head(output)
        return logits, None


class LSTMAddingBaseline(nn.Module):
    """Vanilla LSTM for the adding problem."""

    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = nn.LSTM(2, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.rnn(x)
        out = self.head(output[:, -1])
        return out, None


class LSTMTrendBaseline(nn.Module):
    """Vanilla LSTM for noisy trend tracking."""

    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.rnn(x)
        out = self.head(output)
        return out, None


class MoGRUAddingModel(nn.Module):
    """MoGRU for the adding problem (regression)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = MoGRU(input_size=2, hidden_size=hidden_size, num_layers=1)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _, betas = self.rnn(x)
        out = self.head(output[:, -1])
        return out, betas


class GRUAddingBaseline(nn.Module):
    """Vanilla GRU for the adding problem."""

    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = nn.GRU(2, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.rnn(x)
        out = self.head(output[:, -1])
        return out, None


class MoGRUTrendModel(nn.Module):
    """MoGRU for noisy trend tracking (regression)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = MoGRU(input_size=1, hidden_size=hidden_size, num_layers=1)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _, betas = self.rnn(x)
        out = self.head(output)
        return out, betas


class GRUTrendBaseline(nn.Module):
    """Vanilla GRU for noisy trend tracking."""

    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = nn.GRU(1, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.rnn(x)
        out = self.head(output)
        return out, None


# ===========================================================================
# Training / evaluation
# ===========================================================================

def train_epoch(model, loader, optimizer, device, task_type):
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    total_count = 0

    for batch in loader:
        batch = [b.to(device) for b in batch]

        if task_type in ("copy", "selective_copy"):
            if task_type == "selective_copy":
                tokens, markers, target, mask = batch
                logits, _ = model(tokens, markers)
            else:
                tokens, target, mask = batch
                logits, _ = model(tokens)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target.view(-1), reduction="none"
            )
            loss = (loss * mask.view(-1)).sum() / mask.sum()

            preds = logits.argmax(dim=-1)
            correct = ((preds == target) * mask).sum().item()
            total_metric += correct
            total_count += mask.sum().item()

        elif task_type == "adding":
            input_seq, target = batch
            pred, _ = model(input_seq)
            loss = F.mse_loss(pred, target)
            total_metric += loss.item() * input_seq.size(0)
            total_count += input_seq.size(0)

        elif task_type == "trend":
            input_seq, target = batch
            pred, _ = model(input_seq)
            loss = F.mse_loss(pred, target)
            total_metric += loss.item() * input_seq.size(0)
            total_count += input_seq.size(0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    if task_type in ("copy", "selective_copy"):
        metric = total_metric / total_count  # accuracy
    else:
        metric = total_metric / total_count  # MSE
    return avg_loss, metric


@torch.no_grad()
def evaluate(model, loader, device, task_type):
    model.eval()
    total_metric = 0.0
    total_count = 0
    beta_values = []

    for batch in loader:
        batch = [b.to(device) for b in batch]

        if task_type in ("copy", "selective_copy"):
            if task_type == "selective_copy":
                tokens, markers, target, mask = batch
                logits, betas = model(tokens, markers)
            else:
                tokens, target, mask = batch
                logits, betas = model(tokens)

            preds = logits.argmax(dim=-1)
            correct = ((preds == target) * mask).sum().item()
            total_metric += correct
            total_count += mask.sum().item()

        elif task_type == "adding":
            input_seq, target = batch
            pred, betas = model(input_seq)
            mse = F.mse_loss(pred, target, reduction="sum").item()
            total_metric += mse
            total_count += input_seq.size(0)

        elif task_type == "trend":
            input_seq, target = batch
            pred, betas = model(input_seq)
            mse = F.mse_loss(pred, target, reduction="sum").item()
            total_metric += mse
            total_count += input_seq.size(0)

        if betas is not None:
            beta_values.append(betas.mean(dim=-1))  # average across layers

    metric = total_metric / total_count

    beta_info = None
    if beta_values:
        all_betas = torch.cat(beta_values, dim=0)
        beta_info = {
            "mean": all_betas.mean().item(),
            "std": all_betas.std().item(),
        }

    return metric, beta_info


# ===========================================================================
# Task runners
# ===========================================================================

def build_models(task, vocab_size, hidden_size):
    """Build all 4 models for a given task. Returns dict of {name: model}."""
    if task == "copy":
        models = {
            "MoGRU":  MoGRUCopyModel(vocab_size, hidden_size, task="copy"),
            "GRU":    GRUCopyBaseline(vocab_size, hidden_size, task="copy"),
            "LSTM":   LSTMCopyBaseline(vocab_size, hidden_size, task="copy"),
            "MomGRU": MomGRUCopyBaseline(vocab_size, hidden_size, task="copy"),
        }
    elif task == "selective_copy":
        models = {
            "MoGRU":  MoGRUCopyModel(vocab_size, hidden_size, task="selective_copy"),
            "GRU":    GRUCopyBaseline(vocab_size, hidden_size, task="selective_copy"),
            "LSTM":   LSTMCopyBaseline(vocab_size, hidden_size, task="selective_copy"),
            "MomGRU": MomGRUCopyBaseline(vocab_size, hidden_size, task="selective_copy"),
        }
    elif task == "adding":
        models = {
            "MoGRU":  MoGRUAddingModel(hidden_size),
            "GRU":    GRUAddingBaseline(hidden_size),
            "LSTM":   LSTMAddingBaseline(hidden_size),
            "MomGRU": MomGRUAddingBaseline(hidden_size),
        }
    elif task == "trend":
        models = {
            "MoGRU":  MoGRUTrendModel(hidden_size),
            "GRU":    GRUTrendBaseline(hidden_size),
            "LSTM":   LSTMTrendBaseline(hidden_size),
            "MomGRU": MomGRUTrendBaseline(hidden_size),
        }
    else:
        raise ValueError(f"Unknown task: {task}")
    return models


def build_loaders(task, seq_len, batch_size, vocab_size=16, num_markers=5,
                  train_samples=10_000, val_samples=2_000):
    if task == "copy":
        train_ds = CopyDataset(train_samples, seq_len=seq_len, delay=seq_len, vocab_size=vocab_size)
        val_ds = CopyDataset(val_samples, seq_len=seq_len, delay=seq_len, vocab_size=vocab_size)
    elif task == "selective_copy":
        train_ds = SelectiveCopyDataset(train_samples, seq_len, num_markers, vocab_size)
        val_ds = SelectiveCopyDataset(val_samples, seq_len, num_markers, vocab_size)
    elif task == "adding":
        train_ds = AddingDataset(train_samples, seq_len)
        val_ds = AddingDataset(val_samples, seq_len)
    elif task == "trend":
        train_ds = NoisyTrendDataset(train_samples, seq_len)
        val_ds = NoisyTrendDataset(val_samples, seq_len)
    else:
        raise ValueError(f"Unknown task: {task}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(
    task: str = "selective_copy",
    seq_len: int = 50,
    num_markers: int = 5,
    vocab_size: int = 16,
    hidden_size: int = 128,
    batch_size: int = 64,
    num_epochs: int = 30,
    lr: float = 1e-3,
    device_str: str = "cpu",
    seed: int = 42,
    train_samples: int = 10_000,
    val_samples: int = 2_000,
):
    set_seed(seed)
    device = torch.device(device_str)
    task_type = "adding" if task == "adding" else ("trend" if task == "trend" else task)
    metric_name = "acc" if task in ("copy", "selective_copy") else "mse"

    print(f"Device: {device} | Seed: {seed}")
    print(f"Task: {task} | seq_len={seq_len} | train={train_samples} val={val_samples}")
    print("-" * 70)

    train_loader, val_loader = build_loaders(
        task, seq_len, batch_size, vocab_size, num_markers,
        train_samples, val_samples,
    )
    model_dict = build_models(task, vocab_size, hidden_size)

    param_counts = {}
    for name, model in model_dict.items():
        model_dict[name] = model.to(device)
        param_counts[name] = count_parameters(model)
        print(f"{name:6s} params: {param_counts[name]:,}")
    print("-" * 70)

    models = {
        name: (model, torch.optim.Adam(model.parameters(), lr=lr))
        for name, model in model_dict.items()
    }

    epoch_log = []
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        epoch_entry = {"epoch": epoch}

        for name, (model, optimizer) in models.items():
            t0 = time.time()
            train_loss, train_metric = train_epoch(model, train_loader, optimizer, device, task_type)
            val_metric, beta_info = evaluate(model, val_loader, device, task_type)
            elapsed = time.time() - t0

            entry = {
                "loss": round(train_loss, 6),
                f"train_{metric_name}": round(train_metric, 6),
                f"val_{metric_name}": round(val_metric, 6),
                "time": round(elapsed, 2),
            }
            if beta_info:
                entry["beta_mean"] = round(beta_info["mean"], 4)
                entry["beta_std"] = round(beta_info["std"], 4)
            epoch_entry[name] = entry

            line = f"  {name:5s} | loss={train_loss:.4f} train_{metric_name}={train_metric:.4f} val_{metric_name}={val_metric:.4f} | {elapsed:.1f}s"
            if beta_info:
                line += f" | beta: mean={beta_info['mean']:.3f} std={beta_info['std']:.3f}"
            print(line)

        epoch_log.append(epoch_entry)

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"DONE. Total time: {total_time:.1f}s")

    results = {
        "experiment": {
            "arch": "mogru",
            "task": task,
            "seed": seed,
            "seq_len": seq_len,
            "markers": num_markers,
            "vocab": vocab_size,
            "hidden": hidden_size,
            "batch": batch_size,
            "epochs": num_epochs,
            "lr": lr,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "device": device_str,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": round(total_time, 2),
        },
        "params": param_counts,
        "epochs": epoch_log,
        "final": {name: epoch_log[-1][name] for name in model_dict},
    }
    return results


def compute_param_matched_hidden(target_params, model_type, task, vocab_size=16):
    """Find hidden_size that gives approximately target_params for a model type."""
    for h in range(8, 512):
        if task in ("copy", "selective_copy"):
            if model_type == "GRU":
                m = GRUCopyBaseline(vocab_size, h, task=task)
            elif model_type == "LSTM":
                m = LSTMCopyBaseline(vocab_size, h, task=task)
            else:
                continue
        elif task == "adding":
            if model_type == "GRU":
                m = GRUAddingBaseline(h)
            elif model_type == "LSTM":
                m = LSTMAddingBaseline(h)
            else:
                continue
        elif task == "trend":
            if model_type == "GRU":
                m = GRUTrendBaseline(h)
            elif model_type == "LSTM":
                m = LSTMTrendBaseline(h)
            else:
                continue
        else:
            continue
        if count_parameters(m) >= target_params:
            return h
    return 512


def run_multi_seed(
    tasks,
    seeds,
    seq_lens=None,
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
    """Run experiments across multiple tasks and seeds, aggregate results."""
    import os
    import numpy as np
    from scipy import stats as scipy_stats

    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    DEFAULT_SEQ_LENS = {
        "copy": 50, "adding": 100, "trend": 80, "selective_copy": 50,
    }
    if seq_lens is None:
        seq_lens = {}

    all_results = {}

    for task in tasks:
        task_seq_len = seq_lens.get(task, DEFAULT_SEQ_LENS.get(task, 50))
        metric_name = "acc" if task in ("copy", "selective_copy") else "mse"
        print(f"\n{'='*70}")
        print(f"TASK: {task} (seq_len={task_seq_len})")
        print(f"{'='*70}")

        seed_results = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            result = run_experiment(
                task=task, seq_len=task_seq_len, num_markers=num_markers,
                vocab_size=vocab_size, hidden_size=hidden_size,
                batch_size=batch_size, num_epochs=num_epochs, lr=lr,
                device_str=device_str, seed=seed,
                train_samples=train_samples, val_samples=val_samples,
            )
            seed_results.append(result)

            fname = f"{task}_seed{seed}.json"
            with open(os.path.join(results_dir, fname), "w") as f:
                json.dump(result, f, indent=2)

        model_names = list(seed_results[0]["final"].keys())
        val_key = f"val_{metric_name}"

        aggregated = {}
        for name in model_names:
            vals = [r["final"][name][val_key] for r in seed_results]
            aggregated[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": vals,
                "params": seed_results[0]["params"][name],
            }

        # Statistical tests: MoGRU vs each baseline
        stat_tests = {}
        mogru_vals = [r["final"]["MoGRU"][val_key] for r in seed_results]
        for name in model_names:
            if name == "MoGRU":
                continue
            other_vals = [r["final"][name][val_key] for r in seed_results]
            if len(seeds) >= 3:
                t_stat, p_val = scipy_stats.ttest_rel(mogru_vals, other_vals)
                stat_tests[f"MoGRU_vs_{name}"] = {
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                    "significant_005": bool(p_val < 0.05),
                }

        task_summary = {
            "task": task,
            "seq_len": task_seq_len,
            "seeds": seeds,
            "hidden_size": hidden_size,
            "num_epochs": num_epochs,
            "metric": metric_name,
            "results": aggregated,
            "stat_tests": stat_tests,
        }

        all_results[task] = task_summary

        # Print summary
        print(f"\n{'─'*70}")
        print(f"TASK SUMMARY: {task}")
        print(f"{'─'*70}")
        for name, agg in aggregated.items():
            print(f"  {name:6s}: {val_key}={agg['mean']:.4f} ± {agg['std']:.4f}  ({agg['params']:,} params)")
        for test_name, test_result in stat_tests.items():
            sig = "*" if test_result["significant_005"] else ""
            print(f"  {test_name}: t={test_result['t_stat']:.3f}, p={test_result['p_value']:.4f} {sig}")

    # Save aggregated results
    summary_path = os.path.join(results_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAggregated results saved to {summary_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoGRU Benchmark Suite")
    parser.add_argument("--task", type=str, default="selective_copy",
                        choices=["copy", "adding", "trend", "selective_copy"])
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--markers", type=int, default=5)
    parser.add_argument("--vocab", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=10_000)
    parser.add_argument("--val-samples", type=int, default=2_000)
    parser.add_argument("--save-json", type=str, default=None,
                        help="Path to save results as JSON")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed run (e.g. '42,123,456,789,1337')")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated tasks for multi-seed run (e.g. 'copy,adding,trend')")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory for multi-seed results")
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else [args.task]
        run_multi_seed(
            tasks=tasks,
            seeds=seeds,
            hidden_size=args.hidden,
            vocab_size=args.vocab,
            num_markers=args.markers,
            batch_size=args.batch,
            num_epochs=args.epochs,
            lr=args.lr,
            device_str=args.device,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            results_dir=args.results_dir,
        )
    else:
        results = run_experiment(
            task=args.task,
            seq_len=args.seq_len,
            num_markers=args.markers,
            vocab_size=args.vocab,
            hidden_size=args.hidden,
            batch_size=args.batch,
            num_epochs=args.epochs,
            lr=args.lr,
            device_str=args.device,
            seed=args.seed,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
        )

        if args.save_json:
            with open(args.save_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.save_json}")
