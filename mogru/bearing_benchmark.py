"""
CWRU Bearing Fault Detection Benchmark: MoGRU vs GRU vs LSTM

Real-world test of MoGRU's interference resistance: detect weak periodic
fault impulses buried in broadband vibration noise.

Dataset: Case Western Reserve University Bearing Data Center
  - Drive-end accelerometer, 12kHz, 0 HP motor load
  - 4 classes: Normal, Inner Race, Ball, Outer Race fault
  - Fault sizes: 0.007", 0.014", 0.021" (we use all for training)

The task: classify fixed-length vibration windows into fault type.
Momentum should help by smoothing through broadband noise while
preserving the periodic fault signature.
"""

import sys
import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mogru.mogru import MoGRU, count_parameters


DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cwru")

# Files for 0 HP load (1797 RPM), drive-end 12kHz
FILE_MAP = {
    "normal": ["1797_Normal.npz"],
    "inner_race": [
        "1797_IR_7_DE12.npz",
        "1797_IR_14_DE12.npz",
        "1797_IR_21_DE12.npz",
    ],
    "ball": [
        "1797_B_7_DE12.npz",
        "1797_B_14_DE12.npz",
        "1797_B_21_DE12.npz",
    ],
    "outer_race": [
        "1797_OR@6_7_DE12.npz",
        "1797_OR@6_14_DE12.npz",
        "1797_OR@6_21_DE12.npz",
    ],
}

CLASS_NAMES = ["normal", "inner_race", "ball", "outer_race"]


def load_cwru_data(window_size=1024, stride=512, noise_std=0.0):
    """Load CWRU data and window into fixed-length segments."""
    all_windows = []
    all_labels = []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        for fname in FILE_MAP[class_name]:
            fpath = os.path.join(DATA_DIR, fname)
            data = np.load(fpath)
            signal = data["DE"].flatten()

            # Normalize per-file
            signal = (signal - signal.mean()) / (signal.std() + 1e-8)

            # Window with stride
            n_windows = (len(signal) - window_size) // stride + 1
            for i in range(n_windows):
                start = i * stride
                window = signal[start:start + window_size]
                if noise_std > 0:
                    window = window + np.random.randn(window_size) * noise_std
                all_windows.append(window)
                all_labels.append(label_idx)

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    return X, y


def make_loaders(window_size=1024, stride=512, noise_std=0.0,
                 train_ratio=0.8, batch_size=64, seed=42):
    """Create train/val DataLoaders."""
    np.random.seed(seed)
    X, y = load_cwru_data(window_size, stride, noise_std)

    # Train/val split
    n_train = int(len(X) * train_ratio)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    # Reshape for RNN: (B, T, 1)
    X_train = torch.tensor(X_train).unsqueeze(-1)
    X_val = torch.tensor(X_val).unsqueeze(-1)
    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"Data: {len(X_train)} train, {len(X_val)} val, "
          f"{len(CLASS_NAMES)} classes, window={window_size}")
    print(f"Class distribution (train): {np.bincount(y_train.numpy())}")

    return train_loader, val_loader


# ===========================================================================
# Models
# ===========================================================================

class BearingClassifier(nn.Module):
    """Vibration sequence -> fault class."""

    def __init__(self, rnn_type="gru", hidden_size=64, num_classes=4):
        super().__init__()
        self.rnn_type = rnn_type

        if rnn_type == "gru":
            self.rnn = nn.GRU(1, hidden_size, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(1, hidden_size, batch_first=True)
        elif rnn_type == "mogru":
            self.rnn = MoGRU(1, hidden_size, num_layers=1)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if self.rnn_type == "mogru":
            out, _, betas = self.rnn(x)
            beta_info = {"mean": betas.mean().item(), "std": betas.std().item()}
        else:
            out, _ = self.rnn(x)
            beta_info = None

        # Use last hidden state for classification
        last = out[:, -1, :]
        logits = self.head(last)
        return logits, beta_info


# ===========================================================================
# Training
# ===========================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    beta_info = None
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, bi = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            if bi is not None:
                beta_info = bi
    return correct / total, beta_info


def run_experiment(window_size=1024, hidden_size=64, epochs=20,
                   batch_size=64, lr=1e-3, noise_std=0.0, seed=42):
    """Run MoGRU vs GRU vs LSTM on bearing fault detection."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    print(f"\nCWRU Bearing Fault Detection")
    print(f"window={window_size}, hidden={hidden_size}, epochs={epochs}, "
          f"noise_std={noise_std}, seed={seed}")
    print("=" * 70)

    train_loader, val_loader = make_loaders(
        window_size=window_size, stride=window_size // 2,
        noise_std=noise_std, batch_size=batch_size, seed=seed,
    )

    results = {}
    for rnn_type in ["gru", "lstm", "mogru"]:
        torch.manual_seed(seed)
        model = BearingClassifier(rnn_type, hidden_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        params = count_parameters(model)

        print(f"\n--- {rnn_type.upper()} ({params:,} params) ---")

        best_val_acc = 0.0
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            val_acc, beta_info = evaluate(model, val_loader, device)
            best_val_acc = max(best_val_acc, val_acc)

            if epoch % 5 == 0 or epoch == 1:
                line = (f"  epoch {epoch:2d} | loss={train_loss:.4f} "
                        f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
                if beta_info:
                    line += f" | beta: mean={beta_info['mean']:.3f} std={beta_info['std']:.3f}"
                print(line)

        elapsed = time.time() - t0
        results[rnn_type] = best_val_acc
        print(f"  best val_acc: {best_val_acc:.3f} ({elapsed:.0f}s)")

    # Summary
    print("\n" + "=" * 70)
    print("BEARING FAULT DETECTION SUMMARY")
    print("-" * 40)
    for rnn_type, acc in results.items():
        print(f"  {rnn_type:6s}: val_acc={acc:.4f}")
    winner = max(results, key=results.get)
    print(f"\nWinner: {winner.upper()}")
    return results


def run_noise_sweep(window_size=1024, hidden_size=64, epochs=15,
                    batch_size=64, lr=1e-3, seed=42):
    """Sweep added noise levels to test interference resistance."""
    print("\n" + "=" * 70)
    print("NOISE INJECTION SWEEP (testing interference resistance)")
    print("=" * 70)

    noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    all_results = {}

    for noise_std in noise_levels:
        print(f"\n{'='*50}")
        print(f"Noise std = {noise_std}")
        print(f"{'='*50}")

        results = {}
        for rnn_type in ["gru", "lstm", "mogru"]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            train_loader, val_loader = make_loaders(
                window_size=window_size, stride=window_size // 2,
                noise_std=noise_std, batch_size=batch_size, seed=seed,
            )

            model = BearingClassifier(rnn_type, hidden_size).to(torch.device("cpu"))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            best_acc = 0.0
            for epoch in range(1, epochs + 1):
                train_epoch(model, train_loader, optimizer, torch.device("cpu"))
                val_acc, _ = evaluate(model, val_loader, torch.device("cpu"))
                best_acc = max(best_acc, val_acc)

            results[rnn_type] = best_acc
            print(f"  {rnn_type:6s}: best_acc={best_acc:.4f}")

        all_results[noise_std] = results

    # Summary table
    print("\n" + "=" * 70)
    print("NOISE SWEEP SUMMARY")
    print(f"{'noise':>8} {'GRU':>8} {'LSTM':>8} {'MoGRU':>8} {'Winner':>8}")
    print("-" * 45)
    for noise_std in noise_levels:
        r = all_results[noise_std]
        winner = max(r, key=r.get)
        print(f"{noise_std:>8.1f} {r['gru']:>8.4f} {r['lstm']:>8.4f} "
              f"{r['mogru']:>8.4f} {winner:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["benchmark", "noise_sweep", "both"],
                        default="both")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode in ("benchmark", "both"):
        run_experiment(
            window_size=args.window, hidden_size=args.hidden,
            epochs=args.epochs, batch_size=args.batch,
            lr=args.lr, seed=args.seed,
        )

    if args.mode in ("noise_sweep", "both"):
        run_noise_sweep(
            window_size=args.window, hidden_size=args.hidden,
            epochs=min(args.epochs, 15), batch_size=args.batch,
            lr=args.lr, seed=args.seed,
        )
