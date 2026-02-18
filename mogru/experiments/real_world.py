"""
Real-World Experiments for MoGRU Gate 3

Three datasets:
  1. ETTh1 — Univariate time-series forecasting (predict 96 steps from 96)
  2. ECG5000 — 5-class ECG classification (seq_len=140)
  3. sMNIST — Sequential MNIST (pixel-by-pixel, 784 steps)

All models: MoGRU, GRU, LSTM, SimpleSSM
All experiments: multi-seed, JSON output

Usage:
  python -m mogru.experiments.real_world --experiment etth1
  python -m mogru.experiments.real_world --experiment ecg5000
  python -m mogru.experiments.real_world --experiment smnist
  python -m mogru.experiments.real_world --experiment all
"""

import os
import json
import time
import argparse
import urllib.request
import zipfile
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

from mogru.mogru import MoGRU, count_parameters
from mogru.benchmark import set_seed
from mogru.head_to_head import SSMModel


# ===========================================================================
# Data directory
# ===========================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# ETTh1 Dataset
# ===========================================================================

def download_etth1():
    """Download ETTh1.csv from ETDataset GitHub repo."""
    path = os.path.join(DATA_DIR, "ETTh1.csv")
    if os.path.exists(path):
        return path
    ensure_dirs()
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    print(f"Downloading ETTh1 from {url}...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved to {path}")
    return path


class ETTh1Dataset(Dataset):
    """ETTh1 univariate forecasting: predict next pred_len from lookback window."""

    def __init__(self, data, lookback=96, pred_len=96):
        self.data = data
        self.lookback = lookback
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.lookback - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.lookback]
        y = self.data[idx + self.lookback:idx + self.lookback + self.pred_len]
        return x.unsqueeze(-1), y


def load_etth1_splits(lookback=96, pred_len=96):
    """Load ETTh1 and split into train/val/test (60/20/20)."""
    import csv

    path = download_etth1()
    values = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row["OT"]))

    data = torch.tensor(values, dtype=torch.float32)

    # Normalize
    mean = data[:int(len(data) * 0.6)].mean()
    std = data[:int(len(data) * 0.6)].std()
    data = (data - mean) / (std + 1e-8)

    n = len(data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_ds = ETTh1Dataset(data[:train_end], lookback, pred_len)
    val_ds = ETTh1Dataset(data[train_end:val_end], lookback, pred_len)
    test_ds = ETTh1Dataset(data[val_end:], lookback, pred_len)

    return train_ds, val_ds, test_ds


# ===========================================================================
# ECG5000 Dataset
# ===========================================================================

def download_ecg5000():
    """Download ECG5000 from UCR archive."""
    train_path = os.path.join(DATA_DIR, "ECG5000_TRAIN.tsv")
    test_path = os.path.join(DATA_DIR, "ECG5000_TEST.tsv")
    if os.path.exists(train_path) and os.path.exists(test_path):
        return train_path, test_path
    ensure_dirs()

    url = "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"
    print(f"Downloading ECG5000 from {url}...")
    response = urllib.request.urlopen(url)
    zip_data = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        for name in zf.namelist():
            basename = os.path.basename(name)
            if basename == "ECG5000_TRAIN.tsv":
                with open(train_path, "wb") as f:
                    f.write(zf.read(name))
            elif basename == "ECG5000_TEST.tsv":
                with open(test_path, "wb") as f:
                    f.write(zf.read(name))

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "ECG5000 TSV files not found in zip. "
            "Please download manually from timeseriesclassification.com"
        )

    print(f"Saved to {DATA_DIR}")
    return train_path, test_path


def load_ecg5000_splits():
    """Load ECG5000 and return train/val/test datasets."""
    train_path, test_path = download_ecg5000()

    train_data = np.loadtxt(train_path)
    test_data = np.loadtxt(test_path)

    # First column is label (1-5), rest is time series
    train_labels = train_data[:, 0].astype(int) - 1  # 0-indexed
    train_series = train_data[:, 1:]
    test_labels = test_data[:, 0].astype(int) - 1
    test_series = test_data[:, 1:]

    # Normalize using train stats
    mean = train_series.mean()
    std = train_series.std()
    train_series = (train_series - mean) / (std + 1e-8)
    test_series = (test_series - mean) / (std + 1e-8)

    # Split test into val/test (50/50)
    n_test = len(test_labels)
    val_idx = n_test // 2

    train_ds = TensorDataset(
        torch.tensor(train_series, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(train_labels, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(test_series[:val_idx], dtype=torch.float32).unsqueeze(-1),
        torch.tensor(test_labels[:val_idx], dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(test_series[val_idx:], dtype=torch.float32).unsqueeze(-1),
        torch.tensor(test_labels[val_idx:], dtype=torch.long),
    )

    return train_ds, val_ds, test_ds


# ===========================================================================
# Sequential MNIST Dataset
# ===========================================================================

def download_mnist():
    """Download MNIST using torchvision or manual fallback."""
    mnist_dir = os.path.join(DATA_DIR, "MNIST")

    try:
        from torchvision import datasets
        datasets.MNIST(DATA_DIR, train=True, download=True)
        datasets.MNIST(DATA_DIR, train=False, download=True)
        return mnist_dir
    except ImportError:
        pass

    # Manual download fallback
    os.makedirs(mnist_dir, exist_ok=True)
    import gzip
    import struct

    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        "train-images-idx3-ubyte.gz": "train_images",
        "train-labels-idx1-ubyte.gz": "train_labels",
        "t10k-images-idx3-ubyte.gz": "test_images",
        "t10k-labels-idx1-ubyte.gz": "test_labels",
    }

    for fname in files:
        fpath = os.path.join(mnist_dir, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, fpath)

    return mnist_dir


def _read_mnist_images(path):
    """Read MNIST image file (IDX format)."""
    import gzip
    import struct

    with gzip.open(path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols)


def _read_mnist_labels(path):
    """Read MNIST label file (IDX format)."""
    import gzip
    import struct

    with gzip.open(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_smnist_splits():
    """Load MNIST as sequential task (784 steps of 1 pixel)."""
    ensure_dirs()

    try:
        from torchvision import datasets
        train_full = datasets.MNIST(DATA_DIR, train=True, download=True)
        test_full = datasets.MNIST(DATA_DIR, train=False, download=True)
        train_images = train_full.data.numpy().reshape(-1, 784)
        train_labels = train_full.targets.numpy()
        test_images = test_full.data.numpy().reshape(-1, 784)
        test_labels = test_full.targets.numpy()
    except ImportError:
        mnist_dir = download_mnist()
        train_images = _read_mnist_images(os.path.join(mnist_dir, "train-images-idx3-ubyte.gz"))
        train_labels = _read_mnist_labels(os.path.join(mnist_dir, "train-labels-idx1-ubyte.gz"))
        test_images = _read_mnist_images(os.path.join(mnist_dir, "t10k-images-idx3-ubyte.gz"))
        test_labels = _read_mnist_labels(os.path.join(mnist_dir, "t10k-labels-idx1-ubyte.gz"))

    # Normalize and reshape to (N, 784, 1)
    train_x = torch.tensor(train_images, dtype=torch.float32) / 255.0
    train_x = (train_x - 0.1307) / 0.3081
    train_x = train_x.unsqueeze(-1)
    train_y = torch.tensor(train_labels, dtype=torch.long)

    test_x = torch.tensor(test_images, dtype=torch.float32) / 255.0
    test_x = (test_x - 0.1307) / 0.3081
    test_x = test_x.unsqueeze(-1)
    test_y = torch.tensor(test_labels, dtype=torch.long)

    # Split train into train/val (55000/5000)
    val_x = train_x[55000:]
    val_y = train_y[55000:]
    train_x = train_x[:55000]
    train_y = train_y[:55000]

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    test_ds = TensorDataset(test_x, test_y)

    return train_ds, val_ds, test_ds


# ===========================================================================
# Model builders
# ===========================================================================

class ForecastModel(nn.Module):
    """Sequence-to-sequence forecasting model."""

    def __init__(self, rnn_type, hidden_size, pred_len, input_size=1):
        super().__init__()
        self.rnn_type = rnn_type
        self.pred_len = pred_len

        if rnn_type == "MoGRU":
            self.rnn = MoGRU(input_size, hidden_size)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.head = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        if self.rnn_type == "MoGRU":
            output, _, _ = self.rnn(x)
        else:
            output, _ = self.rnn(x)
        return self.head(output[:, -1])


class ClassificationModel(nn.Module):
    """Sequence classification model (last hidden → logits)."""

    def __init__(self, rnn_type, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn_type = rnn_type

        if rnn_type == "MoGRU":
            self.rnn = MoGRU(input_size, hidden_size)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if self.rnn_type == "MoGRU":
            output, _, _ = self.rnn(x)
        else:
            output, _ = self.rnn(x)
        return self.head(output[:, -1])


class SSMForecastModel(nn.Module):
    """SSM-based forecasting model."""

    def __init__(self, hidden_size, pred_len, input_size=1):
        super().__init__()
        self.ssm = SSMModel(input_size, hidden_size, hidden_size, num_layers=2)
        self.head = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out = self.ssm(x)
        return self.head(out[:, -1])


class SSMClassificationModel(nn.Module):
    """SSM-based classification model."""

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.ssm = SSMModel(input_size, hidden_size, hidden_size, num_layers=2)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.ssm(x)
        return self.head(out[:, -1])


def build_forecast_models(hidden_size, pred_len, input_size=1):
    return {
        "MoGRU": ForecastModel("MoGRU", hidden_size, pred_len, input_size),
        "GRU":   ForecastModel("GRU", hidden_size, pred_len, input_size),
        "LSTM":  ForecastModel("LSTM", hidden_size, pred_len, input_size),
        "SSM":   SSMForecastModel(hidden_size, pred_len, input_size),
    }


def build_classification_models(input_size, hidden_size, num_classes):
    return {
        "MoGRU": ClassificationModel("MoGRU", input_size, hidden_size, num_classes),
        "GRU":   ClassificationModel("GRU", input_size, hidden_size, num_classes),
        "LSTM":  ClassificationModel("LSTM", input_size, hidden_size, num_classes),
        "SSM":   SSMClassificationModel(input_size, hidden_size, num_classes),
    }


# ===========================================================================
# Training loops
# ===========================================================================

def train_forecast(model, train_loader, val_loader, epochs, lr, device):
    """Train a forecasting model, return best val MSE and MAE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_mse = float("inf")
    best_val_mae = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        total_mse, total_mae, count = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                total_mse += F.mse_loss(pred, y, reduction="sum").item()
                total_mae += (pred - y).abs().sum().item()
                count += y.numel()

        val_mse = total_mse / count
        val_mae = total_mae / count
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_mae = val_mae

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d} | val_mse={val_mse:.6f} val_mae={val_mae:.4f}")

    return best_val_mse, best_val_mae


def train_classification(model, train_loader, val_loader, epochs, lr, device):
    """Train a classification model, return best val accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d} | val_acc={val_acc:.4f}")

    return best_val_acc


# ===========================================================================
# Experiment runners
# ===========================================================================

def run_etth1(seeds, hidden_sizes, epochs=50, batch_size=64, lr=1e-3, device_str="cpu"):
    """ETTh1 univariate forecasting experiment."""
    ensure_dirs()
    device = torch.device(device_str)
    pred_len = 96
    lookback = 96

    print("\n" + "=" * 70)
    print("EXPERIMENT: ETTh1 Univariate Forecasting")
    print(f"lookback={lookback}, pred_len={pred_len}, epochs={epochs}")
    print("=" * 70)

    train_ds, val_ds, test_ds = load_etth1_splits(lookback, pred_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    all_results = {}

    for hidden in hidden_sizes:
        print(f"\n--- Hidden={hidden} ---")
        seed_results = {name: [] for name in ["MoGRU", "GRU", "LSTM", "SSM"]}

        for seed in seeds:
            print(f"\n  Seed {seed}:")
            models = build_forecast_models(hidden, pred_len)

            for name, model in models.items():
                set_seed(seed)
                model = model.to(device)
                params = count_parameters(model)
                t0 = time.time()

                print(f"  {name:6s} ({params:,} params)")
                best_mse, best_mae = train_forecast(
                    model, train_loader, val_loader, epochs, lr, device,
                )
                elapsed = time.time() - t0

                seed_results[name].append({
                    "seed": seed,
                    "mse": best_mse,
                    "mae": best_mae,
                    "params": params,
                    "time": round(elapsed, 1),
                })
                print(f"    best: mse={best_mse:.6f} mae={best_mae:.4f} ({elapsed:.1f}s)")

        all_results[f"hidden_{hidden}"] = {
            name: {
                "mean_mse": float(np.mean([r["mse"] for r in runs])),
                "std_mse": float(np.std([r["mse"] for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
                "mean_mae": float(np.mean([r["mae"] for r in runs])),
                "std_mae": float(np.std([r["mae"] for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
                "params": runs[0]["params"],
                "runs": runs,
            }
            for name, runs in seed_results.items()
        }

    result = {
        "experiment": "etth1",
        "lookback": lookback,
        "pred_len": pred_len,
        "epochs": epochs,
        "seeds": seeds,
        "hidden_sizes": hidden_sizes,
        "results": all_results,
    }

    path = os.path.join(RESULTS_DIR, "etth1_results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nETTh1 results saved to {path}")
    return result


def run_ecg5000(seeds, hidden_sizes, epochs=100, batch_size=64, lr=1e-3, device_str="cpu"):
    """ECG5000 classification experiment."""
    ensure_dirs()
    device = torch.device(device_str)

    print("\n" + "=" * 70)
    print("EXPERIMENT: ECG5000 Classification")
    print(f"5 classes, seq_len=140, epochs={epochs}")
    print("=" * 70)

    train_ds, val_ds, test_ds = load_ecg5000_splits()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    all_results = {}

    for hidden in hidden_sizes:
        print(f"\n--- Hidden={hidden} ---")
        seed_results = {name: [] for name in ["MoGRU", "GRU", "LSTM", "SSM"]}

        for seed in seeds:
            print(f"\n  Seed {seed}:")
            models = build_classification_models(1, hidden, 5)

            for name, model in models.items():
                set_seed(seed)
                model = model.to(device)
                params = count_parameters(model)
                t0 = time.time()

                print(f"  {name:6s} ({params:,} params)")
                best_acc = train_classification(
                    model, train_loader, val_loader, epochs, lr, device,
                )
                elapsed = time.time() - t0

                seed_results[name].append({
                    "seed": seed,
                    "accuracy": best_acc,
                    "params": params,
                    "time": round(elapsed, 1),
                })
                print(f"    best: acc={best_acc:.4f} ({elapsed:.1f}s)")

        all_results[f"hidden_{hidden}"] = {
            name: {
                "mean_acc": float(np.mean([r["accuracy"] for r in runs])),
                "std_acc": float(np.std([r["accuracy"] for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
                "params": runs[0]["params"],
                "runs": runs,
            }
            for name, runs in seed_results.items()
        }

    result = {
        "experiment": "ecg5000",
        "num_classes": 5,
        "seq_len": 140,
        "epochs": epochs,
        "seeds": seeds,
        "hidden_sizes": hidden_sizes,
        "results": all_results,
    }

    path = os.path.join(RESULTS_DIR, "ecg5000_results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nECG5000 results saved to {path}")
    return result


def run_smnist(seeds, hidden_sizes, epochs=30, batch_size=128, lr=1e-3, device_str="cpu"):
    """Sequential MNIST experiment (784 steps)."""
    ensure_dirs()
    device = torch.device(device_str)

    print("\n" + "=" * 70)
    print("EXPERIMENT: Sequential MNIST")
    print(f"784 steps, 10 classes, epochs={epochs}")
    print("=" * 70)

    train_ds, val_ds, test_ds = load_smnist_splits()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    all_results = {}

    for hidden in hidden_sizes:
        print(f"\n--- Hidden={hidden} ---")
        seed_results = {name: [] for name in ["MoGRU", "GRU", "LSTM"]}

        for seed in seeds:
            print(f"\n  Seed {seed}:")
            # SSM excluded — 784 sequential steps with per-step scan is too slow
            models = {
                "MoGRU": ClassificationModel("MoGRU", 1, hidden, 10),
                "GRU":   ClassificationModel("GRU", 1, hidden, 10),
                "LSTM":  ClassificationModel("LSTM", 1, hidden, 10),
            }

            for name, model in models.items():
                set_seed(seed)
                model = model.to(device)
                params = count_parameters(model)
                t0 = time.time()

                print(f"  {name:6s} ({params:,} params)")
                best_acc = train_classification(
                    model, train_loader, val_loader, epochs, lr, device,
                )
                elapsed = time.time() - t0

                seed_results[name].append({
                    "seed": seed,
                    "accuracy": best_acc,
                    "params": params,
                    "time": round(elapsed, 1),
                })
                print(f"    best: acc={best_acc:.4f} ({elapsed:.1f}s)")

        all_results[f"hidden_{hidden}"] = {
            name: {
                "mean_acc": float(np.mean([r["accuracy"] for r in runs])),
                "std_acc": float(np.std([r["accuracy"] for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
                "params": runs[0]["params"],
                "runs": runs,
            }
            for name, runs in seed_results.items()
        }

    result = {
        "experiment": "smnist",
        "seq_len": 784,
        "num_classes": 10,
        "epochs": epochs,
        "seeds": seeds,
        "hidden_sizes": hidden_sizes,
        "results": all_results,
    }

    path = os.path.join(RESULTS_DIR, "smnist_results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsMNIST results saved to {path}")
    return result


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoGRU Real-World Experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["etth1", "ecg5000", "smnist", "all"])
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated seeds")
    parser.add_argument("--hidden-sizes", type=str, default="64,128",
                        help="Comma-separated hidden sizes")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override default epochs")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    hidden_sizes = [int(h.strip()) for h in args.hidden_sizes.split(",")]

    if args.experiment in ("etth1", "all"):
        run_etth1(
            seeds=seeds, hidden_sizes=hidden_sizes,
            epochs=args.epochs or 50, batch_size=args.batch,
            lr=args.lr, device_str=args.device,
        )

    if args.experiment in ("ecg5000", "all"):
        run_ecg5000(
            seeds=seeds, hidden_sizes=hidden_sizes,
            epochs=args.epochs or 100, batch_size=args.batch,
            lr=args.lr, device_str=args.device,
        )

    if args.experiment in ("smnist", "all"):
        run_smnist(
            seeds=seeds, hidden_sizes=hidden_sizes,
            epochs=args.epochs or 30, batch_size=128,
            lr=args.lr, device_str=args.device,
        )
