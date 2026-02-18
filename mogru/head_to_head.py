"""
MoGRU vs All: Head-to-Head on MoGRU's Best Tasks

Tests MoGRU against GRU + a simple S4-style SSM baseline on:
  1. Frequency Separation (MoGRU's strongest win: 0.0099 vs 0.0182 GRU)
  2. Interference Resistance (MoGRU's surprise win: 0.457 vs 0.127 GRU)
  3. Trend Reversal (MoGRU's clean win: 0.0047 vs 0.0057 GRU)

The SSM baseline tests whether a linear state-space model (the architecture
class that includes S4/Mamba) already captures MoGRU's gains. If MoGRU
beats SSM here, the momentum mechanism offers something SSMs don't.

Run: python3 head_to_head.py
"""

import sys
import os
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))

from mogru.mogru import MoGRU


# =========================================================================
# Simple S4-style SSM (pure PyTorch, no CUDA kernels)
# =========================================================================

class SimpleSSMCell(nn.Module):
    """
    Minimal diagonal state-space model.
    x_k = A * x_{k-1} + B * u_k
    y_k = C * x_k + D * u_k

    A is diagonal and parameterized as A = exp(-exp(log_A)) for stability.
    This captures the core S4/Mamba inductive bias without CUDA kernels.
    """

    def __init__(self, input_size, state_size, output_size):
        super().__init__()
        self.state_size = state_size

        # Diagonal A (stable by construction)
        self.log_A = nn.Parameter(torch.randn(state_size) * 0.5)
        self.B = nn.Linear(input_size, state_size)
        self.C = nn.Linear(state_size, output_size)
        self.D = nn.Linear(input_size, output_size)

        # Discretization step
        self.log_dt = nn.Parameter(torch.zeros(1) - 1.0)

    def forward(self, u):
        """u: (B, T, input_size) -> y: (B, T, output_size)"""
        B_batch, T, _ = u.shape

        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.log_A)  # negative for stability
        A_bar = torch.exp(A * dt)   # discretized diagonal A

        # Scan
        x = torch.zeros(B_batch, self.state_size, device=u.device)
        outputs = []
        for t in range(T):
            b_input = self.B(u[:, t])
            x = A_bar * x + b_input * dt
            y = self.C(x) + self.D(u[:, t])
            outputs.append(y)

        return torch.stack(outputs, dim=1)


class SSMModel(nn.Module):
    """SSM wrapped with mixing layers for fair comparison."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([
            SimpleSSMCell(hidden_size, hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.input_proj(x)
        for ssm, norm in zip(self.layers, self.norms):
            h = h + ssm(h)  # residual
            h = norm(h)
        out = self.head(h)
        # Squeeze trailing dim if output_size=1 (for continuous tasks)
        if out.size(-1) == 1:
            out = out.squeeze(-1)
        return out


# =========================================================================
# Model builders
# =========================================================================

def build_mogru_continuous(hidden_size):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = MoGRU(1, hidden_size)
            self.head = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _, _ = self.rnn(x)
            return self.head(out).squeeze(-1)
    return M()


def build_gru_continuous(hidden_size):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(1, hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out).squeeze(-1)
    return M()


def build_ssm_continuous(hidden_size):
    return SSMModel(1, hidden_size, 1, num_layers=2)


def build_mogru_discrete(vocab_size, hidden_size):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=0)
            self.rnn = MoGRU(hidden_size, hidden_size)
            self.head = nn.Linear(hidden_size, vocab_size + 1)
        def forward(self, tokens):
            out, _, _ = self.rnn(self.embed(tokens))
            return self.head(out)
    return M()


def build_gru_discrete(vocab_size, hidden_size):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=0)
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, vocab_size + 1)
        def forward(self, tokens):
            out, _ = self.rnn(self.embed(tokens))
            return self.head(out)
    return M()


def build_ssm_discrete(vocab_size, hidden_size):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=0)
            self.ssm = SSMModel(hidden_size, hidden_size, hidden_size, num_layers=2)
            self.head = nn.Linear(hidden_size, vocab_size + 1)
        def forward(self, tokens):
            return self.head(self.ssm(self.embed(tokens)))
    return M()


# =========================================================================
# Datasets (imported from gauntlet concepts, self-contained here)
# =========================================================================

class FrequencySeparationDataset(Dataset):
    def __init__(self, num_samples=4000, seq_len=200, noise_std=0.1):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.noise_std = noise_std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T = self.seq_len
        t = torch.arange(T + 1, dtype=torch.float)
        freq_slow = torch.empty(1).uniform_(0.01, 0.05).item()
        amp_slow = torch.empty(1).uniform_(1.0, 3.0).item()
        phase_slow = torch.empty(1).uniform_(0, 2 * math.pi).item()
        slow = amp_slow * torch.sin(2 * math.pi * freq_slow * t + phase_slow)
        freq_fast = torch.empty(1).uniform_(0.2, 0.5).item()
        amp_fast = torch.empty(1).uniform_(0.3, 1.0).item()
        fast = amp_fast * torch.sin(2 * math.pi * freq_fast * t)
        combined = slow + fast + torch.randn(T + 1) * self.noise_std
        return combined[:T].unsqueeze(-1), slow[1:T + 1]


class InterferenceDataset(Dataset):
    def __init__(self, num_samples=4000, k_items=5, n_distractors=50, vocab_size=32):
        self.num_samples = num_samples
        self.k_items = k_items
        self.n_distractors = n_distractors
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        K, N, V = self.k_items, self.n_distractors, self.vocab_size
        marker_tok, query_tok = V + 1, V + 2
        orig_keys = torch.randint(1, V // 2 + 1, (K,))
        orig_vals = torch.randint(V // 2, V + 1, (K,))
        dist_keys = (orig_keys.repeat(N // K + 1)[:N] + torch.randint(-1, 2, (N,))).clamp(1, V // 2)
        dist_vals = torch.randint(V // 2, V + 1, (N,))
        total_len = K * 3 + N * 2 + K * 2
        tokens = torch.zeros(total_len, dtype=torch.long)
        for i in range(K):
            tokens[i*3], tokens[i*3+1], tokens[i*3+2] = marker_tok, orig_keys[i], orig_vals[i]
        off = K * 3
        for i in range(N):
            tokens[off+i*2], tokens[off+i*2+1] = dist_keys[i], dist_vals[i]
        off = K * 3 + N * 2
        target = torch.zeros(total_len, dtype=torch.long)
        mask = torch.zeros(total_len, dtype=torch.float)
        for i in range(K):
            tokens[off+i*2] = query_tok
            tokens[off+i*2+1] = orig_keys[i]
            target[off+i*2+1] = orig_vals[i]
            mask[off+i*2+1] = 1.0
        return tokens, target, mask


class TrendReversalDataset(Dataset):
    def __init__(self, num_samples=4000, seq_len=100, noise_std=0.1):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.noise_std = noise_std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T = self.seq_len
        slope = torch.empty(1).uniform_(0.02, 0.1).item()
        t_rev = torch.randint(T // 4, 3 * T // 4, (1,)).item()
        clean = torch.zeros(T + 1)
        for i in range(T + 1):
            clean[i] = slope * i if i <= t_rev else slope * t_rev - slope * (i - t_rev)
        noisy = clean + torch.randn(T + 1) * self.noise_std
        return noisy[:T].unsqueeze(-1), clean[1:T + 1]


# =========================================================================
# Training
# =========================================================================

def train_and_eval_continuous(model, train_ds, val_ds, epochs, lr, batch_size, device):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for inp, target in train_loader:
            inp, target = inp.to(device), target.to(device)
            pred = model(inp)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    total_mse, count = 0.0, 0
    with torch.no_grad():
        for inp, target in val_loader:
            inp, target = inp.to(device), target.to(device)
            pred = model(inp)
            total_mse += F.mse_loss(pred, target, reduction="sum").item()
            count += target.numel()
    return total_mse / count


def train_and_eval_discrete(model, train_ds, val_ds, epochs, lr, batch_size, device):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for tokens, target, mask in train_loader:
            tokens, target, mask = tokens.to(device), target.to(device), mask.to(device)
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction="none")
            loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for tokens, target, mask in val_loader:
            tokens, target, mask = tokens.to(device), target.to(device), mask.to(device)
            logits = model(tokens)
            preds = logits.argmax(dim=-1)
            correct += ((preds == target) * mask).sum().item()
            total += mask.sum().item()
    return correct / max(total, 1)


# =========================================================================
# Main
# =========================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run():
    HIDDEN = 96
    BATCH = 64
    LR = 1e-3
    EPOCHS = 25  # more than gauntlet to let models converge
    DEVICE = torch.device("cpu")
    VOCAB = 37  # 32 + special tokens headroom

    print("=" * 65)
    print("  MoGRU vs GRU vs SSM — Head-to-Head")
    print(f"  Hidden: {HIDDEN}, Epochs: {EPOCHS}, Device: {DEVICE}")
    print("=" * 65)

    tasks = [
        {
            "name": "Frequency Separation",
            "type": "continuous",
            "train_ds": FrequencySeparationDataset(4000, 200),
            "val_ds": FrequencySeparationDataset(1000, 200),
            "builders": {
                "MoGRU": lambda: build_mogru_continuous(HIDDEN),
                "GRU":   lambda: build_gru_continuous(HIDDEN),
                "SSM":   lambda: build_ssm_continuous(HIDDEN),
            },
            "metric": "MSE",
            "lower_better": True,
        },
        {
            "name": "Trend Reversal",
            "type": "continuous",
            "train_ds": TrendReversalDataset(4000, 100),
            "val_ds": TrendReversalDataset(1000, 100),
            "builders": {
                "MoGRU": lambda: build_mogru_continuous(HIDDEN),
                "GRU":   lambda: build_gru_continuous(HIDDEN),
                "SSM":   lambda: build_ssm_continuous(HIDDEN),
            },
            "metric": "MSE",
            "lower_better": True,
        },
        {
            "name": "Interference Resistance",
            "type": "discrete",
            "train_ds": InterferenceDataset(4000, 5, 50, 32),
            "val_ds": InterferenceDataset(1000, 5, 50, 32),
            "builders": {
                "MoGRU": lambda: build_mogru_discrete(VOCAB, HIDDEN),
                "GRU":   lambda: build_gru_discrete(VOCAB, HIDDEN),
                "SSM":   lambda: build_ssm_discrete(VOCAB, HIDDEN),
            },
            "metric": "accuracy",
            "lower_better": False,
        },
    ]

    summary = []

    for task in tasks:
        print(f"\n{'─'*65}")
        print(f"TASK: {task['name']} ({task['metric']})")
        print(f"{'─'*65}")

        task_results = {}
        for model_name, builder in task["builders"].items():
            model = builder().to(DEVICE)
            params = count_params(model)
            t0 = time.time()

            if task["type"] == "continuous":
                score = train_and_eval_continuous(
                    model, task["train_ds"], task["val_ds"], EPOCHS, LR, BATCH, DEVICE
                )
            else:
                score = train_and_eval_discrete(
                    model, task["train_ds"], task["val_ds"], EPOCHS, LR, BATCH, DEVICE
                )

            elapsed = time.time() - t0
            task_results[model_name] = score
            arrow = "↓" if task["lower_better"] else "↑"
            print(f"  {model_name:6s} | {task['metric']}={score:.5f} ({arrow} better) | {params:,} params | {elapsed:.1f}s")

        if task["lower_better"]:
            winner = min(task_results, key=task_results.get)
        else:
            winner = max(task_results, key=task_results.get)

        print(f"  Winner: {winner}")
        summary.append((task["name"], task_results, winner))

    # Final summary
    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    print(f"{'Task':<30} {'MoGRU':>10} {'GRU':>10} {'SSM':>10}  Winner")
    print("-" * 75)
    for name, results, winner in summary:
        vals = [f"{results[m]:.5f}" for m in ["MoGRU", "GRU", "SSM"]]
        print(f"{name:<30} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}  ← {winner}")

    wins = {}
    for _, _, w in summary:
        wins[w] = wins.get(w, 0) + 1
    print(f"\nWins: {', '.join(f'{k}={v}' for k, v in sorted(wins.items(), key=lambda x: -x[1]))}")

    # Verdict
    mogru_wins = wins.get("MoGRU", 0)
    ssm_wins = wins.get("SSM", 0)
    print(f"\n{'='*65}")
    if mogru_wins > ssm_wins:
        print("VERDICT: MoGRU beats SSM — momentum offers something SSMs don't.")
        print("This is a publishable result if it holds at scale.")
    elif ssm_wins > mogru_wins:
        print("VERDICT: SSM wins — linear state-space dynamics already capture")
        print("MoGRU's gains. Momentum adds complexity without benefit.")
    else:
        print("VERDICT: Tied — need more tasks or larger scale to differentiate.")
    print(f"{'='*65}")


if __name__ == "__main__":
    run()
