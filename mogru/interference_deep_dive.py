"""
MoGRU Interference Resistance Deep Dive

MoGRU showed strong interference resistance in the diagnostic gauntlet.
This script explores the effect systematically by sweeping:

  1. Number of distractors (10, 25, 50, 100, 200)
  2. Number of items to memorize (3, 5, 8, 10)
  3. Sequence-level difficulty scaling

Goal: build a thorough case that momentum provides inherent resistance
to distractor bombardment, and characterize when/how it helps.
"""

import sys
import os
import math
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mogru.mogru import MoGRU, count_parameters


# ===========================================================================
# Interference Dataset (standalone, from gauntlet)
# ===========================================================================

class InterferenceDataset(Dataset):
    """Memorize K items, survive N distractors, recall originals."""

    def __init__(self, num_samples=5000, k_items=5, n_distractors=50, vocab_size=32):
        self.num_samples = num_samples
        self.k_items = k_items
        self.n_distractors = n_distractors
        self.vocab_size = vocab_size
        self.total_vocab = vocab_size + 3  # +MARKER, +QUERY, +PAD

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        K = self.k_items
        N = self.n_distractors
        V = self.vocab_size

        orig_keys = torch.randint(1, V // 2 + 1, (K,))
        orig_vals = torch.randint(V // 2, V + 1, (K,))

        dist_keys = orig_keys.repeat(N // K + 1)[:N]
        dist_keys = (dist_keys + torch.randint(-1, 2, (N,))).clamp(1, V // 2)
        dist_vals = torch.randint(V // 2, V + 1, (N,))

        marker_tok = V + 1
        query_tok = V + 2

        phase1_len = K * 3
        phase2_len = N * 2
        phase3_len = K * 2
        total_len = phase1_len + phase2_len + phase3_len

        tokens = torch.zeros(total_len, dtype=torch.long)
        for i in range(K):
            tokens[i * 3] = marker_tok
            tokens[i * 3 + 1] = orig_keys[i]
            tokens[i * 3 + 2] = orig_vals[i]

        offset = phase1_len
        for i in range(N):
            tokens[offset + i * 2] = dist_keys[i]
            tokens[offset + i * 2 + 1] = dist_vals[i]

        offset = phase1_len + phase2_len
        for i in range(K):
            tokens[offset + i * 2] = query_tok
            tokens[offset + i * 2 + 1] = orig_keys[i]

        target = torch.zeros(total_len, dtype=torch.long)
        mask = torch.zeros(total_len, dtype=torch.float)
        offset = phase1_len + phase2_len
        for i in range(K):
            target[offset + i * 2 + 1] = orig_vals[i]
            mask[offset + i * 2 + 1] = 1.0

        return tokens, target, mask


# ===========================================================================
# Models
# ===========================================================================

class InterferenceModel(nn.Module):
    """Shared architecture for interference task with swappable RNN."""

    def __init__(self, vocab_size, hidden_size, rnn_type="gru"):
        super().__init__()
        self.rnn_type = rnn_type
        total_vocab = vocab_size + 3
        self.embed = nn.Embedding(total_vocab, hidden_size)

        if rnn_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        elif rnn_type == "mogru":
            self.rnn = MoGRU(hidden_size, hidden_size, num_layers=1)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.head = nn.Linear(hidden_size, total_vocab)

    def forward(self, x):
        emb = self.embed(x)
        if self.rnn_type == "mogru":
            out, _, betas = self.rnn(emb)
        else:
            out, _ = self.rnn(emb)
            betas = None
        logits = self.head(out)
        return logits, betas


def train_and_eval(rnn_type, k_items, n_distractors, vocab_size=32,
                   hidden_size=96, epochs=25, batch_size=64, lr=1e-3, seed=42):
    """Train and evaluate one configuration."""
    torch.manual_seed(seed)

    train_ds = InterferenceDataset(5000, k_items, n_distractors, vocab_size)
    val_ds = InterferenceDataset(1000, k_items, n_distractors, vocab_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = InterferenceModel(vocab_size, hidden_size, rnn_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    params = count_parameters(model)

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for tokens, target, mask in train_loader:
            optimizer.zero_grad()
            logits, _ = model(tokens)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   target.view(-1), reduction="none")
            loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for tokens, target, mask in val_loader:
                logits, _ = model(tokens)
                preds = logits.argmax(dim=-1)
                masked = mask.bool()
                correct += (preds[masked] == target[masked]).sum().item()
                total += masked.sum().item()

        acc = correct / max(total, 1)
        best_acc = max(best_acc, acc)

    elapsed = time.time() - t0
    return best_acc, params, elapsed


# ===========================================================================
# Sweep
# ===========================================================================

def run_sweep():
    MODELS = ["gru", "lstm", "mogru"]

    # Sweep 1: Fix k=5, vary distractors
    print("=" * 70)
    print("SWEEP 1: Distractor count (k=5 items, vary N distractors)")
    print("=" * 70)

    distractor_counts = [10, 25, 50, 100, 200]
    sweep1_results = {}

    for n_dist in distractor_counts:
        print(f"\n--- N={n_dist} distractors ---")
        row = {}
        for rnn_type in MODELS:
            acc, params, elapsed = train_and_eval(
                rnn_type, k_items=5, n_distractors=n_dist
            )
            print(f"  {rnn_type:6s}: acc={acc:.4f} ({elapsed:.0f}s)")
            row[rnn_type] = acc
        sweep1_results[n_dist] = row

    print(f"\n{'N_dist':>8}", end="")
    for m in MODELS:
        print(f"  {m:>8}", end="")
    print("  Winner")
    print("-" * 50)
    for n_dist in distractor_counts:
        row = sweep1_results[n_dist]
        winner = max(row, key=row.get)
        print(f"{n_dist:>8}", end="")
        for m in MODELS:
            marker = " *" if m == winner else "  "
            print(f"  {row[m]:>6.4f}{marker}", end="")
        print(f"  {winner}")

    # Sweep 2: Fix N=50, vary k items
    print("\n" + "=" * 70)
    print("SWEEP 2: Items to memorize (N=50 distractors, vary K items)")
    print("=" * 70)

    k_values = [3, 5, 8, 10]
    sweep2_results = {}

    for k in k_values:
        print(f"\n--- K={k} items ---")
        row = {}
        for rnn_type in MODELS:
            acc, params, elapsed = train_and_eval(
                rnn_type, k_items=k, n_distractors=50
            )
            print(f"  {rnn_type:6s}: acc={acc:.4f} ({elapsed:.0f}s)")
            row[rnn_type] = acc
        sweep2_results[k] = row

    print(f"\n{'K_items':>8}", end="")
    for m in MODELS:
        print(f"  {m:>8}", end="")
    print("  Winner")
    print("-" * 50)
    for k in k_values:
        row = sweep2_results[k]
        winner = max(row, key=row.get)
        print(f"{k:>8}", end="")
        for m in MODELS:
            marker = " *" if m == winner else "  "
            print(f"  {row[m]:>6.4f}{marker}", end="")
        print(f"  {winner}")

    # Overall summary
    mogru_wins = 0
    total_configs = len(distractor_counts) + len(k_values)
    for results in [sweep1_results, sweep2_results]:
        for row in results.values():
            if max(row, key=row.get) == "mogru":
                mogru_wins += 1

    print(f"\n{'='*70}")
    print(f"MoGRU wins {mogru_wins}/{total_configs} configurations")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_sweep()
