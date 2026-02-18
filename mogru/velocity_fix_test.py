"""
Test velocity decay fixes for MoGRU's long-range collapse.

Problem: MoGRU collapses at N>=100 distractors because velocity accumulates
distractor influence. Three fixes tested:

  1. Velocity clipping — bound velocity magnitude (no new params)
  2. Velocity LayerNorm — normalize velocity each step (minimal params)
  3. Velocity write gate — learned gate to ignore distractors (one new gate)

Tested on interference resistance at N=100 and N=200 where baseline MoGRU
currently loses to GRU.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mogru.mogru import MoGRU, count_parameters
from mogru.interference_deep_dive import InterferenceDataset


class InterferenceModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, rnn_type="mogru", **mogru_kwargs):
        super().__init__()
        self.rnn_type = rnn_type
        total_vocab = vocab_size + 3
        self.embed = nn.Embedding(total_vocab, hidden_size)

        if rnn_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        elif rnn_type == "mogru":
            self.rnn = MoGRU(hidden_size, hidden_size, num_layers=1, **mogru_kwargs)
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
                   hidden_size=96, epochs=25, batch_size=64, lr=1e-3,
                   seed=42, **mogru_kwargs):
    torch.manual_seed(seed)

    train_ds = InterferenceDataset(5000, k_items, n_distractors, vocab_size)
    val_ds = InterferenceDataset(1000, k_items, n_distractors, vocab_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = InterferenceModel(vocab_size, hidden_size, rnn_type, **mogru_kwargs)
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


VARIANTS = {
    "gru": {"rnn_type": "gru"},
    "mogru_base": {"rnn_type": "mogru"},
    "mogru_clip_0.5": {"rnn_type": "mogru", "velocity_clip": 0.5},
    "mogru_clip_1.0": {"rnn_type": "mogru", "velocity_clip": 1.0},
    "mogru_v_norm": {"rnn_type": "mogru", "use_velocity_norm": True},
    "mogru_v_gate": {"rnn_type": "mogru", "use_velocity_gate": True},
}


def run():
    test_configs = [
        (5, 100),   # MoGRU baseline: 0.553, GRU: 0.729
        (5, 200),   # MoGRU baseline: 0.063, GRU: 0.293
    ]

    all_results = {}

    for k, n in test_configs:
        print(f"\n{'='*60}")
        print(f"K={k} items, N={n} distractors")
        print(f"{'='*60}")

        results = {}
        for name, kwargs in VARIANTS.items():
            rnn_type = kwargs.pop("rnn_type")
            acc, params, elapsed = train_and_eval(
                rnn_type, k_items=k, n_distractors=n, **kwargs
            )
            kwargs["rnn_type"] = rnn_type  # restore for next run
            print(f"  {name:20s}: acc={acc:.4f}  ({params:,} params, {elapsed:.0f}s)")
            results[name] = acc

        all_results[(k, n)] = results

    # Summary
    print(f"\n{'='*60}")
    print("VELOCITY FIX SUMMARY")
    print(f"{'='*60}")
    for (k, n), results in all_results.items():
        print(f"\nK={k}, N={n}:")
        winner = max(results, key=results.get)
        for name, acc in sorted(results.items(), key=lambda x: -x[1]):
            marker = " <-- BEST" if name == winner else ""
            print(f"  {name:20s}: {acc:.4f}{marker}")

        base = results.get("mogru_base", 0)
        gru = results.get("gru", 0)
        print(f"\n  Baseline MoGRU vs GRU gap: {base - gru:+.4f}")
        for name in ["mogru_clip_0.5", "mogru_clip_1.0", "mogru_v_norm", "mogru_v_gate"]:
            if name in results:
                delta = results[name] - base
                print(f"  {name:20s} vs baseline: {delta:+.4f}")


if __name__ == "__main__":
    run()
