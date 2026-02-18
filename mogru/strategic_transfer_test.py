"""
Strategic Transfer Test: MoGRU vs GRU as Temporal Backbone

Reproduces the architecture from "Cross-Domain Strategic Transfer via
Temporal Integration" (Matthiasson, 2025) and tests whether MoGRU's
momentum dynamics improve cross-domain transfer over vanilla GRU.

Architecture (faithful to the paper):
  - Encoder: MLP (6 → 128 → 64) with ReLU + LayerNorm
  - Target encoder: EMA copy (τ=0.996)
  - Temporal integrator: GRU(64, 64) or MoGRU(64, 64) — the variable
  - Surprise gate: |pred - target|, gating belief updates
  - Predictor: MLP (65 → 32 → 64), K=5 step autoregressive
  - Loss: VICReg (variance + invariance + covariance)

Synthetic domains mimic strategic interaction dynamics:
  - Rising auction: upward price trends with competition
  - Negotiation: oscillatory convergence between parties
  - Public goods: cooperation/defection regime shifts
  - Competitive bidding: aggressive bursts with reversals

Evaluation: LOO cross-domain transfer (train N-1, test on held-out).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import argparse
import copy

from mogru.mogru import MoGRU, count_parameters


# ===========================================================================
# Synthetic Strategic Domains
# ===========================================================================

class StrategicDomainDataset(Dataset):
    """Generates synthetic strategic interaction sequences.

    Each observation is 6D: [value_norm, competition, bid_ratio, won, price_norm, surplus]
    matching the paper's format.
    """

    DOMAINS = ["rising_auction", "negotiation", "public_goods", "competitive_bidding"]

    def __init__(self, domain, num_sequences=2000, seq_len=50, seed=None):
        self.domain = domain
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        rng = np.random.RandomState(seed)
        self.data, self.outcomes = self._generate(rng)

    def _generate(self, rng):
        gen = {
            "rising_auction": self._gen_rising_auction,
            "negotiation": self._gen_negotiation,
            "public_goods": self._gen_public_goods,
            "competitive_bidding": self._gen_competitive_bidding,
        }[self.domain]
        return gen(rng)

    def _gen_rising_auction(self, rng):
        """Upward price trend with increasing competition and noise."""
        seqs, outcomes = [], []
        for _ in range(self.num_sequences):
            T = self.seq_len
            trend = np.linspace(0.1, 0.9, T) + rng.normal(0, 0.05, T)
            competition = np.clip(np.linspace(0.2, 0.8, T) + rng.normal(0, 0.08, T), 0, 1)
            bid_ratio = np.clip(trend * (1 + rng.normal(0, 0.1, T)), 0, 1)
            won = (rng.random(T) < (1 - competition)).astype(float)
            price_norm = np.clip(trend + rng.normal(0, 0.03, T), 0, 1)
            surplus = np.clip(1 - price_norm - rng.uniform(0, 0.2, T), -0.5, 1)
            obs = np.stack([trend, competition, bid_ratio, won, price_norm, surplus], axis=-1)
            seqs.append(obs.astype(np.float32))
            # Outcome: final surplus (regression target for transfer eval)
            outcomes.append(surplus[-1].astype(np.float32))
        return torch.tensor(np.array(seqs)), torch.tensor(np.array(outcomes))

    def _gen_negotiation(self, rng):
        """Oscillatory convergence with decreasing amplitude."""
        seqs, outcomes = [], []
        for _ in range(self.num_sequences):
            T = self.seq_len
            t = np.arange(T, dtype=float)
            # Damped oscillation around 0.5
            decay = np.exp(-t / (T * 0.4))
            osc = 0.5 + 0.4 * decay * np.sin(2 * np.pi * t / 12)
            value_norm = np.clip(osc + rng.normal(0, 0.04, T), 0, 1)
            competition = np.clip(0.5 + 0.2 * np.sin(2 * np.pi * t / 8) + rng.normal(0, 0.05, T), 0, 1)
            bid_ratio = np.clip(value_norm * 0.8 + rng.normal(0, 0.06, T), 0, 1)
            won = (rng.random(T) > 0.5).astype(float)
            price_norm = np.clip(value_norm * 0.9 + rng.normal(0, 0.03, T), 0, 1)
            surplus = np.clip(value_norm - price_norm + rng.normal(0, 0.02, T), -0.5, 1)
            obs = np.stack([value_norm, competition, bid_ratio, won, price_norm, surplus], axis=-1)
            seqs.append(obs.astype(np.float32))
            outcomes.append(surplus[-1].astype(np.float32))
        return torch.tensor(np.array(seqs)), torch.tensor(np.array(outcomes))

    def _gen_public_goods(self, rng):
        """Cooperation/defection regime shifts — sudden transitions."""
        seqs, outcomes = [], []
        for _ in range(self.num_sequences):
            T = self.seq_len
            # Regime shifts: 2-3 shifts per sequence
            regime = np.zeros(T)
            shift_points = sorted(rng.choice(range(5, T - 5), size=2, replace=False))
            current = rng.choice([0.3, 0.7])
            for i in range(T):
                if i in shift_points:
                    current = 1.0 - current  # flip regime
                regime[i] = current
            cooperation = np.clip(regime + rng.normal(0, 0.08, T), 0, 1)
            value_norm = np.clip(cooperation * 0.6 + 0.2 + rng.normal(0, 0.05, T), 0, 1)
            competition = 1 - cooperation
            bid_ratio = np.clip(0.5 + rng.normal(0, 0.1, T), 0, 1)
            won = (rng.random(T) < cooperation).astype(float)
            price_norm = np.clip(0.5 + rng.normal(0, 0.08, T), 0, 1)
            surplus = np.clip(cooperation * 0.5 - 0.1 + rng.normal(0, 0.05, T), -0.5, 1)
            obs = np.stack([value_norm, competition, bid_ratio, won, price_norm, surplus], axis=-1)
            seqs.append(obs.astype(np.float32))
            outcomes.append(surplus[-1].astype(np.float32))
        return torch.tensor(np.array(seqs)), torch.tensor(np.array(outcomes))

    def _gen_competitive_bidding(self, rng):
        """Aggressive bursts with sharp reversals — tests momentum lag."""
        seqs, outcomes = [], []
        for _ in range(self.num_sequences):
            T = self.seq_len
            base = rng.uniform(0.3, 0.7)
            price = np.full(T, base)
            # Random aggressive bursts
            for _ in range(rng.randint(2, 5)):
                start = rng.randint(0, T - 5)
                length = rng.randint(3, 8)
                end = min(start + length, T)
                spike = rng.uniform(0.1, 0.3)
                direction = rng.choice([-1, 1])
                price[start:end] += direction * spike * np.linspace(1, 0.3, end - start)
            price = np.clip(price + rng.normal(0, 0.03, T), 0, 1)
            value_norm = np.clip(price + rng.normal(0, 0.05, T), 0, 1)
            competition = np.clip(0.6 + rng.normal(0, 0.12, T), 0, 1)
            bid_ratio = np.clip(price * 1.1 + rng.normal(0, 0.08, T), 0, 1)
            won = (rng.random(T) < 0.4).astype(float)
            surplus = np.clip(value_norm - price + rng.normal(0, 0.04, T), -0.5, 1)
            obs = np.stack([value_norm, competition, bid_ratio, won, price, surplus], axis=-1)
            seqs.append(obs.astype(np.float32))
            outcomes.append(surplus[-1].astype(np.float32))
        return torch.tensor(np.array(seqs)), torch.tensor(np.array(outcomes))

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return self.data[idx], self.outcomes[idx]


# ===========================================================================
# Architecture (faithful to the paper)
# ===========================================================================

class Encoder(nn.Module):
    """MLP encoder: 6 → 128 → 64, matching the paper."""

    def __init__(self, obs_dim=6, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class SurpriseGate(nn.Module):
    """Surprise-modulated belief update from the paper.

    s_t = |pred_t - target_t|  (prediction error)
    g_t = sigmoid(MLP(stop_grad(s_t)))
    b_t = g_t * h_t + (1 - g_t) * b_{t-1}
    """

    def __init__(self, latent_dim=64):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, h_t, b_prev, pred_z, target_z):
        surprise = torch.abs(pred_z - target_z.detach())
        g = torch.sigmoid(self.gate_mlp(surprise.detach()))  # stop-gradient
        b_t = g * h_t + (1 - g) * b_prev
        return b_t, g


class Predictor(nn.Module):
    """Multi-step predictor: (belief + step_embed) → predicted latent.

    Paper uses 65 → 32 → 64 (64 from belief + 1 for step index).
    """

    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, belief, step_k):
        # step_k: scalar or (B, 1)
        if step_k.dim() == 0:
            step_k = step_k.unsqueeze(0).expand(belief.size(0), 1)
        cat = torch.cat([belief, step_k.float()], dim=-1)
        return self.net(cat)


class StrategicTransferModel(nn.Module):
    """Full architecture from the paper with swappable temporal backbone.

    Components:
      1. Online encoder (MLP)
      2. Target encoder (EMA copy)
      3. Temporal integrator (GRU or MoGRU)
      4. Surprise gate
      5. Predictor (K-step autoregressive)

    Loss: VICReg (variance + invariance + covariance)
    """

    def __init__(self, backbone="gru", latent_dim=64, hidden_dim=128,
                 obs_dim=6, K=5, ema_tau=0.996):
        super().__init__()
        self.backbone_type = backbone
        self.latent_dim = latent_dim
        self.K = K
        self.ema_tau = ema_tau

        # Encoder
        self.encoder = Encoder(obs_dim, hidden_dim, latent_dim)

        # Target encoder (EMA, no grad)
        self.target_encoder = Encoder(obs_dim, hidden_dim, latent_dim)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Temporal integrator
        if backbone == "gru":
            self.rnn = nn.GRU(latent_dim, latent_dim, num_layers=1, batch_first=True)
        elif backbone == "mogru":
            self.rnn = MoGRU(latent_dim, latent_dim, num_layers=1)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Surprise gate
        self.surprise_gate = SurpriseGate(latent_dim)

        # Predictor
        self.predictor = Predictor(latent_dim)

        # Downstream head (linear probe for transfer eval)
        self.probe = nn.Linear(latent_dim, 1)

    @torch.no_grad()
    def _update_target_encoder(self):
        tau = self.ema_tau
        for p_online, p_target in zip(self.encoder.parameters(),
                                       self.target_encoder.parameters()):
            p_target.data.mul_(tau).add_(p_online.data, alpha=1 - tau)

    def forward(self, x):
        """
        Args:
            x: (B, T, 6) strategic observation sequence
        Returns:
            dict with loss components, beliefs, diagnostics
        """
        B, T, _ = x.shape
        device = x.device

        # Encode all timesteps
        x_flat = x.reshape(B * T, -1)
        z_online = self.encoder(x_flat).reshape(B, T, -1)

        with torch.no_grad():
            z_target = self.target_encoder(x_flat).reshape(B, T, -1)

        # Run temporal integrator
        if self.backbone_type == "gru":
            h_all, _ = self.rnn(z_online)
            beta_info = None
        else:
            h_all, _, betas = self.rnn(z_online)
            beta_info = {
                "mean": betas.mean().item(),
                "std": betas.std().item(),
            }

        # Surprise-gated belief update
        beliefs = []
        b_prev = torch.zeros(B, self.latent_dim, device=device)
        total_surprise = 0.0

        for t in range(T):
            h_t = h_all[:, t]
            # One-step prediction from previous belief
            step_k = torch.ones(B, 1, device=device)
            pred_z = self.predictor(b_prev, step_k)
            target_z = z_target[:, t]

            b_t, g_t = self.surprise_gate(h_t, b_prev, pred_z, target_z)
            beliefs.append(b_t)
            total_surprise += g_t.mean().item()
            b_prev = b_t

        beliefs = torch.stack(beliefs, dim=1)  # (B, T, latent_dim)

        # K-step autoregressive prediction loss (VICReg)
        vicreg_loss = torch.tensor(0.0, device=device)
        n_predictions = 0

        for t in range(T - self.K):
            b_t = beliefs[:, t]
            for k in range(1, self.K + 1):
                if t + k >= T:
                    break
                step_k = torch.full((B, 1), k, device=device, dtype=torch.float)
                pred = self.predictor(b_t, step_k)
                target = z_target[:, t + k]
                vicreg_loss = vicreg_loss + self._vicreg_loss(pred, target)
                n_predictions += 1

        if n_predictions > 0:
            vicreg_loss = vicreg_loss / n_predictions

        # Downstream probe (last belief → outcome prediction)
        outcome_pred = self.probe(beliefs[:, -1]).squeeze(-1)

        return {
            "vicreg_loss": vicreg_loss,
            "outcome_pred": outcome_pred,
            "beliefs": beliefs,
            "beta_info": beta_info,
            "mean_surprise": total_surprise / T,
        }

    def _vicreg_loss(self, z1, z2, lam=25.0, mu=25.0, nu=1.0):
        """VICReg: Variance + Invariance + Covariance."""
        # Invariance
        inv_loss = F.mse_loss(z1, z2)

        # Variance
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = (torch.relu(1 - std_z1).mean() + torch.relu(1 - std_z2).mean()) / 2

        # Covariance
        z1_c = z1 - z1.mean(dim=0)
        z2_c = z2 - z2.mean(dim=0)
        N = z1.size(0)
        cov1 = (z1_c.T @ z1_c) / (N - 1)
        cov2 = (z2_c.T @ z2_c) / (N - 1)
        D = z1.size(1)
        cov_loss = (cov1.fill_diagonal_(0).pow(2).sum() / D +
                    cov2.fill_diagonal_(0).pow(2).sum() / D) / 2

        return lam * inv_loss + mu * var_loss + nu * cov_loss


# ===========================================================================
# Training & Evaluation
# ===========================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_probe_loss = 0
    n_batches = 0

    for x, outcomes in loader:
        x, outcomes = x.to(device), outcomes.to(device)
        optimizer.zero_grad()

        result = model(x)
        vicreg = result["vicreg_loss"]
        probe_loss = F.mse_loss(result["outcome_pred"], outcomes)
        loss = vicreg + probe_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model._update_target_encoder()

        total_loss += vicreg.item()
        total_probe_loss += probe_loss.item()
        n_batches += 1

    return total_loss / n_batches, total_probe_loss / n_batches


def evaluate_transfer(model, loader, device):
    """Evaluate downstream transfer: freeze encoder+RNN, only use probe."""
    model.eval()
    total_mse = 0
    n = 0

    with torch.no_grad():
        for x, outcomes in loader:
            x, outcomes = x.to(device), outcomes.to(device)
            result = model(x)
            mse = F.mse_loss(result["outcome_pred"], outcomes, reduction="sum")
            total_mse += mse.item()
            n += outcomes.size(0)

    return total_mse / n


def run_loo_experiment(backbone, domains, seq_len, epochs, batch_size,
                       lr, device, verbose=True):
    """Leave-one-out cross-domain transfer evaluation."""
    results = {}

    for held_out in domains:
        train_domains = [d for d in domains if d != held_out]

        # Build train set: combine all non-held-out domains
        train_datasets = []
        for d in train_domains:
            train_datasets.append(
                StrategicDomainDataset(d, num_sequences=2000, seq_len=seq_len, seed=42)
            )
        train_data = torch.utils.data.ConcatDataset(train_datasets)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # Test set: held-out domain
        test_ds = StrategicDomainDataset(held_out, num_sequences=500, seq_len=seq_len, seed=99)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        # Build model
        model = StrategicTransferModel(backbone=backbone, latent_dim=64,
                                        hidden_dim=128, obs_dim=6).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        if verbose:
            params = count_parameters(model)
            print(f"  [{backbone}] held_out={held_out} | params={params:,}")

        best_transfer_mse = float("inf")
        beta_info = None

        for epoch in range(1, epochs + 1):
            vic_loss, probe_loss = train_epoch(model, train_loader, optimizer, device)
            transfer_mse = evaluate_transfer(model, test_loader, device)
            best_transfer_mse = min(best_transfer_mse, transfer_mse)

            if verbose and (epoch % 5 == 0 or epoch == 1):
                # Get beta info from a forward pass
                model.eval()
                with torch.no_grad():
                    sample_x = next(iter(test_loader))[0].to(device)
                    result = model(sample_x)
                    beta_info = result["beta_info"]
                line = (f"    epoch {epoch:2d} | vicreg={vic_loss:.4f} "
                        f"probe={probe_loss:.4f} transfer_mse={transfer_mse:.4f}")
                if beta_info:
                    line += f" | beta: mean={beta_info['mean']:.3f} std={beta_info['std']:.3f}"
                print(line)

        results[held_out] = best_transfer_mse
        if verbose:
            print(f"  -> best transfer MSE: {best_transfer_mse:.4f}")

    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="MoGRU vs GRU for Strategic Transfer")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    domains = StrategicDomainDataset.DOMAINS

    print(f"Strategic Transfer Test: MoGRU vs GRU")
    print(f"seq_len={args.seq_len}, epochs={args.epochs}, domains={domains}")
    print(f"Device: {device} | Seed: {args.seed}")
    print("=" * 70)

    all_results = {}

    for backbone in ["gru", "mogru"]:
        print(f"\n{'='*70}")
        print(f"Backbone: {backbone.upper()}")
        print(f"{'='*70}")
        results = run_loo_experiment(
            backbone=backbone,
            domains=domains,
            seq_len=args.seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            verbose=True,
        )
        all_results[backbone] = results

    # Summary
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN TRANSFER SUMMARY (lower MSE = better)")
    print("-" * 70)
    print(f"{'Held-out Domain':<25} {'GRU MSE':>10} {'MoGRU MSE':>10} {'Delta':>10}")
    print("-" * 70)

    gru_total, mogru_total = 0, 0
    for domain in domains:
        gru_mse = all_results["gru"][domain]
        mogru_mse = all_results["mogru"][domain]
        delta = mogru_mse - gru_mse
        sign = "+" if delta >= 0 else ""
        print(f"{domain:<25} {gru_mse:>10.4f} {mogru_mse:>10.4f} {sign}{delta:>9.4f}")
        gru_total += gru_mse
        mogru_total += mogru_mse

    gru_avg = gru_total / len(domains)
    mogru_avg = mogru_total / len(domains)
    delta_avg = mogru_avg - gru_avg
    sign = "+" if delta_avg >= 0 else ""
    print("-" * 70)
    print(f"{'AVERAGE':<25} {gru_avg:>10.4f} {mogru_avg:>10.4f} {sign}{delta_avg:>9.4f}")

    winner = "MoGRU" if mogru_avg < gru_avg else "GRU"
    pct = abs(delta_avg) / gru_avg * 100
    print(f"\nWinner: {winner} (by {pct:.1f}%)")
    print("\nExpected: MoGRU should excel on rising_auction and negotiation")
    print("(trending/oscillatory dynamics) but may lag on competitive_bidding")
    print("(sharp reversals where momentum creates inertia).")


if __name__ == "__main__":
    main()
