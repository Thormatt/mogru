"""
Momentum-Gated Recurrent Unit (MoGRU)

A GRU variant that adds a second-order velocity state, giving the hidden
state momentum-like dynamics:

  - Standard GRU gates (reset r, update u)
  - Learned momentum retention gate (beta)
  - Velocity v_t accumulates smoothed deltas over time
  - Hidden state steps via h_prev + u * v_t (not a convex combo)

The key insight: many real sequences have trends, oscillations, and gradual
dynamics. A velocity state lets the model extrapolate and smooth, rather than
reacting to every timestep from scratch.
"""

import torch
import torch.nn as nn


class MoGRUCell(nn.Module):
    """Single MoGRU cell."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_layernorm: bool = True,
        use_damping: bool = False,
        velocity_clip: float | None = None,
        use_velocity_norm: bool = False,
        use_velocity_gate: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_layernorm = use_layernorm
        self.use_damping = use_damping
        self.velocity_clip = velocity_clip
        self.use_velocity_norm = use_velocity_norm
        self.use_velocity_gate = use_velocity_gate

        # --- Fused reset + update gates ---
        self.W_ru = nn.Linear(input_size + hidden_size, 2 * hidden_size)

        # --- Momentum retention gate ---
        self.W_beta = nn.Linear(input_size + hidden_size, hidden_size)

        # --- Candidate (split so reset gate multiplies h only) ---
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

        # --- Optional damping gate ---
        if use_damping:
            self.W_gamma = nn.Linear(input_size + hidden_size, hidden_size)

        # --- Optional velocity LayerNorm ---
        if use_velocity_norm:
            self.v_ln = nn.LayerNorm(hidden_size)

        # --- Optional velocity write gate (selective velocity update) ---
        if use_velocity_gate:
            self.W_alpha = nn.Linear(input_size + hidden_size, hidden_size)

        # --- Optional LayerNorm ---
        if use_layernorm:
            self.ln = nn.LayerNorm(hidden_size)

        self._init_parameters()

    def _init_parameters(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        # High initial momentum (~0.88): velocity persists
        nn.init.constant_(self.W_beta.bias, 2.0)
        # Conservative updates (~0.12): don't overwrite too fast
        # Update gate is the second chunk of W_ru
        with torch.no_grad():
            self.W_ru.bias[self.hidden_size:].fill_(-2.0)
        # Velocity write gate: start open (bias +2 → ~0.88)
        if self.use_velocity_gate:
            nn.init.constant_(self.W_alpha.bias, 2.0)

    def init_state(self, batch_size: int, device: torch.device | None = None):
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        v = torch.zeros(batch_size, self.hidden_size, device=device)
        return h, v

    def forward(self, x_t, h_prev, v_prev):
        """
        Args:
            x_t:    (B, input_size)
            h_prev: (B, hidden_size)
            v_prev: (B, hidden_size)
        Returns:
            h_t:    (B, hidden_size)
            v_t:    (B, hidden_size)
            beta_t: (B, hidden_size) — momentum gate for diagnostics
        """
        # 1) Gates
        cat_xh = torch.cat([x_t, h_prev], dim=-1)
        ru = torch.sigmoid(self.W_ru(cat_xh))
        r, u = ru.chunk(2, dim=-1)

        # 2) Momentum retention
        beta_t = torch.sigmoid(self.W_beta(cat_xh))

        # 3) Candidate
        h_tilde = torch.tanh(self.W_h(x_t) + self.U_h(r * h_prev))

        # 4) Delta and velocity update
        d = h_tilde - h_prev
        if self.use_velocity_gate:
            alpha_t = torch.sigmoid(self.W_alpha(cat_xh))
            v_t = beta_t * v_prev + alpha_t * (1 - beta_t) * d
        else:
            v_t = beta_t * v_prev + (1 - beta_t) * d

        if self.velocity_clip is not None:
            v_t = v_t.clamp(-self.velocity_clip, self.velocity_clip)

        if self.use_velocity_norm:
            v_t = self.v_ln(v_t)

        # 5) Hidden state update
        if self.use_damping:
            gamma = torch.sigmoid(self.W_gamma(cat_xh))
            h_t = (1 - gamma) * h_prev + u * v_t
        else:
            h_t = h_prev + u * v_t

        # 6) LayerNorm
        if self.use_layernorm:
            h_t = self.ln(h_t)

        return h_t, v_t, beta_t


class MoGRU(nn.Module):
    """Multi-layer MoGRU network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        use_layernorm: bool = True,
        use_damping: bool = False,
        velocity_clip: float | None = None,
        use_velocity_norm: bool = False,
        use_velocity_gate: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_in = input_size if i == 0 else hidden_size
            self.cells.append(
                MoGRUCell(
                    cell_in, hidden_size, use_layernorm, use_damping,
                    velocity_clip, use_velocity_norm, use_velocity_gate,
                )
            )

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, states=None):
        """
        Args:
            x:      (B, T, input_size)
            states: list[(h, v)] per layer or None
        Returns:
            output: (B, T, hidden_size)
            states: list[(h, v)] final states
            betas:  (B, T, num_layers) — mean momentum gate per layer
        """
        B, T, _ = x.shape
        device = x.device

        if states is None:
            states = [cell.init_state(B, device) for cell in self.cells]

        outputs, all_betas = [], []

        for t in range(T):
            inp = x[:, t]
            step_betas = []

            for i, cell in enumerate(self.cells):
                h_prev, v_prev = states[i]
                h_t, v_t, beta_t = cell(inp, h_prev, v_prev)
                states[i] = (h_t, v_t)
                step_betas.append(beta_t.mean(dim=-1, keepdim=True))
                inp = self.drop(h_t) if i < self.num_layers - 1 else h_t

            outputs.append(inp)
            all_betas.append(torch.cat(step_betas, dim=-1))

        return (
            torch.stack(outputs, dim=1),
            states,
            torch.stack(all_betas, dim=1),
        )


class MomentumGRUCell(nn.Module):
    """MomentumRNN-style GRU: momentum on INPUT transformation (Nguyen et al., 2020).

    v_t = mu * v_{t-1} + eps * W_x(x_t)
    gates = v_t + W_h(h_{t-1})
    ... then standard GRU gate logic ...

    This is the prior-art baseline. Momentum smooths the input signal,
    but the hidden state update itself is standard first-order GRU.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mu: float = 0.9,
        eps: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mu = mu
        self.eps = eps

        # Input -> 3*hidden (reset, update, candidate gates)
        self.x2h = nn.Linear(input_size, 3 * hidden_size)
        # Hidden -> 3*hidden
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        self._init_parameters()

    def _init_parameters(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def init_state(self, batch_size: int, device: torch.device | None = None):
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        v = torch.zeros(batch_size, 3 * self.hidden_size, device=device)
        return h, v

    def forward(self, x_t, h_prev, v_prev):
        # Momentum on input transformation
        v_t = self.mu * v_prev + self.eps * self.x2h(x_t)

        # Split into gate components
        v_r, v_z, v_n = v_t.chunk(3, dim=-1)
        h_r, h_z, h_n = self.h2h(h_prev).chunk(3, dim=-1)

        r = torch.sigmoid(v_r + h_r)
        z = torch.sigmoid(v_z + h_z)
        n = torch.tanh(v_n + r * h_n)

        h_t = (1 - z) * h_prev + z * n

        # Return mu as a fake "beta" for diagnostic compatibility
        beta_t = torch.full_like(h_t, self.mu)
        return h_t, v_t, beta_t


class MomentumGRU(nn.Module):
    """Multi-layer MomentumRNN-style GRU (Nguyen et al., 2020 approach on GRU)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        mu: float = 0.9,
        eps: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_in = input_size if i == 0 else hidden_size
            self.cells.append(MomentumGRUCell(cell_in, hidden_size, mu, eps))

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, states=None):
        B, T, _ = x.shape
        device = x.device

        if states is None:
            states = [cell.init_state(B, device) for cell in self.cells]

        outputs, all_betas = [], []

        for t in range(T):
            inp = x[:, t]
            step_betas = []

            for i, cell in enumerate(self.cells):
                h_prev, v_prev = states[i]
                h_t, v_t, beta_t = cell(inp, h_prev, v_prev)
                states[i] = (h_t, v_t)
                step_betas.append(beta_t.mean(dim=-1, keepdim=True))
                inp = self.drop(h_t) if i < self.num_layers - 1 else h_t

            outputs.append(inp)
            all_betas.append(torch.cat(step_betas, dim=-1))

        return (
            torch.stack(outputs, dim=1),
            states,
            torch.stack(all_betas, dim=1),
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
