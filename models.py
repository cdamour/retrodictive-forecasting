#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py — Neural network models for Retrodictive Forecasting
=======================================================================

Implements four models, all with matched capacity for fair comparison:

  1. InverseCVAE   — p_θ(x | y, z): learns to reconstruct past from future
  2. ForwardCVAE   — p_θ(y | x, z): learns to predict future from past
  3. ForwardMLP    — ŷ = f_φ(x): deterministic forward baseline
  4. RealNVP       — p_φ(y): normalizing flow prior over future windows

Design principles
-----------------
  - InverseCVAE and ForwardCVAE share the same architecture (MLP encoder/decoder)
    with identical capacity (hidden dim, depth, z_dim).  The ONLY difference is
    the direction: which variable is reconstructed and which is conditioned on.
  - ForwardMLP has the same hidden dim and depth as the CVAE decoders.
  - RealNVP provides a learned prior p(y) for MAP inference, ablated against N(0,I).
  - All models output diagonal Gaussian parameters (mean + log-std) where applicable.

Usage
-----
    from models import InverseCVAE, ForwardCVAE, ForwardMLP, RealNVP
    from config import get_default_config

    cfg = get_default_config()
    inv_cvae = InverseCVAE(n=32, m=16, cfg=cfg.inverse_cvae)
    fwd_cvae = ForwardCVAE(n=32, m=16, cfg=cfg.forward_cvae)
    fwd_mlp  = ForwardMLP(n=32, m=16, cfg=cfg.forward_mlp)
    flow     = RealNVP(dim=16, cfg=cfg.flow_prior)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    InverseCVAEConfig,
    ForwardCVAEConfig,
    ForwardMLPConfig,
    FlowPriorConfig,
)


# ============================================================================
# 1. SHARED BUILDING BLOCKS
# ============================================================================

class MLP(nn.Module):
    """Generic multi-layer perceptron with ReLU activations.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    out_dim : int
        Output dimension.
    hidden : int
        Width of each hidden layer.
    depth : int
        Number of hidden layers (default 2).
    dropout : float
        Dropout rate after each hidden layer (default 0.0 = off).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _gaussian_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    logstd: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood of a diagonal Gaussian, per sample.

    -log p(target | mu, sigma) = 0.5 * Σ_d [ ((target-mu)/sigma)² + 2*logstd + log(2π) ]

    Parameters
    ----------
    target : (B, D)
    mu : (B, D)
    logstd : (B, D)

    Returns
    -------
    nll : (B,)  — summed over dimension D, per sample.
    """
    var = torch.exp(2.0 * logstd)
    return 0.5 * torch.sum(
        ((target - mu) ** 2) / var + 2.0 * logstd + math.log(2.0 * math.pi),
        dim=-1,
    )


def _kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence KL(q(z) || p(z)) with p(z) = N(0, I).

    q(z) = N(mu, diag(exp(logvar)))

    KL = 0.5 * Σ_d [ exp(logvar_d) + mu_d² - 1 - logvar_d ]

    Parameters
    ----------
    mu : (B, z_dim)
    logvar : (B, z_dim)

    Returns
    -------
    kl : (B,)  — summed over latent dimensions, per sample.
    """
    return 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1.0 - logvar, dim=-1)


# ============================================================================
# 2. INVERSE CVAE: p_θ(x | y, z)
# ============================================================================

class InverseCVAE(nn.Module):
    """Conditional VAE that learns the INVERSE conditional p_θ(x | y, z).

    This is the core model of the retrodictive paradigm:
      - Encoder: q_φ(z | x, y)  — inference network (used during training)
      - Decoder: p_θ(x | y, z)  — generative model (used for MAP inference)

    At training time, we optimise the ELBO:
      L = E_q[ -log p_θ(x|y,z) ] + β * KL(q_φ(z|x,y) || p(z))

    At inference time (MAP), the encoder is NOT used.  Instead we
    directly optimise (y, z) to maximise log p_θ(x_obs | y, z) + priors.

    Architecture
    ------------
    Encoder input:  [x, y] ∈ R^{n+m}  →  (μ_z, logvar_z) ∈ R^{2*z_dim}
    Decoder input:  [y, z] ∈ R^{m+z_dim}  →  (μ_x, logstd_x) ∈ R^{2*n}
    """

    def __init__(self, n: int, m: int, cfg: InverseCVAEConfig):
        super().__init__()
        self.n = n
        self.m = m
        self.z_dim = cfg.z_dim

        # Encoder: q(z | x, y)
        self.encoder = MLP(
            in_dim=n + m,
            out_dim=2 * cfg.z_dim,
            hidden=cfg.hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )

        # Decoder: p(x | y, z) — outputs (μ_x, logstd_x)
        self.decoder = MLP(
            in_dim=m + cfg.z_dim,
            out_dim=2 * n,
            hidden=cfg.hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )

        # Clamp bounds for logstd (numerical stability)
        self._logstd_min = -6.0
        self._logstd_max = 2.0

    def encode(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode (x, y) → (μ_z, logvar_z).

        Parameters
        ----------
        x : (B, n) — past windows
        y : (B, m) — future windows

        Returns
        -------
        mu_z : (B, z_dim)
        logvar_z : (B, z_dim)
        """
        h = self.encoder(torch.cat([x, y], dim=-1))
        mu_z, logvar_z = torch.chunk(h, 2, dim=-1)
        return mu_z, logvar_z

    def decode(
        self, y: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode (y, z) → (μ_x, logstd_x).

        Parameters
        ----------
        y : (B, m) — future windows (or optimisation variables)
        z : (B, z_dim) — latent variables

        Returns
        -------
        mu_x : (B, n) — predicted past mean
        logstd_x : (B, n) — predicted past log-std (clamped)
        """
        h = self.decoder(torch.cat([y, z], dim=-1))
        mu_x, logstd_x = torch.chunk(h, 2, dim=-1)
        logstd_x = torch.clamp(logstd_x, self._logstd_min, self._logstd_max)
        return mu_x, logstd_x

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε, ε ~ N(0,I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def elbo(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute negative ELBO (loss to minimise).

        L = E_q[ -log p_θ(x|y,z) ] + β * KL(q || p)

        Parameters
        ----------
        x : (B, n) — observed past
        y : (B, m) — observed future
        beta : float — KL weight (1.0 = standard ELBO)

        Returns
        -------
        loss : scalar tensor (mean over batch)
        stats : dict with 'recon' and 'kl' (float, mean over batch)
        """
        # Encode
        mu_z, logvar_z = self.encode(x, y)
        z = self.reparameterize(mu_z, logvar_z)

        # Decode
        mu_x, logstd_x = self.decode(y, z)

        # Reconstruction loss: -log p(x | y, z)
        recon = _gaussian_nll(x, mu_x, logstd_x)  # (B,)

        # KL divergence
        kl = _kl_standard_normal(mu_z, logvar_z)  # (B,)

        # Total loss
        loss = torch.mean(recon + beta * kl)

        stats = {
            "recon": float(torch.mean(recon).item()),
            "kl": float(torch.mean(kl).item()),
        }
        return loss, stats

    def log_prob_x_given_yz(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Compute log p_θ(x | y, z) for MAP inference.

        Parameters
        ----------
        x : (B, n)
        y : (B, m)
        z : (B, z_dim)

        Returns
        -------
        log_prob : (B,)
        """
        mu_x, logstd_x = self.decode(y, z)
        return -_gaussian_nll(x, mu_x, logstd_x)


# ============================================================================
# 3. FORWARD CVAE: p_θ(y | x, z)  — BASELINE
# ============================================================================

class ForwardCVAE(nn.Module):
    """Conditional VAE that learns the FORWARD conditional p_θ(y | x, z).

    This is the standard CVAE baseline for comparison.
    Same architecture and capacity as InverseCVAE, but the direction
    is reversed:
      - Encoder: q_φ(z | x, y)
      - Decoder: p_θ(y | x, z)

    At inference time, prediction is obtained by:
      ŷ = μ_θ(x, z=0)  (posterior mean with z at prior mode)
    or by sampling z ~ N(0,I) for probabilistic forecasts.
    """

    def __init__(self, n: int, m: int, cfg: ForwardCVAEConfig):
        super().__init__()
        self.n = n
        self.m = m
        self.z_dim = cfg.z_dim

        # Encoder: q(z | x, y) — same as inverse, both see (x, y)
        self.encoder = MLP(
            in_dim=n + m,
            out_dim=2 * cfg.z_dim,
            hidden=cfg.hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )

        # Decoder: p(y | x, z) — reconstructs FUTURE from past + latent
        self.decoder = MLP(
            in_dim=n + cfg.z_dim,
            out_dim=2 * m,
            hidden=cfg.hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )

        self._logstd_min = -6.0
        self._logstd_max = 2.0

    def encode(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode (x, y) → (μ_z, logvar_z)."""
        h = self.encoder(torch.cat([x, y], dim=-1))
        mu_z, logvar_z = torch.chunk(h, 2, dim=-1)
        return mu_z, logvar_z

    def decode(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode (x, z) → (μ_y, logstd_y)."""
        h = self.decoder(torch.cat([x, z], dim=-1))
        mu_y, logstd_y = torch.chunk(h, 2, dim=-1)
        logstd_y = torch.clamp(logstd_y, self._logstd_min, self._logstd_max)
        return mu_y, logstd_y

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def elbo(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute negative ELBO (loss to minimise).

        L = E_q[ -log p_θ(y|x,z) ] + β * KL(q || p)
        """
        mu_z, logvar_z = self.encode(x, y)
        z = self.reparameterize(mu_z, logvar_z)

        mu_y, logstd_y = self.decode(x, z)

        # Reconstruction: -log p(y | x, z)
        recon = _gaussian_nll(y, mu_y, logstd_y)
        kl = _kl_standard_normal(mu_z, logvar_z)

        loss = torch.mean(recon + beta * kl)
        stats = {
            "recon": float(torch.mean(recon).item()),
            "kl": float(torch.mean(kl).item()),
        }
        return loss, stats

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        """Predict future from past (inference mode).

        Parameters
        ----------
        x : (B, n) — observed past windows
        n_samples : int
            If 1, use z=0 (MAP of prior). If >1, sample z ~ N(0,I).

        Returns
        -------
        y_pred : (B, m) if n_samples=1, else (n_samples, B, m)
        """
        self.eval()
        B = x.shape[0]

        if n_samples == 1:
            z = torch.zeros(B, self.z_dim, device=x.device)
            mu_y, _ = self.decode(x, z)
            return mu_y
        else:
            preds = []
            for _ in range(n_samples):
                z = torch.randn(B, self.z_dim, device=x.device)
                mu_y, _ = self.decode(x, z)
                preds.append(mu_y)
            return torch.stack(preds, dim=0)  # (n_samples, B, m)


# ============================================================================
# 4. FORWARD MLP: ŷ = f_φ(x)  — DETERMINISTIC BASELINE
# ============================================================================

class ForwardMLP(nn.Module):
    """Deterministic forward predictor ŷ = f_φ(x).

    Simple MLP trained with MSE loss.  Same hidden dimension and depth
    as the CVAE decoders for fair capacity comparison.

    This is the minimal baseline: if the inverse paradigm cannot beat
    a simple MLP of the same capacity, the paradigm adds no value.
    """

    def __init__(self, n: int, m: int, cfg: ForwardMLPConfig):
        super().__init__()
        self.n = n
        self.m = m

        self.net = MLP(
            in_dim=n,
            out_dim=m,
            hidden=cfg.hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict future from past.

        Parameters
        ----------
        x : (B, n)

        Returns
        -------
        y_pred : (B, m)
        """
        return self.net(x)

    def mse_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MSE loss.

        Parameters
        ----------
        x : (B, n) — past windows
        y : (B, m) — true future windows

        Returns
        -------
        loss : scalar tensor
        stats : dict with 'mse' key
        """
        y_pred = self.forward(x)
        mse = F.mse_loss(y_pred, y)
        return mse, {"mse": float(mse.item())}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict future (inference mode)."""
        self.eval()
        return self.forward(x)


# ============================================================================
# 5. REALNVP NORMALIZING FLOW: p_φ(y)
# ============================================================================

class _CouplingNet(nn.Module):
    """Neural network for RealNVP coupling layer.

    Maps the masked half of the input to (scale, translation) parameters
    for the unmasked half.
    """

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * in_dim),  # outputs (s, t) for full dim
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (log_scale, translation), each of shape (B, in_dim)."""
        st = self.net(x)
        s, t = torch.chunk(st, 2, dim=-1)
        # Clamp scale for stability (tanh bounds to [-1, 1])
        s = torch.tanh(s)
        return s, t


class _RealNVPLayer(nn.Module):
    """Single affine coupling layer for RealNVP.

    Forward:  y_unmasked = x_unmasked * exp(s(x_masked)) + t(x_masked)
    Inverse:  x_unmasked = (y_unmasked - t(y_masked)) * exp(-s(y_masked))

    The mask determines which dimensions are "masked" (pass-through)
    and which are transformed.
    """

    def __init__(self, dim: int, mask: torch.Tensor, hidden: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask)
        self.coupling = _CouplingNet(in_dim=dim, hidden=hidden)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: x → y with log|det(dy/dx)|.

        Returns
        -------
        y : (B, dim)
        log_det : (B,)
        """
        x_masked = x * self.mask
        s, t = self.coupling(x_masked)

        # Only transform unmasked dimensions
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)

        y = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
        log_det = torch.sum(s, dim=-1)

        return y, log_det

    def inverse(
        self, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass: y → x with log|det(dx/dy)|.

        Returns
        -------
        x : (B, dim)
        log_det : (B,)  — negative of forward log_det
        """
        y_masked = y * self.mask
        s, t = self.coupling(y_masked)

        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)

        x = y_masked + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        log_det = -torch.sum(s, dim=-1)

        return x, log_det


class RealNVP(nn.Module):
    """RealNVP normalizing flow for learning the prior p_φ(y).

    Transforms a base distribution N(0, I) into a flexible learned
    distribution over future windows y.

    Provides:
      - log_prob(y): exact log-likelihood (for MAP inference prior term)
      - sample(n):   generate samples from the learned distribution
                     (for multi-start initialisation)

    Architecture: stack of affine coupling layers with alternating
    checkerboard masks.

    References
    ----------
    Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017).
    Density estimation using Real-NVP. ICLR 2017.
    """

    def __init__(self, dim: int, cfg: FlowPriorConfig):
        super().__init__()
        self.dim = dim

        layers: List[_RealNVPLayer] = []
        for k in range(cfg.n_layers):
            # Alternating checkerboard mask
            mask = torch.zeros(dim)
            mask[k % 2::2] = 1.0
            layers.append(_RealNVPLayer(dim=dim, mask=mask, hidden=cfg.hidden))
        self.layers = nn.ModuleList(layers)

    def forward_transform(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: data space → latent space.

        x (data) → z (base) with accumulated log|det|.
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer.forward(z)
            log_det_total = log_det_total + ld
        return z, log_det_total

    def inverse_transform(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse: latent space → data space.

        z (base) → x (data).
        """
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            log_det_total = log_det_total + ld
        return x, log_det_total

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute exact log p_φ(x) via change of variables.

        log p(x) = log p_base(f(x)) + log |det(df/dx)|

        where f is the forward transform and p_base = N(0, I).

        Parameters
        ----------
        x : (B, dim)

        Returns
        -------
        log_prob : (B,)
        """
        z, log_det = self.forward_transform(x)
        # Base distribution: standard normal
        log_base = -0.5 * torch.sum(
            z ** 2 + math.log(2.0 * math.pi), dim=-1
        )
        return log_base + log_det

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample from the learned distribution.

        Parameters
        ----------
        n : int — number of samples
        device : torch.device

        Returns
        -------
        x : (n, dim)
        """
        z = torch.randn(n, self.dim, device=device)
        x, _ = self.inverse_transform(z)
        return x


# ============================================================================
# 6. TRAINING UTILITIES
# ============================================================================

def train_cvae(
    model: InverseCVAE | ForwardCVAE,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    *,
    epochs: int,
    lr: float,
    beta: float,
    grad_clip: float,
    device: torch.device,
    verbose: bool = True,
    print_every: int = 10,
) -> Dict[str, List[float]]:
    """Train a CVAE (inverse or forward) and return training history.

    Parameters
    ----------
    model : InverseCVAE or ForwardCVAE
    train_loader, val_loader : DataLoaders
    epochs, lr, beta, grad_clip : training hyperparameters
    device : torch.device
    verbose : print progress
    print_every : print every N epochs

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'train_recon',
              'val_recon', 'train_kl', 'val_kl'
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "val_loss": [],
        "train_recon": [], "val_recon": [],
        "train_kl": [], "val_kl": [],
    }

    for ep in range(epochs):
        # --- Train ---
        model.train()
        batch_losses, batch_recon, batch_kl = [], [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss, stats = model.elbo(xb, yb, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            batch_losses.append(loss.item())
            batch_recon.append(stats["recon"])
            batch_kl.append(stats["kl"])

        train_loss = float(sum(batch_losses) / len(batch_losses))
        train_recon = float(sum(batch_recon) / len(batch_recon))
        train_kl = float(sum(batch_kl) / len(batch_kl))

        # --- Validate ---
        model.eval()
        val_losses, val_recon_list, val_kl_list = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss, stats = model.elbo(xb, yb, beta=beta)
                val_losses.append(loss.item())
                val_recon_list.append(stats["recon"])
                val_kl_list.append(stats["kl"])

        val_loss = float(sum(val_losses) / len(val_losses))
        val_recon = float(sum(val_recon_list) / len(val_recon_list))
        val_kl = float(sum(val_kl_list) / len(val_kl_list))

        # Record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_recon"].append(train_recon)
        history["val_recon"].append(val_recon)
        history["train_kl"].append(train_kl)
        history["val_kl"].append(val_kl)

        if verbose and ((ep + 1) % print_every == 0 or ep == 0):
            print(
                f"  [{type(model).__name__}] Epoch {ep+1:3d}/{epochs}: "
                f"train={train_loss:.4f} (recon={train_recon:.2f}, kl={train_kl:.2f})  "
                f"val={val_loss:.4f}"
            )

    return history


def train_mlp(
    model: ForwardMLP,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    *,
    epochs: int,
    lr: float,
    grad_clip: float,
    device: torch.device,
    verbose: bool = True,
    print_every: int = 10,
) -> Dict[str, List[float]]:
    """Train a ForwardMLP with MSE loss and return training history.

    Returns
    -------
    history : dict with keys 'train_mse', 'val_mse'
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_mse": [], "val_mse": []}

    for ep in range(epochs):
        # --- Train ---
        model.train()
        batch_mse = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss, stats = model.mse_loss(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            batch_mse.append(stats["mse"])

        train_mse = float(sum(batch_mse) / len(batch_mse))

        # --- Validate ---
        model.eval()
        val_mse_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, stats = model.mse_loss(xb, yb)
                val_mse_list.append(stats["mse"])

        val_mse = float(sum(val_mse_list) / len(val_mse_list))

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)

        if verbose and ((ep + 1) % print_every == 0 or ep == 0):
            print(
                f"  [ForwardMLP] Epoch {ep+1:3d}/{epochs}: "
                f"train_mse={train_mse:.6f}  val_mse={val_mse:.6f}"
            )

    return history


def train_flow(
    flow: RealNVP,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    cfg: FlowPriorConfig,
    device: torch.device,
    verbose: bool = True,
    print_every: int = 10,
) -> Dict[str, any]:
    """Train RealNVP flow prior with early stopping.

    Parameters
    ----------
    flow : RealNVP model
    y_train, y_val : numpy arrays of standardised future windows
    cfg : FlowPriorConfig
    device : torch.device

    Returns
    -------
    history : dict with 'train_nll', 'val_nll', 'best_epoch', 'best_val_nll'
    """
    import numpy as np
    from torch.utils.data import DataLoader

    flow.to(device)
    optimizer = torch.optim.Adam(
        flow.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    train_loader = DataLoader(
        torch.from_numpy(y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        torch.from_numpy(y_val),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    history = {"train_nll": [], "val_nll": []}
    best_val_nll = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for ep in range(cfg.epochs):
        # --- Train ---
        flow.train()
        batch_nll = []
        for yb in train_loader:
            yb = yb.to(device)
            nll = -flow.log_prob(yb).mean()

            optimizer.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), cfg.grad_clip)
            optimizer.step()

            batch_nll.append(nll.item())

        train_nll = float(sum(batch_nll) / len(batch_nll))

        # --- Validate ---
        flow.eval()
        val_nll_list = []
        with torch.no_grad():
            for yb in val_loader:
                yb = yb.to(device)
                val_nll_list.append((-flow.log_prob(yb).mean()).item())

        val_nll = float(sum(val_nll_list) / len(val_nll_list))

        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)

        # --- Early stopping ---
        if val_nll < best_val_nll - cfg.min_delta:
            best_val_nll = val_nll
            best_epoch = ep + 1
            best_state = {k: v.cpu().clone() for k, v in flow.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and ((ep + 1) % print_every == 0 or ep == 0):
            print(
                f"  [RealNVP] Epoch {ep+1:3d}/{cfg.epochs}: "
                f"train_nll={train_nll:.4f}  val_nll={val_nll:.4f}  "
                f"(best={best_val_nll:.4f} @ ep{best_epoch})"
            )

        if patience_counter >= cfg.patience:
            if verbose:
                print(
                    f"  [RealNVP] Early stopping at epoch {ep+1} "
                    f"(best val_nll={best_val_nll:.4f} @ epoch {best_epoch})"
                )
            break

    # Restore best model
    if best_state is not None:
        flow.load_state_dict(best_state)
        flow.to(device)

    history["best_epoch"] = best_epoch
    history["best_val_nll"] = best_val_nll

    return history


# ============================================================================
# 7. MODEL PARAMETER COUNTING
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(
    n: int, m: int,
    inv_cfg: InverseCVAEConfig,
    fwd_cfg: ForwardCVAEConfig,
    mlp_cfg: ForwardMLPConfig,
    flow_cfg: FlowPriorConfig,
) -> Dict[str, int]:
    """Create all models and report parameter counts.

    Verifies that all models have comparable capacity for fair comparison.
    """
    inv = InverseCVAE(n, m, inv_cfg)
    fwd = ForwardCVAE(n, m, fwd_cfg)
    mlp = ForwardMLP(n, m, mlp_cfg)
    flow = RealNVP(m, flow_cfg)

    return {
        "InverseCVAE": count_parameters(inv),
        "ForwardCVAE": count_parameters(fwd),
        "ForwardMLP": count_parameters(mlp),
        "RealNVP": count_parameters(flow),
    }


# ============================================================================
# 8. SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import numpy as np
    from config import get_default_config

    cfg = get_default_config()
    n, m = cfg.experiment.n, cfg.experiment.m

    print("=" * 60)
    print("Retrodictive Forecasting — Model Architecture Summary")
    print("=" * 60)

    # Parameter counts
    params = model_summary(
        n, m,
        cfg.inverse_cvae, cfg.forward_cvae,
        cfg.forward_mlp, cfg.flow_prior,
    )
    print(f"\nWindow sizes: n={n} (past), m={m} (future)")
    print(f"\nTrainable parameters:")
    for name, count in params.items():
        print(f"  {name:<16}: {count:>8,d}")

    # Quick shape checks with dummy data
    B = 4
    x = torch.randn(B, n)
    y = torch.randn(B, m)
    z = torch.randn(B, cfg.inverse_cvae.z_dim)

    print(f"\n--- InverseCVAE shape checks ---")
    inv = InverseCVAE(n, m, cfg.inverse_cvae)
    mu_z, logvar_z = inv.encode(x, y)
    print(f"  encode(x,y) -> mu_z: {mu_z.shape}, logvar_z: {logvar_z.shape}")
    mu_x, logstd_x = inv.decode(y, z)
    print(f"  decode(y,z) -> mu_x: {mu_x.shape}, logstd_x: {logstd_x.shape}")
    loss, stats = inv.elbo(x, y)
    print(f"  elbo -> loss: {loss.item():.4f}, recon: {stats['recon']:.4f}, kl: {stats['kl']:.4f}")
    lp = inv.log_prob_x_given_yz(x, y, z)
    print(f"  log_prob_x_given_yz -> shape: {lp.shape}")

    print(f"\n--- ForwardCVAE shape checks ---")
    fwd = ForwardCVAE(n, m, cfg.forward_cvae)
    loss, stats = fwd.elbo(x, y)
    print(f"  elbo -> loss: {loss.item():.4f}, recon: {stats['recon']:.4f}, kl: {stats['kl']:.4f}")
    y_pred = fwd.predict(x)
    print(f"  predict(x) -> shape: {y_pred.shape}")
    y_samples = fwd.predict(x, n_samples=5)
    print(f"  predict(x, n_samples=5) -> shape: {y_samples.shape}")

    print(f"\n--- ForwardMLP shape checks ---")
    mlp = ForwardMLP(n, m, cfg.forward_mlp)
    y_pred = mlp.predict(x)
    print(f"  predict(x) -> shape: {y_pred.shape}")
    loss, stats = mlp.mse_loss(x, y)
    print(f"  mse_loss -> loss: {loss.item():.6f}")

    print(f"\n--- RealNVP shape checks ---")
    flow = RealNVP(m, cfg.flow_prior)
    lp = flow.log_prob(y)
    print(f"  log_prob(y) -> shape: {lp.shape}")
    samples = flow.sample(8, device=torch.device("cpu"))
    print(f"  sample(8) -> shape: {samples.shape}")

    print("\n All shape checks passed.")
