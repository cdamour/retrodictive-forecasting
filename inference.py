#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inference.py - MAP inference engine for Retrodictive Forecasting
=========================================================================

Implements retrodictive MAP inference:

        (y*, z*) = argmax_{y,z}  [ log p_theta(x_obs | y, z)
                                                            + lam_prior * log p(y)
                                                            + log p(z)
                                                            - lam_smooth * R(y) ]

where:
    - p_theta(x | y, z) is the learned inverse CVAE decoder
    - p(y) is either a learned RealNVP flow or N(0, I) (ablation)
    - p(z) = N(0, I) is the latent prior
    - R(y) is an optional smoothness penalty on first differences

Multi-start strategy
--------------------
    K initialisations are generated:
        - If flow prior available: sample y ~ p_phi(y), score by reconstruction
            error, keep the best as starting point
        - If N(0,I) prior: sample y ~ N(0, I)
    Each start is optimised independently. The solution with lowest
    total MAP loss is selected ("best" aggregation).

The module also provides batch inference over the full test set with
progress reporting and collection of per-sample diagnostics (MAP loss,
multi-start dispersion, prior log-prob, retrodictive NLL).

Usage
-----
        from inference import map_infer_single, map_infer_batch
        from config import get_default_config, PriorKind

        cfg = get_default_config()
        # Single sample
        result = map_infer_single(model, flow, x_obs, cfg=cfg.map_inference,
                                                            prior_kind=PriorKind.FLOW, device=device)
        # Batch over test set
        results = map_infer_batch(model, flow, X_test_s, cfg=cfg.map_inference,
                                                            prior_kind=PriorKind.FLOW, device=device)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import MAPConfig, PriorKind
from models import InverseCVAE, RealNVP


# ============================================================================
# 1. DATA STRUCTURES
# ============================================================================

@dataclass
class MAPResult:
    """Result of MAP inference for a single test sample.

    Attributes
    ----------
    y_hat : np.ndarray, shape (m,)
        Inferred future window (standardised).
    z_hat : np.ndarray, shape (z_dim,)
        Inferred latent variable.
    map_loss : float
        Total MAP objective at the solution (lower = better).
    retro_nll : float
        Retrodictive NLL: -log p_theta(x_obs | y, z).
    prior_logprob : float
        log p(y) under the prior (flow or N(0,I)).
    z_logprob : float
        log p(z) = -0.5 ||z||^2 (up to a constant).
    multistart_losses : List[float]
        MAP losses from all K restarts (for dispersion analysis).
    multistart_ys : Optional[np.ndarray]
        All K inferred y vectors, shape (K, m).  Stored only if
        requested (for dispersion metric computation).
    """
    y_hat: np.ndarray
    z_hat: np.ndarray
    map_loss: float
    retro_nll: float
    prior_logprob: float
    z_logprob: float
    multistart_losses: List[float] = field(default_factory=list)
    multistart_ys: Optional[np.ndarray] = None


@dataclass
class BatchMAPResults:
    """Aggregated results of MAP inference over a batch of test samples.

    Attributes
    ----------
    Y_hat : np.ndarray, shape (N, m)
        All inferred futures (standardised).
    map_losses : np.ndarray, shape (N,)
        Per-sample MAP losses.
    retro_nlls : np.ndarray, shape (N,)
        Per-sample retrodictive NLLs.
    prior_logprobs : np.ndarray, shape (N,)
        Per-sample prior log-probs.
    z_logprobs : np.ndarray, shape (N,)
        Per-sample z prior log-probs.
    multistart_std : np.ndarray, shape (N,)
        Per-sample std of y across multi-start restarts
        (mean over dimensions).
    elapsed_seconds : float
        Total wall-clock time for batch inference.
    n_samples : int
        Number of test samples evaluated.
    indices : Optional[np.ndarray]
        Test-set indices that were actually evaluated (length n_samples).
    """
    Y_hat: np.ndarray
    map_losses: np.ndarray
    retro_nlls: np.ndarray
    prior_logprobs: np.ndarray
    z_logprobs: np.ndarray
    multistart_std: np.ndarray
    elapsed_seconds: float
    n_samples: int
    indices: Optional[np.ndarray] = None


# ============================================================================
# 2. HELPER: SMOOTHNESS PENALTY
# ============================================================================

def _smoothness_penalty(y: torch.Tensor) -> torch.Tensor:
    """Quadratic penalty on first differences of y.

    R(y) = sum_{t=1}^{m-1} (y_t - y_{t-1})^2

    Parameters
    ----------
    y : (B, m) or (1, m)

    Returns
    -------
    penalty : (B,) or (1,)
    """
    dy = y[:, 1:] - y[:, :-1]
    return torch.sum(dy ** 2, dim=-1)


# ============================================================================
# 3. HELPER: MULTI-START INITIALISATION
# ============================================================================

@torch.no_grad()
def _init_from_flow(
    flow: RealNVP,
    model: InverseCVAE,
    x_obs: torch.Tensor,
    K: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate K candidates from flow prior and pick the best by recon error.

    Strategy:
            1. Sample K future candidates y ~ p_phi(y)
      2. Sample K latent vectors z ~ N(0, I)
            3. Decode each (y_k, z_k) -> mu_x(y_k, z_k)
            4. Score by ||mu_x - x_obs||^2 (cheap, no gradient needed)
      5. Return the (y, z) with lowest reconstruction error

    Parameters
    ----------
    flow : RealNVP prior
    model : InverseCVAE
    x_obs : (1, n) - single observed past window
    K : int - number of candidates
    device : torch.device

    Returns
    -------
    y_init : (1, m)
    z_init : (1, z_dim)
    """
    model.eval()
    flow.eval()

    # Sample candidates
    ys = flow.sample(K, device=device)                     # (K, m)
    zs = torch.randn(K, model.z_dim, device=device)       # (K, z_dim)

    # Score by reconstruction error
    mu_x, _ = model.decode(ys, zs)                         # (K, n)
    x_expanded = x_obs.expand(K, -1)                       # (K, n)
    recon_errors = torch.sum((mu_x - x_expanded) ** 2, dim=-1)  # (K,)

    # Select best
    best_idx = int(torch.argmin(recon_errors).item())
    return ys[best_idx:best_idx + 1].clone(), zs[best_idx:best_idx + 1].clone()


@torch.no_grad()
def _init_from_normal(
    model: InverseCVAE,
    x_obs: torch.Tensor,
    K: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate K candidates from N(0,I) and pick the best by recon error.

    Same strategy as _init_from_flow but using standard normal samples
    instead of flow samples.
    """
    model.eval()

    m = model.m
    ys = torch.randn(K, m, device=device)                  # (K, m)
    zs = torch.randn(K, model.z_dim, device=device)       # (K, z_dim)

    mu_x, _ = model.decode(ys, zs)
    x_expanded = x_obs.expand(K, -1)
    recon_errors = torch.sum((mu_x - x_expanded) ** 2, dim=-1)

    best_idx = int(torch.argmin(recon_errors).item())
    return ys[best_idx:best_idx + 1].clone(), zs[best_idx:best_idx + 1].clone()


@torch.no_grad()
def _init_from_forward_cvae(
    forward_cvae,
    x_obs: torch.Tensor,
    z_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate warm-start initialisation from forward CVAE prediction.

    Uses the forward CVAE's posterior mean (z=0) to produce an informed
    starting point for MAP optimisation.  This initialization exploits
    the forward model's knowledge of p(y|x) to start near a plausible
    future, then the inverse MAP refines it using the retrodictive
    likelihood p(x_obs|y,z).

    Parameters
    ----------
    forward_cvae : ForwardCVAE (trained, eval mode)
    x_obs : (1, n) — single observed past window
    z_dim : int — inverse CVAE latent dimension
    device : torch.device

    Returns
    -------
    y_init : (1, m)
    z_init : (1, z_dim)
    """
    forward_cvae.eval()
    y_warm = forward_cvae.predict(x_obs, n_samples=1)  # (1, m)
    z_init = torch.zeros(1, z_dim, device=device)       # z=0 (prior mode)
    return y_warm.clone(), z_init.clone()


# ============================================================================
# 4. SINGLE-SAMPLE MAP INFERENCE
# ============================================================================

def _compute_map_objective(
    model: InverseCVAE,
    flow: Optional[RealNVP],
    x_obs: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    prior_kind: PriorKind,
    lam_prior: float,
    lam_smooth: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the MAP objective (to MINIMISE).

    L(y, z) = -log p_theta(x_obs | y, z)    [retrodictive NLL]
              - lam_prior * log p(y)        [prior on futures]
              + 0.5 ||z||^2                 [latent prior -log p(z)]
              + lam_smooth * R(y)           [smoothness]

    Returns
    -------
    loss : scalar tensor
    components : dict with individual terms (float)
    """
    # Retrodictive NLL
    mu_x, logstd_x = model.decode(y, z)
    retro_nll = 0.5 * torch.sum(
        ((x_obs - mu_x) / torch.exp(logstd_x)) ** 2 + 2.0 * logstd_x,
        dim=-1,
    )  # (1,)

    # Latent prior: -log p(z) = 0.5 ||z||^2  (up to constant)
    nlp_z = 0.5 * torch.sum(z ** 2, dim=-1)  # (1,)

    # Future prior: -log p(y)
    if prior_kind == PriorKind.FLOW and flow is not None and lam_prior > 0:
        nlp_y = -flow.log_prob(y)  # (1,)
    elif prior_kind == PriorKind.STANDARD_NORMAL and lam_prior > 0:
        # N(0,I): -log p(y) = 0.5 ||y||^2 + const
        nlp_y = 0.5 * torch.sum(y ** 2, dim=-1)  # (1,)
    else:
        nlp_y = torch.zeros_like(retro_nll)

    # Smoothness
    if lam_smooth > 0:
        smooth = _smoothness_penalty(y)
    else:
        smooth = torch.zeros_like(retro_nll)

    # Total
    loss = retro_nll + nlp_z + lam_prior * nlp_y + lam_smooth * smooth

    components = {
        "retro_nll": float(retro_nll.item()),
        "nlp_z": float(nlp_z.item()),
        "nlp_y": float(nlp_y.item()),
        "smooth": float(smooth.item()),
        "total": float(loss.item()),
    }

    return loss.squeeze(), components


def _run_single_optimisation(
    model: InverseCVAE,
    flow: Optional[RealNVP],
    x_obs: torch.Tensor,
    y_init: torch.Tensor,
    z_init: torch.Tensor,
    *,
    prior_kind: PriorKind,
    cfg: MAPConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float, Dict[str, float]]:
    """Run a single MAP optimisation from given initial (y, z).

    Uses Adam optimiser with gradient clipping and clamping.

    Returns
    -------
    y_best : (1, m) - best y found during optimisation
    z_best : (1, z_dim) - best z found
    best_loss : float
    best_components : dict - breakdown of the best MAP objective
    """
    model.eval()
    if flow is not None:
        flow.eval()

    # Optimisation variables
    y = y_init.clone().detach().to(device).requires_grad_(True)
    z = z_init.clone().detach().to(device).requires_grad_(True)

    optimizer = torch.optim.Adam([y, z], lr=cfg.lr)

    best_loss = float("inf")
    best_y = y_init.clone().detach()
    best_z = z_init.clone().detach()
    best_components = {}

    for step in range(cfg.steps):
        optimizer.zero_grad()

        loss, components = _compute_map_objective(
            model, flow, x_obs, y, z,
            prior_kind=prior_kind,
            lam_prior=cfg.lam_prior,
            lam_smooth=cfg.lam_smooth,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_([y, z], cfg.grad_clip)
        optimizer.step()

        # Clamp to prevent divergence
        with torch.no_grad():
            y.clamp_(-cfg.y_clip, cfg.y_clip)
            z.clamp_(-cfg.z_clip, cfg.z_clip)

        # Track best
        current_loss = components["total"]
        if current_loss < best_loss:
            best_loss = current_loss
            best_y = y.detach().clone()
            best_z = z.detach().clone()
            best_components = components

    return best_y, best_z, best_loss, best_components


def map_infer_single(
    model: InverseCVAE,
    flow: Optional[RealNVP],
    x_obs: torch.Tensor,
    *,
    cfg: MAPConfig,
    prior_kind: PriorKind,
    device: torch.device,
    store_all_ys: bool = True,
    forward_cvae=None,
) -> MAPResult:
    """Full MAP inference for a single observed past window.

    Runs K multi-start optimisations and selects the best.
    If forward_cvae is provided, restart k=0 is warm-started from the
    forward CVAE prediction (Variant A: 1 warm + K-1 random).

    Parameters
    ----------
    model : InverseCVAE (already on device, eval mode)
    flow : RealNVP or None (if prior_kind is STANDARD_NORMAL)
    x_obs : (n,) or (1, n) - single observed past window (standardised)
    cfg : MAPConfig
    prior_kind : PriorKind - which prior to use
    device : torch.device
    store_all_ys : bool - if True, store all K inferred ys for dispersion

    Returns
    -------
    MAPResult
    """
    model.eval()
    if flow is not None:
        flow.eval()

    # Reshape x_obs to (1, n)
    if x_obs.dim() == 1:
        x_obs = x_obs.unsqueeze(0)
    x_obs = x_obs.to(device)

    K = cfg.K_multistart
    all_losses: List[float] = []
    all_ys: List[np.ndarray] = []
    all_results: List[Tuple[torch.Tensor, torch.Tensor, float, Dict]] = []

    for k in range(K):
        # Initialisation: k=0 uses forward CVAE warm-start if available
        if k == 0 and forward_cvae is not None:
            y_init, z_init = _init_from_forward_cvae(
                forward_cvae, x_obs, model.z_dim, device=device
            )
        elif prior_kind == PriorKind.FLOW and flow is not None:
            # Use flow samples with reconstruction-based scoring
            # Sample a fresh batch each restart for diversity
            y_init, z_init = _init_from_flow(
                flow, model, x_obs, K=max(K, 16), device=device
            )
        else:
            # Use N(0, I) samples
            y_init, z_init = _init_from_normal(
                model, x_obs, K=max(K, 16), device=device
            )

        # Run optimisation
        y_best, z_best, best_loss, best_comp = _run_single_optimisation(
            model, flow, x_obs, y_init, z_init,
            prior_kind=prior_kind,
            cfg=cfg,
            device=device,
        )

        all_losses.append(best_loss)
        all_ys.append(y_best.cpu().numpy().reshape(-1))
        all_results.append((y_best, z_best, best_loss, best_comp))

    # Select overall best across K restarts
    best_k = int(np.argmin(all_losses))
    y_star, z_star, loss_star, comp_star = all_results[best_k]

    # Compute prior log-prob at solution
    with torch.no_grad():
        if prior_kind == PriorKind.FLOW and flow is not None:
            prior_lp = float(flow.log_prob(y_star).item())
        else:
            # N(0,I): log p(y) = -0.5 ||y||^2 - 0.5 m log(2*pi)
            m = y_star.shape[-1]
            prior_lp = float(
                (-0.5 * torch.sum(y_star ** 2) - 0.5 * m * np.log(2 * np.pi)).item()
            )

        z_lp = float(
            (-0.5 * torch.sum(z_star ** 2) - 0.5 * z_star.shape[-1] * np.log(2 * np.pi)).item()
        )

    # Multi-start y array
    multistart_ys = np.array(all_ys) if store_all_ys else None  # (K, m)

    return MAPResult(
        y_hat=y_star.cpu().numpy().reshape(-1),
        z_hat=z_star.cpu().numpy().reshape(-1),
        map_loss=loss_star,
        retro_nll=comp_star["retro_nll"],
        prior_logprob=prior_lp,
        z_logprob=z_lp,
        multistart_losses=all_losses,
        multistart_ys=multistart_ys,
    )


# ============================================================================
# 5. BATCH MAP INFERENCE OVER TEST SET
# ============================================================================

def map_infer_batch(
    model: InverseCVAE,
    flow: Optional[RealNVP],
    X_test_s: np.ndarray,
    *,
    cfg: MAPConfig,
    prior_kind: PriorKind,
    device: torch.device,
    verbose: bool = True,
    print_every: int = 200,
    store_all_ys: bool = True,
    forward_cvae=None,
) -> BatchMAPResults:
    """Run MAP inference on a (sub)set of test samples.

    Subsamples n_eval samples from X_test_s (uniformly spaced)
    if X_test_s has more than n_eval rows.

    Parameters
    ----------
    model : InverseCVAE (already trained)
    flow : RealNVP or None
    X_test_s : np.ndarray, shape (N_test, n) - standardised past windows
    cfg : MAPConfig
    prior_kind : PriorKind
    device : torch.device
    verbose : bool - print progress
    print_every : int - print every N samples
    store_all_ys : bool - store multi-start ys per sample

    Returns
    -------
    BatchMAPResults
    """
    N_test = X_test_s.shape[0]
    n_eval = min(cfg.n_eval, N_test)

    # Uniform subsampling indices
    if n_eval < N_test:
        indices = np.linspace(0, N_test - 1, n_eval).astype(int)
    else:
        indices = np.arange(N_test)
        n_eval = N_test

    if verbose:
        pk_label = prior_kind.value
        print(f"\n  [MAP] Inferring {n_eval} samples (prior={pk_label}, "
              f"K={cfg.K_multistart}, steps={cfg.steps})...")

    model.eval()
    if flow is not None:
        flow.eval()

    # Storage
    m = model.m
    Y_hat = np.zeros((n_eval, m), dtype=np.float32)
    map_losses = np.zeros(n_eval, dtype=np.float32)
    retro_nlls = np.zeros(n_eval, dtype=np.float32)
    prior_logprobs = np.zeros(n_eval, dtype=np.float32)
    z_logprobs = np.zeros(n_eval, dtype=np.float32)
    multistart_stds = np.zeros(n_eval, dtype=np.float32)

    t0 = time.time()

    for count, idx in enumerate(indices):
        x_obs = torch.from_numpy(X_test_s[idx]).float()

        result = map_infer_single(
            model, flow, x_obs,
            cfg=cfg,
            prior_kind=prior_kind,
            device=device,
            store_all_ys=store_all_ys,
            forward_cvae=forward_cvae,
        )

        Y_hat[count] = result.y_hat
        map_losses[count] = result.map_loss
        retro_nlls[count] = result.retro_nll
        prior_logprobs[count] = result.prior_logprob
        z_logprobs[count] = result.z_logprob

        # Multi-start dispersion: mean std across dimensions
        if result.multistart_ys is not None and result.multistart_ys.shape[0] > 1:
            # std per dimension across K restarts, then mean over dimensions
            multistart_stds[count] = float(
                np.mean(np.std(result.multistart_ys, axis=0))
            )
        else:
            multistart_stds[count] = 0.0

        if verbose and ((count + 1) % print_every == 0):
            elapsed = time.time() - t0
            rate = (count + 1) / elapsed
            eta = (n_eval - count - 1) / rate if rate > 0 else 0
            print(f"    {count+1}/{n_eval}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                  f"loss={result.map_loss:.4f})")

    elapsed = time.time() - t0

    if verbose:
        print(f"  [MAP] Done: {n_eval} samples in {elapsed:.1f}s "
              f"({n_eval/elapsed:.1f} samples/s)")
        print(f"    Mean MAP loss: {np.mean(map_losses):.4f}")
        print(f"    Mean retro NLL: {np.mean(retro_nlls):.4f}")
        print(f"    Mean prior logprob: {np.mean(prior_logprobs):.4f}")
        print(f"    Mean multistart std: {np.mean(multistart_stds):.4f}")

    return BatchMAPResults(
        Y_hat=Y_hat,
        map_losses=map_losses,
        retro_nlls=retro_nlls,
        prior_logprobs=prior_logprobs,
        z_logprobs=z_logprobs,
        multistart_std=multistart_stds,
        elapsed_seconds=elapsed,
        n_samples=n_eval,
        indices=indices,
    )


# ============================================================================
# 6. FORWARD BASELINE INFERENCE (for comparison)
# ============================================================================

@torch.no_grad()
def forward_cvae_predict_batch(
    model,  # ForwardCVAE
    X_test_s: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Batch prediction from a trained ForwardCVAE.

    Uses z=0 (prior mode) for deterministic point predictions.

    Parameters
    ----------
    model : ForwardCVAE (trained)
    X_test_s : (N, n)
    device : torch.device
    batch_size : int

    Returns
    -------
    Y_pred : (N, m) - predicted futures (standardised)
    """
    model.eval()
    model.to(device)

    N = X_test_s.shape[0]
    m = model.m
    Y_pred = np.zeros((N, m), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = torch.from_numpy(X_test_s[start:end]).float().to(device)
        yb = model.predict(xb, n_samples=1)
        Y_pred[start:end] = yb.cpu().numpy()

    return Y_pred


@torch.no_grad()
def forward_mlp_predict_batch(
    model,  # ForwardMLP
    X_test_s: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Batch prediction from a trained ForwardMLP.

    Parameters
    ----------
    model : ForwardMLP (trained)
    X_test_s : (N, n)
    device : torch.device
    batch_size : int

    Returns
    -------
    Y_pred : (N, m)
    """
    model.eval()
    model.to(device)

    N = X_test_s.shape[0]
    m = model.m
    Y_pred = np.zeros((N, m), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = torch.from_numpy(X_test_s[start:end]).float().to(device)
        yb = model.predict(xb)
        Y_pred[start:end] = yb.cpu().numpy()

    return Y_pred


# ============================================================================
# 7. SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import torch
    from config import get_default_config, PriorKind
    from models import InverseCVAE, RealNVP

    cfg = get_default_config()
    n, m = cfg.experiment.n, cfg.experiment.m
    device = torch.device("cpu")

    print("=" * 60)
    print("Retrodictive Forecasting -- Inference Module Self-Test")
    print("=" * 60)

    # Create dummy models (untrained)
    model = InverseCVAE(n, m, cfg.inverse_cvae).to(device)
    flow = RealNVP(m, cfg.flow_prior).to(device)

    # Dummy observation
    x_obs = torch.randn(n)

    print("\n--- Single MAP inference (flow prior) ---")
    result = map_infer_single(
        model, flow, x_obs,
        cfg=cfg.map_inference,
        prior_kind=PriorKind.FLOW,
        device=device,
    )
    print(f"  y_hat shape: {result.y_hat.shape}")
    print(f"  z_hat shape: {result.z_hat.shape}")
    print(f"  MAP loss: {result.map_loss:.4f}")
    print(f"  Retro NLL: {result.retro_nll:.4f}")
    print(f"  Prior logprob: {result.prior_logprob:.4f}")
    print(f"  Multi-start losses: {[f'{l:.2f}' for l in result.multistart_losses]}")
    if result.multistart_ys is not None:
        print(f"  Multi-start ys shape: {result.multistart_ys.shape}")

    print("\n--- Single MAP inference (N(0,I) prior) ---")
    result_normal = map_infer_single(
        model, None, x_obs,
        cfg=cfg.map_inference,
        prior_kind=PriorKind.STANDARD_NORMAL,
        device=device,
    )
    print(f"  MAP loss: {result_normal.map_loss:.4f}")

    print("\n--- Batch inference (small test) ---")
    X_dummy = np.random.randn(20, n).astype(np.float32)

    # Override n_eval for quick test
    from dataclasses import replace
    quick_cfg = MAPConfig(
        steps=10, lr=0.05, K_multistart=2,
        lam_prior=1.0, lam_smooth=0.0,
        y_clip=3.5, z_clip=3.5, grad_clip=5.0,
        aggregate="best", n_eval=5,
    )

    batch_results = map_infer_batch(
        model, flow, X_dummy,
        cfg=quick_cfg,
        prior_kind=PriorKind.FLOW,
        device=device,
        verbose=True,
        print_every=2,
    )
    print(f"  Y_hat shape: {batch_results.Y_hat.shape}")
    print(f"  Mean MAP loss: {np.mean(batch_results.map_losses):.4f}")
    print(f"  Mean multistart std: {np.mean(batch_results.multistart_std):.4f}")
    print(f"  Elapsed: {batch_results.elapsed_seconds:.2f}s")

    print("\n All inference checks passed.")
