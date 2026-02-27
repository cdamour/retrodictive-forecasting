#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py — Central configuration for the Retrodictive Forecasting PoC
=======================================================================

Defines:
    - 4 synthetic cases A–D (2 GO + 2 NO-GO) with full parametrisation
    - 2 : real-data case ERA5 (GO) configured via NetCDF path and ERA_ssrd (GO) configured via NetCDF path + daylight filter
  - Global experimental parameters (window sizes, splits, seeds)
  - Model hyperparameters (Inverse CVAE, Forward CVAE, Forward MLP, RealNVP)
  - MAP inference settings
  - Arrow-of-time diagnostic settings
  - Ablation configurations (flow prior vs N(0,I))
  - 4 falsifiable predictions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


# ============================================================================
# 1. SYNTHETIC CASE DEFINITIONS
# ============================================================================

class Verdict(Enum):
    """Expected arrow-of-time diagnostic outcome."""
    GO = "GO"
    NOGO = "NO-GO"


@dataclass(frozen=True)
class CaseSpec:
    """Full specification of a synthetic process for the PoC suite.

    Each case is designed to probe a specific aspect of the retrodictive
    paradigm.  GO cases should exhibit strong temporal asymmetry
    (Δ_arrow >> 0) and benefit from inverse inference.  NO-GO cases
    should be time-reversible (Δ_arrow ≈ 0) and show no advantage.

    Attributes
    ----------
    key : str
        Short identifier (A–D), used as CLI argument.
    name : str
        Descriptive name used in file outputs and figures.
    verdict : Verdict
        Expected GO / NO-GO outcome.
    description : str
        One-line description for logs and tables.
    analogy : str
        Cross-disciplinary analogy (for the paper narrative).
    irreversibility_source : str
        Mechanism creating (or not) temporal asymmetry.
    params : dict
        Process-specific parameters passed to the generator.
    """
    key: str
    name: str
    verdict: Verdict
    description: str
    analogy: str
    irreversibility_source: str
    params: dict


CASES: Dict[str, CaseSpec] = {

    # ------------------------------------------------------------------
    # Case A — GO: Dissipative nonlinear autoregressive with
    #          state-dependent (multiplicative) noise.
    #
    # s_t = α tanh(s_{t-1}) + γ s_{t-1}² + σ(s_{t-1}) ε_t
    # σ(s) = σ_0 + σ_1 |s|
    # ε_t ~ N(0,1)  (symmetric innovations!)
    #
    # Irreversibility arises from:
    #   (i)  tanh contraction (large values compressed, small not)
    #   (ii) multiplicative noise (volatility tied to amplitude)
    # NOT from asymmetric noise distribution.
    # ------------------------------------------------------------------
    "A": CaseSpec(
        key="A",
        name="GO_dissipative_nlar",
        verdict=Verdict.GO,
        description="Nonlinear tanh AR(1) with strong multiplicative noise + cubic dissipation",
        analogy="Dissipative physical system (thermostat, local atmospheric dynamics)",
        irreversibility_source="Nonlinear contraction (tanh) + strong multiplicative noise + cubic dissipation",
        params=dict(
            alpha=0.7,       # tanh coefficient (contraction strength)
            gamma_quad=0.05, # quadratic drift term
            gamma_cubic=-0.08, # cubic dissipation (asymmetric restoring for large |s|)
            sigma_0=0.3,     # baseline noise level
            sigma_1=0.35,    # state-dependent noise amplification
        ),
    ),

    # ------------------------------------------------------------------
    # Case B — NO-GO: Pure symmetric random walk.
    #
    # s_t = s_{t-1} + σ ε_t
    # ε_t ~ N(0,1)
    #
    # Time-reversible: increments are i.i.d., so
    #   p(Δs_1,...,Δs_n) = p(Δs_n,...,Δs_1)
    # The inverse paradigm has NO structural advantage.
    # ------------------------------------------------------------------
    "B": CaseSpec(
        key="B",
        name="NOGO_rw_symmetric",
        verdict=Verdict.NOGO,
        description="Pure symmetric random walk (i.i.d. Gaussian increments)",
        analogy="Efficient market hypothesis, pure diffusion",
        irreversibility_source="None — process is time-reversible by construction",
        params=dict(
            sigma=0.5,       # increment standard deviation
        ),
    ),

    # ------------------------------------------------------------------
    # Case C — GO: Shot noise with exponential relaxation.
    #
    # s_t = λ s_{t-1} + A_t
    # A_t = J_t with prob p_shot, else 0
    # J_t ~ Exp(μ_J)
    # Observed: s̃_t = s_t + σ_obs ε_t,  ε_t ~ N(0,1)
    #
    # Strongly irreversible: forward = sharp rises + slow decay;
    # time-reversed = slow exponential rise + sharp drops (unphysical).
    # Direct connection to thermodynamic fluctuation theorems
    # (Kawai–Parrondo–Van den Broeck, PRL 2007).
    # ------------------------------------------------------------------
    "C": CaseSpec(
        key="C",
        name="GO_shotnoise_relaxation",
        verdict=Verdict.GO,
        description="Shot noise with exponential relaxation + symmetric obs. noise",
        analogy="Cloud passages (solar irradiance), neural spikes, equipment faults",
        irreversibility_source="Causal excitation-relaxation (energy dissipation)",
        params=dict(
            decay=0.95,      # exponential decay factor λ (retention per timestep)
            shot_prob=0.04,  # probability of a shot event at each timestep
            shot_scale=1.5,  # scale parameter of the exponential shot amplitude
            sigma_obs=0.05,  # symmetric observation noise std
        ),
    ),

    # ------------------------------------------------------------------
    # Case D — NO-GO: Pure sinusoid with symmetric i.i.d. noise.
    #
    # s_t = A sin(2π t / P) + σ ε_t,   ε_t ~ N(0,1)
    #
    # Time-reversible: sin is odd, so the time-reversed series is just
    # a phase-shifted sinusoid + symmetric noise.  The forward direction
    # is equally informative as the backward direction.
    #
    # σ = 0.50 (calibrated so that arrow-of-time diagnostic returns
    # NO-GO; at σ = 0.25 the periodicity at large window lengths
    # created a false-positive GO at LEVEL representation).
    # ------------------------------------------------------------------
    "D": CaseSpec(
        key="D",
        name="NOGO_sinusoid_symmetric",
        verdict=Verdict.NOGO,
        description="Pure sinusoid with symmetric i.i.d. Gaussian noise",
        analogy="Periodic mechanical oscillation, idealised seasonal cycle",
        irreversibility_source="None — deterministic component is time-reversible, noise is symmetric",
        params=dict(
            amplitude=1.0,   # sinusoid amplitude
            period=40,       # sinusoid period (in timesteps)
            sigma=0.50,      # observation noise std (calibrated: NO-GO at LEVEL)
        ),
    ),

    # ------------------------------------------------------------------
    # Case ERA5 — GO: Real-world ERA5 box-mean 10m wind speed (u10/v10).
    #
    # This is not a synthetic generator. The time series is loaded from
    # a NetCDF file (already present in the repo by default).
    # ------------------------------------------------------------------
    "ERA5": CaseSpec(
        key="ERA5",
        name="ERA5_w10_2023_northsea",
        verdict=Verdict.GO,
        description="ERA5 10m wind speed (box-mean), 2023 North Sea (u10/v10)",
        analogy="Real atmospheric dynamics (reanalysis)",
        irreversibility_source="Driven-dissipative geophysical dynamics (weather)",
        params=dict(
            # Default example file shipped with this workspace
            nc_path="era5_wind_2023_northsea_56N_3E/88b77ea627f174bbc75547e3c9040acb.nc",
            year=2023,
            lat=56.0,
            lon=3.0,
            halfspan_deg=0.25,
            series="w10_boxmean",
        ),
    ),
    # ------------------------------------------------------------------
    # Case ERA_ssrd — GO: Real-world ERA5 box-mean solar irradiance from SSRD.
    #
    # - Loads `ssrd` from a NetCDF file (hourly single levels).
    # - Converts accumulated energy to mean hourly flux: W/m² = ssrd / 3600.
    # - Applies a conservative daylight filter for North Sea (~54°N): 06–20 UTC.
    # - Intended as a real-data GO case with strong diurnal structure and
    #   physically constrained support (non-negative, daytime-only series).
    # ------------------------------------------------------------------
    "ERA_ssrd": CaseSpec(
        key="ERA_ssrd",
        name="ERA5_ssrd_2023_northsea_daylight",
        verdict=Verdict.GO,
        description="ERA5 SSRD-derived irradiance (W/m², box-mean), daylight-only, 2023 North Sea",
        analogy="Solar irradiance forecasting under cloud variability (reanalysis proxy)",
        irreversibility_source="Driven atmospheric radiative forcing + cloud intermittency; constrained support (daylight-only)",
        params=dict(
            # User-provided NetCDF path (downloaded from CDS via browser)
            nc_path="fb90d84b3f9f55aea635f4d4ad93ebc3.nc",
            year=2023,
            lat=54.0,              # indicative center (metadata only; used in the paper text)
            lon=5.0,
            halfspan_deg=0.25,      # metadata only (box already applied when downloading)
            series="ssrd_boxmean_wm2_daylight",
            # Daylight filtering policy (conservative and fully reproducible)
            daylight_mode="utc_window",  # {"utc_window","threshold","none"}
            utc_start=6,
            utc_end=20,
            # Used only if daylight_mode="threshold"
            thr_wm2=10.0,
            # Spatial reduction within the downloaded box:
            # - "mean" (default) can look clear-sky-like due to cloud smoothing.
            # - "point" selects a fixed grid cell (ilat/ilon).
            # - "max_variability" automatically selects the rampiest grid cell (recommended).
            spatial_mode="max_variability",  # "mean" | "point" | "max_variability"
            ilat=0,
            ilon=0,
        ),
    ),

}



# Ordered lists for reproducible iteration across all pipeline stages
SYNTH_CASE_ORDER: List[str] = ["A", "B", "C", "D"]
CASE_ORDER: List[str] = SYNTH_CASE_ORDER + ["ERA5", "ERA_ssrd"]


# ============================================================================
# 2. GLOBAL EXPERIMENTAL PARAMETERS
# ============================================================================

@dataclass(frozen=True)
class ExperimentConfig:
    """Global parameters shared across all cases.

    Chosen to be large enough for statistical significance
    while remaining CPU-tractable without specialised hardware.
    """
    # --- Data generation ---
    T: int = 20_000                # total series length
    seed: int = 42                 # global random seed
    n: int = 32                    # past window length (input to inverse model)
    m: int = 16                    # future window length (variable to infer)

    # --- Train / val / test split (chronological) ---
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac = 1 - train_frac - val_frac = 0.15

    # --- Standardisation ---
    standardize: bool = True       # z-score using train statistics only

    # --- Device ---
    device: str = "cpu"            # "cpu" or "cuda"

    # --- Output ---
    outdir: str = "outputs"


# ============================================================================
# 3. MODEL HYPERPARAMETERS
# ============================================================================

@dataclass(frozen=True)
class InverseCVAEConfig:
    """Inverse CVAE: learns p_θ(x | y, z).

    Architecture: MLP encoder q(z|x,y), MLP decoder p(x|y,z).
    Both use the same hidden dimension and depth for simplicity.
    """
    z_dim: int = 8                 # latent dimension
    hidden: int = 128              # hidden layer width
    depth: int = 2                 # number of hidden layers
    dropout: float = 0.0           # dropout rate (0 = off)

    epochs: int = 80               # training epochs
    batch_size: int = 256
    lr: float = 2e-3               # Adam learning rate
    beta: float = 1.0              # KL weight (β-VAE; 1.0 = standard ELBO)
    grad_clip: float = 5.0         # gradient clipping norm


@dataclass(frozen=True)
class ForwardCVAEConfig:
    """Forward CVAE baseline: learns p_θ(y | x, z).

    Same capacity as InverseCVAE for fair comparison.
    The only difference is the direction: encoder q(z|x,y),
    decoder p(y|x,z).
    """
    z_dim: int = 8
    hidden: int = 128
    depth: int = 2
    dropout: float = 0.0

    epochs: int = 80
    batch_size: int = 256
    lr: float = 2e-3
    beta: float = 1.0
    grad_clip: float = 5.0


@dataclass(frozen=True)
class ForwardMLPConfig:
    """Forward MLP baseline: deterministic y = f_φ(x).

    Same hidden dimension as the CVAE decoder for fair capacity
    comparison.  Trained with MSE loss.
    """
    hidden: int = 128
    depth: int = 2
    dropout: float = 0.0

    epochs: int = 80
    batch_size: int = 256
    lr: float = 2e-3
    grad_clip: float = 5.0


# ============================================================================
# 4. PRIOR OVER FUTURES p(y)
# ============================================================================

@dataclass(frozen=True)
class FlowPriorConfig:
    """RealNVP normalizing flow for learning p_ψ(y).

    Used as the informed prior in MAP inference.
    Ablated against N(0,I) to demonstrate prior importance.
    """
    n_layers: int = 4              # number of RealNVP coupling layers
    hidden: int = 128
    epochs: int = 60
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 0.0
    patience: int = 8              # early stopping patience
    min_delta: float = 1e-4        # early stopping min improvement
    grad_clip: float = 5.0


class PriorKind(Enum):
    """Which prior to use for p(y) in MAP inference."""
    FLOW = "flow"                  # learned RealNVP
    STANDARD_NORMAL = "normal"     # N(0, I) — ablation baseline


# ============================================================================
# 5. MAP INFERENCE SETTINGS
# ============================================================================

@dataclass(frozen=True)
class MAPConfig:
    """MAP inference: (y*, z*) = argmax log p_θ(x_obs|y,z) + λ log p(y) + log p(z).

    Uses Adam optimiser with multi-start initialisation.
    """
    steps: int = 200               # optimisation steps per inference
    lr: float = 0.05               # Adam lr for (y, z)
    K_multistart: int = 5          # number of random initialisations
    lam_prior: float = 2.0         # weight on log p(y) term
    lam_smooth: float = 0.0        # smoothness penalty (usually 0)
    y_clip: float = 3.5            # clamp y to [-clip, clip] (standardised)
    z_clip: float = 3.5            # clamp z to [-clip, clip]
    grad_clip: float = 5.0         # gradient clipping for (y, z)
    aggregate: str = "best"        # "best" = keep lowest-loss restart

    # Number of test samples to evaluate (subsample for speed)
    n_eval: int = 2000


# ============================================================================
# 6. ARROW-OF-TIME DIAGNOSTIC
# ============================================================================

@dataclass(frozen=True)
class ArrowOfTimeConfig:
    """KL-based arrow-of-time diagnostic Δ_arrow.

    Uses kNN estimator (Pérez-Cruz, ISIT 2008) on forward/backward
    conditional embedding windows.

    GO declared if median(J(m)) > τ AND count(J(m)>τ) >= C_min.
    """
    window_lengths: List[int] = field(default_factory=lambda: [2, 4, 8])
    k_nn: int = 5                  # kNN parameter
    n_subsample: int = 2000        # max windows to subsample (speed)
    n_bootstrap: int = 200         # bootstrap replicates for CI
    tau: float = 0.05              # decision threshold
    C_min: int = 2                 # minimum scales exceeding τ


# ============================================================================
# 7. ABLATION MATRIX
# ============================================================================

@dataclass(frozen=True)
class AblationRun:
    """One row of the ablation matrix: case × prior_kind."""
    case_key: str
    prior_kind: PriorKind


def build_ablation_matrix() -> List[AblationRun]:
    """Full ablation: each case × {flow, N(0,I)}."""
    runs = []
    for key in CASE_ORDER:
        for pk in PriorKind:
            runs.append(AblationRun(case_key=key, prior_kind=pk))
    return runs


# ============================================================================
# 8. FALSIFIABLE PREDICTIONS (for the paper)
# ============================================================================

# Falsifiable predictions stated a priori before any experiment is run.
# Each entry maps a prediction ID to its statement, relevant cases, and
# the metric used to evaluate it.
PREDICTIONS = {
    "P1": {
        "statement": "Arrow-of-time diagnostic matches expected verdicts",
        "cases_positive": ["A", "C", "ERA5", "ERA_ssrd"],
        "cases_negative": ["B", "D"],
        "metric": "arrow_of_time_verdict",
    },
    "P2": {
        "statement": "Flow prior improves inverse inference vs N(0,I) on GO cases",
        "cases": ["A", "C"],
        "metric": "rmse_flow_vs_normal",
    },
    "P3": {
        "statement": "Inverse MAP does not offer a meaningful advantage on NO-GO cases",
        "cases": ["B", "D"],
        "metric": "rmse_inv_vs_mlp",
    },
    "P4": {
        "statement": "Inverse MAP competitive with or beats MLP forward on GO cases",
        "cases": ["A", "C", "ERA5", "ERA_ssrd"],
        "metric": "rmse_ratio_inv_mlp",
    },
}


# ============================================================================
# 9. CONVENIENCE: default configs bundle
# ============================================================================

@dataclass
class FullConfig:
    """Bundle of all configuration objects for easy passing."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    inverse_cvae: InverseCVAEConfig = field(default_factory=InverseCVAEConfig)
    forward_cvae: ForwardCVAEConfig = field(default_factory=ForwardCVAEConfig)
    forward_mlp: ForwardMLPConfig = field(default_factory=ForwardMLPConfig)
    flow_prior: FlowPriorConfig = field(default_factory=FlowPriorConfig)
    map_inference: MAPConfig = field(default_factory=MAPConfig)
    arrow_of_time: ArrowOfTimeConfig = field(default_factory=ArrowOfTimeConfig)


def get_default_config() -> FullConfig:
    """Return the default configuration bundle."""
    return FullConfig()


# ============================================================================
# 10. SELF-TEST
# ============================================================================

if __name__ == "__main__":
    cfg = get_default_config()
    print("=" * 60)
    print("Retrodictive Forecasting — Configuration Summary")
    print("=" * 60)

    print(
        f"\nGlobal: T={cfg.experiment.T}, n={cfg.experiment.n}, "
        f"m={cfg.experiment.m}, seed={cfg.experiment.seed}"
    )
    print(
        f"Split: {cfg.experiment.train_frac:.0%} / "
        f"{cfg.experiment.val_frac:.0%} / "
        f"{1-cfg.experiment.train_frac-cfg.experiment.val_frac:.0%}"
    )

    print(f"\n{'Case':<6} {'Name':<28} {'Verdict':<8} {'Irrev. source'}")
    print("-" * 80)
    for key in CASE_ORDER:
        c = CASES[key]
        print(
            f"  {c.key:<4} {c.name:<28} {c.verdict.value:<8} "
            f"{c.irreversibility_source[:40]}"
        )

    print(
        f"\nInverse CVAE: z_dim={cfg.inverse_cvae.z_dim}, "
        f"hidden={cfg.inverse_cvae.hidden}, epochs={cfg.inverse_cvae.epochs}"
    )
    print(
        f"Forward CVAE: z_dim={cfg.forward_cvae.z_dim}, "
        f"hidden={cfg.forward_cvae.hidden}, epochs={cfg.forward_cvae.epochs}"
    )
    print(
        f"Forward MLP:  hidden={cfg.forward_mlp.hidden}, "
        f"epochs={cfg.forward_mlp.epochs}"
    )
    print(
        f"Flow prior:   layers={cfg.flow_prior.n_layers}, "
        f"hidden={cfg.flow_prior.hidden}, epochs={cfg.flow_prior.epochs}"
    )
    print(
        f"MAP:          steps={cfg.map_inference.steps}, "
        f"K={cfg.map_inference.K_multistart}, "
        f"λ_prior={cfg.map_inference.lam_prior}"
    )
    print(
        f"Arrow-of-time: windows={cfg.arrow_of_time.window_lengths}, "
        f"τ={cfg.arrow_of_time.tau}"
    )

    print(
        f"\nAblation matrix: {len(build_ablation_matrix())} runs "
        f"({len(CASE_ORDER)} cases × {len(PriorKind)} priors)"
    )

    print(f"\nFalsifiable predictions: {len(PREDICTIONS)}")
    for pid, p in PREDICTIONS.items():
        print(f"  {pid}: {p['statement']}")

    print("\n✓ Configuration valid.")
