#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnostics.py — Arrow-of-Time diagnostic with block-permutation test
=====================================================================

Determines whether a time series exhibits sufficient temporal asymmetry
to benefit from the retrodictive paradigm (GO vs NO-GO decision).

Method (block permutation)
-------------------------------
For each window length w, build forward/backward embeddings and compute
the symmetrised KL divergence J_obs = D_KL(F||B) + D_KL(B||F) using a
kNN estimator.  Statistical significance is assessed via a block
permutation test on the pooled samples [F; B]:

  1. Randomly permute contiguous blocks of embeddings
  2. Split back into two pseudo-groups F', B'
  3. Compute J_perm on (F', B')
  4. Repeat n_perm times → null distribution
  5. p-value = (1 + #{J_perm >= J_obs}) / (n_perm + 1)

Block permutation preserves local temporal dependence within blocks,
yielding valid p-values even when successive embedding windows overlap.

Dual diagnostic
---------------
Two representations are tested:
  - LEVEL: embeddings built from x_t directly
  - DIFF:  embeddings built from Δx_t = x_t - x_{t-1}

LEVEL captures state-dependent irreversibility (e.g., multiplicative noise).
DIFF captures transition-level irreversibility (e.g., excitation-relaxation).

Combined decision rule:
  GO if LEVEL = GO   (amplitude-dependent irreversibility)
  OR  if DIFF  = GO   (transition-level irreversibility)

Each sub-diagnostic declares GO if count_w(p_w < α) >= C_min.

References
----------
  - Pérez-Cruz, F. (2008). Kullback-Leibler divergence estimation of
    continuous distributions. IEEE ISIT.
    DOI: 10.1109/ISIT.2008.4595271

  - Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap.
    JASA, 89(428), 1303–1313.
    DOI: 10.1080/01621459.1994.10476870

  - Kawai, R., Parrondo, J.M.R., Van den Broeck, C. (2007).
    Dissipation: The phase-space perspective.
    Physical Review Letters, 98(8), 080602.
    DOI: 10.1103/PhysRevLett.98.080602

Usage
-----
    from diagnostics import arrow_of_time_diagnostic, ArrowOfTimeResult
    from config import get_default_config

    cfg = get_default_config()
    result = arrow_of_time_diagnostic(series, cfg.arrow_of_time)
    print(result.verdict)        # "GO" or "NO-GO"
    print(result.level_result)   # sub-diagnostic on x_t
    print(result.diff_result)    # sub-diagnostic on Δx_t
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from config import ArrowOfTimeConfig, Verdict


# ============================================================================
# 1. kNN KL DIVERGENCE ESTIMATOR
# ============================================================================

def _kl_knn(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 5,
    eps: float = 1e-12,
) -> float:
    """Estimate D_KL(P_X || P_Y) using kNN distances (Pérez-Cruz, 2008).

    Parameters
    ----------
    X : (n, d) — samples from P
    Y : (m, d) — samples from Q
    k : int — number of nearest neighbours
    eps : float — small constant to prevent log(0)

    Returns
    -------
    kl : float — estimated D_KL (can be negative due to estimation noise)
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    n, d = X.shape
    m = Y.shape[0]

    if n <= k + 1 or m <= k + 1:
        return 0.0

    # k-th NN in X (exclude self → query k+1)
    nn_x = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    dist_x, _ = nn_x.kneighbors(X, return_distance=True)
    r_k = dist_x[:, -1] + eps  # k-th NN distance (self excluded)

    # k-th NN in Y
    nn_y = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(Y)
    dist_y, _ = nn_y.kneighbors(X, return_distance=True)
    s_k = dist_y[:, -1] + eps

    return float(np.log(m / (n - 1.0)) + d * np.mean(np.log(s_k / r_k)))


def _j_divergence(F: np.ndarray, B: np.ndarray, k: int = 5) -> float:
    """Symmetrised KL divergence: J(F, B) = D_KL(F||B) + D_KL(B||F)."""
    return float(_kl_knn(F, B, k=k) + _kl_knn(B, F, k=k))


# ============================================================================
# 2. FORWARD / BACKWARD EMBEDDINGS
# ============================================================================

def _build_embeddings(
    x: np.ndarray,
    w: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build forward and backward conditional embedding vectors.

    Forward:  e^f_t = [x_t, ..., x_{t+w-1},  x_{t+w}, ..., x_{t+2w-1}]
                    = [past_w, future_w]

    Backward: e^b_t = [rev(future_w), rev(past_w)]
              (time-reversed version of the same window)

    Parameters
    ----------
    x : (T,) — time series (raw or differenced)
    w : int — half-window length
    stride : int — step between consecutive windows

    Returns
    -------
    F, B : (N, 2w) — forward and backward embeddings
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    T = len(x)
    max_t = T - 2 * w

    if max_t <= 0:
        raise ValueError(f"Series too short for w={w}: need T > 2w, got T={T}")

    idx = np.arange(0, max_t, stride, dtype=int)
    n = len(idx)

    F = np.zeros((n, 2 * w), dtype=np.float32)
    B = np.zeros((n, 2 * w), dtype=np.float32)

    for i, t in enumerate(idx):
        past = x[t: t + w]
        fut = x[t + w: t + 2 * w]

        F[i, :w] = past
        F[i, w:] = fut

        # Time reversal: reversed future then reversed past
        B[i, :w] = fut[::-1]
        B[i, w:] = past[::-1]

    return F, B


# ============================================================================
# 3. BLOCK PERMUTATION TEST
# ============================================================================

def _make_block_permutation(
    n: int,
    block_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a block permutation of indices [0, ..., n-1].

    Splits the index range into contiguous blocks of length `block_len`,
    then shuffles the block order (preserving within-block structure).
    """
    if block_len <= 1:
        return rng.permutation(n)

    blocks = []
    start = 0
    while start < n:
        end = min(n, start + block_len)
        blocks.append(np.arange(start, end))
        start = end

    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def _block_permutation_pvalue(
    F: np.ndarray,
    B: np.ndarray,
    k_nn: int,
    n_perm: int,
    block_len: int,
    rng: np.random.Generator,
) -> Tuple[float, float, np.ndarray]:
    """Compute p-value for J_obs via block permutation of pooled samples.

    Under H0 (time-reversibility), F and B are exchangeable.
    We pool them, apply block permutations, and re-split.

    Parameters
    ----------
    F, B : (n, 2w) — forward and backward embeddings
    k_nn : int — kNN parameter
    n_perm : int — number of permutations
    block_len : int — block size for permutation
    rng : Generator

    Returns
    -------
    p_value : float
    J_obs : float — observed symmetrised KL
    J_null : (n_perm,) — null distribution of J under permutation
    """
    n = min(F.shape[0], B.shape[0])
    F = F[:n]
    B = B[:n]

    J_obs = _j_divergence(F, B, k=k_nn)

    pooled = np.concatenate([F, B], axis=0)
    N = pooled.shape[0]  # 2n

    J_null = np.empty(n_perm, dtype=np.float32)
    for i in range(n_perm):
        perm = _make_block_permutation(N, block_len=block_len, rng=rng)
        pseudo = pooled[perm]
        Fp = pseudo[:n]
        Bp = pseudo[n: 2 * n]
        J_null[i] = _j_divergence(Fp, Bp, k=k_nn)

    # One-sided p-value (test: J_obs significantly larger than null)
    p_value = float((1.0 + np.sum(J_null >= J_obs)) / (n_perm + 1.0))

    return p_value, float(J_obs), J_null


# ============================================================================
# 4. DATACLASSES — preserving public interface
# ============================================================================

@dataclass
class ArrowScaleResult:
    """Result of the arrow-of-time diagnostic at a single window scale.

    Attributes
    ----------
    w : int — window half-length
    J_median : float — observed J divergence (J_obs from the permutation test)
    J_ci_low : float — 2.5th percentile of null distribution
    J_ci_high : float — 97.5th percentile of null distribution
    exceeds_tau : bool — True if p_value < alpha (significant)
    p_value : float — permutation test p-value
    n_samples : int — number of embedding pairs used
    J_samples : Optional[np.ndarray] — null distribution samples (if stored)
    """
    w: int
    J_median: float                    # observed J_obs value
    J_ci_low: float                    # null distribution 2.5th percentile
    J_ci_high: float                   # null distribution 97.5th percentile
    exceeds_tau: bool                  # True if p_value < alpha
    p_value: float = 0.0              # permutation p-value
    n_samples: int = 0                # number of embedding pairs
    J_samples: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class SubDiagnosticResult:
    """Result from one representation (LEVEL or DIFF).

    Attributes
    ----------
    tag : str — "LEVEL" or "DIFF"
    verdict : Verdict — GO or NO-GO for this representation
    n_reject : int — number of scales with p < alpha
    scales : List[ArrowScaleResult]
    """
    tag: str
    verdict: Verdict
    n_reject: int
    scales: List[ArrowScaleResult]


@dataclass
class ArrowOfTimeResult:
    """Complete arrow-of-time diagnostic result.

    Preserves the public interface expected by evaluation.py, plotting.py,
    run_single.py, and run_all.py:
      .verdict, .overall_median, .scale_results, .n_exceeding,
      .tau, .C_min, .decision_rule

    Additionally exposes the dual sub-diagnostics:
      .level_result, .diff_result
    """
    verdict: Verdict
    overall_median: float              # = J_obs from the deciding sub-diagnostic
    scale_results: List[ArrowScaleResult]  # = scales of the deciding sub-diagnostic
    n_exceeding: int                   # = n_reject of the deciding sub-diagnostic
    tau: float                         # = alpha (significance level)
    C_min: int
    decision_rule: str

    # Sub-diagnostics for each representation
    level_result: Optional[SubDiagnosticResult] = None
    diff_result: Optional[SubDiagnosticResult] = None


# ============================================================================
# 5. SINGLE-REPRESENTATION SUB-DIAGNOSTIC
# ============================================================================

def _run_sub_diagnostic(
    x: np.ndarray,
    tag: str,
    window_lengths: List[int],
    k_nn: int,
    C_min: int,
    alpha: float,
    n_perm: int,
    block_len: int,
    stride_mode: str,
    seed: int,
    verbose: bool = True,
    store_samples: bool = False,
) -> SubDiagnosticResult:
    """Run block-permutation arrow-of-time test on one representation.

    Parameters
    ----------
    x : (T,) — input signal (raw level or Δx)
    tag : str — "LEVEL" or "DIFF"
    window_lengths : list of int
    k_nn, C_min, alpha, n_perm, block_len, stride_mode : diagnostic params
    seed : int
    verbose, store_samples : flags

    Returns
    -------
    SubDiagnosticResult
    """
    rng = np.random.default_rng(seed)
    scales: List[ArrowScaleResult] = []
    rejects: List[bool] = []

    for w in window_lengths:
        stride = w if stride_mode == "w" else int(stride_mode)

        F, B = _build_embeddings(x, w=w, stride=stride)

        # Auto block length: sqrt(n_samples), clipped to [3, 50]
        if block_len == 0:
            bl = int(np.clip(np.sqrt(F.shape[0]), 3, 50))
        else:
            bl = int(block_len)

        p_value, J_obs, J_null = _block_permutation_pvalue(
            F=F, B=B,
            k_nn=k_nn,
            n_perm=n_perm,
            block_len=bl,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )

        reject = bool(p_value < alpha)
        rejects.append(reject)

        # Null distribution quantiles (for backward-compat CI fields)
        ci_low = float(np.percentile(J_null, 2.5))
        ci_high = float(np.percentile(J_null, 97.5))

        n_samples = min(F.shape[0], B.shape[0])

        scales.append(ArrowScaleResult(
            w=int(w),
            J_median=float(J_obs),         # J_obs value stored under J_median field name
            J_ci_low=ci_low,               # = null 2.5th pctile
            J_ci_high=ci_high,             # = null 97.5th pctile
            exceeds_tau=reject,            # = p < alpha
            p_value=float(p_value),
            n_samples=int(n_samples),
            J_samples=J_null if store_samples else None,
        ))

        if verbose:
            flag = "REJECT" if reject else "ok"
            print(f"    [{tag}] w={w:2d} | N={n_samples:5d} | "
                  f"J_obs={J_obs:9.4f} | p={p_value:.4f} | {flag}")

    n_reject = int(sum(rejects))
    verdict = Verdict.GO if n_reject >= C_min else Verdict.NOGO

    if verbose:
        print(f"    [{tag}] → {verdict.value} "
              f"(n_reject={n_reject}/{len(scales)}, C_min={C_min})")

    return SubDiagnosticResult(
        tag=tag,
        verdict=verdict,
        n_reject=n_reject,
        scales=scales,
    )


# ============================================================================
# 6. MAIN PUBLIC API — arrow_of_time_diagnostic()
# ============================================================================

def arrow_of_time_diagnostic(
    series: np.ndarray,
    cfg: ArrowOfTimeConfig,
    seed: int = 42,
    verbose: bool = True,
    store_samples: bool = False,
) -> ArrowOfTimeResult:
    """Run the dual arrow-of-time diagnostic (LEVEL + DIFF) on a series.

    Decision rule (combined):
      GO if LEVEL = GO  OR  DIFF = GO
      NO-GO otherwise

    Parameters
    ----------
    series : (T,) — raw time series
    cfg : ArrowOfTimeConfig
    seed : int
    verbose : bool
    store_samples : bool

    Returns
    -------
    ArrowOfTimeResult — with .level_result and .diff_result populated
    """
    series = np.asarray(series, dtype=np.float32).ravel()

    # Default permutation test parameters
    n_perm = getattr(cfg, "n_perm", 200)
    block_len = getattr(cfg, "block_len", 0)     # 0 = auto
    stride_mode = getattr(cfg, "stride_mode", "w")
    alpha = getattr(cfg, "alpha", cfg.tau)  # use tau as alpha for compat

    if verbose:
        print(f"\n  [Arrow-of-Time V4] Block permutation test "
              f"(windows={cfg.window_lengths}, k={cfg.k_nn}, "
              f"n_perm={n_perm}, alpha={alpha:.3f})")

    # --- LEVEL diagnostic: on x_t ---
    if verbose:
        print(f"  --- LEVEL (on x_t) ---")
    level_result = _run_sub_diagnostic(
        x=series,
        tag="LEVEL",
        window_lengths=list(cfg.window_lengths),
        k_nn=int(cfg.k_nn),
        C_min=int(cfg.C_min),
        alpha=float(alpha),
        n_perm=int(n_perm),
        block_len=int(block_len),
        stride_mode=str(stride_mode),
        seed=seed + 1000,
        verbose=verbose,
        store_samples=store_samples,
    )

    # --- DIFF diagnostic: on Δx_t ---
    dx = np.diff(series)
    if verbose:
        print(f"  --- DIFF (on Δx_t) ---")
    diff_result = _run_sub_diagnostic(
        x=dx,
        tag="DIFF",
        window_lengths=list(cfg.window_lengths),
        k_nn=int(cfg.k_nn),
        C_min=int(cfg.C_min),
        alpha=float(alpha),
        n_perm=int(n_perm),
        block_len=int(block_len),
        stride_mode=str(stride_mode),
        seed=seed + 2000,
        verbose=verbose,
        store_samples=store_samples,
    )

    # --- Combined decision: GO if LEVEL=GO OR DIFF=GO ---
    combined_go = (level_result.verdict == Verdict.GO
                   or diff_result.verdict == Verdict.GO)
    verdict = Verdict.GO if combined_go else Verdict.NOGO

    # Choose the "primary" sub-diagnostic for backward-compat fields
    # Priority: if both GO, pick the one with more rejections
    if level_result.verdict == Verdict.GO and diff_result.verdict == Verdict.GO:
        primary = (level_result if level_result.n_reject >= diff_result.n_reject
                   else diff_result)
    elif level_result.verdict == Verdict.GO:
        primary = level_result
    elif diff_result.verdict == Verdict.GO:
        primary = diff_result
    else:
        # Both NO-GO: pick the one with the smallest p-value
        min_p_level = min((s.p_value for s in level_result.scales), default=1.0)
        min_p_diff = min((s.p_value for s in diff_result.scales), default=1.0)
        primary = level_result if min_p_level <= min_p_diff else diff_result

    # Build backward-compatible fields from primary
    overall_median = float(np.median([s.J_median for s in primary.scales]))
    n_exceeding = primary.n_reject

    decision_rule = (
        f"LEVEL={level_result.verdict.value}({level_result.n_reject}/{len(level_result.scales)}), "
        f"DIFF={diff_result.verdict.value}({diff_result.n_reject}/{len(diff_result.scales)}) "
        f"→ {verdict.value} "
        f"[rule: GO if LEVEL=GO OR DIFF=GO, alpha={alpha}, C_min={cfg.C_min}]"
    )

    if verbose:
        print(f"  [Arrow-of-Time V4] {decision_rule}")

    return ArrowOfTimeResult(
        verdict=verdict,
        overall_median=overall_median,
        scale_results=primary.scales,
        n_exceeding=n_exceeding,
        tau=float(alpha),
        C_min=int(cfg.C_min),
        decision_rule=decision_rule,
        level_result=level_result,
        diff_result=diff_result,
    )


# ============================================================================
# 7. BATCH DIAGNOSTIC (all cases)
# ============================================================================

def run_all_diagnostics(
    series_dict: dict,
    cfg: ArrowOfTimeConfig,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run arrow-of-time diagnostic on all cases.

    Parameters
    ----------
    series_dict : dict mapping case_key → np.ndarray
    cfg : ArrowOfTimeConfig
    seed : int
    verbose : bool

    Returns
    -------
    results : dict mapping case_key → ArrowOfTimeResult
    """
    results = {}
    for key, series in series_dict.items():
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Case {key}")
            print(f"{'=' * 60}")
        results[key] = arrow_of_time_diagnostic(
            series, cfg, seed=seed, verbose=verbose,
        )
    return results


def diagnostic_summary_table(
    results: dict,
    case_specs: Optional[dict] = None,
) -> str:
    """Format diagnostic results as a readable ASCII table.

    Parameters
    ----------
    results : dict mapping case_key → ArrowOfTimeResult
    case_specs : optional dict mapping case_key → CaseSpec

    Returns
    -------
    table : str
    """
    lines = []
    header = (
        f"{'Case':<6} {'Name':<28} {'Expected':<10} "
        f"{'LEVEL':<8} {'DIFF':<8} {'Combined':<10} {'Match':<6}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for key in sorted(results.keys()):
        r = results[key]

        if case_specs and key in case_specs:
            spec = case_specs[key]
            name = spec.name
            expected = spec.verdict.value
            match = "✓" if r.verdict == spec.verdict else "✗"
        else:
            name = "?"
            expected = "?"
            match = "?"

        lv = r.level_result.verdict.value if r.level_result else "?"
        dv = r.diff_result.verdict.value if r.diff_result else "?"

        lines.append(
            f"  {key:<4} {name:<28} {expected:<10} "
            f"{lv:<8} {dv:<8} {r.verdict.value:<10} {match:<6}"
        )

    return "\n".join(lines)


# ============================================================================
# 8. SELF-TEST
# ============================================================================

if __name__ == "__main__":
    from config import CASES, SYNTH_CASE_ORDER, get_default_config
    from generators import generate_series

    cfg = get_default_config()
    exp = cfg.experiment

    print("=" * 70)
    print("Retrodictive Forecasting — Arrow-of-Time Diagnostic (Block Permutation)")
    print("=" * 70)

    # Generate all series
    series_dict = {}
    for key in SYNTH_CASE_ORDER:
        series_dict[key] = generate_series(key, T=exp.T, seed=exp.seed)

    # Run diagnostics
    results = run_all_diagnostics(
        series_dict, cfg.arrow_of_time, seed=exp.seed, verbose=True,
    )

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    table = diagnostic_summary_table(results, CASES)
    print(table)

    # Check P1
    print("\n--- Prediction P1 check ---")
    go_keys = [k for k in SYNTH_CASE_ORDER if CASES[k].verdict == Verdict.GO]
    nogo_keys = [k for k in SYNTH_CASE_ORDER if CASES[k].verdict == Verdict.NOGO]

    all_match = True
    for k in go_keys:
        ok = results[k].verdict == Verdict.GO
        status = "✓" if ok else "✗"
        lv = results[k].level_result.verdict.value if results[k].level_result else "?"
        dv = results[k].diff_result.verdict.value if results[k].diff_result else "?"
        print(f"  {k} ({CASES[k].name}): expected GO, "
              f"got {results[k].verdict.value} [L={lv}, D={dv}] {status}")
        if not ok:
            all_match = False

    for k in nogo_keys:
        ok = results[k].verdict == Verdict.NOGO
        status = "✓" if ok else "✗"
        lv = results[k].level_result.verdict.value if results[k].level_result else "?"
        dv = results[k].diff_result.verdict.value if results[k].diff_result else "?"
        print(f"  {k} ({CASES[k].name}): expected NO-GO, "
              f"got {results[k].verdict.value} [L={lv}, D={dv}] {status}")
        if not ok:
            all_match = False

    if all_match:
        print("\n  P1 PASSED: all verdicts match expectations")
    else:
        print("\n  P1 FAILED: some verdicts do not match")

    print("\n✓ All diagnostic checks completed.")
