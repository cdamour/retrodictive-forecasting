#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plotting.py — Publication-quality figures for Retrodictive Forecasting
===================================================================

Generates all figures for the article.  Each function produces
one figure and saves it as both PNG (300 dpi) and PDF.

Figures
-------
  Fig 1.  Raw synthetic series overview (4 subplots, one per case)
    Fig 2.  Arrow-of-time dual diagnostic (LEVEL + DIFF, J_obs + p-values x case)
    Fig 3.  Training curves (loss vs epoch, per model x case)
  Fig 4.  Per-case example reconstructions (best/median/worst MAP)
    Fig 5.  Cross-case RMSE comparison barplot (5 methods x 4 cases)
    Fig 6.  RMSE per forecast horizon (line plot, per method x case)
  Fig 7.  Multi-start dispersion boxplot (GO vs NO-GO)
  Fig 8.  Prior ablation barplot (flow vs N(0,I))
  Fig 9.  Prediction verification summary (pass/fail heatmap)

Style
-----
  - Matplotlib with seaborn palette for consistency
  - Colourblind-friendly palette (tab10 / ColorBrewer)
  - Font size suitable for two-column journal format
  - All labels in English

Usage
-----
    from plotting import plot_all
    plot_all(comparisons, arrow_results, histories, datasets, outdir="outputs/figures")
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import CASES, CASE_ORDER, Verdict
from evaluation import CaseComparison, PredictionCheck
from diagnostics import ArrowOfTimeResult


# ============================================================================
# 0. STYLE CONFIGURATION
# ============================================================================

# Publication-compatible style (two-column journal format)
STYLE = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.2,
}

# Colourblind-friendly palette
COLORS = {
    "Naive mean":          "#999999",   # grey
    "Forward MLP":         "#E69F00",   # orange
    "Forward CVAE":        "#56B4E9",   # sky blue
    "Inverse MAP (flow)":  "#009E73",   # green
    "Inverse MAP (N0I)":   "#CC79A7",   # pink
}

CASE_COLORS = {
    "A": "#0072B2",  # blue
    "B": "#D55E00",  # vermillion
    "C": "#009E73",  # green
    "D": "#F0E442",  # yellow
}

VERDICT_COLORS = {
    Verdict.GO: "#009E73",    # green
    Verdict.NOGO: "#D55E00",  # vermillion
}

METHOD_ORDER = [
    "Naive mean", "Forward MLP", "Forward CVAE",
    "Inverse MAP (flow)", "Inverse MAP (N0I)",
]


def _ordered_case_keys(keys: List[str]) -> List[str]:
    """Return keys ordered by CASE_ORDER first, then remaining keys."""
    keys_set = set(keys)
    ordered = [k for k in CASE_ORDER if k in keys_set]
    for k in keys:
        if k not in ordered:
            ordered.append(k)
    return ordered


def _case_name(
    key: str,
    *,
    comparisons: Optional[Dict[str, CaseComparison]] = None,
    arrow_results: Optional[Dict[str, ArrowOfTimeResult]] = None,
) -> str:
    if key in CASES:
        return CASES[key].name
    if comparisons is not None and key in comparisons:
        return comparisons[key].case_name
    # arrow_results doesn't store a human-readable name; fall back to key
    _ = arrow_results
    return str(key)


def _case_verdict(
    key: str,
    *,
    comparisons: Optional[Dict[str, CaseComparison]] = None,
    arrow_results: Optional[Dict[str, ArrowOfTimeResult]] = None,
) -> Optional[Verdict]:
    if key in CASES:
        return CASES[key].verdict
    if arrow_results is not None and key in arrow_results:
        return arrow_results[key].verdict
    if comparisons is not None and key in comparisons:
        ar = comparisons[key].arrow_result
        if ar is not None:
            return ar.verdict
        return comparisons[key].verdict_expected
    return None


def _verdict_color(verdict: Optional[Verdict]) -> str:
    if verdict is None:
        return "#333333"
    return VERDICT_COLORS.get(verdict, "#333333")


def _apply_style():
    """Apply publication style to matplotlib."""
    plt.rcParams.update(STYLE)


def _savefig(fig, filepath: str, formats: Tuple[str, ...] = ("png", "pdf")):
    """Save figure in multiple formats."""
    base, _ = os.path.splitext(filepath)
    for fmt in formats:
        path = f"{base}.{fmt}"
        fig.savefig(path, format=fmt)
    plt.close(fig)


def _ensure_dir(outdir: str):
    """Create output directory if needed."""
    os.makedirs(outdir, exist_ok=True)


# ============================================================================
# 1. RAW SERIES OVERVIEW
# ============================================================================

def plot_series_overview(
    series_dict: Dict[str, np.ndarray],
    outdir: str,
    n_points: int = 2000,
) -> str:
    """Fig 1: Overview of synthetic time series.

    Shows the first n_points of each series with case name,
    verdict, and irreversibility source annotation.

    Parameters
    ----------
    series_dict : dict case_key -> np.ndarray
    outdir : str
    n_points : int — number of points to show

    Returns
    -------
    str — filepath of saved figure
    """
    _apply_style()
    _ensure_dir(outdir)

    cases = [k for k in CASE_ORDER if k in series_dict]
    if not cases:
        cases = _ordered_case_keys(list(series_dict.keys()))

    fig, axes = plt.subplots(
        len(cases), 1, figsize=(7, 8), sharex=True,
    )

    if len(cases) == 1:
        axes = [axes]

    for i, key in enumerate(cases):
        ax = axes[i]
        s = series_dict[key][:n_points]
        t = np.arange(len(s))

        verdict = _case_verdict(key)
        color = _verdict_color(verdict)
        ax.plot(t, s, color=color, linewidth=0.5, alpha=0.8)
        ax.set_ylabel(f"Case {key}", fontweight="bold")

        # Annotation
        verdict_str = verdict.value if verdict is not None else "?"
        name = _case_name(key)
        ax.text(
            0.98, 0.92, f"{name}\n[{verdict_str}]",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    axes[-1].set_xlabel("Time step t")
    fig.suptitle("Synthetic Test Suite — Raw Series", fontweight="bold", y=1.01)
    fig.tight_layout()

    filename = "fig1_series_overview"
    if len(cases) == 1:
        filename = f"{filename}_{cases[0]}"
    path = os.path.join(outdir, filename)
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# 2. ARROW-OF-TIME DIAGNOSTIC
# ============================================================================

def _plot_arrow_panel(
    ax_j: plt.Axes,
    ax_p: plt.Axes,
    sub_results: dict,
    cases: list,
    tag: str,
    alpha: float,
) -> None:
    """Draw one panel (LEVEL or DIFF) of the arrow-of-time figure.

    Top subplot: J_obs bars (grouped by case x window).
    Bottom subplot: -log10(p-value) bars with significance line.
    """
    n_cases = len(cases)

    # Collect all window lengths
    all_ws = sorted({sr.w for k in cases for sr in sub_results[k].scales})
    n_ws = len(all_ws)

    bar_width = 0.8 / max(n_ws, 1)
    x_base = np.arange(n_cases)

    for j, w in enumerate(all_ws):
        j_obs_vals = []
        p_vals = []
        ci_lows = []
        ci_highs = []
        colors = []

        for k in cases:
            sr_match = [sr for sr in sub_results[k].scales if sr.w == w]
            if sr_match:
                sr = sr_match[0]
                j_obs_vals.append(sr.J_median)       # J_obs
                p_vals.append(sr.p_value)
                ci_lows.append(max(sr.J_median - sr.J_ci_low, 0))
                ci_highs.append(max(sr.J_ci_high - sr.J_median, 0))
            else:
                j_obs_vals.append(0.0)
                p_vals.append(1.0)
                ci_lows.append(0)
                ci_highs.append(0)
            colors.append(_verdict_color(sub_results[k].verdict if sub_results.get(k) else None))

        x = x_base + j * bar_width - 0.4 + bar_width / 2

        # Top panel: J_obs bars with null CI as error bars
        ax_j.bar(
            x, j_obs_vals, width=bar_width, label=f"w={w}",
            color=colors, alpha=0.7, edgecolor="white", linewidth=0.5,
            yerr=[ci_lows, ci_highs], capsize=2, error_kw={"linewidth": 0.8},
        )

        # Bottom panel: -log10(p) bars
        neg_log_p = [-np.log10(max(p, 1e-6)) for p in p_vals]
        ax_p.bar(
            x, neg_log_p, width=bar_width,
            color=colors, alpha=0.7, edgecolor="white", linewidth=0.5,
        )

    # Significance line on p-value panel
    sig_line = -np.log10(alpha)
    ax_p.axhline(y=sig_line, color="red", linestyle="--", linewidth=1,
                 label=f"$\\alpha = {alpha}$")

    # Format top panel
    ax_j.set_xticks(x_base)
    ax_j.set_xticklabels([])  # labels on bottom panel only
    ax_j.set_ylabel("$J_{obs}(w)$")
    ax_j.set_title(f"{tag}", fontweight="bold")
    ax_j.legend(loc="upper right", fontsize=6)
    ax_j.set_ylim(bottom=0)

    # Format bottom panel
    ax_p.set_xticks(x_base)
    ax_p.set_xticklabels(
        [f"{k}" for k in cases],
        fontsize=6,
    )
    ax_p.set_ylabel("$-\\log_{10}(p)$")
    ax_p.legend(loc="upper right", fontsize=6)
    ax_p.set_ylim(bottom=0)

    # Annotate verdicts below x-axis
    for i, k in enumerate(cases):
        sub = sub_results[k]
        v_str = sub.verdict.value
        v_col = _verdict_color(sub.verdict)
        ax_p.annotate(
            v_str, xy=(i, 0), xytext=(0, -22),
            textcoords="offset points", ha="center", fontsize=6,
            fontweight="bold", color=v_col,
        )


def plot_arrow_of_time(
    arrow_results: Dict[str, ArrowOfTimeResult],
    outdir: str,
) -> str:
    """Fig 2: Dual arrow-of-time diagnostic (LEVEL + DIFF).

    Two side-by-side panels, each with J_obs bars (top) and
    -log10(p-value) bars (bottom).  Significance line at alpha.

    Parameters
    ----------
    arrow_results : dict case_key -> ArrowOfTimeResult
    outdir : str

    Returns
    -------
    str -- filepath
    """
    _apply_style()
    _ensure_dir(outdir)

    cases = [k for k in CASE_ORDER if k in arrow_results]
    if not cases:
        cases = _ordered_case_keys(list(arrow_results.keys()))

    # Check if dual sub-diagnostics are available
    has_dual = all(
        getattr(ar, "level_result", None) is not None
        and getattr(ar, "diff_result", None) is not None
        for ar in arrow_results.values()
    )

    if not has_dual:
        # Fallback: single-panel plot using scale_results (no dual sub-diagnostics)
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 4.5),
                                              height_ratios=[2, 1],
                                              sharex=True)
        from diagnostics import SubDiagnosticResult
        fake_subs = {}
        for k in cases:
            ar = arrow_results[k]
            fake_subs[k] = SubDiagnosticResult(
                tag="ALL", verdict=ar.verdict,
                n_reject=ar.n_exceeding, scales=ar.scale_results,
            )
        alpha = arrow_results[cases[0]].tau
        _plot_arrow_panel(ax_top, ax_bot, fake_subs, cases, "Arrow-of-Time", alpha)
        fig.tight_layout()
        path = os.path.join(outdir, "fig2_arrow_of_time")
        _savefig(fig, path)
        return path + ".png"

    # --- Dual panels: LEVEL (left) and DIFF (right) ---
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.15, wspace=0.3)

    ax_lj = fig.add_subplot(gs[0, 0])  # LEVEL J_obs
    ax_lp = fig.add_subplot(gs[1, 0])  # LEVEL p-values
    ax_dj = fig.add_subplot(gs[0, 1])  # DIFF J_obs
    ax_dp = fig.add_subplot(gs[1, 1])  # DIFF p-values

    alpha = arrow_results[cases[0]].tau

    level_subs = {k: arrow_results[k].level_result for k in cases}
    diff_subs = {k: arrow_results[k].diff_result for k in cases}

    _plot_arrow_panel(ax_lj, ax_lp, level_subs, cases, "LEVEL ($x_t$)", alpha)
    _plot_arrow_panel(ax_dj, ax_dp, diff_subs, cases, "DIFF ($\\Delta x_t$)", alpha)

    fig.suptitle("Arrow-of-Time Diagnostic (Block Permutation Test)",
                 fontweight="bold", fontsize=11, y=1.01)
    fig.tight_layout()

    path = os.path.join(outdir, "fig2_arrow_of_time")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# 3. TRAINING CURVES
# ============================================================================

def plot_training_curves(
    histories: Dict[str, Dict[str, Dict[str, List[float]]]],
    outdir: str,
) -> str:
    """Fig 3: Training curves for all models on all cases.

    Grid layout: rows = cases, columns = model types.

    Parameters
    ----------
    histories : nested dict
        histories[case_key][model_name] -> dict with loss keys
        Model names: "inverse_cvae", "forward_cvae", "forward_mlp", "flow"
    outdir : str

    Returns
    -------
    str — filepath
    """
    _apply_style()
    _ensure_dir(outdir)

    model_names = ["inverse_cvae", "forward_cvae", "forward_mlp", "flow"]
    model_labels = ["Inverse CVAE", "Forward CVAE", "Forward MLP", "RealNVP Flow"]

    cases = [k for k in CASE_ORDER if k in histories]
    if not cases:
        cases = _ordered_case_keys(list(histories.keys()))
    n_cases = len(cases)
    n_models = len(model_names)

    fig, axes = plt.subplots(
        n_cases, n_models, figsize=(3.5 * n_models, 2.2 * n_cases),
        squeeze=False,
    )

    for i, key in enumerate(cases):
        for j, (mname, mlabel) in enumerate(zip(model_names, model_labels)):
            ax = axes[i][j]

            if mname not in histories[key]:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_title(f"{mlabel}" if i == 0 else "")
                continue

            h = histories[key][mname]

            if mname in ("inverse_cvae", "forward_cvae"):
                ax.plot(h.get("train_loss", []), label="train", linewidth=0.8)
                ax.plot(h.get("val_loss", []), label="val", linewidth=0.8)
                ax.set_ylabel("ELBO loss")
            elif mname == "forward_mlp":
                ax.plot(h.get("train_mse", []), label="train", linewidth=0.8)
                ax.plot(h.get("val_mse", []), label="val", linewidth=0.8)
                ax.set_ylabel("MSE")
            elif mname == "flow":
                ax.plot(h.get("train_nll", []), label="train", linewidth=0.8)
                ax.plot(h.get("val_nll", []), label="val", linewidth=0.8)
                if "best_epoch" in h:
                    ax.axvline(x=h["best_epoch"] - 1, color="red",
                               linestyle=":", linewidth=0.8, label="best")
                ax.set_ylabel("NLL")

            if i == 0:
                ax.set_title(mlabel, fontweight="bold")
            if i == n_cases - 1:
                ax.set_xlabel("Epoch")
            if j == 0:
                ax.text(
                    -0.25, 0.5, f"Case {key}", transform=ax.transAxes,
                    ha="center", va="center", rotation=90, fontweight="bold",
                )
            ax.legend(fontsize=6, loc="upper right")

    fig.suptitle("Training Curves", fontweight="bold", y=1.01)
    fig.tight_layout()

    path = os.path.join(outdir, "fig3_training_curves")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# 4. PER-CASE EXAMPLE RECONSTRUCTIONS
# ============================================================================

def plot_example_reconstructions(
    case_key: str,
    X_test_s: np.ndarray,
    Y_test_s: np.ndarray,
    Y_hat_s: np.ndarray,
    Y_naive_s: np.ndarray,
    Y_mlp_s: np.ndarray,
    outdir: str,
    n_examples: int = 3,
    case_name: Optional[str] = None,
    verdict: Optional[Verdict] = None,
) -> str:
    """Fig 4 (per case): Example reconstructions showing best/median/worst.

    Selects examples based on MAP RMSE (best, median, worst).
    Shows past window, true future, and predicted future for:
      - Naive, MLP, Inverse MAP

    Parameters
    ----------
    case_key : str
    X_test_s : (N, n) — observed past (standardised)
    Y_test_s : (N, m) — true future (standardised)
    Y_hat_s : (N_eval, m) — inverse MAP predictions
    Y_naive_s : (N, m) — naive predictions
    Y_mlp_s : (N, m) — MLP predictions
    outdir : str
    n_examples : int — number of examples (3 = best/median/worst)

    Returns
    -------
    str — filepath
    """
    _apply_style()
    _ensure_dir(outdir)

    if case_name is None:
        case_name = _case_name(case_key)
    if verdict is None:
        verdict = _case_verdict(case_key)
    N_eval = Y_hat_s.shape[0]
    N_test = Y_test_s.shape[0]
    n = X_test_s.shape[1]
    m = Y_test_s.shape[1]

    # Match subsampled indices
    if N_eval < N_test:
        eval_indices = np.linspace(0, N_test - 1, N_eval).astype(int)
    else:
        eval_indices = np.arange(N_test)

    # RMSE per sample for inverse MAP
    rmse_per_sample = np.sqrt(np.mean(
        (Y_test_s[eval_indices] - Y_hat_s) ** 2, axis=1
    ))
    sorted_idx = np.argsort(rmse_per_sample)

    # Select best, median, worst
    picks = [
        ("Best", sorted_idx[0]),
        ("Median", sorted_idx[len(sorted_idx) // 2]),
        ("Worst", sorted_idx[-1]),
    ]

    fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 3))
    if n_examples == 1:
        axes = [axes]

    for col, (label, eval_idx) in enumerate(picks):
        ax = axes[col]
        test_idx = eval_indices[eval_idx]

        # Past
        t_past = np.arange(n)
        ax.plot(t_past, X_test_s[test_idx], color="black", linewidth=1.2,
                label="Observed past")

        # Future: true
        t_future = np.arange(n, n + m)
        ax.plot(t_future, Y_test_s[test_idx], color="black", linestyle="--",
                linewidth=1.2, label="True future")

        # Future: naive
        ax.plot(t_future, Y_naive_s[test_idx], color=COLORS["Naive mean"],
                linewidth=0.8, alpha=0.7, label="Naive")

        # Future: MLP
        ax.plot(t_future, Y_mlp_s[test_idx], color=COLORS["Forward MLP"],
                linewidth=0.8, label="MLP")

        # Future: Inverse MAP
        ax.plot(t_future, Y_hat_s[eval_idx], color=COLORS["Inverse MAP (flow)"],
                linewidth=1.2, label="Inverse MAP")

        # Vertical separator
        ax.axvline(x=n - 0.5, color="gray", linestyle=":", linewidth=0.6)

        ax.set_title(f"{label} (RMSE={rmse_per_sample[eval_idx]:.3f})", fontsize=8)
        ax.set_xlabel("Position")
        if col == 0:
            ax.set_ylabel("Value (standardised)")
            ax.legend(fontsize=6, loc="upper left")

    fig.suptitle(
        f"Case {case_key}: {case_name} ({verdict.value if verdict is not None else '?'}) — Example Reconstructions",
        fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    path = os.path.join(outdir, f"fig4_examples_{case_key}")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# 5. CROSS-CASE RMSE COMPARISON
# ============================================================================

def plot_cross_case_rmse(
    comparisons: Dict[str, CaseComparison],
    outdir: str,
) -> str:
    """Fig 5: Cross-case RMSE barplot (grouped by case, coloured by method).

    Parameters
    ----------
    comparisons : dict case_key -> CaseComparison
    outdir : str

    Returns
    -------
    str — filepath
    """
    _apply_style()
    _ensure_dir(outdir)

    cases = [k for k in CASE_ORDER if k in comparisons]
    if not cases:
        cases = _ordered_case_keys(list(comparisons.keys()))
    methods = [m for m in METHOD_ORDER if any(
        m in comparisons[k].methods for k in cases
    )]
    n_cases = len(cases)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(8, 4))

    bar_width = 0.8 / n_methods
    x_base = np.arange(n_cases)

    for j, mname in enumerate(methods):
        rmses = []
        for k in cases:
            comp = comparisons[k]
            if mname in comp.methods:
                rmses.append(comp.methods[mname].rmse_s)
            else:
                rmses.append(0)

        x = x_base + j * bar_width - 0.4 + bar_width / 2
        ax.bar(
            x, rmses, width=bar_width, label=mname,
            color=COLORS.get(mname, "#333333"), alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(
        [f"{k}\n({_case_verdict(k, comparisons=comparisons).value if _case_verdict(k, comparisons=comparisons) is not None else '?'})" for k in cases],
    )
    ax.set_ylabel("RMSE (standardised)")
    ax.set_title("Cross-Case RMSE Comparison", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    path = os.path.join(outdir, "fig5_cross_case_rmse")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# 6. RMSE PER FORECAST HORIZON
# ============================================================================

def plot_rmse_per_horizon(
    comparisons: Dict[str, CaseComparison],
    outdir: str,
) -> str:
    """Fig 6: RMSE per forecast horizon position (line plots).

    Grid: one subplot per case.

    Parameters
    ----------
    comparisons : dict case_key -> CaseComparison
    outdir : str

    Returns
    -------
    str — filepath
    """
    _apply_style()
    _ensure_dir(outdir)

    cases = [k for k in CASE_ORDER if k in comparisons]
    if not cases:
        cases = _ordered_case_keys(list(comparisons.keys()))
    n_cases = len(cases)
    ncols = min(3, n_cases)
    nrows = (n_cases + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, key in enumerate(cases):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        comp = comparisons[key]
        name = _case_name(key, comparisons=comparisons)
        verdict = _case_verdict(key, comparisons=comparisons)

        for mname in METHOD_ORDER:
            if mname in comp.methods:
                mr = comp.methods[mname]
                if mr.rmse_per_pos_s is not None:
                    horizons = np.arange(1, len(mr.rmse_per_pos_s) + 1)
                    ax.plot(
                        horizons, mr.rmse_per_pos_s,
                        color=COLORS.get(mname, "#333333"),
                        linewidth=1.0, label=mname, marker=".", markersize=3,
                    )

        ax.set_title(f"Case {key}: {name[:20]} ({verdict.value if verdict is not None else '?'})", fontsize=8)
        ax.set_xlabel("Horizon h")
        ax.set_ylabel("RMSE(h)")
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    # Hide unused subplots
    for idx in range(n_cases, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("RMSE per Forecast Horizon", fontweight="bold", y=1.01)
    fig.tight_layout()

    path = os.path.join(outdir, "fig6_rmse_per_horizon")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# 7. MULTI-START DISPERSION
# ============================================================================

def plot_multistart_dispersion(
    comparisons: Dict[str, CaseComparison],
    outdir: str,
) -> str:
    """Fig 7: Multi-start dispersion comparison (GO vs NO-GO).

    Boxplot of mean multi-start std per sample, grouped by verdict.

    Parameters
    ----------
    comparisons : dict case_key -> CaseComparison
    outdir : str

    Returns
    -------
    str — filepath
    """
    _apply_style()
    _ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    data_go = []
    data_nogo = []
    labels_go = []
    labels_nogo = []

    keys = [k for k in CASE_ORDER if k in comparisons]
    if not keys:
        keys = _ordered_case_keys(list(comparisons.keys()))

    for key in keys:
        comp = comparisons[key]
        verdict = _case_verdict(key, comparisons=comparisons)
        inv = comp.methods.get("Inverse MAP (flow)")
        if inv and inv.mean_multistart_std is not None:
            if verdict == Verdict.GO:
                data_go.append(inv.mean_multistart_std)
                labels_go.append(f"{key}")
            else:
                data_nogo.append(inv.mean_multistart_std)
                labels_nogo.append(f"{key}")

    # Bar plot
    all_labels = labels_go + labels_nogo
    all_values = data_go + data_nogo
    all_colors = ([VERDICT_COLORS[Verdict.GO]] * len(data_go) +
                  [VERDICT_COLORS[Verdict.NOGO]] * len(data_nogo))

    if all_labels:
        x = np.arange(len(all_labels))
        ax.bar(x, all_values, color=all_colors, alpha=0.8, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels)

        # Group labels
        if data_go:
            ax.text(
                (len(data_go) - 1) / 2, max(all_values) * 1.05, "GO",
                ha="center", fontweight="bold", color=VERDICT_COLORS[Verdict.GO],
            )
        if data_nogo:
            ax.text(
                len(data_go) + (len(data_nogo) - 1) / 2,
                max(all_values) * 1.05, "NO-GO",
                ha="center", fontweight="bold", color=VERDICT_COLORS[Verdict.NOGO],
            )

    ax.set_ylabel("Mean multi-start std(y)")
    ax.set_title("Multi-Start Dispersion", fontweight="bold")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    path = os.path.join(outdir, "fig7_multistart_dispersion")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# 8. PRIOR ABLATION
# ============================================================================

def plot_prior_ablation(
    comparisons: Dict[str, CaseComparison],
    outdir: str,
) -> str:
    """Fig 8: Prior ablation — flow vs N(0,I) RMSE comparison.

    Grouped bars: flow vs N(0,I) for each case.

    Parameters
    ----------
    comparisons : dict case_key -> CaseComparison
    outdir : str

    Returns
    -------
    str — filepath
    """
    _apply_style()
    _ensure_dir(outdir)

    cases = [k for k in CASE_ORDER if k in comparisons]
    if not cases:
        cases = _ordered_case_keys(list(comparisons.keys()))
    n_cases = len(cases)

    fig, ax = plt.subplots(figsize=(6, 3.5))

    bar_width = 0.35
    x = np.arange(n_cases)

    rmse_flow = []
    rmse_n0i = []
    for k in cases:
        comp = comparisons[k]
        flow_r = comp.methods.get("Inverse MAP (flow)")
        n0i_r = comp.methods.get("Inverse MAP (N0I)")
        rmse_flow.append(flow_r.rmse_s if flow_r else 0)
        rmse_n0i.append(n0i_r.rmse_s if n0i_r else 0)

    ax.bar(x - bar_width / 2, rmse_flow, bar_width,
           label="Flow prior", color=COLORS["Inverse MAP (flow)"], alpha=0.85)
    ax.bar(x + bar_width / 2, rmse_n0i, bar_width,
           label="N(0,I) prior", color=COLORS["Inverse MAP (N0I)"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{k}\n({_case_verdict(k, comparisons=comparisons).value if _case_verdict(k, comparisons=comparisons) is not None else '?'})" for k in cases]
    )
    ax.set_ylabel("RMSE (standardised)")
    ax.set_title("Prior Ablation: Flow vs N(0,I) (P2)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    path = os.path.join(outdir, "fig8_prior_ablation")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# 9. PREDICTION VERIFICATION HEATMAP
# ============================================================================

def plot_prediction_summary(
    checks: List[PredictionCheck],
    outdir: str,
) -> str:
    """Fig 9: Falsifiable predictions pass/fail summary.

    Simple horizontal bar showing P1–P4 as green (pass) or red (fail).

    Parameters
    ----------
    checks : List[PredictionCheck]
    outdir : str

    Returns
    -------
    str — filepath
    """
    _apply_style()
    _ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(6, 3))

    pids = [c.pid for c in checks]
    passed = [1 if c.passed else 0 for c in checks]
    colors = ["#009E73" if p else "#D55E00" for p in passed]

    y = np.arange(len(pids))
    ax.barh(y, [1] * len(pids), color=colors, alpha=0.8, edgecolor="white")

    for i, c in enumerate(checks):
        status = "PASS" if c.passed else "FAIL"
        ax.text(0.02, i, f"{c.pid}: {c.statement}",
                va="center", fontsize=7, fontweight="bold", color="white")
        ax.text(0.98, i, status, va="center", ha="right",
                fontsize=8, fontweight="bold", color="white")

    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Falsifiable Predictions Verification", fontweight="bold")

    n_passed = sum(passed)
    n_total = len(passed)
    ax.text(0.5, -0.8, f"{n_passed}/{n_total} passed",
            ha="center", fontsize=10, fontweight="bold",
            transform=ax.transData)

    fig.tight_layout()

    path = os.path.join(outdir, "fig9_prediction_summary")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# 10. CONVENIENCE: PLOT ALL
# ============================================================================

def plot_all(
    comparisons: Dict[str, CaseComparison],
    arrow_results: Dict[str, ArrowOfTimeResult],
    histories: Dict[str, Dict[str, Dict[str, List[float]]]],
    series_dict: Dict[str, np.ndarray],
    checks: List[PredictionCheck],
    test_data: Optional[Dict[str, dict]] = None,
    outdir: str = "outputs/figures",
) -> List[str]:
    """Generate all figures (Fig 1–9).

    Parameters
    ----------
    comparisons : dict case_key -> CaseComparison
    arrow_results : dict case_key -> ArrowOfTimeResult
    histories : nested dict case_key -> model_name -> history dict
    series_dict : dict case_key -> raw series
    checks : List[PredictionCheck]
    test_data : optional dict case_key -> {X_test_s, Y_test_s, Y_hat_s,
                Y_naive_s, Y_mlp_s} for example reconstructions
    outdir : str

    Returns
    -------
    List[str] — list of saved filepaths
    """
    _ensure_dir(outdir)
    paths = []

    # Fig 1: Series overview
    paths.append(plot_series_overview(series_dict, outdir))

    # Fig 2: Arrow-of-time
    paths.append(plot_arrow_of_time(arrow_results, outdir))

    # Fig 3: Training curves
    paths.append(plot_training_curves(histories, outdir))

    # Fig 4: Per-case examples (if test data provided)
    if test_data:
        for key in CASE_ORDER:
            if key in test_data:
                td = test_data[key]
                paths.append(plot_example_reconstructions(
                    case_key=key,
                    X_test_s=td["X_test_s"],
                    Y_test_s=td["Y_test_s"],
                    Y_hat_s=td["Y_hat_s"],
                    Y_naive_s=td["Y_naive_s"],
                    Y_mlp_s=td["Y_mlp_s"],
                    outdir=outdir,
                ))

    # Fig 5: Cross-case RMSE
    paths.append(plot_cross_case_rmse(comparisons, outdir))

    # Fig 6: RMSE per horizon
    paths.append(plot_rmse_per_horizon(comparisons, outdir))

    # Fig 7: Multi-start dispersion
    paths.append(plot_multistart_dispersion(comparisons, outdir))

    # Fig 8: Prior ablation
    paths.append(plot_prior_ablation(comparisons, outdir))

    # Fig 9: Prediction summary
    paths.append(plot_prediction_summary(checks, outdir))

    print(f"\n  [Plotting] {len(paths)} figures saved to {outdir}/")
    return paths


# ============================================================================
# 11. SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Retrodictive Forecasting — Plotting Module Self-Test")
    print("=" * 60)
    print("  This module requires data from run_all.py.")
    print("  Run `python run_all.py` first to generate all figures.")
    print("  Individual plot functions can be called with appropriate data.")
    print("\n  Available functions:")
    for name in [
        "plot_series_overview", "plot_arrow_of_time",
        "plot_training_curves", "plot_example_reconstructions",
        "plot_cross_case_rmse", "plot_rmse_per_horizon",
        "plot_multistart_dispersion", "plot_prior_ablation",
        "plot_prediction_summary", "plot_all",
    ]:
        print(f"    - {name}()")
    print("\n  All plot functions accept an outdir parameter.")


# ============================================================================
# ADDITIONAL FIGURES: Fig 10–14
# ============================================================================

from matplotlib.patches import Patch as _Patch
try:
    from scipy import stats as _sp_stats
    _HAS_SCIPY_PLOT = True
except ImportError:
    _HAS_SCIPY_PLOT = False


# ---------------------------------------------------------------------------
# Fig 10 — J_obs strength summary per case, sorted by intensity
# ---------------------------------------------------------------------------

def plot_jobs_summary(
    arrow_results: Dict[str, ArrowOfTimeResult],
    outdir: str,
) -> str:
    """Fig 10: Horizontal barplot J_obs median per case, ordered by strength.

    Coloured by GO (green) / NO-GO (orange-red). Shows how each case scores
    on the arrow-of-time diagnostic. Provides a compact 'scoreboard' for the
    Results section.

    Parameters
    ----------
    arrow_results : dict case_key -> ArrowOfTimeResult
    outdir : str

    Returns
    -------
    str — saved filepath (.png)
    """
    _apply_style()
    _ensure_dir(outdir)

    cases = [k for k in CASE_ORDER if k in arrow_results]
    if not cases:
        cases = _ordered_case_keys(list(arrow_results.keys()))

    # Sort by J_obs descending (strongest irreversibility first)
    sorted_cases = sorted(cases,
                          key=lambda k: arrow_results[k].overall_median,
                          reverse=True)

    j_max = max(arrow_results[k].overall_median for k in cases)

    fig, ax = plt.subplots(figsize=(6.5, 0.9 * len(sorted_cases) + 1.8))

    for i, key in enumerate(sorted_cases):
        ar = arrow_results[key]
        color = _verdict_color(ar.verdict)
        j_val = ar.overall_median
        ax.barh(i, j_val, color=color, alpha=0.82, edgecolor="white", height=0.6)
        ax.text(j_val + j_max * 0.01, i, f"{j_val:.2f}",
                va="center", fontsize=7.5, fontweight="bold")

    ax.set_yticks(range(len(sorted_cases)))
    ax.set_yticklabels(
        [f"{k}: {_case_name(k)[:20]}  [{arrow_results[k].verdict.value}]"
         for k in sorted_cases],
        fontsize=7.5
    )
    ax.set_xlabel(r"$J_{obs}$ median (all windows & scales)", fontsize=8)
    ax.set_title("Arrow-of-Time Strength per Case (P1)", fontweight="bold")
    ax.set_xlim(left=0)
    ax.axvline(x=0, color="black", linewidth=0.8)

    legend_elements = [
        _Patch(facecolor=VERDICT_COLORS[Verdict.GO],   label="GO — paradigm applicable"),
        _Patch(facecolor=VERDICT_COLORS[Verdict.NOGO], label="NO-GO — time-reversible"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
    fig.tight_layout()

    path = os.path.join(outdir, "fig10_jobs_summary")
    _savefig(fig, path)
    return path + ".png"


# ---------------------------------------------------------------------------
# Fig 11 — RMSE per horizon: GO cases overlay (CVAE vs Inverse MAP)
# ---------------------------------------------------------------------------

def plot_rmse_horizon_go_overlay(
    comparisons: Dict[str, CaseComparison],
    go_case_keys: List[str],
    outdir: str,
) -> str:
    """Fig 11: RMSE per horizon — GO cases grid, CVAE vs Inverse MAP.

    Shows head-to-head RMSE(h) profiles. The green shaded region marks
    horizons where the inverse MAP outperforms the forward CVAE — the
    'retrodictive advantage window'.

    Parameters
    ----------
    comparisons   : dict case_key -> CaseComparison
    go_case_keys  : list of GO case keys
    outdir        : str

    Returns
    -------
    str — filepath
    """
    _apply_style()
    _ensure_dir(outdir)

    if not go_case_keys:
        return ""

    ncols = min(2, len(go_case_keys))
    nrows = (len(go_case_keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows),
                              squeeze=False)
    fig.suptitle(
        "RMSE per Forecast Horizon — GO Cases\n"
        "Forward CVAE vs Inverse MAP (shaded = inverse advantage)",
        fontweight="bold", y=1.02, fontsize=9,
    )

    for idx, key in enumerate(go_case_keys):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        comp = comparisons[key]
        name = _case_name(key, comparisons=comparisons)
        verdict = _case_verdict(key, comparisons=comparisons)

        # Plot MLP (reference, dashed), CVAE and Inverse
        plot_specs = [
            ("Forward MLP",        COLORS["Forward MLP"],        "--", 0.8),
            ("Forward CVAE",       COLORS["Forward CVAE"],       "-",  1.4),
            ("Inverse MAP (flow)", COLORS["Inverse MAP (flow)"], "-",  1.4),
        ]
        for mname, color, ls, lw in plot_specs:
            if mname in comp.methods:
                mr = comp.methods[mname]
                if mr.rmse_per_pos_s is not None:
                    h = np.arange(1, len(mr.rmse_per_pos_s) + 1)
                    ax.plot(h, mr.rmse_per_pos_s, color=color,
                            linewidth=lw, linestyle=ls, label=mname,
                            marker=".", markersize=3)

        # Shade the retrodictive advantage window
        inv_mr  = comp.methods.get("Inverse MAP (flow)")
        cvae_mr = comp.methods.get("Forward CVAE")
        if (inv_mr and cvae_mr
                and inv_mr.rmse_per_pos_s is not None
                and cvae_mr.rmse_per_pos_s is not None):
            h   = np.arange(1, len(inv_mr.rmse_per_pos_s) + 1)
            inv_r  = inv_mr.rmse_per_pos_s
            cvae_r = cvae_mr.rmse_per_pos_s
            ax.fill_between(h, inv_r, cvae_r,
                            where=(inv_r <= cvae_r),
                            alpha=0.20,
                            color=COLORS["Inverse MAP (flow)"],
                            label="Inv ≤ CVAE")

        ax.set_title(
            f"Case {key}: {name[:22]} [{verdict.value if verdict else '?'}]",
            fontsize=8,
        )
        ax.set_xlabel("Horizon h")
        ax.set_ylabel("RMSE(h)")
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    for idx in range(len(go_case_keys), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    path = os.path.join(outdir, "fig11_rmse_horizon_go_overlay")
    _savefig(fig, path)
    return path + ".png"


# ---------------------------------------------------------------------------
# Fig 12 — Scatter RetroNLL vs per-sample RMSE
# ---------------------------------------------------------------------------

def plot_retronll_vs_rmse(
    case_keys: List[str],
    retro_nlls_dict: Dict[str, np.ndarray],
    Y_true_dict: Dict[str, np.ndarray],
    Y_hat_dict: Dict[str, np.ndarray],
    outdir: str,
) -> str:
    """Fig 12: Per-sample scatter of RetroNLL vs prediction RMSE.

    A significant negative correlation validates that the MAP objective
    (retrodictive NLL) is a good proxy for prediction quality — i.e. the
    paradigm exploits causal information correctly.

    Parameters
    ----------
    case_keys       : GO case keys to include
    retro_nlls_dict : case_key -> (n_eval,) retrodictive NLL per sample
    Y_true_dict     : case_key -> (n_eval, m) ground truth aligned to eval
    Y_hat_dict      : case_key -> (n_eval, m) MAP predictions
    outdir          : str

    Returns
    -------
    str — filepath
    """
    _apply_style()
    _ensure_dir(outdir)

    n_cases = len(case_keys)
    if n_cases == 0:
        return ""

    ncols = min(2, n_cases)
    nrows = (n_cases + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.8 * nrows),
                              squeeze=False)
    fig.suptitle(
        "RetroNLL vs Per-Sample RMSE (GO cases)\n"
        r"Validates: low $-\log p_\theta(x|y^*)$ ↔ accurate prediction",
        fontweight="bold", y=1.02, fontsize=9,
    )

    for idx, key in enumerate(case_keys):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        retro_nlls     = retro_nlls_dict[key]
        Y_true         = Y_true_dict[key]
        Y_hat          = Y_hat_dict[key]
        rmse_per_sample = np.sqrt(np.mean((Y_true - Y_hat) ** 2, axis=1))

        ax.scatter(retro_nlls, rmse_per_sample,
                   alpha=0.25, s=7,
                   color=COLORS["Inverse MAP (flow)"], linewidths=0)

        # Regression line + statistics
        mask = np.isfinite(retro_nlls) & np.isfinite(rmse_per_sample)
        if mask.sum() > 10 and _HAS_SCIPY_PLOT:
            slope, intercept, r_val, p_val, _ = _sp_stats.linregress(
                retro_nlls[mask], rmse_per_sample[mask]
            )
            x_line = np.linspace(retro_nlls[mask].min(),
                                  retro_nlls[mask].max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color="crimson", linewidth=1.3, linestyle="--",
                    label=f"r = {r_val:.3f}  (p = {p_val:.3f})")
            ax.legend(fontsize=7)

        ax.set_xlabel(r"RetroNLL $= -\log\,p_\theta(x_{obs}\,|\,y^*, z^*)$",
                      fontsize=7.5)
        ax.set_ylabel("Per-sample RMSE (standardised)", fontsize=7.5)
        ax.set_title(f"Case {key}: {_case_name(key)[:22]}", fontsize=8)

    for idx in range(n_cases, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    path = os.path.join(outdir, "fig12_retronll_vs_rmse")
    _savefig(fig, path)
    return path + ".png"


# ---------------------------------------------------------------------------
# Fig 13 — MAP loss distribution (flow vs N(0,I))
# ---------------------------------------------------------------------------

def plot_map_loss_distribution(
    case_key: str,
    map_losses_flow: np.ndarray,
    map_losses_n0i: np.ndarray,
    verdict,
    outdir: str,
) -> str:
    """Fig 13: Density histogram of MAP loss values per case.

    Compares the distribution of final MAP objective values between the
    flow prior and the N(0,I) ablation.  A tighter / lower distribution
    for the flow prior (P2) confirms better optimisation landscape.

    Parameters
    ----------
    case_key        : str
    map_losses_flow : (n,)
    map_losses_n0i  : (n,)
    verdict         : Verdict
    outdir          : str

    Returns
    -------
    str — filepath
    """
    _apply_style()
    _ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bins = max(20, min(60, len(map_losses_flow) // 15 + 5))

    ax.hist(map_losses_flow, bins=bins, alpha=0.65,
            color=COLORS["Inverse MAP (flow)"], label="Flow prior",
            density=True, edgecolor="none")
    ax.hist(map_losses_n0i, bins=bins, alpha=0.50,
            color=COLORS["Inverse MAP (N0I)"], label="N(0,I) prior",
            density=True, edgecolor="none")

    med_flow = float(np.median(map_losses_flow))
    med_n0i  = float(np.median(map_losses_n0i))
    ax.axvline(med_flow, color=COLORS["Inverse MAP (flow)"],
               linestyle="--", linewidth=1.1,
               label=f"Flow median: {med_flow:.1f}")
    ax.axvline(med_n0i,  color=COLORS["Inverse MAP (N0I)"],
               linestyle="--", linewidth=1.1,
               label=f"N(0,I) median: {med_n0i:.1f}")

    verdict_str = verdict.value if verdict is not None else "?"
    ax.set_xlabel("MAP Loss (total objective, lower = better)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Case {case_key}: {_case_name(case_key)[:28]} [{verdict_str}]\n"
        "MAP Loss Distribution (Prior Ablation — P2)",
        fontsize=8, fontweight="bold",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()

    path = os.path.join(outdir, f"fig13_map_loss_dist_{case_key}")
    _savefig(fig, path)
    return path + ".png"


# ---------------------------------------------------------------------------
# Fig 14 — FIC contribution diagnostic
# ---------------------------------------------------------------------------

def plot_fic_contribution(
    case_key: str,
    map_losses: np.ndarray,
    multistart_std: np.ndarray,
    outdir: str,
) -> str:
    """Fig 14: Forward-Inverse Chaining (FIC) contribution diagnostic.

    Two panels:
    - Left: scatter multi-start std vs MAP loss — distinguishes easy vs
      rugged optimisation landscapes.
    - Right: CDF of MAP loss — shows P10/P90 spread.

    Parameters
    ----------
    case_key       : str
    map_losses     : (n,) final MAP losses (best restart)
    multistart_std : (n,) per-sample std of y across K restarts
    outdir         : str

    Returns
    -------
    str — filepath
    """
    _apply_style()
    _ensure_dir(outdir)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    fig.suptitle(
        f"Case {case_key}: FIC (Forward-Inverse Chaining) Landscape Diagnostic",
        fontweight="bold", y=1.02, fontsize=9,
    )

    # ---- Panel A: landscape scatter ----
    ax = axes[0]
    ax.scatter(multistart_std, map_losses,
               alpha=0.22, s=7,
               color=COLORS["Inverse MAP (flow)"], linewidths=0)

    ms_med = float(np.median(multistart_std))
    ml_med = float(np.median(map_losses))
    ax.axvline(ms_med, color="gray", linestyle=":", linewidth=0.9)
    ax.axhline(ml_med, color="gray", linestyle=":", linewidth=0.9)
    ax.text(0.03, 0.05, "Easy landscape\n→ good solution",
            transform=ax.transAxes, fontsize=6.5, color="#009E73", va="bottom")
    ax.text(0.97, 0.95, "Rugged landscape\n→ suboptimal",
            transform=ax.transAxes, fontsize=6.5, color="#D55E00",
            va="top", ha="right")
    ax.set_xlabel("Multi-start std (optimisation roughness)")
    ax.set_ylabel("MAP Loss (best of K restarts)")
    ax.set_title("Landscape Complexity vs Solution Quality", fontsize=8)

    # ---- Panel B: CDF ----
    ax2 = axes[1]
    sorted_losses = np.sort(map_losses)
    cdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
    ax2.plot(sorted_losses, cdf,
             color=COLORS["Inverse MAP (flow)"], linewidth=1.3)
    p10 = float(np.percentile(map_losses, 10))
    p90 = float(np.percentile(map_losses, 90))
    ax2.axvline(p10, color="#0072B2", linestyle="--", linewidth=0.9,
                label=f"P10 = {p10:.1f}")
    ax2.axvline(p90, color="#D55E00", linestyle="--", linewidth=0.9,
                label=f"P90 = {p90:.1f}")
    ax2.set_xlabel("MAP Loss")
    ax2.set_ylabel("CDF")
    ax2.set_title("Cumulative Distribution of MAP Loss (flow prior)", fontsize=8)
    ax2.legend(fontsize=7)

    fig.tight_layout()
    path = os.path.join(outdir, f"fig14_fic_contribution_{case_key}")
    _savefig(fig, path)
    return path + ".png"


# ---------------------------------------------------------------------------
# Updated plot_all (replaces the original defined above)
# ---------------------------------------------------------------------------

def plot_all(
    comparisons: Dict[str, CaseComparison],
    arrow_results: Dict[str, ArrowOfTimeResult],
    histories: Dict[str, Dict[str, Dict[str, List[float]]]],
    series_dict: Dict[str, np.ndarray],
    checks: List[PredictionCheck],
    test_data: Optional[Dict[str, dict]] = None,
    map_data: Optional[Dict[str, dict]] = None,
    outdir: str = "outputs/figures",
) -> List[str]:
    """Generate all figures (Fig 1–14).

    Parameters
    ----------
    comparisons  : dict case_key -> CaseComparison
    arrow_results: dict case_key -> ArrowOfTimeResult
    histories    : nested dict case_key -> model_name -> history dict
    series_dict  : dict case_key -> raw series
    checks       : List[PredictionCheck]
    test_data    : optional dict case_key -> {X_test_s, Y_test_s, Y_hat_s,
                   Y_naive_s, Y_mlp_s}
    map_data     : optional dict case_key -> {map_results_flow, map_results_n0i,
                   Y_fwd_mlp, Y_fwd_cvae, dataset}
    outdir       : str

    Returns
    -------
    List[str]
    """
    _ensure_dir(outdir)
    paths: List[str] = []

    # ---- Existing figures (1–9) ----
    paths.append(plot_series_overview(series_dict, outdir))
    paths.append(plot_arrow_of_time(arrow_results, outdir))
    paths.append(plot_training_curves(histories, outdir))

    if test_data:
        for key in CASE_ORDER:
            if key in test_data:
                td = test_data[key]
                paths.append(plot_example_reconstructions(
                    case_key=key,
                    X_test_s=td["X_test_s"],
                    Y_test_s=td["Y_test_s"],
                    Y_hat_s=td["Y_hat_s"],
                    Y_naive_s=td["Y_naive_s"],
                    Y_mlp_s=td["Y_mlp_s"],
                    outdir=outdir,
                ))

    paths.append(plot_cross_case_rmse(comparisons, outdir))
    paths.append(plot_rmse_per_horizon(comparisons, outdir))
    paths.append(plot_multistart_dispersion(comparisons, outdir))
    paths.append(plot_prior_ablation(comparisons, outdir))
    paths.append(plot_prediction_summary(checks, outdir))

    # ---- New figures (10–14) ----

    # Fig 10: J_obs strength summary
    paths.append(plot_jobs_summary(arrow_results, outdir))

    # Fig 11: GO overlay RMSE per horizon
    go_cases = [
        k for k in CASE_ORDER
        if k in comparisons
        and _case_verdict(k, comparisons=comparisons) == Verdict.GO
    ]
    if go_cases:
        paths.append(plot_rmse_horizon_go_overlay(comparisons, go_cases, outdir))

    if map_data:
        # Prepare aligned arrays for scatter and FIC figures
        retro_nlls_dict: Dict[str, np.ndarray] = {}
        Y_true_dict:     Dict[str, np.ndarray] = {}
        Y_hat_dict:      Dict[str, np.ndarray] = {}

        for key in go_cases:
            if key not in map_data:
                continue
            md = map_data[key]
            ds = md["dataset"]
            N_test = ds.X_test_s.shape[0]
            n_eval = md["map_results_flow"].n_samples
            eval_idx = (
                np.linspace(0, N_test - 1, n_eval).astype(int)
                if n_eval < N_test else np.arange(N_test)
            )
            retro_nlls_dict[key] = md["map_results_flow"].retro_nlls
            Y_true_dict[key]     = ds.Y_test_s[eval_idx]
            Y_hat_dict[key]      = md["map_results_flow"].Y_hat

        # Fig 12: RetroNLL vs RMSE scatter
        if retro_nlls_dict:
            paths.append(plot_retronll_vs_rmse(
                list(retro_nlls_dict.keys()),
                retro_nlls_dict, Y_true_dict, Y_hat_dict, outdir,
            ))

        # Fig 13: MAP loss distribution (all cases)
        for key in CASE_ORDER:
            if key not in map_data:
                continue
            md = map_data[key]
            verdict = _case_verdict(key, comparisons=comparisons)
            paths.append(plot_map_loss_distribution(
                key,
                md["map_results_flow"].map_losses,
                md["map_results_n0i"].map_losses,
                verdict, outdir,
            ))

        # Fig 14: FIC contribution (GO cases)
        for key in go_cases:
            if key not in map_data:
                continue
            md = map_data[key]
            paths.append(plot_fic_contribution(
                key,
                md["map_results_flow"].map_losses,
                md["map_results_flow"].multistart_std,
                outdir,
            ))

    print(f"\n  [Plotting] {len(paths)} figures saved to {outdir}/")
    return paths
