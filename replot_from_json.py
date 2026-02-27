#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replot_from_json.py — Regenerate all figures from results_all.json
=========================================================================

Given a JSON produced by export_results_json_enriched(), all 14 figures
are available without re-running training or inference.

Available figures:
  Fig  1 : Raw time-series overview              (requires series_data)
  Fig  2 : Arrow-of-time diagnostic              (always available)
  Fig  3 : Training curves                       (requires histories)
  Fig  4 : Example reconstructions (per case)    (requires arrays)
  Fig  5 : Cross-case RMSE (with CI if present)  (always available)
  Fig  6 : RMSE per forecast horizon             (always available)
  Fig  7 : Multi-start dispersion                (always available)
  Fig  8 : Flow vs N(0,I) prior ablation         (always available)
  Fig  9 : Prediction scorecard P1-P4            (always available)
  Fig 10 : J_obs strength summary (GO/NO-GO)     (always available)
  Fig 11 : RMSE per horizon overlay GO cases     (always available)
  Fig 12 : RetroNLL vs RMSE scatter              (requires arrays)
  Fig 13 : MAP loss distribution per case        (requires arrays)
  Fig 14 : FIC contribution scatter/CDF per case (requires arrays)

Usage:
    python replot_from_json.py --json results_all.json --outdir figures/
    python replot_from_json.py --json results_all.json --outdir figures/ --figs 1 3 4 12 13 14
    python replot_from_json.py --json results_all.json --outdir figures/ --figs 4 --cases A ERA5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Path projet
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

try:
    from config import CASES, CASE_ORDER, Verdict
except ImportError:
    class Verdict:
        class GO:
            value = "GO"
        class NOGO:
            value = "NO-GO"
    CASE_ORDER = ["A", "B", "C", "D", "ERA5", "ERA_ssrd"]
    CASES = {}

# ============================================================================
# PALETTE (identique a plotting.py)
# ============================================================================

COLORS = {
    "Naive mean":          "#999999",
    "Forward MLP":         "#E69F00",
    "Forward CVAE":        "#56B4E9",
    "Inverse MAP (flow)":  "#009E73",
    "Inverse MAP (N0I)":   "#CC79A7",
}

VERDICT_COLORS = {
    "GO":    "#009E73",
    "NO-GO": "#D55E00",
}

METHOD_ORDER = [
    "Naive mean", "Forward MLP", "Forward CVAE",
    "Inverse MAP (flow)", "Inverse MAP (N0I)",
]

MODEL_DISPLAY = {
    "inverse_cvae": ("Inverse CVAE", "#009E73"),
    "forward_cvae": ("Forward CVAE", "#56B4E9"),
    "forward_mlp":  ("Forward MLP",  "#E69F00"),
    "flow":         ("RealNVP Flow", "#CC79A7"),
}

# ============================================================================
# DATACLASSES RECONSTRUCTED FROM JSON
# ============================================================================

class _Verdict:
    def __init__(self, value: str):
        self.value = value
    def __eq__(self, other):
        if isinstance(other, _Verdict):
            return self.value == other.value
        return False
    def __repr__(self):
        return f"Verdict({self.value})"


def _verdict_color(v) -> str:
    if v is None:
        return "#333333"
    val = v.value if hasattr(v, "value") else str(v)
    return VERDICT_COLORS.get(val, "#333333")


class ArrowScaleResult:
    def __init__(self, d: dict):
        self.w           = d["w"]
        self.J_median    = d.get("J_obs", d.get("J_median", 0.0))
        self.J_ci_low    = d.get("J_null_ci_low",  d.get("J_ci_low",  0.0))
        self.J_ci_high   = d.get("J_null_ci_high", d.get("J_ci_high", 0.0))
        self.p_value     = d.get("p_value", 1.0)
        self.exceeds_tau = d.get("reject", False)
        self.n_samples   = d.get("n_samples", 0)


class SubDiagnosticResult:
    def __init__(self, d: dict):
        self.tag      = d["tag"]
        self.verdict  = _Verdict(d["verdict"])
        self.n_reject = d.get("n_reject", 0)
        self.scales   = [ArrowScaleResult(s) for s in d.get("scales", [])]


class ArrowOfTimeResult:
    def __init__(self, d: dict):
        self.verdict        = _Verdict(d["verdict"])
        self.overall_median = d.get("overall_median", 0.0)
        self.n_exceeding    = d.get("n_exceeding", 0)
        self.tau            = d.get("tau", 0.05)
        self.C_min          = d.get("C_min", 2)
        self.decision_rule  = d.get("decision_rule", "")
        self.scale_results  = [ArrowScaleResult(s) for s in d.get("scale_results", [])]
        self.level_result   = (SubDiagnosticResult(d["level"])
                               if "level" in d and d["level"] else None)
        self.diff_result    = (SubDiagnosticResult(d["diff"])
                               if "diff"  in d and d["diff"]  else None)


class MethodResult:
    def __init__(self, d: dict, method_name: str, case_key: str):
        self.method_name          = method_name
        self.case_key             = case_key
        self.rmse_s               = d.get("rmse_s", 0.0)
        self.mae_s                = d.get("mae_s", 0.0)
        self.rmse_orig            = d.get("rmse_orig", 0.0)
        self.mae_orig             = d.get("mae_orig", 0.0)
        self.skill_vs_naive       = d.get("skill_vs_naive", 0.0)
        self.mean_retro_nll       = d.get("mean_retro_nll")
        self.mean_prior_logprob   = d.get("mean_prior_logprob")
        self.mean_multistart_std  = d.get("mean_multistart_std")
        self.mean_map_loss        = d.get("mean_map_loss")
        self.n_samples            = d.get("n_samples", 0)
        rpp = d.get("rmse_per_pos_s")
        self.rmse_per_pos_s = np.array(rpp) if rpp is not None else None
        self.rmse_s_ci_low  = d.get("rmse_s_ci_low")
        self.rmse_s_ci_high = d.get("rmse_s_ci_high")
        self.Y_pred_s       = None   # non stocke au niveau method


class CaseComparison:
    def __init__(self, d: dict):
        self.case_key         = d["case_key"]
        self.case_name        = d.get("case_name", d["case_key"])
        self.verdict_expected = _Verdict(d.get("verdict_expected", "GO"))
        self.methods = {
            mname: MethodResult(mdata, mname, d["case_key"])
            for mname, mdata in d.get("methods", {}).items()
        }
        self.arrow_result = None
        # Per-sample arrays (from enriched JSON export)
        arr = d.get("arrays", {})
        self.X_test_s        = np.array(arr["X_test_s"])        if "X_test_s"        in arr else None
        self.Y_test_s        = np.array(arr["Y_test_s"])        if "Y_test_s"        in arr else None
        self.Y_naive_s       = np.array(arr["Y_naive_s"])       if "Y_naive_s"       in arr else None
        self.Y_mlp_s         = np.array(arr["Y_mlp_s"])         if "Y_mlp_s"         in arr else None
        self.Y_hat_flow      = np.array(arr["Y_hat_flow"])      if "Y_hat_flow"      in arr else None
        self.Y_hat_n0i       = np.array(arr["Y_hat_n0i"])       if "Y_hat_n0i"       in arr else None
        self.retro_nlls_flow = np.array(arr["retro_nlls_flow"]) if "retro_nlls_flow" in arr else None
        self.map_losses_flow = np.array(arr["map_losses_flow"]) if "map_losses_flow" in arr else None
        self.map_losses_n0i  = np.array(arr["map_losses_n0i"])  if "map_losses_n0i"  in arr else None
        self.multistart_std  = np.array(arr["multistart_std"])  if "multistart_std"  in arr else None


class PredictionCheck:
    def __init__(self, d: dict):
        self.pid       = d.get("pid", "")
        self.statement = d.get("statement", "")
        self.passed    = d.get("passed", False)
        self.details   = d.get("details", "")


# ============================================================================
# LOADING FROM JSON
# ============================================================================

def load_from_json(json_path: str):
    """Load results_all.json and reconstruct all result objects."""
    with open(json_path) as f:
        data = json.load(f)

    arrow_results: Dict[str, ArrowOfTimeResult] = {
        k: ArrowOfTimeResult(v)
        for k, v in data.get("arrow_of_time", {}).items()
    }

    comparisons: Dict[str, CaseComparison] = {
        k: CaseComparison(v)
        for k, v in data.get("comparisons", {}).items()
    }
    for k, comp in comparisons.items():
        if k in arrow_results:
            comp.arrow_result = arrow_results[k]

    pred_raw = data.get("predictions", {})
    if isinstance(pred_raw, dict):
        checks: List[PredictionCheck] = [
            PredictionCheck({"pid": pid, **v}) for pid, v in pred_raw.items()
        ]
    else:
        checks = [PredictionCheck(v) for v in pred_raw]

    series_data: Dict[str, np.ndarray] = {
        k: np.array(v) for k, v in data.get("series_data", {}).items()
    }
    histories: Dict[str, Dict] = data.get("histories", {})

    return arrow_results, comparisons, checks, series_data, histories


# ============================================================================
# HELPERS COMMUNS
# ============================================================================

def _apply_style():
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "font.family":       "sans-serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
    })


def _savefig(fig, filepath: str):
    for ext in ("png", "pdf"):
        fig.savefig(f"{filepath}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {filepath}.png")


def _ensure_dir(outdir: str):
    os.makedirs(outdir, exist_ok=True)


def _ordered_keys(keys):
    order = {k: i for i, k in enumerate(CASE_ORDER)}
    return sorted(keys, key=lambda k: order.get(k, 999))


def _case_name(key: str, comparisons=None) -> str:
    if CASES and key in CASES:
        return CASES[key].name
    if comparisons and key in comparisons:
        return comparisons[key].case_name
    return key


# ============================================================================
# FIG 1 -- Apercu des series temporelles
# ============================================================================

def plot_series_overview(
    series_data: Dict[str, np.ndarray],
    arrow_results: Dict[str, ArrowOfTimeResult],
    comparisons: Dict[str, CaseComparison],
    outdir: str,
    n_points: int = 2000,
) -> str:
    if not series_data:
        print("    [SKIP] series_data absent du JSON")
        return ""

    _apply_style()
    _ensure_dir(outdir)

    cases = _ordered_keys([k for k in series_data if k in arrow_results])
    if not cases:
        cases = _ordered_keys(list(series_data.keys()))

    fig, axes = plt.subplots(len(cases), 1,
                             figsize=(8, 1.6 * len(cases) + 1.0),
                             sharex=True)
    if len(cases) == 1:
        axes = [axes]

    for i, key in enumerate(cases):
        ax = axes[i]
        s  = series_data[key][:n_points]
        t  = np.arange(len(s))

        verdict = arrow_results[key].verdict if key in arrow_results else None
        color   = _verdict_color(verdict)
        ax.plot(t, s, color=color, linewidth=0.5, alpha=0.85)
        ax.set_ylabel(f"Case {key}", fontweight="bold", fontsize=8)

        name        = _case_name(key, comparisons)
        verdict_str = verdict.value if verdict is not None else "?"
        ax.text(0.98, 0.90, f"{name}\n[{verdict_str}]",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[-1].set_xlabel("Time step $t$")
    fig.suptitle("Time Series Overview", fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(outdir, "fig1_series_overview")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# FIG 2 -- Arrow-of-time diagnostic (bug corrige : labels x-axis)
# ============================================================================

def _plot_arrow_panel(ax_j, ax_p, cases, sub_results, alpha=0.05):
    all_ws    = sorted({sr.w for k in cases for sr in sub_results[k].scales})
    bar_width = 0.8 / max(len(all_ws), 1)
    x_base    = np.arange(len(cases))

    for j, w in enumerate(all_ws):
        j_obs_vals, p_vals, ci_lows, ci_highs, colors = [], [], [], [], []
        for k in cases:
            sr_match = [sr for sr in sub_results[k].scales if sr.w == w]
            if sr_match:
                sr = sr_match[0]
                j_obs_vals.append(sr.J_median)
                p_vals.append(sr.p_value)
                ci_lows.append(max(sr.J_median - sr.J_ci_low, 0))
                ci_highs.append(max(sr.J_ci_high - sr.J_median, 0))
            else:
                j_obs_vals.append(0.0); p_vals.append(1.0)
                ci_lows.append(0); ci_highs.append(0)
            colors.append(_verdict_color(sub_results[k].verdict))

        x = x_base + j * bar_width - 0.4 + bar_width / 2
        ax_j.bar(x, j_obs_vals, width=bar_width, label=f"w={w}",
                 color=colors, alpha=0.7, edgecolor="white", linewidth=0.5,
                 yerr=[ci_lows, ci_highs], capsize=2,
                 error_kw={"linewidth": 0.8})
        neg_log_p = [-np.log10(max(p, 1e-6)) for p in p_vals]
        ax_p.bar(x, neg_log_p, width=bar_width,
                 color=colors, alpha=0.7, edgecolor="white", linewidth=0.5)

    ax_p.axhline(y=-np.log10(alpha), color="red", linestyle="--",
                 linewidth=1, label=f"alpha={alpha}")

    ax_j.set_xticks(x_base); ax_j.set_xticklabels([])
    ax_j.set_ylabel("$J_{obs}(w)$")
    ax_j.legend(loc="upper right", fontsize=6)
    ax_j.set_ylim(bottom=0)

    ax_p.set_xticks(x_base)
    ax_p.set_xticklabels([f"{k}" for k in cases], fontsize=7)   # cle courte uniquement
    ax_p.set_ylabel("$-\\log_{10}(p)$")
    ax_p.legend(loc="upper right", fontsize=6)
    ax_p.set_ylim(bottom=0)

    for i, k in enumerate(cases):
        v = sub_results[k].verdict
        ax_p.annotate(v.value, xy=(i, 0), xytext=(0, -22),
                      textcoords="offset points", ha="center", fontsize=6,
                      fontweight="bold", color=_verdict_color(v))


def plot_arrow_of_time(arrow_results: Dict, outdir: str) -> str:
    _apply_style(); _ensure_dir(outdir)

    cases = _ordered_keys(list(arrow_results.keys()))
    if not cases:
        return ""

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("Arrow-of-Time Diagnostic (Block Permutation Test)",
                 fontweight="bold", y=1.01)

    for col, (attr, tag) in enumerate([
        ("level_result", "LEVEL ($x_t$)"),
        ("diff_result",  "DIFF ($\\Delta x_t$)"),
    ]):
        gs  = gridspec.GridSpec(2, 2, figure=fig,
                                left=0.07 + col * 0.5, right=0.47 + col * 0.5,
                                hspace=0.08, height_ratios=[2, 1])
        ax_j = fig.add_subplot(gs[0, :])
        ax_p = fig.add_subplot(gs[1, :])

        sub_results = {k: getattr(arrow_results[k], attr)
                       for k in cases if getattr(arrow_results[k], attr, None) is not None}
        valid = [k for k in cases if k in sub_results]
        if valid:
            _plot_arrow_panel(ax_j, ax_p, valid, sub_results)
            ax_j.set_title(tag, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(outdir, "fig2_arrow_of_time")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# FIG 3 -- Courbes d'apprentissage
# ============================================================================

def plot_training_curves(
    histories: Dict[str, Dict],
    comparisons: Dict[str, CaseComparison],
    outdir: str,
) -> str:
    if not histories:
        print("    [SKIP] histories absent du JSON")
        return ""

    _apply_style(); _ensure_dir(outdir)

    model_names = ["inverse_cvae", "forward_cvae", "forward_mlp", "flow"]
    cases       = _ordered_keys([k for k in histories if histories[k]])
    if not cases:
        return ""

    nrows = len(cases)
    ncols = len(model_names)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 2.5 * nrows),
                             squeeze=False)
    fig.suptitle("Training Curves", fontweight="bold", y=1.01)

    for row, key in enumerate(cases):
        hist_case = histories[key]
        case_name = _case_name(key, comparisons)
        for col, mname in enumerate(model_names):
            ax = axes[row][col]
            display, color = MODEL_DISPLAY.get(mname, (mname, "#333333"))

            if mname not in hist_case:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#aaaaaa")
                if row == 0:
                    ax.set_title(display, fontsize=8, fontweight="bold", color=color)
                continue

            h = hist_case[mname]
            train_key = (next((k for k in h if "train" in k and "loss" in k), None)
                         or next((k for k in h if "train" in k), None))
            val_key   = (next((k for k in h if "val" in k   and "loss" in k), None)
                         or next((k for k in h if "val" in k),   None))

            if train_key and h.get(train_key):
                ep = np.arange(1, len(h[train_key]) + 1)
                ax.plot(ep, h[train_key], color=color, linewidth=1.2, label="train")
            if val_key and h.get(val_key):
                ep = np.arange(1, len(h[val_key]) + 1)
                ax.plot(ep, h[val_key], color=color, linewidth=1.2,
                        linestyle="--", alpha=0.7, label="val")

            best_ep = h.get("best_epoch")
            if best_ep is not None and train_key and h.get(train_key):
                if 0 < int(best_ep) <= len(h[train_key]):
                    ax.axvline(x=int(best_ep), color="red",
                               linestyle=":", linewidth=0.8, alpha=0.7)

            if row == 0:
                ax.set_title(display, fontsize=8, fontweight="bold", color=color)
            if col == 0:
                ax.set_ylabel(f"Case {key}\n{case_name[:14]}", fontsize=7)
            ax.set_xlabel("Epoch", fontsize=7)
            ax.legend(fontsize=5, loc="upper right")

    fig.tight_layout()
    path = os.path.join(outdir, "fig3_training_curves")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# FIG 4 -- Exemples de reconstructions (par cas)
# ============================================================================

def plot_example_reconstructions(
    case_key: str,
    comp: CaseComparison,
    outdir: str,
    n_examples: int = 3,
) -> str:
    needed = [comp.X_test_s, comp.Y_test_s, comp.Y_hat_flow]
    if any(a is None for a in needed):
        print(f"    [SKIP Fig 4 {case_key}] arrays per-sample absents du JSON")
        return ""

    _apply_style(); _ensure_dir(outdir)

    X_test_s   = comp.X_test_s
    Y_test_s   = comp.Y_test_s
    Y_hat_flow = comp.Y_hat_flow
    Y_naive_s  = comp.Y_naive_s
    Y_mlp_s    = comp.Y_mlp_s

    N_eval = Y_hat_flow.shape[0]
    N_test = X_test_s.shape[0]
    n      = X_test_s.shape[1]
    m      = Y_test_s.shape[1]

    if N_eval < N_test:
        eval_idx = np.linspace(0, N_test - 1, N_eval).astype(int)
    else:
        eval_idx = np.arange(N_test)

    Y_true_eval = Y_test_s[eval_idx]
    rmse_per    = np.sqrt(np.mean((Y_true_eval - Y_hat_flow) ** 2, axis=1))
    sorted_idx  = np.argsort(rmse_per)
    n_ex        = min(n_examples, N_eval)
    sel_sorted  = [0, N_eval // 2, N_eval - 1][:n_ex]
    labels      = ["Best", "Median", "Worst"][:n_ex]

    t_past   = np.arange(-n, 0)
    t_future = np.arange(0, m)

    fig, axes = plt.subplots(1, n_ex, figsize=(4.5 * n_ex, 3.5), squeeze=False)
    fig.suptitle(
        f"Case {case_key}: {comp.case_name[:25]} "
        f"[{comp.verdict_expected.value}] -- Example Reconstructions",
        fontweight="bold", y=1.02,
    )

    for col, (si, label) in enumerate(zip(sel_sorted, labels)):
        ax = axes[0][col]
        ei = sorted_idx[si]
        ti = eval_idx[ei]

        ax.plot(t_past,   X_test_s[ti],   color="black",                linewidth=1.2, label="Past $x$")
        ax.plot(t_future, Y_test_s[ti],   color="black", linestyle="--",linewidth=1.2, label="True $y$")
        if Y_naive_s is not None:
            ax.plot(t_future, Y_naive_s[ti], color=COLORS["Naive mean"],
                    linewidth=0.8, alpha=0.7, linestyle=":", label="Naive")
        if Y_mlp_s is not None:
            ax.plot(t_future, Y_mlp_s[ti],   color=COLORS["Forward MLP"],
                    linewidth=0.8, alpha=0.8, linestyle="--", label="MLP")
        ax.plot(t_future, Y_hat_flow[ei], color=COLORS["Inverse MAP (flow)"],
                linewidth=1.5, label="Inverse MAP")

        ax.axvline(x=0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(f"{label} (RMSE={rmse_per[ei]:.3f})", fontsize=8)
        ax.set_xlabel("Relative time step")
        if col == 0:
            ax.set_ylabel("Standardised value")
            ax.legend(fontsize=6, loc="upper left")

    fig.tight_layout()
    path = os.path.join(outdir, f"fig4_examples_{case_key}")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# FIG 5 -- Cross-case RMSE barplot
# ============================================================================

def plot_cross_case_rmse(comparisons: Dict, outdir: str) -> str:
    _apply_style(); _ensure_dir(outdir)

    cases   = _ordered_keys(list(comparisons.keys()))
    methods = [m for m in METHOD_ORDER
               if any(m in comparisons[k].methods for k in cases)]

    x         = np.arange(len(cases))
    bar_width = 0.8 / len(methods)
    fig, ax   = plt.subplots(figsize=(max(8, len(cases) * 1.5), 4.5))

    for j, mname in enumerate(methods):
        rmses    = [comparisons[k].methods[mname].rmse_s
                    if mname in comparisons[k].methods else np.nan for k in cases]
        ci_low   = [comparisons[k].methods[mname].rmse_s_ci_low  or 0
                    if mname in comparisons[k].methods else 0 for k in cases]
        ci_high  = [comparisons[k].methods[mname].rmse_s_ci_high or 0
                    if mname in comparisons[k].methods else 0 for k in cases]
        yerr_low  = [max(r - l, 0) if r and l else 0 for r, l in zip(rmses, ci_low)]
        yerr_high = [max(h - r, 0) if r and h else 0 for r, h in zip(rmses, ci_high)]
        xpos = x + j * bar_width - 0.4 + bar_width / 2
        ax.bar(xpos, rmses, width=bar_width,
               color=COLORS.get(mname, "#333333"), alpha=0.85,
               label=mname, edgecolor="white",
               yerr=[yerr_low, yerr_high], capsize=2,
               error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{k}\n({'GO' if comparisons[k].verdict_expected.value == 'GO' else 'NO-GO'})"
         for k in cases],
        fontsize=8,
    )
    ax.set_ylabel("RMSE (standardised)")
    ax.set_title("Cross-Case RMSE Comparison", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    path = os.path.join(outdir, "fig5_cross_case_rmse")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# FIG 6 -- RMSE per horizon
# ============================================================================

def plot_rmse_per_horizon(comparisons: Dict, outdir: str) -> str:
    _apply_style(); _ensure_dir(outdir)

    cases   = _ordered_keys(list(comparisons.keys()))
    methods = [m for m in METHOD_ORDER
               if any(m in comparisons[k].methods for k in cases)]

    ncols = min(3, len(cases))
    nrows = (len(cases) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)

    for idx, key in enumerate(cases):
        row, col = divmod(idx, ncols)
        ax       = axes[row][col]
        comp     = comparisons[key]

        for mname in methods:
            if mname not in comp.methods:
                continue
            rpp = comp.methods[mname].rmse_per_pos_s
            if rpp is None:
                continue
            h = np.arange(1, len(rpp) + 1)
            ax.plot(h, rpp, color=COLORS.get(mname, "#333333"),
                    linewidth=1.0, label=mname, marker=".", markersize=3,
                    linestyle="--" if mname == "Forward MLP" else "-")

        ax.set_title(
            f"Case {key}: {comp.case_name[:20]} ({comp.verdict_expected.value})",
            fontsize=8,
        )
        ax.set_xlabel("Horizon h")
        ax.set_ylabel("RMSE(h)")
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    for idx in range(len(cases), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("RMSE per Forecast Horizon", fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(outdir, "fig6_rmse_per_horizon")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# FIG 7 -- Multi-start dispersion
# ============================================================================

def plot_multistart_dispersion(comparisons: Dict, outdir: str) -> str:
    _apply_style(); _ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    data_go, data_nogo     = [], []
    labels_go, labels_nogo = [], []

    for key in _ordered_keys(list(comparisons.keys())):
        comp    = comparisons[key]
        verdict = comp.verdict_expected.value
        inv     = comp.methods.get("Inverse MAP (flow)")
        if inv and inv.mean_multistart_std is not None:
            name_short = comp.case_name[:12]
            if verdict == "GO":
                data_go.append(inv.mean_multistart_std)
                labels_go.append(f"{key}\n({name_short})")
            else:
                data_nogo.append(inv.mean_multistart_std)
                labels_nogo.append(f"{key}\n({name_short})")

    all_labels = labels_go + labels_nogo
    all_values = data_go   + data_nogo
    all_colors = ([VERDICT_COLORS["GO"]]    * len(data_go) +
                  [VERDICT_COLORS["NO-GO"]] * len(data_nogo))

    if all_labels:
        x = np.arange(len(all_labels))
        ax.bar(x, all_values, color=all_colors, alpha=0.8, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, fontsize=7)
        vmax = max(all_values) if all_values else 1.0
        if data_go:
            ax.text((len(data_go) - 1) / 2, vmax * 1.08, "GO",
                    ha="center", fontweight="bold", color=VERDICT_COLORS["GO"])
        if data_nogo:
            ax.text(len(data_go) + (len(data_nogo) - 1) / 2, vmax * 1.08,
                    "NO-GO", ha="center", fontweight="bold",
                    color=VERDICT_COLORS["NO-GO"])

    ax.set_ylabel("Mean multi-start std(y)")
    ax.set_title("Multi-Start Dispersion", fontweight="bold")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = os.path.join(outdir, "fig7_multistart_dispersion")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# FIG 8 -- Prior ablation (flow vs N0I)
# ============================================================================

def plot_prior_ablation(comparisons: Dict, outdir: str) -> str:
    _apply_style(); _ensure_dir(outdir)

    cases     = _ordered_keys(list(comparisons.keys()))
    x         = np.arange(len(cases))
    bar_width = 0.35
    fig, ax   = plt.subplots(figsize=(max(6, len(cases) * 1.2), 4))

    rmse_flow = [comparisons[k].methods.get("Inverse MAP (flow)") for k in cases]
    rmse_n0i  = [comparisons[k].methods.get("Inverse MAP (N0I)")  for k in cases]

    ax.bar(x - bar_width / 2,
           [m.rmse_s if m else np.nan for m in rmse_flow],
           bar_width, label="Flow prior",
           color=COLORS["Inverse MAP (flow)"], alpha=0.85)
    ax.bar(x + bar_width / 2,
           [m.rmse_s if m else np.nan for m in rmse_n0i],
           bar_width, label="N(0,I) prior",
           color=COLORS["Inverse MAP (N0I)"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{k}\n({comparisons[k].verdict_expected.value})" for k in cases],
        fontsize=8,
    )
    ax.set_ylabel("RMSE (standardised)")
    ax.set_title("Prior Ablation: Flow vs N(0,I) (P2)", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "fig8_prior_ablation")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# FIG 9 -- Prediction scorecard
# ============================================================================

def plot_prediction_summary(checks: List[PredictionCheck], outdir: str) -> str:
    _apply_style(); _ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(8, max(3, len(checks) * 0.9 + 1)))
    n = len(checks)

    for i, chk in enumerate(checks):
        color = "#009E73" if chk.passed else "#D55E00"
        ax.barh(i, 1, height=0.6, color=color, alpha=0.85)
        ax.text(0.02, i, f"{chk.pid}: {chk.statement}",
                va="center", ha="left", fontsize=8.5, color="white", fontweight="bold")
        ax.text(0.98, i, "PASS" if chk.passed else "FAIL",
                va="center", ha="right", fontsize=9, color="white", fontweight="bold")

    n_pass = sum(1 for c in checks if c.passed)
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_xlabel(f"{n_pass}/{n} passed", fontsize=10, fontweight="bold")
    ax.set_title("Falsifiable Predictions Verification", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path = os.path.join(outdir, "fig9_prediction_summary")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# FIG 10 -- J_obs summary (bug corrige : key -> k dans set_yticklabels)
# ============================================================================

def plot_jobs_summary(
    arrow_results: Dict,
    outdir: str,
    comparisons: Dict = None,
) -> str:
    _apply_style(); _ensure_dir(outdir)

    cases        = _ordered_keys(list(arrow_results.keys()))
    sorted_cases = sorted(cases,
                          key=lambda k: arrow_results[k].overall_median,
                          reverse=True)
    j_max = max(arrow_results[k].overall_median for k in cases) or 1.0

    fig, ax = plt.subplots(figsize=(6.5, 0.9 * len(sorted_cases) + 1.8))

    for i, k in enumerate(sorted_cases):
        ar    = arrow_results[k]
        j_val = ar.overall_median
        ax.barh(i, j_val, color=_verdict_color(ar.verdict),
                alpha=0.82, edgecolor="white", height=0.6)
        ax.text(j_val + j_max * 0.01, i, f"{j_val:.2f}",
                va="center", fontsize=7.5, fontweight="bold")

    ax.set_yticks(range(len(sorted_cases)))
    ax.set_yticklabels(                                              # k correct
        [f"{k}: {_case_name(k, comparisons)[:24]}  [{arrow_results[k].verdict.value}]"
         for k in sorted_cases],
        fontsize=7.5,
    )
    ax.set_xlabel(r"$J_{obs}$ median (all windows & scales)", fontsize=8)
    ax.set_title("Arrow-of-Time Strength per Case (P1)", fontweight="bold")
    ax.set_xlim(left=0)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.legend(handles=[
        Patch(facecolor=VERDICT_COLORS["GO"],    label="GO"),
        Patch(facecolor=VERDICT_COLORS["NO-GO"], label="NO-GO"),
    ], fontsize=7, loc="lower right")
    fig.tight_layout()
    path = os.path.join(outdir, "fig10_jobs_summary")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# FIG 11 -- RMSE horizon GO overlay
# ============================================================================

def plot_rmse_horizon_go_overlay(comparisons: Dict, outdir: str) -> str:
    _apply_style(); _ensure_dir(outdir)

    go_cases = _ordered_keys([
        k for k, c in comparisons.items()
        if c.verdict_expected.value == "GO"
    ])
    if not go_cases:
        return ""

    ncols = min(2, len(go_cases))
    nrows = (len(go_cases) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle("RMSE per Horizon -- GO Cases\nShaded: Inverse MAP <= Forward CVAE",
                 fontweight="bold", y=1.02)

    plot_methods = [
        ("Forward MLP",        COLORS["Forward MLP"],        "--", 0.8),
        ("Forward CVAE",       COLORS["Forward CVAE"],       "-",  1.4),
        ("Inverse MAP (flow)", COLORS["Inverse MAP (flow)"], "-",  1.4),
    ]

    for idx, key in enumerate(go_cases):
        row, col = divmod(idx, ncols)
        ax   = axes[row][col]
        comp = comparisons[key]
        rpp  = {}

        for mname, color, ls, lw in plot_methods:
            if mname not in comp.methods:
                continue
            r = comp.methods[mname].rmse_per_pos_s
            if r is not None:
                rpp[mname] = np.array(r)
                h = np.arange(1, len(r) + 1)
                ax.plot(h, r, color=color, linewidth=lw,
                        linestyle=ls, label=mname, marker=".", markersize=3)

        if "Inverse MAP (flow)" in rpp and "Forward CVAE" in rpp:
            inv_r  = rpp["Inverse MAP (flow)"]
            cvae_r = rpp["Forward CVAE"]
            h      = np.arange(1, len(inv_r) + 1)
            mask   = inv_r <= cvae_r
            if mask.any():
                ax.fill_between(h, inv_r, cvae_r, where=mask, alpha=0.15,
                                color=COLORS["Inverse MAP (flow)"], label="Inv<=CVAE")

        ax.set_title(f"Case {key}: {comp.case_name[:18]} [GO]", fontsize=8)
        ax.set_xlabel("Horizon h"); ax.set_ylabel("RMSE(h)")
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    for idx in range(len(go_cases), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    path = os.path.join(outdir, "fig11_rmse_horizon_go_overlay")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# FIG 12 -- RetroNLL vs RMSE scatter
# ============================================================================

def plot_retronll_vs_rmse(comparisons: Dict, outdir: str) -> str:
    go_cases = _ordered_keys([
        k for k, c in comparisons.items()
        if c.verdict_expected.value == "GO"
           and c.retro_nlls_flow is not None
           and c.Y_test_s is not None
           and c.Y_hat_flow is not None
    ])
    if not go_cases:
        print("    [SKIP Fig 12] arrays retro_nlls/Y_hat absents du JSON")
        return ""

    _apply_style(); _ensure_dir(outdir)

    ncols = min(2, len(go_cases))
    nrows = (len(go_cases) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle(
        "RetroNLL vs Prediction RMSE (per sample, GO cases)\n"
        "Negative correlation validates MAP objective",
        fontweight="bold", y=1.02,
    )

    for idx, key in enumerate(go_cases):
        row, col = divmod(idx, ncols)
        ax   = axes[row][col]
        comp = comparisons[key]

        N_eval = comp.Y_hat_flow.shape[0]
        N_test = comp.Y_test_s.shape[0]
        eval_idx = (np.linspace(0, N_test - 1, N_eval).astype(int)
                    if N_eval < N_test else np.arange(N_test))

        Y_true_eval = comp.Y_test_s[eval_idx]
        rmse_per    = np.sqrt(np.mean((Y_true_eval - comp.Y_hat_flow) ** 2, axis=1))
        retro_nlls  = comp.retro_nlls_flow

        mask = np.isfinite(retro_nlls) & np.isfinite(rmse_per)
        if mask.sum() < 5:
            ax.text(0.5, 0.5, "Insufficient data",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        x_plot, y_plot = retro_nlls[mask], rmse_per[mask]
        ax.scatter(x_plot, y_plot, color=COLORS["Inverse MAP (flow)"],
                   alpha=0.3, s=8, linewidths=0)

        slope, intercept, r, p_val, _ = scipy_stats.linregress(x_plot, y_plot)
        x_line = np.linspace(x_plot.min(), x_plot.max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                color="red", linewidth=1.2, linestyle="--",
                label=f"r={r:.2f}, p={p_val:.2e}")

        ax.set_title(f"Case {key}: {comp.case_name[:18]}", fontsize=8)
        ax.set_xlabel("RetroNLL (MAP objective)")
        ax.set_ylabel("RMSE per sample")
        ax.legend(fontsize=6)

    for idx in range(len(go_cases), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    path = os.path.join(outdir, "fig12_retronll_vs_rmse")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# FIG 13 -- Distribution MAP losses (par cas)
# ============================================================================

def plot_map_loss_distribution(comparisons: Dict, outdir: str) -> str:
    eligible = _ordered_keys([
        k for k, c in comparisons.items()
        if c.map_losses_flow is not None
    ])
    if not eligible:
        print("    [SKIP Fig 13] map_losses absents du JSON")
        return ""

    _apply_style(); _ensure_dir(outdir)

    ncols = min(3, len(eligible))
    nrows = (len(eligible) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
    fig.suptitle("MAP Loss Distribution: Flow Prior vs N(0,I) (P2)",
                 fontweight="bold", y=1.01)

    for idx, key in enumerate(eligible):
        row, col = divmod(idx, ncols)
        ax   = axes[row][col]
        comp = comparisons[key]

        losses_flow = comp.map_losses_flow
        losses_n0i  = comp.map_losses_n0i
        bins        = max(20, min(60, len(losses_flow) // 15 + 5))

        ax.hist(losses_flow, bins=bins, alpha=0.65,
                color=COLORS["Inverse MAP (flow)"], label="Flow prior", density=True)
        if losses_n0i is not None:
            ax.hist(losses_n0i, bins=bins, alpha=0.50,
                    color=COLORS["Inverse MAP (N0I)"], label="N(0,I) prior", density=True)

        med_flow = float(np.median(losses_flow))
        ax.axvline(med_flow, color=COLORS["Inverse MAP (flow)"],
                   linestyle="--", linewidth=1.0, label=f"Median={med_flow:.2f}")
        if losses_n0i is not None:
            med_n0i = float(np.median(losses_n0i))
            ax.axvline(med_n0i, color=COLORS["Inverse MAP (N0I)"],
                       linestyle=":", linewidth=1.0, label=f"N0I Med={med_n0i:.2f}")

        ax.set_title(
            f"Case {key}: {comp.case_name[:18]} [{comp.verdict_expected.value}]",
            fontsize=8,
        )
        ax.set_xlabel("Final MAP loss")
        ax.set_ylabel("Density")
        ax.legend(fontsize=5)

    for idx in range(len(eligible), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    path = os.path.join(outdir, "fig13_map_loss_dist")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# FIG 14 -- FIC contribution scatter + CDF (par cas)
# ============================================================================

def plot_fic_contribution(comparisons: Dict, outdir: str) -> str:
    eligible = _ordered_keys([
        k for k, c in comparisons.items()
        if c.map_losses_flow is not None and c.multistart_std is not None
    ])
    if not eligible:
        print("    [SKIP Fig 14] map_losses/multistart_std absents du JSON")
        return ""

    _apply_style(); _ensure_dir(outdir)

    ncols = min(3, len(eligible))
    nrows = (len(eligible) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2,
                             figsize=(4.0 * ncols * 2, 3.2 * nrows), squeeze=False)
    fig.suptitle("Forward-Inverse Chaining (FIC) -- MAP Landscape",
                 fontweight="bold", y=1.01)

    for idx, key in enumerate(eligible):
        row      = idx // ncols
        col_base = (idx % ncols) * 2
        comp     = comparisons[key]

        map_losses   = comp.map_losses_flow
        ms_std       = comp.multistart_std
        color        = COLORS["Inverse MAP (flow)"]
        title_prefix = f"Case {key}: {comp.case_name[:14]} [{comp.verdict_expected.value}]"

        # Scatter : multistart_std vs MAP loss
        ax_s = axes[row][col_base]
        ax_s.scatter(ms_std, map_losses, color=color, alpha=0.3, s=8, linewidths=0)
        ax_s.axvline(float(np.median(ms_std)),   color="gray", linestyle="--",
                     linewidth=0.8, alpha=0.7)
        ax_s.axhline(float(np.median(map_losses)), color="gray", linestyle="--",
                     linewidth=0.8, alpha=0.7)
        ax_s.set_xlabel("Multi-start std(y)")
        ax_s.set_ylabel("MAP loss")
        ax_s.set_title(f"{title_prefix}\nScatter", fontsize=7)

        # CDF MAP loss
        ax_c = axes[row][col_base + 1]
        sorted_losses = np.sort(map_losses)
        ax_c.plot(sorted_losses, np.linspace(0, 1, len(sorted_losses)),
                  color=color, linewidth=1.5)
        p10 = float(np.percentile(map_losses, 10))
        p90 = float(np.percentile(map_losses, 90))
        ax_c.axvline(p10, color="orange", linestyle=":", linewidth=1.0,
                     label=f"P10={p10:.2f}")
        ax_c.axvline(p90, color="red",    linestyle=":", linewidth=1.0,
                     label=f"P90={p90:.2f}")
        ax_c.set_xlabel("MAP loss"); ax_c.set_ylabel("CDF")
        ax_c.set_title(f"{title_prefix}\nCDF", fontsize=7)
        ax_c.legend(fontsize=5); ax_c.set_ylim(0, 1)

    for idx in range(len(eligible), nrows * ncols):
        row = idx // ncols; col_base = (idx % ncols) * 2
        axes[row][col_base].set_visible(False)
        axes[row][col_base + 1].set_visible(False)

    fig.tight_layout()
    path = os.path.join(outdir, "fig14_fic_contribution")
    _savefig(fig, path)
    return path + ".png"


# ============================================================================
# REGISTRY ET MAIN
# ============================================================================

def _build_registry(arrow_results, comparisons, checks, series_data, histories):
    return {
        1:  ("Series overview",          lambda d: plot_series_overview(series_data, arrow_results, comparisons, d)),
        2:  ("Arrow-of-time diagnostic", lambda d: plot_arrow_of_time(arrow_results, d)),
        3:  ("Training curves",          lambda d: plot_training_curves(histories, comparisons, d)),
        4:  ("Example reconstructions",  lambda d: any(
                plot_example_reconstructions(k, comparisons[k], d)
                for k in _ordered_keys(list(comparisons.keys()))
            )),
        5:  ("Cross-case RMSE",          lambda d: plot_cross_case_rmse(comparisons, d)),
        6:  ("RMSE per horizon",         lambda d: plot_rmse_per_horizon(comparisons, d)),
        7:  ("Multi-start dispersion",   lambda d: plot_multistart_dispersion(comparisons, d)),
        8:  ("Prior ablation",           lambda d: plot_prior_ablation(comparisons, d)),
        9:  ("Prediction scorecard",     lambda d: plot_prediction_summary(checks, d)),
        10: ("J_obs summary",            lambda d: plot_jobs_summary(arrow_results, d, comparisons=comparisons)),
        11: ("RMSE horizon GO overlay",  lambda d: plot_rmse_horizon_go_overlay(comparisons, d)),
        12: ("RetroNLL vs RMSE",         lambda d: plot_retronll_vs_rmse(comparisons, d)),
        13: ("MAP loss distribution",    lambda d: plot_map_loss_distribution(comparisons, d)),
        14: ("FIC contribution",         lambda d: plot_fic_contribution(comparisons, d)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate all 14 figures from an enriched results_all.json"
    )
    parser.add_argument("--json",   default="results_all.json")
    parser.add_argument("--outdir", default="figures_replot")
    parser.add_argument("--figs",   nargs="*", type=int,
                        help="Figure numbers to generate. Ex: --figs 1 3 4")
    parser.add_argument("--cases",  nargs="*",
                        help="Filtrer cas pour Fig 4. Ex: --cases A ERA5")
    args = parser.parse_args()

    print(f"\nChargement de {args.json} ...")
    arrow_results, comparisons, checks, series_data, histories = load_from_json(args.json)

    n_arrays = sum(1 for c in comparisons.values() if c.Y_hat_flow is not None)
    print(f"  {len(arrow_results)} cas  |  {len(checks)} predictions  |  "
          f"series_data: {len(series_data)}  |  histories: {len(histories)}  |  "
          f"per-sample arrays: {n_arrays}/{len(comparisons)}")

    if args.cases:
        comparisons = {k: v for k, v in comparisons.items() if k in args.cases}

    registry     = _build_registry(arrow_results, comparisons, checks, series_data, histories)
    figs_to_plot = args.figs if args.figs else sorted(registry.keys())
    figs_to_plot = [f for f in figs_to_plot if f in registry]

    print(f"\nGeneration de {len(figs_to_plot)} figure(s) --> {args.outdir}/\n")
    _ensure_dir(args.outdir)

    ok, skipped, errors = [], [], []
    for fig_num in figs_to_plot:
        label, fn = registry[fig_num]
        print(f"  Fig {fig_num:2d}: {label} ...")
        try:
            result = fn(args.outdir)
            if not result:
                skipped.append(fig_num)
            else:
                ok.append(fig_num)
        except Exception as e:
            print(f"    ERREUR: {e}")
            errors.append((fig_num, str(e)))

    print(f"\n{'─'*55}")
    print(f"  OK      : {len(ok)}  {ok}")
    if skipped:
        print(f"  Skipped : {len(skipped)}  {skipped}  (missing data)")
    if errors:
        print(f"  Erreurs : {len(errors)}")
        for fn, msg in errors:
            print(f"    Fig {fn}: {msg}")
    if skipped:
        print("\n  Relancer avec un JSON genere par export_results_json_enriched()")
    print(f"\n  Figures dans : {args.outdir}/")


if __name__ == "__main__":
    main()
