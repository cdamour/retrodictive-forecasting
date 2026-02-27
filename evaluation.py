#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py - Metrics and comparison for Retrodictive Forecasting
=============================================================================

Computes all metrics and verifies falsifiable
predictions P1-P4.

Metrics per method
------------------
  - RMSE (root mean squared error, per-position and global)
  - MAE  (mean absolute error, per-position and global)
  - Retrodictive NLL (from MAP inference only)
  - Prior log-prob (from MAP inference only)
  - Multi-start dispersion (std across K restarts, from MAP only)

Methods compared
----------------
  1. Naive mean baseline (unconditional train mean of y)
  2. Forward MLP         (deterministic y = f(x))
  3. Forward CVAE        (y = mu_theta(x, z=0))
  4. Inverse MAP (flow)  (retrodictive MAP with RealNVP prior)
  5. Inverse MAP (N0I)   (retrodictive MAP with N(0,I) prior - ablation)

Output
------
  - Per-case comparison table
  - Cross-case summary table
  - Falsifiable predictions verification (P1–P4)
  - JSON export of all results for reproducibility

Usage
-----
    from evaluation import (
        compute_metrics, build_comparison_table,
        verify_predictions, export_results_json,
    )
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    CASES,
    CASE_ORDER,
    PREDICTIONS,
    Verdict,
    FullConfig,
)
from generators import Standardizer
from inference import BatchMAPResults
from diagnostics import ArrowOfTimeResult


# ============================================================================
# 1. CORE METRICS
# ============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (global, across all positions and samples).

    Parameters
    ----------
    y_true : (N, m) - ground truth
    y_pred : (N, m) - predictions

    Returns
    -------
    float
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (global).

    Parameters
    ----------
    y_true : (N, m)
    y_pred : (N, m)

    Returns
    -------
    float
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_per_position(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """RMSE per forecast horizon position.

    Parameters
    ----------
    y_true : (N, m)
    y_pred : (N, m)

    Returns
    -------
    np.ndarray, shape (m,) - RMSE at each position t+1, ..., t+m
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def mae_per_position(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAE per forecast horizon position.

    Parameters
    ----------
    y_true : (N, m)
    y_pred : (N, m)

    Returns
    -------
    np.ndarray, shape (m,)
    """
    return np.mean(np.abs(y_true - y_pred), axis=0)


def skill_score(rmse_model: float, rmse_baseline: float) -> float:
    """RMSE skill score relative to baseline.

    SS = 1 - RMSE_model / RMSE_baseline

    SS > 0 : model improves over baseline
    SS = 0 : same as baseline
    SS < 0 : worse than baseline

    Parameters
    ----------
    rmse_model : float
    rmse_baseline : float - typically naive mean

    Returns
    -------
    float
    """
    if rmse_baseline < 1e-12:
        return 0.0
    return 1.0 - rmse_model / rmse_baseline


# ============================================================================
# 2. METHOD RESULT CONTAINER
# ============================================================================

@dataclass
class MethodResult:
    """Evaluation results for a single method on a single case.

    All metrics are computed on the TEST set in STANDARDISED space
    unless otherwise noted.
    """
    method_name: str
    case_key: str

    # Global metrics (standardised space)
    rmse_s: float = 0.0
    mae_s: float = 0.0

    # Global metrics (original scale, after inverse transform)
    rmse_orig: float = 0.0
    mae_orig: float = 0.0

    # Per-position RMSE (standardised)
    rmse_per_pos_s: Optional[np.ndarray] = None

    # Skill score vs naive
    skill_vs_naive: float = 0.0

    # MAP-specific metrics (only for inverse methods)
    mean_retro_nll: Optional[float] = None
    mean_prior_logprob: Optional[float] = None
    mean_multistart_std: Optional[float] = None
    mean_map_loss: Optional[float] = None

    # Predictions (standardised)
    Y_pred_s: Optional[np.ndarray] = None

    # Number of test samples evaluated
    n_samples: int = 0


# ============================================================================
# 3. COMPUTE METRICS FOR ALL METHODS
# ============================================================================

def compute_naive_metrics(
    Y_test_s: np.ndarray,
    y_mean_s: np.ndarray,
    sy: Standardizer,
    Y_test_raw: np.ndarray,
    case_key: str,
) -> MethodResult:
    """Evaluate the naive mean baseline.

    y_naive = mean(Y_train) for all test samples.

    Parameters
    ----------
    Y_test_s : (N, m) - true test futures (standardised)
    y_mean_s : (1, m) - train mean future (standardised)
    sy : Standardizer - for inverse transform
    Y_test_raw : (N, m) - true test futures (original scale)
    case_key : str

    Returns
    -------
    MethodResult
    """
    N = Y_test_s.shape[0]
    Y_pred_s = np.tile(y_mean_s, (N, 1))  # (N, m)

    # Original scale
    Y_pred_orig = sy.inverse(Y_pred_s)

    return MethodResult(
        method_name="Naive mean",
        case_key=case_key,
        rmse_s=rmse(Y_test_s, Y_pred_s),
        mae_s=mae(Y_test_s, Y_pred_s),
        rmse_orig=rmse(Y_test_raw, Y_pred_orig),
        mae_orig=mae(Y_test_raw, Y_pred_orig),
        rmse_per_pos_s=rmse_per_position(Y_test_s, Y_pred_s),
        skill_vs_naive=0.0,  # by definition
        Y_pred_s=Y_pred_s,
        n_samples=N,
    )


def compute_forward_metrics(
    Y_test_s: np.ndarray,
    Y_pred_s: np.ndarray,
    sy: Standardizer,
    Y_test_raw: np.ndarray,
    rmse_naive: float,
    method_name: str,
    case_key: str,
) -> MethodResult:
    """Evaluate a forward prediction method (MLP or CVAE).

    Parameters
    ----------
    Y_test_s : (N, m) - true test futures (standardised)
    Y_pred_s : (N, m) - predicted test futures (standardised)
    sy : Standardizer
    Y_test_raw : (N, m)
    rmse_naive : float - RMSE of naive baseline (for skill score)
    method_name : str
    case_key : str

    Returns
    -------
    MethodResult
    """
    N = Y_test_s.shape[0]
    Y_pred_orig = sy.inverse(Y_pred_s)

    rmse_val = rmse(Y_test_s, Y_pred_s)

    return MethodResult(
        method_name=method_name,
        case_key=case_key,
        rmse_s=rmse_val,
        mae_s=mae(Y_test_s, Y_pred_s),
        rmse_orig=rmse(Y_test_raw, Y_pred_orig),
        mae_orig=mae(Y_test_raw, Y_pred_orig),
        rmse_per_pos_s=rmse_per_position(Y_test_s, Y_pred_s),
        skill_vs_naive=skill_score(rmse_val, rmse_naive),
        Y_pred_s=Y_pred_s,
        n_samples=N,
    )


def compute_inverse_metrics(
    Y_test_s: np.ndarray,
    batch_results: BatchMAPResults,
    sy: Standardizer,
    Y_test_raw: np.ndarray,
    rmse_naive: float,
    method_name: str,
    case_key: str,
    eval_indices: Optional[np.ndarray] = None,
) -> MethodResult:
    """Evaluate inverse MAP inference results.

    Parameters
    ----------
    Y_test_s : (N_test, m) - full test set ground truth (standardised)
    batch_results : BatchMAPResults - from map_infer_batch
    sy : Standardizer
    Y_test_raw : (N_test, m) - full test set (original scale)
    rmse_naive : float
    method_name : str
    case_key : str
    eval_indices : optional (n_eval,) - which test indices were evaluated
                   (if subsampled in map_infer_batch)

    Returns
    -------
    MethodResult
    """
    n_eval = batch_results.n_samples
    Y_hat_s = batch_results.Y_hat  # (n_eval, m)

    # Get corresponding ground truth
    if eval_indices is not None:
        Y_true_s = Y_test_s[eval_indices]
        Y_true_raw = Y_test_raw[eval_indices]
    else:
        # Assume uniform subsampling (same as map_infer_batch)
        N_test = Y_test_s.shape[0]
        if n_eval < N_test:
            indices = np.linspace(0, N_test - 1, n_eval).astype(int)
        else:
            indices = np.arange(N_test)
        Y_true_s = Y_test_s[indices]
        Y_true_raw = Y_test_raw[indices]

    # Original scale predictions
    Y_hat_orig = sy.inverse(Y_hat_s)

    rmse_val = rmse(Y_true_s, Y_hat_s)

    return MethodResult(
        method_name=method_name,
        case_key=case_key,
        rmse_s=rmse_val,
        mae_s=mae(Y_true_s, Y_hat_s),
        rmse_orig=rmse(Y_true_raw, Y_hat_orig),
        mae_orig=mae(Y_true_raw, Y_hat_orig),
        rmse_per_pos_s=rmse_per_position(Y_true_s, Y_hat_s),
        skill_vs_naive=skill_score(rmse_val, rmse_naive),
        mean_retro_nll=float(np.mean(batch_results.retro_nlls)),
        mean_prior_logprob=float(np.mean(batch_results.prior_logprobs)),
        mean_multistart_std=float(np.mean(batch_results.multistart_std)),
        mean_map_loss=float(np.mean(batch_results.map_losses)),
        Y_pred_s=Y_hat_s,
        n_samples=n_eval,
    )


# ============================================================================
# 4. COMPARISON TABLES
# ============================================================================

@dataclass
class CaseComparison:
    """All method results for a single case."""
    case_key: str
    case_name: str
    verdict_expected: Verdict
    arrow_result: Optional[ArrowOfTimeResult] = None
    methods: Dict[str, MethodResult] = field(default_factory=dict)


def build_comparison_table(comparison: CaseComparison) -> str:
    """Format a single-case comparison as an ASCII table.

    Parameters
    ----------
    comparison : CaseComparison

    Returns
    -------
    str - formatted table
    """
    lines = []
    lines.append(f"Case {comparison.case_key}: {comparison.case_name} "
                  f"(expected: {comparison.verdict_expected.value})")

    if comparison.arrow_result:
        ar = comparison.arrow_result
        # Show combined verdict + sub-diagnostics
        lv = ar.level_result.verdict.value if ar.level_result else "?"
        dv = ar.diff_result.verdict.value if ar.diff_result else "?"
        lines.append(f"  Arrow-of-time: {ar.verdict.value} "
                      f"[LEVEL={lv}, DIFF={dv}] "
                      f"({ar.decision_rule})")

    lines.append("")
    header = (
        f"  {'Method':<22} {'RMSE(s)':<10} {'MAE(s)':<10} "
        f"{'Skill%':<8} {'RetroNLL':<10} {'PriorLP':<10} {'MS-std':<8} "
        f"{'N':<6}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for name in ["Naive mean", "Forward MLP", "Forward CVAE",
                  "Inverse MAP (flow)", "Inverse MAP (N0I)"]:
        if name not in comparison.methods:
            continue
        m = comparison.methods[name]

        retro_str = f"{m.mean_retro_nll:.4f}" if m.mean_retro_nll is not None else "-"
        prior_str = f"{m.mean_prior_logprob:.4f}" if m.mean_prior_logprob is not None else "-"
        ms_str = f"{m.mean_multistart_std:.4f}" if m.mean_multistart_std is not None else "-"

        lines.append(
            f"  {name:<22} {m.rmse_s:<10.4f} {m.mae_s:<10.4f} "
            f"{m.skill_vs_naive*100:<8.1f} {retro_str:<10} {prior_str:<10} "
            f"{ms_str:<8} {m.n_samples:<6}"
        )

    return "\n".join(lines)


def build_cross_case_table(comparisons: Dict[str, CaseComparison]) -> str:
    """Build a cross-case summary table (key metric per method x case).

    Shows RMSE(standardised) and skill score for each method.

    Parameters
    ----------
    comparisons : dict mapping case_key -> CaseComparison

    Returns
    -------
    str - formatted table
    """
    methods_order = [
        "Naive mean", "Forward MLP", "Forward CVAE",
        "Inverse MAP (flow)", "Inverse MAP (N0I)",
    ]

    lines = []
    lines.append("Cross-case RMSE (standardised) and Skill Score (%)")
    lines.append("")

    # Header
    case_keys = [k for k in CASE_ORDER if k in comparisons]
    header = f"  {'Method':<22}"
    for k in case_keys:
        header += f" {k}:RMSE  {k}:SS% "
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for mname in methods_order:
        row = f"  {mname:<22}"
        for k in case_keys:
            comp = comparisons[k]
            if mname in comp.methods:
                m = comp.methods[mname]
                row += f" {m.rmse_s:6.4f} {m.skill_vs_naive*100:5.1f}%"
            else:
                row += f" {'-':>6} {'-':>5} "
        lines.append(row)

    return "\n".join(lines)


# ============================================================================
# 5. FALSIFIABLE PREDICTIONS VERIFICATION
# ============================================================================

@dataclass
class PredictionCheck:
    """Result of checking one falsifiable prediction."""
    pid: str
    statement: str
    passed: bool
    details: str


def verify_predictions(
    comparisons: Dict[str, CaseComparison],
    arrow_results: Dict[str, ArrowOfTimeResult],
    tau: float = 0.05,
    tolerance: float = 0.05,
) -> List[PredictionCheck]:
    """Verify all 4 falsifiable predictions.

    Parameters
    ----------
    comparisons : dict case_key -> CaseComparison
    arrow_results : dict case_key -> ArrowOfTimeResult
    tau : float -- threshold for P1
    tolerance : float -- relative tolerance band (5%), used symmetrically:
        - P3 (NO-GO): RMSE_inv / RMSE_MLP >= (1 - tolerance) = 0.95
        - P4 (GO):    RMSE_inv / RMSE_MLP <= (1 + tolerance) = 1.05
    Returns
    -------
    List[PredictionCheck] -- one per prediction (P1-P4)
    """
    checks: List[PredictionCheck] = []

    # --- P1: Arrow-of-time diagnostic matches expected verdicts ---
    p1_pass = True
    p1_details = []
    for k in ["A", "C"]:
        if k in arrow_results:
            ar = arrow_results[k]
            ok = ar.verdict == Verdict.GO
            lv = ar.level_result.verdict.value if ar.level_result else "?"
            dv = ar.diff_result.verdict.value if ar.diff_result else "?"
            sym = "OK" if ok else "NO"
            p1_details.append(f"{k}:{ar.verdict.value}[L={lv},D={dv}]({sym})")
            if not ok:
                p1_pass = False
    for k in ["B", "D"]:
        if k in arrow_results:
            ar = arrow_results[k]
            ok = ar.verdict == Verdict.NOGO
            lv = ar.level_result.verdict.value if ar.level_result else "?"
            dv = ar.diff_result.verdict.value if ar.diff_result else "?"
            sym = "OK" if ok else "NO"
            p1_details.append(f"{k}:{ar.verdict.value}[L={lv},D={dv}]({sym})")
            if not ok:
                p1_pass = False
    checks.append(PredictionCheck(
        "P1", PREDICTIONS["P1"]["statement"], p1_pass, ", ".join(p1_details),
    ))

    # --- P2: Flow prior improves inverse inference vs N(0,I) on GO ---
    p2_pass = True
    p2_details = []
    for k in ["A", "C"]:
        if k in comparisons:
            comp = comparisons[k]
            inv_flow = comp.methods.get("Inverse MAP (flow)")
            inv_n0i = comp.methods.get("Inverse MAP (N0I)")
            if inv_flow and inv_n0i:
                ok = inv_flow.rmse_s < inv_n0i.rmse_s
                mark = "OK" if ok else "NO"
                p2_details.append(
                    f"{k}: flow={inv_flow.rmse_s:.4f} vs N0I={inv_n0i.rmse_s:.4f} "
                    f"({mark})"
                )
                if not ok:
                    p2_pass = False
    checks.append(PredictionCheck(
        "P2", PREDICTIONS["P2"]["statement"], p2_pass, "; ".join(p2_details),
    ))

    # --- P3: Inverse MAP does not offer a meaningful advantage on NO-GO cases ---
    p3_pass = True
    p3_details = []
    for k in ["B", "D"]:
        if k in comparisons:
            comp = comparisons[k]
            mlp = comp.methods.get("Forward MLP")
            inv = comp.methods.get("Inverse MAP (flow)")
            if mlp and inv:
                ratio = inv.rmse_s / (mlp.rmse_s + 1e-12)
                ok = ratio >= 0.95
                mark = "OK" if ok else "NO"
                p3_details.append(
                    f"{k}: inv={inv.rmse_s:.4f} vs MLP={mlp.rmse_s:.4f} "
                    f"({mark})"
                )
                if not ok:
                    p3_pass = False
    checks.append(PredictionCheck(
        "P3", PREDICTIONS["P3"]["statement"], p3_pass, "; ".join(p3_details),
    ))


    # --- P4: Inverse MAP competitive with or beats MLP on GO ---
    p4_pass = True
    p4_details = []
    for k in ["A", "C", "ERA5", "ERA_ssrd"]:
        if k in comparisons:
            comp = comparisons[k]
            mlp = comp.methods.get("Forward MLP")
            inv = comp.methods.get("Inverse MAP (flow)")
            if mlp and inv:
                # "competitive" = within 5% or better
                ratio = inv.rmse_s / (mlp.rmse_s + 1e-12)
                ok = ratio <= 1.05
                mark = "OK" if ok else "NO"
                p4_details.append(
                    f"{k}: inv={inv.rmse_s:.4f} vs MLP={mlp.rmse_s:.4f} "
                    f"ratio={ratio:.3f} ({mark})"
                )
                if not ok:
                    p4_pass = False
    checks.append(PredictionCheck(
        "P4", PREDICTIONS["P4"]["statement"], p4_pass, "; ".join(p4_details),
    ))

    return checks




def format_prediction_checks(checks: List[PredictionCheck]) -> str:
    """Format prediction verification results as readable text.

    Parameters
    ----------
    checks : List[PredictionCheck]

    Returns
    -------
    str
    """
    lines = []
    lines.append("Falsifiable Predictions Verification")
    lines.append("=" * 60)

    n_passed = sum(1 for c in checks if c.passed)
    n_total = len(checks)

    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"\n  {c.pid}: {c.statement}")
        lines.append(f"      {status}")
        lines.append(f"      {c.details}")

    lines.append(f"\n  Overall: {n_passed}/{n_total} predictions passed")

    return "\n".join(lines)


# ============================================================================
# 6. JSON EXPORT
# ============================================================================

def _to_serializable(obj):
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.generic,)):
        # Fallback for other NumPy scalars
        return obj.item()
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, set):
        return [_to_serializable(v) for v in sorted(obj)]
    elif isinstance(obj, Verdict):
        return obj.value
    elif hasattr(obj, "__dataclass_fields__"):
        return _to_serializable(asdict(obj))
    elif hasattr(obj, "detach") and hasattr(obj, "cpu"):
        # torch.Tensor (best-effort)
        try:
            return obj.detach().cpu().numpy().tolist()
        except Exception:
            return str(obj)
    else:
        return obj


def export_results_json(
    comparisons: Dict[str, CaseComparison],
    arrow_results: Dict[str, ArrowOfTimeResult],
    checks: List[PredictionCheck],
    filepath: str,
) -> None:
    """Export all results to a JSON file for reproducibility.

    Parameters
    ----------
    comparisons : dict case_key -> CaseComparison
    arrow_results : dict case_key -> ArrowOfTimeResult
    checks : List[PredictionCheck]
    filepath : str - output path
    """
    results = {
        "arrow_of_time": {},
        "comparisons": {},
        "predictions": [],
    }

    # Arrow-of-time results with dual sub-diagnostics
    for k, ar in arrow_results.items():
        def _serialize_scales(scales):
            return [
                {
                    "w": sr.w,
                    "J_obs": sr.J_median,
                    "J_null_ci_low": sr.J_ci_low,
                    "J_null_ci_high": sr.J_ci_high,
                    "p_value": getattr(sr, "p_value", None),
                    "reject": sr.exceeds_tau,
                    "n_samples": getattr(sr, "n_samples", None),
                }
                for sr in scales
            ]

        def _serialize_sub(sub):
            if sub is None:
                return None
            return {
                "tag": sub.tag,
                "verdict": sub.verdict.value,
                "n_reject": sub.n_reject,
                "scales": _serialize_scales(sub.scales),
            }

        results["arrow_of_time"][k] = {
            "verdict": ar.verdict.value,
            "decision_rule": ar.decision_rule,
            "overall_median": ar.overall_median,
            "n_exceeding": ar.n_exceeding,
            "tau": ar.tau,
            "C_min": ar.C_min,
            "level": _serialize_sub(getattr(ar, "level_result", None)),
            "diff": _serialize_sub(getattr(ar, "diff_result", None)),
            "scale_results": _serialize_scales(ar.scale_results),
        }

    # Comparison results (without large arrays)
    for k, comp in comparisons.items():
        comp_dict = {
            "case_key": comp.case_key,
            "case_name": comp.case_name,
            "verdict_expected": comp.verdict_expected.value,
            "methods": {},
        }
        for mname, mr in comp.methods.items():
            comp_dict["methods"][mname] = {
                "rmse_s": mr.rmse_s,
                "mae_s": mr.mae_s,
                "rmse_orig": mr.rmse_orig,
                "mae_orig": mr.mae_orig,
                "skill_vs_naive": mr.skill_vs_naive,
                "mean_retro_nll": mr.mean_retro_nll,
                "mean_prior_logprob": mr.mean_prior_logprob,
                "mean_multistart_std": mr.mean_multistart_std,
                "mean_map_loss": mr.mean_map_loss,
                "n_samples": mr.n_samples,
                "rmse_per_pos_s": (
                    mr.rmse_per_pos_s.tolist()
                    if mr.rmse_per_pos_s is not None else None
                ),
            }
        results["comparisons"][k] = comp_dict

    # Prediction checks
    for c in checks:
        results["predictions"].append({
            "pid": c.pid,
            "statement": c.statement,
            "passed": c.passed,
            "details": c.details,
        })

    # Summary stats
    n_passed = sum(1 for c in checks if c.passed)
    results["summary"] = {
        "n_predictions_passed": n_passed,
        "n_predictions_total": len(checks),
    }

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(_to_serializable(results), f, indent=2)


# ============================================================================
# 7. SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Retrodictive Forecasting - Evaluation Module Self-Test")
    print("=" * 60)

    # Dummy data
    N, m = 100, 16
    Y_true = np.random.randn(N, m).astype(np.float32)
    Y_pred_good = Y_true + 0.1 * np.random.randn(N, m).astype(np.float32)
    Y_pred_bad = np.random.randn(N, m).astype(np.float32)

    print(f"\n--- Core metrics ---")
    print(f"  RMSE (good pred): {rmse(Y_true, Y_pred_good):.4f}")
    print(f"  RMSE (bad pred):  {rmse(Y_true, Y_pred_bad):.4f}")
    print(f"  MAE (good pred):  {mae(Y_true, Y_pred_good):.4f}")
    print(f"  MAE (bad pred):   {mae(Y_true, Y_pred_bad):.4f}")

    rmse_pp = rmse_per_position(Y_true, Y_pred_good)
    print(f"  RMSE per position shape: {rmse_pp.shape}")
    print(f"  RMSE per position (first 4): {rmse_pp[:4]}")

    ss = skill_score(rmse(Y_true, Y_pred_good), rmse(Y_true, Y_pred_bad))
    print(f"  Skill score (good vs bad as baseline): {ss:.4f}")

    print(f"\n--- Prediction checks (dummy) ---")
    # Create minimal dummy structures for testing
    from diagnostics import ArrowOfTimeResult, ArrowScaleResult

    dummy_arrow_go = ArrowOfTimeResult(
        verdict=Verdict.GO, overall_median=0.15,
        scale_results=[], n_exceeding=3, tau=0.05, C_min=2,
        decision_rule="test",
    )
    dummy_arrow_nogo = ArrowOfTimeResult(
        verdict=Verdict.NOGO, overall_median=0.01,
        scale_results=[], n_exceeding=0, tau=0.05, C_min=2,
        decision_rule="test",
    )

    dummy_comparisons = {}
    for k in CASE_ORDER:
        comp = CaseComparison(
            case_key=k, case_name=CASES[k].name,
            verdict_expected=CASES[k].verdict,
        )
        comp.methods["Naive mean"] = MethodResult(
            method_name="Naive mean", case_key=k, rmse_s=1.0, n_samples=100,
        )
        comp.methods["Forward MLP"] = MethodResult(
            method_name="Forward MLP", case_key=k, rmse_s=0.8, n_samples=100,
        )
        comp.methods["Inverse MAP (flow)"] = MethodResult(
            method_name="Inverse MAP (flow)", case_key=k,
            rmse_s=0.7 if CASES[k].verdict == Verdict.GO else 1.05,
            mean_multistart_std=0.1 if CASES[k].verdict == Verdict.GO else 0.5,
            n_samples=100,
        )
        comp.methods["Inverse MAP (N0I)"] = MethodResult(
            method_name="Inverse MAP (N0I)", case_key=k,
            rmse_s=0.75 if CASES[k].verdict == Verdict.GO else 1.1,
            n_samples=100,
        )
        dummy_comparisons[k] = comp

    dummy_arrow_results = {
        "A": dummy_arrow_go, "C": dummy_arrow_go,
        "ERA5": dummy_arrow_go, "ERA_ssrd": dummy_arrow_go,
        "B": dummy_arrow_nogo, "D": dummy_arrow_nogo,
    }

    checks = verify_predictions(dummy_comparisons, dummy_arrow_results)
    print(format_prediction_checks(checks))

    # Test JSON export
    test_path = "/tmp/test_eval_results.json"
    export_results_json(dummy_comparisons, dummy_arrow_results, checks, test_path)
    print(f"\n  JSON exported to {test_path}")

    print("\n All evaluation checks passed.")


# ============================================================================
# ADDITIONAL METRICS: Bootstrap CI, Diebold-Mariano test, RMSE ratios
# ============================================================================

# Additional imports (used by bootstrap and statistical tests below)
import datetime as _datetime
try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def bootstrap_rmse_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    B: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap 95% CI on global RMSE (sample-level resampling, B=1000).

    Non-parametric bootstrap resampling at the sample level (rows of y_true /
    y_pred). Appropriate for i.i.d. or weakly dependent test samples; provides
    a conservative approximation for moderately autocorrelated data.

    Reference: Efron & Tibshirani (1993), "An Introduction to the Bootstrap".

    Parameters
    ----------
    y_true : (N, m)
    y_pred : (N, m)
    B      : bootstrap replicates
    ci     : confidence level (default 0.95)
    seed   : reproducibility

    Returns
    -------
    (ci_low, ci_high) : Tuple[float, float]
    """
    rng = np.random.default_rng(seed)
    N = y_true.shape[0]
    rmse_boot = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        rmse_boot[b] = float(np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2)))
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(rmse_boot, alpha)), float(np.quantile(rmse_boot, 1.0 - alpha))


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    h: int = 1,
) -> Tuple[float, float]:
    """Diebold-Mariano (1995) test with Harvey-Leybourne-Newbold correction.

    H0: E[d_t] = 0,  d_t = MSE(method1) - MSE(method2).
    dm_stat > 0  =>  method1 has higher loss than method2.

    References
    ----------
    Diebold & Mariano (1995), JBES.
    Harvey, Leybourne & Newbold (1997), IJF — small-sample correction.

    Parameters
    ----------
    y_true  : (N, m)
    y_pred1 : (N, m)  — "challenger" (e.g. Inverse MAP)
    y_pred2 : (N, m)  — "baseline"   (e.g. Forward MLP)
    h       : forecast horizon for autocorrelation window (default 1)

    Returns
    -------
    (dm_stat, p_value) : Tuple[float, float]
    """
    e1 = np.mean((y_true - y_pred1) ** 2, axis=1)
    e2 = np.mean((y_true - y_pred2) ** 2, axis=1)
    d = e1 - e2
    N = len(d)
    d_mean = np.mean(d)

    # Newey-West long-run variance
    nw_var = np.var(d, ddof=1)
    for lag in range(1, h):
        gamma_lag = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
        nw_var += 2.0 * (1.0 - lag / h) * gamma_lag
    nw_var = max(nw_var, 1e-14)

    # HLN correction factor
    hlnb = np.sqrt((N + 1 - 2 * h + h * (h - 1) / N) / N)
    dm_stat = float(hlnb * d_mean / np.sqrt(nw_var / N))

    if _HAS_SCIPY:
        p_value = float(2.0 * (1.0 - _scipy_stats.t.cdf(abs(dm_stat), df=N - 1)))
    else:
        # Normal approximation fallback
        import math
        p_value = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(dm_stat) / math.sqrt(2)))))
    return dm_stat, p_value


def compute_rmse_ratios(
    comparisons: Dict[str, "CaseComparison"],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute RMSE ratios inv/MLP, inv/CVAE, inv/naive for all cases.

    Returns
    -------
    dict case_key -> {
        "inv_flow_vs_mlp"  : float | None,
        "inv_flow_vs_cvae" : float | None,
        "inv_flow_vs_naive": float | None,
    }
    """
    ratios: Dict[str, Dict[str, Optional[float]]] = {}
    for k, comp in comparisons.items():
        inv   = comp.methods.get("Inverse MAP (flow)")
        mlp   = comp.methods.get("Forward MLP")
        cvae  = comp.methods.get("Forward CVAE")
        naive = comp.methods.get("Naive mean")
        ratios[k] = {
            "inv_flow_vs_mlp":   float(inv.rmse_s / (mlp.rmse_s   + 1e-12)) if (inv and mlp)   else None,
            "inv_flow_vs_cvae":  float(inv.rmse_s / (cvae.rmse_s  + 1e-12)) if (inv and cvae)  else None,
            "inv_flow_vs_naive": float(inv.rmse_s / (naive.rmse_s + 1e-12)) if (inv and naive) else None,
        }
    return ratios


def compute_dm_tests_from_arrays(
    Y_true_s: np.ndarray,
    predictions: Dict[str, np.ndarray],
    eval_indices: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute Diebold-Mariano tests from raw prediction arrays.

    All predictions are aligned to eval_indices (subsampled test set used
    for MAP inference).

    Parameters
    ----------
    Y_true_s   : (N_test, m) full ground truth (standardised)
    predictions: dict method_name -> (n_eval or N_test, m)
    eval_indices: optional (n_eval,) indices of evaluated test samples

    Returns
    -------
    dict pair_label -> {method1, method2, dm_stat, p_value,
                        significant_at_05, method1_rmse, method2_rmse}
    """
    N_full = Y_true_s.shape[0]

    if eval_indices is not None:
        Y_true_eval = Y_true_s[eval_indices]
        n_eval = len(eval_indices)
    else:
        Y_true_eval = Y_true_s
        n_eval = N_full

    def _align(pred: np.ndarray) -> np.ndarray:
        """Subsample forward predictions to match eval_indices length."""
        if len(pred) == n_eval:
            return pred
        # Forward methods predict on all N_test; subsample same way as MAP
        idx = np.linspace(0, len(pred) - 1, n_eval).astype(int)
        return pred[idx]

    pairs = [
        ("inv_vs_mlp",  "Inverse MAP (flow)", "Forward MLP"),
        ("inv_vs_cvae", "Inverse MAP (flow)", "Forward CVAE"),
        ("inv_vs_n0i",  "Inverse MAP (flow)", "Inverse MAP (N0I)"),
        ("cvae_vs_mlp", "Forward CVAE",       "Forward MLP"),
    ]

    results: Dict[str, Dict[str, float]] = {}
    for pair_label, m1, m2 in pairs:
        if m1 not in predictions or m2 not in predictions:
            continue
        p1 = _align(predictions[m1])
        p2 = _align(predictions[m2])
        n  = min(len(p1), len(p2), len(Y_true_eval))
        yt, p1, p2 = Y_true_eval[:n], p1[:n], p2[:n]
        try:
            dm_stat, p_val = diebold_mariano_test(yt, p1, p2)
            results[pair_label] = {
                "method1": m1,
                "method2": m2,
                "dm_stat": round(dm_stat, 6),
                "p_value": round(p_val, 6),
                "significant_at_05": bool(p_val < 0.05),
                "method1_rmse": float(np.sqrt(np.mean((yt - p1) ** 2))),
                "method2_rmse": float(np.sqrt(np.mean((yt - p2) ** 2))),
            }
        except Exception as exc:
            results[pair_label] = {"error": str(exc)}
    return results


# ---------------------------------------------------------------------------
# export_results_json: enriched version with CI, ratios, DM test, run config
# ---------------------------------------------------------------------------



def export_results_json(
    comparisons: Dict[str, "CaseComparison"],
    arrow_results: Dict[str, "ArrowOfTimeResult"],
    checks: "List[PredictionCheck]",
    filepath: str,
    dm_tests: Optional[Dict[str, Dict]] = None,
    run_config: Optional[Dict] = None,
) -> None:
    """Export all results to JSON — enriched version.

    Extends the original export with:
    - Bootstrap 95% CI on RMSE per method
    - RMSE ratios (inv/MLP, inv/CVAE, inv/naive) per case
    - Diebold-Mariano test results (if provided)
    - Run configuration / hyperparameters (if provided)
    - Timestamp
    """
    import json as _json
    import os as _os

    # ------------------------------------------------------------------
    # Rebuild results dict from scratch (mirrors original structure +
    # adds new fields) to avoid calling the original which has a
    # different signature.
    # ------------------------------------------------------------------

    def _to_serial(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_serial(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serial(v) for v in obj]
        if isinstance(obj, set):
            return [_to_serial(v) for v in sorted(obj)]
        if isinstance(obj, Verdict):
            return obj.value
        if hasattr(obj, "__dataclass_fields__"):
            from dataclasses import asdict as _asdict
            return _to_serial(_asdict(obj))
        if hasattr(obj, "detach") and hasattr(obj, "cpu"):
            try:
                return obj.detach().cpu().numpy().tolist()
            except Exception:
                return str(obj)
        return obj

    results: Dict = {
        "arrow_of_time": {},
        "comparisons":   {},
        "predictions":   [],
    }

    # Arrow-of-time
    for k, ar in arrow_results.items():
        def _ser_scales(scales):
            return [
                {"w": sr.w, "J_obs": sr.J_median,
                 "J_null_ci_low": sr.J_ci_low, "J_null_ci_high": sr.J_ci_high,
                 "p_value": getattr(sr, "p_value", None),
                 "reject": sr.exceeds_tau,
                 "n_samples": getattr(sr, "n_samples", None)}
                for sr in scales
            ]
        def _ser_sub(sub):
            if sub is None:
                return None
            return {"tag": sub.tag, "verdict": sub.verdict.value,
                    "n_reject": sub.n_reject, "scales": _ser_scales(sub.scales)}
        results["arrow_of_time"][k] = {
            "verdict": ar.verdict.value,
            "decision_rule": ar.decision_rule,
            "overall_median": ar.overall_median,
            "n_exceeding": ar.n_exceeding,
            "tau": ar.tau,
            "C_min": ar.C_min,
            "level": _ser_sub(getattr(ar, "level_result", None)),
            "diff":  _ser_sub(getattr(ar, "diff_result", None)),
            "scale_results": _ser_scales(ar.scale_results),
        }

    # Comparisons
    for k, comp in comparisons.items():
        comp_dict: Dict = {
            "case_key": comp.case_key,
            "case_name": comp.case_name,
            "verdict_expected": comp.verdict_expected.value,
            "methods": {},
        }
        for mname, mr in comp.methods.items():
            comp_dict["methods"][mname] = {
                "rmse_s":            mr.rmse_s,
                "rmse_s_ci_low":     getattr(mr, "rmse_s_ci_low",  None),
                "rmse_s_ci_high":    getattr(mr, "rmse_s_ci_high", None),
                "mae_s":             mr.mae_s,
                "rmse_orig":         mr.rmse_orig,
                "mae_orig":          mr.mae_orig,
                "skill_vs_naive":    mr.skill_vs_naive,
                "mean_retro_nll":    mr.mean_retro_nll,
                "mean_prior_logprob":mr.mean_prior_logprob,
                "mean_multistart_std": mr.mean_multistart_std,
                "mean_map_loss":     mr.mean_map_loss,
                "n_samples":         mr.n_samples,
                "rmse_per_pos_s": (mr.rmse_per_pos_s.tolist()
                                   if mr.rmse_per_pos_s is not None else None),
            }
        results["comparisons"][k] = comp_dict

    # Predictions
    for c in checks:
        results["predictions"].append({
            "pid": c.pid, "statement": c.statement,
            "passed": c.passed, "details": c.details,
        })

    # RMSE ratios per case
    results["rmse_ratios"] = compute_rmse_ratios(comparisons)

    # Diebold-Mariano test results
    if dm_tests is not None:
        results["dm_tests"] = dm_tests

    # Summary
    n_passed = sum(1 for c in checks if c.passed)
    results["summary"] = {
        "n_predictions_passed": n_passed,
        "n_predictions_total":  len(checks),
        "timestamp":            _datetime.datetime.now().isoformat(),
    }

    # Run configuration
    if run_config is not None:
        results["run_config"] = run_config

    _os.makedirs(_os.path.dirname(filepath) if _os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w") as f:
        _json.dump(_to_serial(results), f, indent=2)
    print(f"  [JSON] Exported → {filepath}")


# ============================================================================
# ENRICHED JSON EXPORT — per-sample arrays for full figure reproducibility
# ============================================================================

def export_results_json_enriched(
    comparisons:        Dict[str, "CaseComparison"],
    arrow_results:      Dict[str, "ArrowOfTimeResult"],
    checks:             "List[PredictionCheck]",
    filepath:           str,
    # ── Standard fields ────────────────────────────────────────────────────
    dm_tests:           Optional[Dict[str, Dict]]       = None,
    run_config:         Optional[Dict]                  = None,
    # ── Extended fields: per-sample arrays ───────────────────────────────────
    series_data:        Optional[Dict[str, np.ndarray]] = None,
    histories:          Optional[Dict[str, Dict]]       = None,
    map_results_flow:   Optional[Dict[str, "BatchMAPResults"]] = None,
    map_results_n0i:    Optional[Dict[str, "BatchMAPResults"]] = None,
    Y_test_s_dict:      Optional[Dict[str, np.ndarray]] = None,
    X_test_s_dict:      Optional[Dict[str, np.ndarray]] = None,
    Y_naive_s_dict:     Optional[Dict[str, np.ndarray]] = None,
    Y_mlp_s_dict:       Optional[Dict[str, np.ndarray]] = None,
    n_series_points:    int = 2000,
) -> None:
    """Enriched JSON export — extends export_results_json() with per-sample arrays.

    New sections in the JSON output:
    - ``series_data[case]``        ndarray(T,)          → time series
    - ``histories[case][model]``   dict of lists        → training curves
    - ``comparisons[case].arrays`` per-sample arrays    → prediction plots
    - ``rmse_s_ci_low/high``       bootstrap 95% CI     → error bars

    Parameters
    ----------
    comparisons       : dict case_key -> CaseComparison
    arrow_results     : dict case_key -> ArrowOfTimeResult
    checks            : list of PredictionCheck
    filepath          : output JSON file path
    dm_tests          : Diebold-Mariano test results per case
    run_config        : run configuration / hyperparameters
    series_data       : dict case_key -> ndarray(T,) — raw series
    histories         : dict case_key -> dict model_name -> history dict
    map_results_flow  : dict case_key -> BatchMAPResults (flow prior)
    map_results_n0i   : dict case_key -> BatchMAPResults (N(0,I) prior)
    Y_test_s_dict     : dict case_key -> ndarray(N, m) — ground truth
    X_test_s_dict     : dict case_key -> ndarray(N, n) — observed past
    Y_naive_s_dict    : dict case_key -> ndarray(N, m) — naive baseline
    Y_mlp_s_dict      : dict case_key -> ndarray(N, m) — baseline MLP
    n_series_points   : maximum number of time-series points to store
    """
    import datetime as _dt
    import json
    import os

    def _to_serial(obj):
        """Convert common scientific/PyTorch objects into JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_serial(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serial(v) for v in obj]
        if isinstance(obj, set):
            return [_to_serial(v) for v in sorted(obj)]
        if isinstance(obj, Verdict):
            return obj.value
        if hasattr(obj, "__dataclass_fields__"):
            from dataclasses import asdict as _asdict
            return _to_serial(_asdict(obj))
        if hasattr(obj, "detach") and hasattr(obj, "cpu"):
            try:
                return obj.detach().cpu().numpy().tolist()
            except Exception:
                return str(obj)
        return obj

    def _ser_scales(scales):
        return [
            {
                "w":              sr.w,
                "J_obs":          sr.J_median,
                "J_null_ci_low":  sr.J_ci_low,
                "J_null_ci_high": sr.J_ci_high,
                "p_value":        getattr(sr, "p_value", None),
                "reject":         sr.exceeds_tau,
                "n_samples":      getattr(sr, "n_samples", None),
            }
            for sr in scales
        ]

    def _ser_sub(sub):
        if sub is None:
            return None
        return {
            "tag":      sub.tag,
            "verdict":  sub.verdict.value,
            "n_reject": sub.n_reject,
            "scales":   _ser_scales(sub.scales),
        }

    results: Dict = {
        "arrow_of_time": {},
        "comparisons":   {},
        "predictions":   [],
    }

    # ── Arrow-of-time results ────────────────────────────────────────────────────────
    for k, ar in arrow_results.items():
        results["arrow_of_time"][k] = {
            "verdict":        ar.verdict.value,
            "decision_rule":  ar.decision_rule,
            "overall_median": ar.overall_median,
            "n_exceeding":    ar.n_exceeding,
            "tau":            ar.tau,
            "C_min":          ar.C_min,
            "level":          _ser_sub(getattr(ar, "level_result", None)),
            "diff":           _ser_sub(getattr(ar, "diff_result",  None)),
            "scale_results":  _ser_scales(ar.scale_results),
        }

    # ── Comparisons + metrics + bootstrap CI + per-sample arrays ──────────
    for k, comp in comparisons.items():
        comp_dict: Dict = {
            "case_key":         comp.case_key,
            "case_name":        comp.case_name,
            "verdict_expected": comp.verdict_expected.value,
            "methods":          {},
        }

        for mname, mr in comp.methods.items():
            # Bootstrap CI — computed if Y_pred_s and Y_test_s are available
            ci_low, ci_high = None, None
            if (Y_test_s_dict and k in Y_test_s_dict
                    and mr.Y_pred_s is not None):
                try:
                    ci_low, ci_high = bootstrap_rmse_ci(
                        Y_test_s_dict[k], mr.Y_pred_s, B=1000, seed=42
                    )
                except Exception:
                    pass

            comp_dict["methods"][mname] = {
                "rmse_s":              mr.rmse_s,
                "rmse_s_ci_low":       ci_low  if ci_low  is not None
                                       else getattr(mr, "rmse_s_ci_low",  None),
                "rmse_s_ci_high":      ci_high if ci_high is not None
                                       else getattr(mr, "rmse_s_ci_high", None),
                "mae_s":               mr.mae_s,
                "rmse_orig":           mr.rmse_orig,
                "mae_orig":            mr.mae_orig,
                "skill_vs_naive":      mr.skill_vs_naive,
                "mean_retro_nll":      mr.mean_retro_nll,
                "mean_prior_logprob":  mr.mean_prior_logprob,
                "mean_multistart_std": mr.mean_multistart_std,
                "mean_map_loss":       mr.mean_map_loss,
                "n_samples":           mr.n_samples,
                "rmse_per_pos_s": (
                    mr.rmse_per_pos_s.tolist()
                    if mr.rmse_per_pos_s is not None else None
                ),
            }

        # ── Per-sample arrays ───────────────────────────────────────────
        arrays: Dict = {}

        if X_test_s_dict and k in X_test_s_dict:
            arrays["X_test_s"]  = X_test_s_dict[k].tolist()
        if Y_test_s_dict and k in Y_test_s_dict:
            arrays["Y_test_s"]  = Y_test_s_dict[k].tolist()
        if Y_naive_s_dict and k in Y_naive_s_dict:
            arrays["Y_naive_s"] = Y_naive_s_dict[k].tolist()
        if Y_mlp_s_dict and k in Y_mlp_s_dict:
            arrays["Y_mlp_s"]   = Y_mlp_s_dict[k].tolist()

        if map_results_flow and k in map_results_flow:
            br = map_results_flow[k]
            arrays["Y_hat_flow"]      = br.Y_hat.tolist()
            arrays["retro_nlls_flow"] = br.retro_nlls.tolist()
            arrays["map_losses_flow"] = br.map_losses.tolist()
            arrays["multistart_std"]  = br.multistart_std.tolist()

        if map_results_n0i and k in map_results_n0i:
            br2 = map_results_n0i[k]
            arrays["Y_hat_n0i"]      = br2.Y_hat.tolist()
            arrays["map_losses_n0i"] = br2.map_losses.tolist()

        if arrays:
            comp_dict["arrays"] = arrays

        results["comparisons"][k] = comp_dict

    # ── Prediction checks ─────────────────────────────────────────────────────────
    for c in checks:
        results["predictions"].append({
            "pid":       c.pid,
            "statement": c.statement,
            "passed":    c.passed,
            "details":   c.details,
        })

    # ── Series data ────────────────────────────────────────────────
    if series_data:
        results["series_data"] = {
            k: (arr[:n_series_points].tolist()
                if isinstance(arr, np.ndarray) else list(arr[:n_series_points]))
            for k, arr in series_data.items()
        }

    # ── Training histories ──────────────────────────────────────────────────
    if histories:
        results["histories"] = _to_serial(histories)

    # ── RMSE ratios and DM tests ───────────────────────────────────────────────
    results["rmse_ratios"] = compute_rmse_ratios(comparisons)

    if dm_tests is not None:
        results["dm_tests"] = _to_serial(dm_tests)

    # ── Summary and run configuration ─────────────────────────────────────────────────
    n_passed = sum(1 for c in checks if c.passed)
    results["summary"] = {
        "n_predictions_passed": n_passed,
        "n_predictions_total":  len(checks),
        "timestamp":            _dt.datetime.now().isoformat(),
    }

    if run_config is not None:
        results["run_config"] = _to_serial(run_config)

    # ── Write output file ─────────────────────────────────────────────────────────────
    os.makedirs(
        os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
        exist_ok=True,
    )
    with open(filepath, "w") as f:
        json.dump(_to_serial(results), f, indent=2)

    # Summary report
    size_mb = os.path.getsize(filepath) / 1024 / 1024
    n_with_arrays = sum(
        1 for v in results["comparisons"].values() if "arrays" in v
    )
    print(f"  [JSON] Enriched export → {filepath}")
    print(f"         Size: {size_mb:.1f} MB | "
          f"arrays: {n_with_arrays}/{len(comparisons)} cases | "
          f"series_data: {'yes' if 'series_data' in results else 'no'} | "
          f"histories: {'yes' if 'histories' in results else 'no'}")
