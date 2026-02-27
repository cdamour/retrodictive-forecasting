"""
export_results_json_enriched.py
================================
Standalone export function that serialises ALL data needed by
replot_from_json.py to regenerate all 14 figures without re-running
training or inference.

NEW JSON SECTIONS:
  - series_data[case]          → ndarray(T,)         Fig 1
  - histories[case][model]     → dict of lists        Fig 3
  - comparisons[case].arrays   → per-sample arrays    Figs 4, 12, 13, 14
  - comparisons[case].methods[m].rmse_s_ci_low/high   Fig 5

USAGE in run_single.py:
  Replace the existing call:
      export_results_json(comparisons_dict, arrow_dict, checks, json_path, ...)
  with:
      export_results_json_enriched(
          comparisons_dict, arrow_dict, checks, json_path,
          dm_tests={case_key: dm_tests},
          run_config=run_config,
          # --- Additional arguments ---
          series_data={case_key: series},             # ndarray(T,)
          histories={case_key: histories},            # dict[model -> hist_dict]
          map_results_flow={case_key: map_results_flow},   # BatchMAPResults
          map_results_n0i={case_key: map_results_n0i},     # BatchMAPResults
          Y_test_s_dict={case_key: Y_test_s},         # ndarray(N, m)
          X_test_s_dict={case_key: X_test_s},         # ndarray(N, n)
          Y_naive_s_dict={case_key: Y_naive_s},       # ndarray(N, m)
          Y_mlp_s_dict={case_key: Y_fwd_mlp},         # ndarray(N, m)
      )

USAGE in run_all.py:
  Same logic — collect per-case dicts across the case loop.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Dict, List, Optional, Any

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Universal serialiser
# ──────────────────────────────────────────────────────────────────────────────

def _to_serial(obj: Any) -> Any:
    """Recursively convert any object to JSON-serialisable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serial(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serial(v) for v in obj]
    if isinstance(obj, set):
        return [_to_serial(v) for v in sorted(obj)]
    # Dataclasses (Verdict, ArrowScaleResult, MethodResult, …)
    if hasattr(obj, "value"):           # Enum-like (Verdict)
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict as _asdict
        return _to_serial(_asdict(obj))
    # PyTorch tensors (if present)
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        try:
            return obj.detach().cpu().numpy().tolist()
        except Exception:
            return str(obj)
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Arrow-of-time serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ser_scales(scales) -> list:
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


def _ser_sub(sub) -> Optional[dict]:
    if sub is None:
        return None
    return {
        "tag":     sub.tag,
        "verdict": sub.verdict.value,
        "n_reject": sub.n_reject,
        "scales":  _ser_scales(sub.scales),
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def export_results_json_enriched(
    comparisons:        Dict,
    arrow_results:      Dict,
    checks:             List,
    filepath:           str,
    # ── Standard fields ────────────────────────────────────────────────────
    dm_tests:           Optional[Dict] = None,
    run_config:         Optional[Dict] = None,
    # ── Extended fields: per-sample data for all figures ─────────────────────────
    series_data:        Optional[Dict[str, np.ndarray]] = None,
    histories:          Optional[Dict[str, Dict]]       = None,
    map_results_flow:   Optional[Dict]                  = None,
    map_results_n0i:    Optional[Dict]                  = None,
    Y_test_s_dict:      Optional[Dict[str, np.ndarray]] = None,
    X_test_s_dict:      Optional[Dict[str, np.ndarray]] = None,
    Y_naive_s_dict:     Optional[Dict[str, np.ndarray]] = None,
    Y_mlp_s_dict:       Optional[Dict[str, np.ndarray]] = None,
    n_series_points:    int = 2000,   # Maximum time-series points to store (Fig 1)
) -> None:
    """Export all results and per-sample data to a single JSON file.

    Parameters
    ----------
    comparisons       : dict case_key -> CaseComparison
    arrow_results     : dict case_key -> ArrowOfTimeResult
    checks            : list of PredictionCheck
    filepath          : output JSON file path
    dm_tests          : Diebold-Mariano test results
    run_config        : run configuration / hyperparameters
    series_data       : dict case_key -> ndarray(T,) — raw series (Fig 1)
    histories         : dict case_key -> dict model_name -> history dict (Fig 3)
    map_results_flow  : dict case_key -> BatchMAPResults flow prior (Figs 12-14)
    map_results_n0i   : dict case_key -> BatchMAPResults N(0,I) prior (Fig 13)
    Y_test_s_dict     : dict case_key -> ndarray(N, m) — ground truth (Figs 4, 12)
    X_test_s_dict     : dict case_key -> ndarray(N, n) — observed past (Fig 4)
    Y_naive_s_dict    : dict case_key -> ndarray(N, m) — naive baseline (Fig 4)
    Y_mlp_s_dict      : dict case_key -> ndarray(N, m) — baseline MLP (Fig 4)
    n_series_points   : maximum time-series points to store
    """

    results: Dict = {
        "arrow_of_time": {},
        "comparisons":   {},
        "predictions":   [],
    }

    # ── Arrow-of-Time ────────────────────────────────────────────────────────
    for k, ar in arrow_results.items():
        results["arrow_of_time"][k] = {
            "verdict":       ar.verdict.value,
            "decision_rule": ar.decision_rule,
            "overall_median": ar.overall_median,
            "n_exceeding":   ar.n_exceeding,
            "tau":           ar.tau,
            "C_min":         ar.C_min,
            "level":         _ser_sub(getattr(ar, "level_result", None)),
            "diff":          _ser_sub(getattr(ar, "diff_result",  None)),
            "scale_results": _ser_scales(ar.scale_results),
        }

    # ── Comparisons + arrays per-sample ─────────────────────────────────────
    for k, comp in comparisons.items():
        comp_dict: Dict = {
            "case_key":        comp.case_key,
            "case_name":       comp.case_name,
            "verdict_expected": comp.verdict_expected.value,
            "methods":         {},
        }

        for mname, mr in comp.methods.items():
            comp_dict["methods"][mname] = {
                # ── Standard metrics ─────────────────────────────────────
                "rmse_s":             mr.rmse_s,
                "rmse_s_ci_low":      getattr(mr, "rmse_s_ci_low",  None),
                "rmse_s_ci_high":     getattr(mr, "rmse_s_ci_high", None),
                "mae_s":              mr.mae_s,
                "rmse_orig":          mr.rmse_orig,
                "mae_orig":           mr.mae_orig,
                "skill_vs_naive":     mr.skill_vs_naive,
                "mean_retro_nll":     mr.mean_retro_nll,
                "mean_prior_logprob": mr.mean_prior_logprob,
                "mean_multistart_std": mr.mean_multistart_std,
                "mean_map_loss":      mr.mean_map_loss,
                "n_samples":          mr.n_samples,
                "rmse_per_pos_s": (
                    mr.rmse_per_pos_s.tolist()
                    if mr.rmse_per_pos_s is not None else None
                ),
            }

        # ── Per-sample arrays (all methods) ────────────────────────────────────
        # Stored in comparisons[case].arrays — separate from method metrics
        # to preserve the existing structure.
        arrays: Dict = {}

        # Shared arrays aligned to the evaluation index
        if X_test_s_dict and k in X_test_s_dict:
            arrays["X_test_s"] = X_test_s_dict[k].tolist()    # (N, n)
        if Y_test_s_dict and k in Y_test_s_dict:
            arrays["Y_test_s"] = Y_test_s_dict[k].tolist()    # (N, m)
        if Y_naive_s_dict and k in Y_naive_s_dict:
            arrays["Y_naive_s"] = Y_naive_s_dict[k].tolist()  # (N, m)
        if Y_mlp_s_dict and k in Y_mlp_s_dict:
            arrays["Y_mlp_s"] = Y_mlp_s_dict[k].tolist()      # (N, m)

        # MAP predictions — flow prior
        if map_results_flow and k in map_results_flow:
            br_flow = map_results_flow[k]
            arrays["Y_hat_flow"]      = br_flow.Y_hat.tolist()        # (N, m)
            arrays["retro_nlls_flow"] = br_flow.retro_nlls.tolist()   # (N,)
            arrays["map_losses_flow"] = br_flow.map_losses.tolist()   # (N,)
            arrays["multistart_std"]  = br_flow.multistart_std.tolist() # (N,)

        # MAP predictions — N(0,I) prior (ablation)
        if map_results_n0i and k in map_results_n0i:
            br_n0i = map_results_n0i[k]
            arrays["Y_hat_n0i"]       = br_n0i.Y_hat.tolist()         # (N, m)
            arrays["map_losses_n0i"]  = br_n0i.map_losses.tolist()    # (N,)

        if arrays:
            comp_dict["arrays"] = arrays

        results["comparisons"][k] = comp_dict

    # ── Predictions ─────────────────────────────────────────────────────────
    for c in checks:
        results["predictions"].append({
            "pid":       c.pid,
            "statement": c.statement,
            "passed":    c.passed,
            "details":   c.details,
        })

    # ── Series data (Fig 1) ──────────────────────────────────────────────────
    # Fig 1 : plot_series_overview() — stocke jusqu'à n_series_points points
    if series_data:
        results["series_data"] = {
            k: arr[:n_series_points].tolist() if isinstance(arr, np.ndarray)
               else arr[:n_series_points]
            for k, arr in series_data.items()
        }

    # ── Training histories (Fig 3) ────────────────────────────────────────────
    # Fig 3 : plot_training_curves()
    # Structure : histories[case][model_name] -> {train_loss: [...], val_loss: [...], ...}
    if histories:
        results["histories"] = _to_serial(histories)

    # ── RMSE ratios and DM tests ──────────────────────────────────────────────
    # Calculer rmse_ratios si evaluation.py est importé
    try:
        from evaluation import compute_rmse_ratios
        results["rmse_ratios"] = compute_rmse_ratios(comparisons)
    except ImportError:
        pass

    if dm_tests is not None:
        results["dm_tests"] = _to_serial(dm_tests)

    # ── Summary + run_config ─────────────────────────────────────────────────
    n_passed = sum(1 for c in checks if c.passed)
    results["summary"] = {
        "n_predictions_passed": n_passed,
        "n_predictions_total":  len(checks),
        "timestamp":            datetime.datetime.now().isoformat(),
    }

    if run_config is not None:
        results["run_config"] = _to_serial(run_config)

    # ── Write output file ─────────────────────────────────────────────────────
    os.makedirs(
        os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
        exist_ok=True
    )
    with open(filepath, "w") as f:
        json.dump(_to_serial(results), f, indent=2)

    # Summary report
    size_mb = os.path.getsize(filepath) / 1024 / 1024
    n_arrays = sum(
        1 for k in results.get("comparisons", {}).values()
        if "arrays" in k
    )
    print(f"  [JSON] Written: {filepath}")
    print(f"  Size      : {size_mb:.1f} MB")
    print(f"  Cases with per-sample arrays: {n_arrays}")
    print(f"  series_data : {'yes' if 'series_data' in results else 'no'}")
    print(f"  histories   : {'yes' if 'histories' in results else 'no'}")


# ──────────────────────────────────────────────────────────────────────────────
# Patch de run_single.py — extrait à copier-coller
# ──────────────────────────────────────────────────────────────────────────────

PATCH_RUN_SINGLE = """
# In run_single.py — replace the EXPORT JSON block with:

from export_results_json_enriched import export_results_json_enriched

json_path = os.path.join(outdir, f"results_{case_key}.json")
export_results_json_enriched(
    comparisons_dict,
    arrow_dict,
    checks,
    json_path,
    dm_tests    = {case_key: dm_tests},
    run_config  = run_config,
    # ── Additional data ──────────────────────────────────────
    series_data       = {case_key: series},
    histories         = {case_key: histories},
    map_results_flow  = {case_key: map_results_flow},
    map_results_n0i   = {case_key: map_results_n0i},
    Y_test_s_dict     = {case_key: Y_test_s},
    X_test_s_dict     = {case_key: X_test_s},
    Y_naive_s_dict    = {case_key: Y_naive_s},
    Y_mlp_s_dict      = {case_key: Y_fwd_mlp},
)

# NOTE: Y_test_s and X_test_s are available from the dataset object:
#   dataset = result['dataset']
#   X_test_s = dataset.X_test_s
#   Y_test_s = dataset.Y_test_s
#   Y_naive_s: extract from comparison.methods['Naive mean'].Y_pred_s
#   Y_fwd_mlp: extract from comparison.methods['Forward MLP'].Y_pred_s
"""


PATCH_RUN_ALL = """
# In run_all.py — populate per-case dicts across the case loop:

series_all       = {}
map_flow_all     = {}
map_n0i_all      = {}
Y_test_all       = {}
X_test_all       = {}
Y_naive_all      = {}
Y_mlp_all        = {}

for key in cases:
    result = run_case(key, ...)
    histories_all[key]    = result["histories"]
    series_all[key]       = result["series"]
    map_flow_all[key]     = result["map_results_flow"]
    map_n0i_all[key]      = result["map_results_n0i"]
    # Extract from dataset or MethodResult.Y_pred_s
    Y_test_all[key]       = result["dataset"].Y_test_s
    X_test_all[key]       = result["dataset"].X_test_s
    Y_naive_all[key]      = comparisons[key].methods["Naive mean"].Y_pred_s
    Y_mlp_all[key]        = comparisons[key].methods["Forward MLP"].Y_pred_s

export_results_json_enriched(
    comparisons, arrow_results, checks,
    os.path.join(outdir, "results_all.json"),
    dm_tests         = dm_tests_all,
    run_config       = run_config,
    series_data      = series_all,
    histories        = histories_all,
    map_results_flow = map_flow_all,
    map_results_n0i  = map_n0i_all,
    Y_test_s_dict    = Y_test_all,
    X_test_s_dict    = X_test_all,
    Y_naive_s_dict   = Y_naive_all,
    Y_mlp_s_dict     = Y_mlp_all,
)
"""


if __name__ == "__main__":
    print("export_results_json_enriched.py — integration reference")
    print("Copy this file into your project directory.")
    print()
    print("=== Integration snippet: run_single.py ===")
    print(PATCH_RUN_SINGLE)
    print("=== Integration snippet: run_all.py ===")
    print(PATCH_RUN_ALL)
