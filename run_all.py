#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_all.py - Run the full retrodictive forecasting pipeline for all cases
=====================================================================

Orchestrates the complete experimental protocol:
  1. Run each case via run_single.run_single_case()
  2. Aggregate cross-case results
  3. Verify all 4 falsifiable predictions (P1-P4)
  4. Generate cross-case figures
  5. Export consolidated results to JSON
  6. Print final summary

Usage
-----
    # Full run (default settings, CPU) — includes ERA5 if available
    python run_all.py

    # Quick mode (pipeline testing) — includes ERA5 if available
    python run_all.py --quick

    # Specific cases only
    python run_all.py --cases A C D

    # GPU
    python run_all.py --device cuda

    # Custom output
    python run_all.py --outdir my_results
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

from config import (
    CASES,
    CASE_ORDER,
    SYNTH_CASE_ORDER,
    FullConfig,
    ExperimentConfig,
    InverseCVAEConfig,
    ForwardCVAEConfig,
    ForwardMLPConfig,
    FlowPriorConfig,
    MAPConfig,
    Verdict,
    get_default_config,
)
from diagnostics import ArrowOfTimeResult
from evaluation import (
    CaseComparison,
    build_comparison_table,
    build_cross_case_table,
    verify_predictions,
    format_prediction_checks,
    export_results_json,
    export_results_json_enriched,
    compute_rmse_ratios,
)
from plotting import plot_all
from run_single import run_single_case, _quick_config


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_all(
    cfg: FullConfig,
    device: torch.device,
    outdir: str,
    cases: List[str],
    verbose: bool = True,
) -> Dict:
    """Run the full retrodictive forecasting pipeline for all specified cases.

    Parameters
    ----------
    cfg : FullConfig
    device : torch.device
    outdir : str
    cases : List[str] - case keys to run
    verbose : bool

    Returns
    -------
    dict with keys:
        "comparisons" : dict case_key -> CaseComparison
        "arrow_results" : dict case_key -> ArrowOfTimeResult
        "checks" : List[PredictionCheck]
        "all_results" : dict case_key -> full result dict from run_single
    """
    os.makedirs(outdir, exist_ok=True)

    t_total = time.time()

    def _print_box(lines: List[str], width: int = 68) -> None:
        print("+" + "-" * width + "+")
        for line in lines:
            print("|" + line.ljust(width) + "|")
        print("+" + "-" * width + "+")

    _print_box([
        "  RETRODICTIVE FORECASTING - Proof of Concept",
        "  Full Experimental Protocol",
    ])
    print(f"\n  Cases: {', '.join(cases)}")
    print(f"  Device: {device}")
    print(f"  Output: {outdir}")
    print(f"  T={cfg.experiment.T}, n={cfg.experiment.n}, m={cfg.experiment.m}, "
          f"seed={cfg.experiment.seed}")

    # ------------------------------------------------------------------
    # 1. RUN EACH CASE
    # ------------------------------------------------------------------
    all_results = {}
    comparisons = {}
    arrow_results = {}
    histories_all = {}
    series_dict = {}
    dm_tests_all: dict = {}
    ran_cases: List[str] = []

    for i, key in enumerate(cases):
        print(f"\n\n{'#' * 70}")
        print(f"  CASE {key} ({i+1}/{len(cases)}): {CASES[key].name}")
        print(f"{'#' * 70}")

        case_outdir = os.path.join(outdir, CASES[key].name)

        try:
            result = run_single_case(
                case_key=key,
                cfg=cfg,
                device=device,
                outdir=case_outdir,
                verbose=verbose,
            )
        except (ImportError, FileNotFoundError) as e:
            if key == "ERA5":
                print(
                    "[!] Skipping ERA5: missing optional dependency or NetCDF file.\n"
                    f"    {type(e).__name__}: {e}"
                )
                continue
            raise

        all_results[key] = result
        comparisons[key] = result["comparison"]
        arrow_results[key] = result["arrow_result"]
        histories_all[key] = result["histories"]
        series_dict[key] = result["series"]
        if "dm_tests" in result:
            dm_tests_all[key] = result["dm_tests"]
        ran_cases.append(key)

    # ------------------------------------------------------------------
    # 2. CROSS-CASE ANALYSIS
    # ------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("  CROSS-CASE ANALYSIS")
    print(f"{'='*70}")

    # Per-case tables
    for key in ran_cases:
        print(f"\n{build_comparison_table(comparisons[key])}")

    # Cross-case table
    print(f"\n{build_cross_case_table(comparisons)}")

    # ------------------------------------------------------------------
    # 3. VERIFY PREDICTIONS P1-P4
    # ------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("  FALSIFIABLE PREDICTIONS VERIFICATION")
    print(f"{'='*70}")

    checks = verify_predictions(comparisons, arrow_results)
    print(f"\n{format_prediction_checks(checks)}")

    # ------------------------------------------------------------------
    # 4. CROSS-CASE FIGURES
    # ------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("  GENERATING CROSS-CASE FIGURES")
    print(f"{'='*70}")

    fig_dir = os.path.join(outdir, "figures_cross_case")

    # Build test_data and map_data dicts for figures
    test_data = {}
    map_data  = {}
    for key in ran_cases:
        r  = all_results[key]
        ds = r["dataset"]
        N_test = ds.X_test_s.shape[0]
        Y_naive_all = np.tile(ds.y_mean_s, (N_test, 1))

        test_data[key] = {
            "X_test_s": ds.X_test_s,
            "Y_test_s": ds.Y_test_s,
            "Y_hat_s":  r["map_results_flow"].Y_hat,
            "Y_naive_s": Y_naive_all,
            "Y_mlp_s":  r["Y_fwd_mlp"],
        }
        map_data[key] = {
            "map_results_flow": r["map_results_flow"],
            "map_results_n0i":  r["map_results_n0i"],
            "Y_fwd_mlp":        r["Y_fwd_mlp"],
            "Y_fwd_cvae":       r["Y_fwd_cvae"],
            "dataset":          ds,
        }

    plot_all(
        comparisons=comparisons,
        arrow_results=arrow_results,
        histories=histories_all,
        series_dict=series_dict,
        checks=checks,
        test_data=test_data,
        map_data=map_data,
        outdir=fig_dir,
    )

    # ------------------------------------------------------------------
    # 5. EXPORT CONSOLIDATED RESULTS
    # ------------------------------------------------------------------
    # Global run config
    run_config_global = {
        "cases": ran_cases,
        "experiment": {
            "T": cfg.experiment.T, "n": cfg.experiment.n,
            "m": cfg.experiment.m, "seed": cfg.experiment.seed,
        },
        "map_inference": {
            "steps":        cfg.map_inference.steps,
            "K_multistart": cfg.map_inference.K_multistart,
            "lam_prior":    cfg.map_inference.lam_prior,
            "lr":           cfg.map_inference.lr,
            "n_eval":       cfg.map_inference.n_eval,
        },
        "models": {
            "inverse_cvae": {"epochs": cfg.inverse_cvae.epochs,
                             "z_dim":  cfg.inverse_cvae.z_dim},
            "forward_cvae": {"epochs": cfg.forward_cvae.epochs,
                             "z_dim":  cfg.forward_cvae.z_dim},
            "forward_mlp":  {"epochs": cfg.forward_mlp.epochs},
            "flow_prior":   {"epochs": cfg.flow_prior.epochs,
                             "n_layers": cfg.flow_prior.n_layers},
        },
        "case_params": {k: CASES[k].params for k in ran_cases if k in CASES},
    }

    # Print RMSE ratios summary
    ratios = compute_rmse_ratios(comparisons)
    print(f"\n  RMSE Ratios (inv/MLP):")
    for k, r in ratios.items():
        if r.get("inv_flow_vs_mlp") is not None:
            print(f"    {k}: inv/MLP={r['inv_flow_vs_mlp']:.4f}  "
                  f"inv/CVAE={r.get('inv_flow_vs_cvae', 'N/A'):.4f}")

    json_path = os.path.join(outdir, "results_all.json")
    export_results_json_enriched(
        comparisons, arrow_results, checks, json_path,
        dm_tests         = dm_tests_all if dm_tests_all else None,
        run_config       = run_config_global,
        series_data      = series_dict,
        histories        = histories_all,
        map_results_flow = {k: all_results[k]["map_results_flow"] for k in ran_cases},
        map_results_n0i  = {k: all_results[k]["map_results_n0i"]  for k in ran_cases},
        Y_test_s_dict    = {k: test_data[k]["Y_test_s"]  for k in ran_cases},
        X_test_s_dict    = {k: test_data[k]["X_test_s"]  for k in ran_cases},
        Y_naive_s_dict   = {k: test_data[k]["Y_naive_s"] for k in ran_cases},
        Y_mlp_s_dict     = {k: test_data[k]["Y_mlp_s"]   for k in ran_cases},
    )
    print(f"\n  Consolidated results exported to {json_path}")

    # ------------------------------------------------------------------
    # 6. FINAL SUMMARY
    # ------------------------------------------------------------------
    elapsed = time.time() - t_total

    print("\n")
    _print_box([" FINAL SUMMARY"], width=68)

    # Arrow-of-time verdicts
    print(f"\n  Arrow-of-Time Verdicts:")
    for key in ran_cases:
        ar = arrow_results[key]
        expected = CASES[key].verdict
        match = "OK" if ar.verdict == expected else "MISMATCH"
        lv = ar.level_result.verdict.value if ar.level_result else "?"
        dv = ar.diff_result.verdict.value if ar.diff_result else "?"
        print(f"    {key} ({CASES[key].name[:25]:<25}): "
              f"{ar.verdict.value:<6} [L={lv}, D={dv}] "
              f"(expected {expected.value}) {match}")

    # Best method per case
    print(f"\n  Best Method per Case (by RMSE):")
    for key in ran_cases:
        comp = comparisons[key]
        best_name = None
        best_rmse = float("inf")
        for mname, mr in comp.methods.items():
            if mname == "Naive mean":
                continue  # skip baseline
            if mr.rmse_s < best_rmse:
                best_rmse = mr.rmse_s
                best_name = mname
        naive_rmse = comp.methods["Naive mean"].rmse_s
        print(f"    {key}: {best_name} (RMSE={best_rmse:.4f}, "
              f"naive={naive_rmse:.4f}, "
              f"skill={100*(1-best_rmse/naive_rmse):.1f}%)")

    # Prediction summary
    n_passed = sum(1 for c in checks if c.passed)
    n_total = len(checks)
    print(f"\n  Predictions: {n_passed}/{n_total} passed")
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"    {c.pid}: {status} - {c.statement}")

    print(f"\n  Total execution time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Output directory: {outdir}")

    return {
        "comparisons": comparisons,
        "arrow_results": arrow_results,
        "checks": checks,
        "all_results": all_results,
    }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the retrodictive forecasting pipeline for all cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_all.py                        # Full run (A–D + ERA5 if available)
    python run_all.py --quick                # Quick test mode (A–D + ERA5 if available)
    python run_all.py --cases A C D          # Specific cases only
    python run_all.py --cases ERA5           # Run ERA5 only (requires NetCDF path in config)
    python run_all.py --device cuda          # Use GPU
    python run_all.py --outdir my_results    # Custom output dir
        """,
    )
    parser.add_argument(
        "--cases", type=str, nargs="+", default=None,
        choices=CASE_ORDER,
        help="Case keys to run (default: A–D + ERA5 if available)",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory (default: 'outputs')",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer epochs, less data (for testing)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed (default: 42)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce verbosity",
    )

    args = parser.parse_args()

    # Configuration
    if args.quick:
        cfg = _quick_config()
        print("[!] QUICK MODE: reduced epochs and data for pipeline testing")
    else:
        cfg = get_default_config()

    # Seed override
    if args.seed is not None:
        cfg = FullConfig(
            experiment=ExperimentConfig(
                T=cfg.experiment.T,
                seed=args.seed,
                n=cfg.experiment.n,
                m=cfg.experiment.m,
            ),
            inverse_cvae=cfg.inverse_cvae,
            forward_cvae=cfg.forward_cvae,
            forward_mlp=cfg.forward_mlp,
            flow_prior=cfg.flow_prior,
            map_inference=cfg.map_inference,
            arrow_of_time=cfg.arrow_of_time,
        )

    # Cases
    if args.cases:
        cases = args.cases
    else:
        # Default behaviour: run everything, but skip ERA5 if unavailable.
        cases = list(CASE_ORDER)

    # Output directory
    outdir = args.outdir if args.outdir else cfg.experiment.outdir

    # Device
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # Set seeds
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    # Run
    run_all(
        cfg=cfg,
        device=device,
        outdir=outdir,
        cases=cases,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
