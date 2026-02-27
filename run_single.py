#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_single.py - Run the full retrodictive forecasting pipeline for ONE case
=====================================================================

Executes the complete pipeline for a single synthetic case:
  1. Generate synthetic time series
  2. Prepare dataset (windowing, standardisation, splitting)
  3. Run arrow-of-time diagnostic
  4. Train all models (Inverse CVAE, Forward CVAE, Forward MLP, RealNVP)
  5. Run MAP inference (flow prior + N(0,I) ablation)
  6. Run forward baselines
  7. Compute metrics and comparison
  8. Generate per-case figures
  9. Export results to JSON

Usage
-----
    # Run case A with default settings
    python run_single.py --case A

    # Run case C with custom output directory
    python run_single.py --case C --outdir results_C

    # Quick test mode (fewer epochs, less data)
    python run_single.py --case A --quick

    # Specify device
    python run_single.py --case D --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch

from config import (
    CASES,
    CASE_ORDER,
    FullConfig,
    ExperimentConfig,
    InverseCVAEConfig,
    ForwardCVAEConfig,
    ForwardMLPConfig,
    FlowPriorConfig,
    MAPConfig,
    PriorKind,
    Verdict,
    get_default_config,
)
from generators import (
    generate_series,
    prepare_dataset,
    make_dataloaders,
    series_summary,
    PreparedDataset,
)
from models import (
    InverseCVAE,
    ForwardCVAE,
    ForwardMLP,
    RealNVP,
    train_cvae,
    train_mlp,
    train_flow,
    count_parameters,
)
from inference import (
    map_infer_batch,
    forward_cvae_predict_batch,
    forward_mlp_predict_batch,
    BatchMAPResults,
)
from diagnostics import (
    arrow_of_time_diagnostic,
    ArrowOfTimeResult,
)
from evaluation import (
    compute_naive_metrics,
    compute_forward_metrics,
    compute_inverse_metrics,
    CaseComparison,
    MethodResult,
    build_comparison_table,
    export_results_json,
    export_results_json_enriched,
    verify_predictions,
    format_prediction_checks,
    compute_dm_tests_from_arrays,
)
from plotting import (
    plot_series_overview,
    plot_arrow_of_time,
    plot_training_curves,
    plot_example_reconstructions,
    plot_cross_case_rmse,
    plot_rmse_per_horizon,
    plot_multistart_dispersion,
    plot_prior_ablation,
    plot_retronll_vs_rmse,
    plot_map_loss_distribution,
    plot_fic_contribution,
)


# ============================================================================
# QUICK MODE OVERRIDES (for testing pipeline logic)
# ============================================================================

def _quick_config() -> FullConfig:
    """Return a fast configuration for pipeline testing."""
    return FullConfig(
        experiment=ExperimentConfig(T=2000, seed=42, n=32, m=16),
        inverse_cvae=InverseCVAEConfig(epochs=5, batch_size=128, z_dim=4, hidden=64, depth=1),
        forward_cvae=ForwardCVAEConfig(epochs=5, batch_size=128, z_dim=4, hidden=64, depth=1),
        forward_mlp=ForwardMLPConfig(epochs=5, batch_size=128, hidden=64, depth=1),
        flow_prior=FlowPriorConfig(epochs=5, batch_size=128, n_layers=2, hidden=64, patience=3),
        map_inference=MAPConfig(steps=20, K_multistart=3, n_eval=50, lr=0.05),
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}min"


def _log(msg: str, *, verbose: bool) -> None:
    if verbose:
        print(f"[{_ts()}] {msg}")

def run_single_case(
    case_key: str,
    cfg: FullConfig,
    device: torch.device,
    outdir: str,
    verbose: bool = True,
) -> Dict:
    """Run the full pipeline for a single case.

    Parameters
    ----------
    case_key : str - one of "A", "B", "C", "D", "ERA5"
    cfg : FullConfig
    device : torch.device
    outdir : str
    verbose : bool

    Returns
    -------
    dict with keys:
        "comparison" : CaseComparison
        "arrow_result" : ArrowOfTimeResult
        "histories" : dict model_name -> history
        "dataset" : PreparedDataset
        "series" : np.ndarray
    """
    assert case_key in CASES, f"Unknown case: {case_key}"
    spec = CASES[case_key]
    exp = cfg.experiment

    os.makedirs(outdir, exist_ok=True)
    fig_dir = os.path.join(outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    t_total = time.time()

    _log(
        f"Start: case={case_key}, T={exp.T}, n={exp.n}, m={exp.m}, seed={exp.seed}, device={device}",
        verbose=verbose,
    )
    _log(
        "Cfg: "
        f"inv_epochs={cfg.inverse_cvae.epochs}, fwd_epochs={cfg.forward_cvae.epochs}, "
        f"mlp_epochs={cfg.forward_mlp.epochs}, flow_epochs={cfg.flow_prior.epochs}, "
        f"map_steps={cfg.map_inference.steps}, map_K={cfg.map_inference.K_multistart}, map_n_eval={cfg.map_inference.n_eval}",
        verbose=verbose,
    )

    print("=" * 70)
    print(f"  RETRODICTIVE FORECASTING - Case {case_key}: {spec.name}")
    print(f"  Expected verdict: {spec.verdict.value}")
    print(f"  Device: {device}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. GENERATE SERIES
    # ------------------------------------------------------------------
    print(f"\n[1/9] Loading/generating series (T={exp.T})...")
    t_step = time.time()
    series = generate_series(case_key, T=exp.T, seed=exp.seed)
    stats = series_summary(series, name=spec.name)
    if verbose:
        print(f"  Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, "
              f"Skew={stats['skewness']:.4f}, Kurt={stats['kurtosis_excess']:.4f}")
        print(f"  Done in {_fmt_seconds(time.time() - t_step)}")

    # ------------------------------------------------------------------
    # 2. PREPARE DATASET
    # ------------------------------------------------------------------
    print(f"\n[2/9] Preparing dataset (n={exp.n}, m={exp.m})...")
    t_step = time.time()
    dataset = prepare_dataset(series, cfg, case_key=case_key)
    print(f"  Windows: train={dataset.n_train}, val={dataset.n_val}, test={dataset.n_test}")

    if verbose:
        print(
            "  Shapes: "
            f"X_train_s={dataset.X_train_s.shape}, Y_train_s={dataset.Y_train_s.shape}, "
            f"X_val_s={dataset.X_val_s.shape}, Y_val_s={dataset.Y_val_s.shape}, "
            f"X_test_s={dataset.X_test_s.shape}, Y_test_s={dataset.Y_test_s.shape}"
        )

    train_loader, val_loader = make_dataloaders(
        dataset, batch_size=cfg.inverse_cvae.batch_size,
    )

    if verbose:
        print(f"  DataLoaders ready in {_fmt_seconds(time.time() - t_step)}")

    # ------------------------------------------------------------------
    # 3. ARROW-OF-TIME DIAGNOSTIC
    # ------------------------------------------------------------------
    print(f"\n[3/9] Arrow-of-time diagnostic...")
    t_step = time.time()
    arrow_result = arrow_of_time_diagnostic(
        series, cfg.arrow_of_time, seed=exp.seed, verbose=verbose,
    )
    lv = arrow_result.level_result.verdict.value if arrow_result.level_result else "?"
    dv = arrow_result.diff_result.verdict.value if arrow_result.diff_result else "?"
    match_sym = "\u2713" if arrow_result.verdict == spec.verdict else "\u2717"
    print(f"  -> Verdict: {arrow_result.verdict.value} "
          f"[LEVEL={lv}, DIFF={dv}] "
          f"(expected: {spec.verdict.value}) {match_sym}")
    if verbose:
        print(f"  Done in {_fmt_seconds(time.time() - t_step)}")

    # ------------------------------------------------------------------
    # 4. TRAIN MODELS
    # ------------------------------------------------------------------
    histories = {}

    # 4a. Inverse CVAE
    print(f"\n[4/9] Training models...")
    print(f"\n  --- Inverse CVAE ---")
    cvae_print_every = 1 if cfg.inverse_cvae.epochs <= 20 else 10
    inv_cvae = InverseCVAE(exp.n, exp.m, cfg.inverse_cvae)
    print(f"  Parameters: {count_parameters(inv_cvae):,d}")
    t_step = time.time()
    histories["inverse_cvae"] = train_cvae(
        inv_cvae, train_loader, val_loader,
        epochs=cfg.inverse_cvae.epochs,
        lr=cfg.inverse_cvae.lr,
        beta=cfg.inverse_cvae.beta,
        grad_clip=cfg.inverse_cvae.grad_clip,
        device=device,
        verbose=verbose,
        print_every=cvae_print_every,
    )
    if verbose:
        print(f"  [InverseCVAE] Done in {_fmt_seconds(time.time() - t_step)}")

    # 4b. Forward CVAE
    print(f"\n  --- Forward CVAE ---")
    cvae_print_every = 1 if cfg.forward_cvae.epochs <= 20 else 10
    fwd_cvae = ForwardCVAE(exp.n, exp.m, cfg.forward_cvae)
    print(f"  Parameters: {count_parameters(fwd_cvae):,d}")
    t_step = time.time()
    histories["forward_cvae"] = train_cvae(
        fwd_cvae, train_loader, val_loader,
        epochs=cfg.forward_cvae.epochs,
        lr=cfg.forward_cvae.lr,
        beta=cfg.forward_cvae.beta,
        grad_clip=cfg.forward_cvae.grad_clip,
        device=device,
        verbose=verbose,
        print_every=cvae_print_every,
    )
    if verbose:
        print(f"  [ForwardCVAE] Done in {_fmt_seconds(time.time() - t_step)}")

    # 4c. Forward MLP
    print(f"\n  --- Forward MLP ---")
    mlp_print_every = 1 if cfg.forward_mlp.epochs <= 20 else 10
    fwd_mlp = ForwardMLP(exp.n, exp.m, cfg.forward_mlp)
    print(f"  Parameters: {count_parameters(fwd_mlp):,d}")
    t_step = time.time()
    histories["forward_mlp"] = train_mlp(
        fwd_mlp, train_loader, val_loader,
        epochs=cfg.forward_mlp.epochs,
        lr=cfg.forward_mlp.lr,
        grad_clip=cfg.forward_mlp.grad_clip,
        device=device,
        verbose=verbose,
        print_every=mlp_print_every,
    )
    if verbose:
        print(f"  [ForwardMLP] Done in {_fmt_seconds(time.time() - t_step)}")

    # 4d. RealNVP flow prior
    print(f"\n  --- RealNVP Flow Prior ---")
    flow_print_every = 1 if cfg.flow_prior.epochs <= 20 else 10
    flow = RealNVP(exp.m, cfg.flow_prior)
    print(f"  Parameters: {count_parameters(flow):,d}")
    t_step = time.time()
    histories["flow"] = train_flow(
        flow, dataset.Y_train_s, dataset.Y_val_s,
        cfg=cfg.flow_prior,
        device=device,
        verbose=verbose,
        print_every=flow_print_every,
    )
    if verbose:
        print(f"  [RealNVP] Done in {_fmt_seconds(time.time() - t_step)}")

    # ------------------------------------------------------------------
    # 5. MAP INFERENCE (flow prior)
    # ------------------------------------------------------------------
    print(f"\n[5/9] MAP inference (flow prior)...")
    map_print_every = max(1, int(cfg.map_inference.n_eval // 5))
    inv_cvae.to(device).eval()
    flow.to(device).eval()

    map_results_flow = map_infer_batch(
        inv_cvae, flow, dataset.X_test_s,
        cfg=cfg.map_inference,
        prior_kind=PriorKind.FLOW,
        device=device,
        verbose=verbose,
        print_every=map_print_every,
        forward_cvae=fwd_cvae,
    )

    # ------------------------------------------------------------------
    # 6. MAP INFERENCE (N(0,I) ablation)
    # ------------------------------------------------------------------
    print(f"\n[6/9] MAP inference (N(0,I) prior - ablation)...")
    map_results_n0i = map_infer_batch(
        inv_cvae, None, dataset.X_test_s,
        cfg=cfg.map_inference,
        prior_kind=PriorKind.STANDARD_NORMAL,
        device=device,
        verbose=verbose,
        print_every=map_print_every,
        forward_cvae=fwd_cvae,
    )

    # ------------------------------------------------------------------
    # 7. FORWARD BASELINES
    # ------------------------------------------------------------------
    print(f"\n[7/9] Forward baseline predictions...")
    Y_fwd_cvae = forward_cvae_predict_batch(fwd_cvae, dataset.X_test_s, device=device)
    Y_fwd_mlp = forward_mlp_predict_batch(fwd_mlp, dataset.X_test_s, device=device)

    # ------------------------------------------------------------------
    # 8. EVALUATION
    # ------------------------------------------------------------------
    print(f"\n[8/9] Computing metrics...")

    # Naive baseline
    naive_result = compute_naive_metrics(
        dataset.Y_test_s, dataset.y_mean_s, dataset.sy,
        dataset.Y_test, case_key,
    )
    rmse_naive = naive_result.rmse_s

    # Forward MLP
    mlp_result = compute_forward_metrics(
        dataset.Y_test_s, Y_fwd_mlp, dataset.sy,
        dataset.Y_test, rmse_naive, "Forward MLP", case_key,
    )

    # Forward CVAE
    fwd_cvae_result = compute_forward_metrics(
        dataset.Y_test_s, Y_fwd_cvae, dataset.sy,
        dataset.Y_test, rmse_naive, "Forward CVAE", case_key,
    )

    # Inverse MAP (flow)
    inv_flow_result = compute_inverse_metrics(
        dataset.Y_test_s, map_results_flow, dataset.sy,
        dataset.Y_test, rmse_naive, "Inverse MAP (flow)", case_key,
    )

    # Inverse MAP (N(0,I))
    inv_n0i_result = compute_inverse_metrics(
        dataset.Y_test_s, map_results_n0i, dataset.sy,
        dataset.Y_test, rmse_naive, "Inverse MAP (N0I)", case_key,
    )

    # Build comparison
    comparison = CaseComparison(
        case_key=case_key,
        case_name=spec.name,
        verdict_expected=spec.verdict,
        arrow_result=arrow_result,
        methods={
            "Naive mean": naive_result,
            "Forward MLP": mlp_result,
            "Forward CVAE": fwd_cvae_result,
            "Inverse MAP (flow)": inv_flow_result,
            "Inverse MAP (N0I)": inv_n0i_result,
        },
    )

    # Print table
    print(f"\n{'='*70}")
    print(build_comparison_table(comparison))
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # 9. FIGURES & EXPORT
    # ------------------------------------------------------------------
    print(f"\n[9/9] Generating figures and exporting results...")

    # Series overview (single case)
    plot_series_overview({case_key: series}, fig_dir)

    # Arrow-of-time
    plot_arrow_of_time({case_key: arrow_result}, fig_dir)

    # Training curves
    plot_training_curves({case_key: histories}, fig_dir)

    # Example reconstructions
    N_eval = map_results_flow.n_samples
    N_test = dataset.X_test_s.shape[0]
    if N_eval < N_test:
        eval_indices = np.linspace(0, N_test - 1, N_eval).astype(int)
    else:
        eval_indices = np.arange(N_test)

    # Naive predictions for all test samples (for examples plot)
    Y_naive_all = np.tile(dataset.y_mean_s, (N_test, 1))

    plot_example_reconstructions(
        case_key=case_key,
        X_test_s=dataset.X_test_s,
        Y_test_s=dataset.Y_test_s,
        Y_hat_s=map_results_flow.Y_hat,
        Y_naive_s=Y_naive_all,
        Y_mlp_s=Y_fwd_mlp,
        outdir=fig_dir,
    )

    # Cross-case plots (single case still useful)
    plot_cross_case_rmse({case_key: comparison}, fig_dir)
    plot_rmse_per_horizon({case_key: comparison}, fig_dir)
    plot_multistart_dispersion({case_key: comparison}, fig_dir)
    plot_prior_ablation({case_key: comparison}, fig_dir)

    # ----------------------------------------------------------------
    # ADDITIONAL FIGURES (Fig 12-14)
    # ----------------------------------------------------------------
    # Eval indices (shared)
    N_eval_flow = map_results_flow.n_samples
    N_test_total = dataset.X_test_s.shape[0]
    if N_eval_flow < N_test_total:
        eval_indices_new = np.linspace(0, N_test_total - 1, N_eval_flow).astype(int)
    else:
        eval_indices_new = np.arange(N_test_total)

    # Fig 12: RetroNLL vs RMSE (GO only)
    if spec.verdict == Verdict.GO:
        Y_true_eval = dataset.Y_test_s[eval_indices_new]
        try:
            plot_retronll_vs_rmse(
                case_keys=[case_key],
                retro_nlls_dict={case_key: map_results_flow.retro_nlls},
                Y_true_dict={case_key: Y_true_eval},
                Y_hat_dict={case_key: map_results_flow.Y_hat},
                outdir=fig_dir,
            )
        except Exception as _e:
            print(f"  [Warn] plot_retronll_vs_rmse: {_e}")

    # Fig 13: MAP loss distribution
    try:
        plot_map_loss_distribution(
            case_key=case_key,
            map_losses_flow=map_results_flow.map_losses,
            map_losses_n0i=map_results_n0i.map_losses,
            verdict=spec.verdict,
            outdir=fig_dir,
        )
    except Exception as _e:
        print(f"  [Warn] plot_map_loss_distribution: {_e}")

    # Fig 14: FIC contribution (GO only)
    if spec.verdict == Verdict.GO:
        try:
            plot_fic_contribution(
                case_key=case_key,
                map_losses=map_results_flow.map_losses,
                multistart_std=map_results_flow.multistart_std,
                outdir=fig_dir,
            )
        except Exception as _e:
            print(f"  [Warn] plot_fic_contribution: {_e}")

    # ----------------------------------------------------------------
    # DM TESTS
    # ----------------------------------------------------------------
    predictions_for_dm = {
        "Inverse MAP (flow)": map_results_flow.Y_hat,
        "Inverse MAP (N0I)":  map_results_n0i.Y_hat,
        "Forward MLP":        Y_fwd_mlp,
        "Forward CVAE":       Y_fwd_cvae,
    }
    dm_tests = compute_dm_tests_from_arrays(
        Y_true_s=dataset.Y_test_s,
        predictions=predictions_for_dm,
        eval_indices=eval_indices_new,
    )
    print(f"\n  DM Tests (case {case_key}):")
    for pair, res in dm_tests.items():
        if "dm_stat" in res:
            sig = " (*)" if res["significant_at_05"] else ""
            print(f"    {pair:<20}: DM={res['dm_stat']:+.3f}  p={res['p_value']:.4f}{sig}"
                  f"  [{res['method1']} vs {res['method2']}]")

    # ----------------------------------------------------------------
    # RUN CONFIG BLOCK
    # ----------------------------------------------------------------
    run_config = {
        "case_key": case_key,
        "case_params": spec.params,
        "experiment": {
            "T": exp.T, "n": exp.n, "m": exp.m, "seed": exp.seed,
        },
        "map_inference": {
            "steps":        cfg.map_inference.steps,
            "K_multistart": cfg.map_inference.K_multistart,
            "lam_prior":    cfg.map_inference.lam_prior,
            "lr":           cfg.map_inference.lr,
            "n_eval":       cfg.map_inference.n_eval,
        },
        "models": {
            "inverse_cvae": {
                "epochs": cfg.inverse_cvae.epochs,
                "z_dim":  cfg.inverse_cvae.z_dim,
                "hidden": cfg.inverse_cvae.hidden,
                "depth":  cfg.inverse_cvae.depth,
            },
            "forward_cvae": {
                "epochs": cfg.forward_cvae.epochs,
                "z_dim":  cfg.forward_cvae.z_dim,
                "hidden": cfg.forward_cvae.hidden,
                "depth":  cfg.forward_cvae.depth,
            },
            "forward_mlp": {
                "epochs": cfg.forward_mlp.epochs,
                "hidden": cfg.forward_mlp.hidden,
                "depth":  cfg.forward_mlp.depth,
            },
            "flow_prior": {
                "epochs":   cfg.flow_prior.epochs,
                "n_layers": cfg.flow_prior.n_layers,
                "hidden":   cfg.flow_prior.hidden,
            },
        },
    }

    # ----------------------------------------------------------------
    # EXPORT JSON
    # ----------------------------------------------------------------
    comparisons_dict = {case_key: comparison}
    arrow_dict = {case_key: arrow_result}

    checks = verify_predictions(comparisons_dict, arrow_dict)
    print(f"\n{format_prediction_checks(checks)}")

    json_path = os.path.join(outdir, f"results_{case_key}.json")
    export_results_json_enriched(
        comparisons_dict, arrow_dict, checks, json_path,
        dm_tests    = {case_key: dm_tests},
        run_config  = run_config,
        series_data      = {case_key: series},
        histories        = {case_key: histories},
        map_results_flow = {case_key: map_results_flow},
        map_results_n0i  = {case_key: map_results_n0i},
        Y_test_s_dict    = {case_key: dataset.Y_test_s},
        X_test_s_dict    = {case_key: dataset.X_test_s},
        Y_naive_s_dict   = {case_key: Y_naive_all},
        Y_mlp_s_dict     = {case_key: Y_fwd_mlp},
    )
    print(f"\n  Results exported to {json_path}")

    elapsed = time.time() - t_total
    print(f"\n  Total time for case {case_key}: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    return {
        "comparison":      comparison,
        "arrow_result":    arrow_result,
        "histories":       histories,
        "dataset":         dataset,
        "series":          series,
        "map_results_flow": map_results_flow,
        "map_results_n0i":  map_results_n0i,
        "Y_fwd_mlp":       Y_fwd_mlp,
        "Y_fwd_cvae":      Y_fwd_cvae,
        "dm_tests":        dm_tests,
        "run_config":      run_config,
    }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the retrodictive forecasting pipeline for a single case",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_single.py --case A
  python run_single.py --case C --outdir results_C
  python run_single.py --case A --quick
  python run_single.py --case D --device cuda
    python run_single.py --case ERA5 --quick
        """,
    )
    parser.add_argument(
        "--case", type=str, required=True,
        choices=CASE_ORDER,
                help="Case key (A, B, C, D, or ERA5)",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory (default: outputs/<case_name>)",
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
        "--quiet", action="store_true",
        help="Reduce console output (disable verbose progress prints)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed (default: 42)",
    )

    args = parser.parse_args()

    # Configuration
    if args.quick:
        cfg = _quick_config()
        print("[!] QUICK MODE: reduced epochs and data for testing")
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

    # Output directory
    if args.outdir is None:
        outdir = os.path.join(
            cfg.experiment.outdir,
            CASES[args.case].name,
        )
    else:
        outdir = args.outdir

    # Device
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # Set seeds
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    # Run
    run_single_case(
        case_key=args.case,
        cfg=cfg,
        device=device,
        outdir=outdir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
