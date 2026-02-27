# Retrodictive Forecasting — Proof of Concept

**Cédric Damour**  
ENERGY-Lab, Université de La Réunion, Saint-Denis, La Réunion, France

---

## Overview

This repository contains the complete implementation supporting the article:

> **Retrodictive Forecasting: A Proof-of-Concept for Exploiting Temporal Asymmetry in Time Series Prediction*  
> Cédric Damour, *arXiv*, 2026 `[Arxiv_DOI_PLACEHOLDER]` 
> DOI: `[ZENODO_DOI_PLACEHOLDER]`

The core idea inverts the conventional forecasting paradigm: instead of
predicting the future from the past, we identify the future state *y\** that
would most plausibly explain the presently observed window *x_obs* through
Maximum A Posteriori (MAP) optimisation on a Conditional Variational
Autoencoder (CVAE):

```
y* = argmax_y  log p_θ(x_obs | y, z*) + λ · log p_φ(z*)
```

A necessary precondition — temporal asymmetry of the process — is assessed
via a block-permutation test on the J-divergence (arrow-of-time diagnostic).
This yields a Go / No-Go decision before any model is trained.

---

## Repository Structure

```
.
├── config.py                        # All hyperparameters and case definitions
├── generators.py                    # Synthetic series generators + ERA5 loader
├── models.py                        # InverseCVAE, ForwardCVAE, MLP, RealNVP
├── inference.py                     # MAP inference (multi-start L-BFGS + FIC)
├── diagnostics.py                   # Arrow-of-time diagnostic (J-divergence)
├── evaluation.py                    # Metrics, DM tests, bootstrap CI
├── plotting.py                      # All 14 publication figures
├── run_single.py                    # Single-case pipeline entry point
├── run_all.py                       # Full experimental protocol (all cases)
├── replot_from_json.py              # Regenerate figures from saved JSON
├── export_results_json_enriched.py  # Enriched JSON serialisation
├── reproduce_all.sh                 # One-command reproduction script
├── requirements.txt
└── LICENSE
```

### Experimental Cases

| Key | Type | Process | Expected verdict |
|-----|------|---------|-----------------|
| A | Synthetic | NLAR dissipative | GO |
| B | Synthetic | Random walk | NO-GO |
| C | Synthetic | Shot noise | NO-GO |
| D | Synthetic | Sinusoid | GO |
| ERA5 | Real | North Sea wind speed (10 m) | GO |
| ERA_ssrd | Real | North Sea solar irradiance | GO |

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CPU or GPU)

```bash
git clone https://github.com/[GITHUB_USERNAME]/retrodictive-forecasting.git
cd retrodictive-forecasting
pip install -r requirements.txt
```

For GPU support, replace the `torch` line in `requirements.txt` with the
appropriate CUDA wheel from https://pytorch.org/get-started/locally/.

---

## ERA5 Data

ERA5 reanalysis data are **not redistributed** in this repository (Copernicus
licence). To reproduce the ERA5 cases:

1. Create a free account at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).
2. Install the CDS API client: `pip install cdsapi`.
3. Download the required variables (10-m wind speed `u10`/`v10`, surface solar
   radiation downwards `ssrd`) for the North Sea region:
   - Latitude: 54°N – 58°N, Longitude: 3°E – 8°E
   - Period: 2010–2020, hourly
4. Set the NetCDF file paths in `config.py`:
   ```python
   CASES["ERA5"].params["nc_path"]      = "/path/to/era5_wind.nc"
   CASES["ERA_ssrd"].params["nc_path"]  = "/path/to/era5_ssrd.nc"
   ```

Without ERA5 files the four synthetic cases (A–D) run fully and reproduce all
falsifiable predictions P1–P5.

---

## Reproduction

### One-command reproduction (recommended)

```bash
bash reproduce_all.sh          # full run, ~2–4 h on CPU
bash reproduce_all.sh --quick  # pipeline test, ~5 min, reduced epochs
bash reproduce_all.sh --device cuda  # GPU
```

### Manual step-by-step

```bash
# Run a single case
python run_single.py --case A
python run_single.py --case A --quick    # fast test mode
python run_single.py --case ERA5         # requires NetCDF file

# Run all cases
python run_all.py
python run_all.py --cases A C D          # subset
python run_all.py --outdir my_results

# Regenerate all 14 figures from saved JSON (no retraining)
python replot_from_json.py --json outputs/results_all.json --outdir figures/
python replot_from_json.py --json outputs/results_all.json --figs 4 12 13 14
```

---

## Outputs

| Path | Content |
|------|---------|
| `outputs/results_all.json` | All metrics, arrow-of-time verdicts, DM tests, bootstrap CI, per-sample arrays |
| `outputs/figures_cross_case/` | Figs 1–14 as PNG (300 dpi) + PDF |
| `outputs/<case_name>/` | Per-case JSON and figures |

### Figure index

| Fig | Description |
|-----|-------------|
| 1 | Raw synthetic series overview |
| 2 | Arrow-of-time diagnostic (J-divergence, LEVEL + DIFF) |
| 3 | Training curves |
| 4 | Example reconstructions (best / median / worst MAP) |
| 5 | Cross-case RMSE comparison |
| 6 | RMSE per forecast horizon |
| 7 | Multi-start dispersion (GO vs NO-GO) |
| 8 | Prior ablation (flow prior vs N(0,I)) |
| 9 | Falsifiable predictions pass/fail summary |
| 10 | J_obs strength per case (arrow-of-time scoreboard) |
| 11 | RMSE per horizon — GO cases overlay |
| 12 | RetroNLL vs per-sample RMSE scatter |
| 13 | MAP loss distribution (flow vs N(0,I)) |
| 14 | FIC landscape diagnostic (scatter + CDF) |

---

## Key Hyperparameters

All hyperparameters are centralised in `config.py` and documented in the
`get_default_config()` function. Critical values used in the paper:

| Parameter | Value | Role |
|-----------|-------|------|
| `n` | 32 | Past window length (time steps) |
| `m` | 16 | Forecast horizon (time steps) |
| `z_dim` | 8 | CVAE latent dimension |
| `K_multistart` | 5 | MAP restarts per sample |
| `lam_prior` | 0.3 | Prior regularisation weight λ |
| `map_steps` | 200 | Adam iterations per restart |
| `arrow tau` | 0.05 | Significance threshold (Go/No-Go) |

---

## Architecture Summary

```
Inverse CVAE:
  Encoder q_φ(z | x_obs, y)  →  z ~ N(μ, σ²)
  Decoder p_θ(x_obs | y, z)  →  reconstructed past

MAP inference:
  y* = argmin_y  -log p_θ(x_obs | y, z*(y)) + λ · ||z*(y)||²
  Warm-started by forward CVAE prediction (FIC: Forward-Inverse Chaining)

Arrow-of-time diagnostic:
  J(w) = KL(p_fwd || p_bwd) + KL(p_bwd || p_fwd)  [J-divergence, window w]
  Verdict: GO if J(w) > threshold for ≥ C_min windows, else NO-GO
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{damour2026retrodictive,
  title   = {Retrodictive Forecasting: Inverse Inference for Time Series
             via Conditional Variational Autoencoders},
  author  = {Damour, C{\'e}dric},
  journal = {arXiv},
  year    = {2026},
  doi     = {[PAPER_DOI_PLACEHOLDER]}
}
```

---

## License

This software is released under the MIT License — see [LICENSE](LICENSE) for details.

ERA5 reanalysis data are subject to the Copernicus Licence and are **not**
included in this repository.

---

## Contact

Cédric Damour  
ENERGY-Lab, Université de La Réunion  
Saint-Denis, La Réunion, France  
`cedric.damour@univ-reunion.fr` 
ORCID : https://orcid.org/0000-0002-1399-2729