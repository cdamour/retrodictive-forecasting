#!/usr/bin/env bash
# =============================================================================
# reproduce_all.sh
# Retrodictive Forecasting — Full Reproduction Script
# Cédric Damour, ENERGY-Lab, Université de La Réunion
# =============================================================================
#
# Reproduces all results and figures reported in the article.
# Requires Python >= 3.9 and all packages listed in requirements.txt.
#
# USAGE
# -----
#   bash reproduce_all.sh            # full run  (~2–4 h on CPU)
#   bash reproduce_all.sh --quick    # fast test (~5 min, reduced epochs)
#   bash reproduce_all.sh --device cuda   # GPU run
#
# OUTPUT
# ------
#   outputs/results_all.json          consolidated results
#   outputs/figures_cross_case/       Figs 1–14 (PNG + PDF)
#   outputs/<case_name>/figures/      per-case figures
#   outputs/<case_name>/results_<case>.json
#
# NOTES
# -----
#   - ERA5 cases require NetCDF files.  See README.md § ERA5 Data.
#   - Set ERA5_NC_PATH and ERA5_SSRD_NC_PATH below if files are available.
#   - Without ERA5 files the synthetic cases A–D still run fully.
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cpu}"
OUTDIR="${OUTDIR:-outputs}"
QUICK=""

# ── Parse arguments ───────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --quick)  QUICK="--quick" ;;
    --device) DEVICE="${2:-cpu}"; shift ;;
    --device=*) DEVICE="${arg#*=}" ;;
    --outdir=*) OUTDIR="${arg#*=}" ;;
    *) echo "[!] Unknown argument: $arg"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  Retrodictive Forecasting — Reproduction Script"
echo "  Device : $DEVICE"
echo "  Output : $OUTDIR"
if [ -n "$QUICK" ]; then
  echo "  Mode   : QUICK (reduced epochs — for pipeline testing)"
else
  echo "  Mode   : FULL"
fi
echo "============================================================"

# ── Environment check ─────────────────────────────────────────────────────────
echo ""
echo "[0/3] Checking environment..."
$PYTHON -c "import torch, numpy, scipy, sklearn, matplotlib; \
    print(f'  Python  : OK'); \
    print(f'  PyTorch : {torch.__version__}'); \
    print(f'  NumPy   : {numpy.__version__}'); \
    print(f'  SciPy   : {scipy.__version__}'); \
    print(f'  Sklearn : {sklearn.__version__}'); \
    print(f'  Matplotlib: {matplotlib.__version__}')"

# Optional: xarray for ERA5
$PYTHON -c "import xarray; print(f'  xarray  : {xarray.__version__}')" 2>/dev/null \
  || echo "  xarray  : not installed — ERA5 cases will be skipped"

# ── Step 1: Run full pipeline ──────────────────────────────────────────────────
echo ""
echo "[1/3] Running full pipeline (run_all.py)..."
$PYTHON run_all.py \
  --device "$DEVICE" \
  --outdir "$OUTDIR" \
  $QUICK

# ── Step 2: Regenerate figures from JSON ──────────────────────────────────────
echo ""
echo "[2/3] Regenerating all figures from JSON (replot_from_json.py)..."
$PYTHON replot_from_json.py \
  --json  "$OUTDIR/results_all.json" \
  --outdir "$OUTDIR/figures_replot"

# ── Step 3: Summary ───────────────────────────────────────────────────────────
echo ""
echo "[3/3] Done."
echo ""
echo "  Main results   : $OUTDIR/results_all.json"
echo "  Cross-case figs: $OUTDIR/figures_cross_case/"
echo "  Replotted figs : $OUTDIR/figures_replot/"
echo ""
echo "To inspect results:"
echo "  $PYTHON -c \"import json; d=json.load(open('$OUTDIR/results_all.json')); \\"
echo "    [print(k, d['comparisons'][k]['methods'].get('Inverse MAP (flow)',{}).get('rmse_s','N/A')) \\"
echo "    for k in d['comparisons']]\""
echo "============================================================"
