#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generators.py — Time series generators and loaders for Retrodictive Forecasting
==================================================================================

Implements the 4 synthetic processes (2 GO + 2 NO-GO) defined in config.py,
plus a real-data loader for the ERA5 case, and data preparation utilities
(windowing, standardisation, splitting).

Processes
---------
    A    GO_dissipative_nlar      — Nonlinear tanh AR(1) + multiplicative noise
    B    NOGO_rw_symmetric        — Pure symmetric random walk
    C    GO_shotnoise_relaxation  — Shot noise + exponential decay + obs. noise
    D    NOGO_sinusoid_symmetric  — Pure sinusoid + symmetric i.i.d. noise
    ERA5 ERA5_w10_2023_northsea   — Real data: box-mean 10m wind speed from NetCDF
    ERA_ssrd ERA5_ssrd_2023_northsea  — Real data: box-mean SSRD converted to W/m² + daylight filter

Design principles
-----------------
  - ALL processes use symmetric innovations epsilon_t ~ N(0,1).
  - Irreversibility (GO cases) arises from DYNAMICS, not from noise distribution.
  - Each generator is deterministic given (case_key, T, seed).
  - Generators return raw float32 numpy arrays of shape (T,).

Usage
-----
    from generators import generate_series, prepare_dataset
    from config import CASES, get_default_config

    cfg = get_default_config()
    series = generate_series("A", T=cfg.experiment.T, seed=cfg.experiment.seed)
    dataset = prepare_dataset(series, cfg, case_key="A")
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    CASES,
    CaseSpec,
    ExperimentConfig,
    FullConfig,
    get_default_config,
)


# Optional deps: only needed for ERA5 NetCDF loading
try:
    import xarray as xr
except Exception:  # pragma: no cover
    xr = None


# ============================================================================
# 1. INDIVIDUAL GENERATORS
# ============================================================================

def _generate_A(T: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    """Case A — GO: Dissipative nonlinear AR(1) with strong multiplicative noise.

    s_t = alpha*tanh(s_{t-1}) + gamma_quad*s_{t-1}^2 + gamma_cubic*s_{t-1}^3 + sigma(s_{t-1})*epsilon_t
    sigma(s) = sigma_0 + sigma_1*|s|
    epsilon_t ~ N(0, 1)

    Irreversibility mechanism:
      - tanh creates asymmetric contraction (large values compressed)
            - Strong multiplicative noise (sigma_1=0.35) couples volatility to state
      - Cubic dissipation creates asymmetric restoring force
            - Forward: low state -> low noise -> predictable
                Backward: high state -> high noise -> uncertain origin
            - The strong sigma_1 creates a marked informational asymmetry that
        the retrodictive paradigm can exploit

    Parameters (from config.py):
            alpha=0.7, gamma_quad=0.05, gamma_cubic=-0.08, sigma_0=0.3, sigma_1=0.35
    """
    alpha = params["alpha"]
    gamma_quad = params["gamma_quad"]
    gamma_cubic = params["gamma_cubic"]
    sigma_0 = params["sigma_0"]
    sigma_1 = params["sigma_1"]

    s = np.zeros(T, dtype=np.float32)
    eps = rng.standard_normal(T).astype(np.float32)

    for t in range(1, T):
        sigma_t = sigma_0 + sigma_1 * abs(s[t - 1])
        s[t] = (alpha * np.tanh(s[t - 1])
                + gamma_quad * s[t - 1] ** 2
                + gamma_cubic * s[t - 1] ** 3
                + sigma_t * eps[t])
        # Safety clamp to prevent divergence from cubic term
        s[t] = np.clip(s[t], -10.0, 10.0)

    return s


def _generate_B(T: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    """Case B — NO-GO: Pure symmetric random walk.

        s_t = s_{t-1} + sigma*epsilon_t
        epsilon_t ~ N(0, 1)

        Time-reversibility:
            Increments Delta s_t = sigma*epsilon_t are i.i.d. symmetric.
            p(Delta s_1, ..., Delta s_n) = p(Delta s_n, ..., Delta s_1)  (exchangeable)
            -> Delta_arrow ~= 0 by construction.

    Note: the raw series s_t is non-stationary (variance grows with t).
    After windowing + standardisation, the windows are approximately
    stationary within their local context, which is the operational regime
    used in this work.
    """
    sigma = params["sigma"]

    eps = rng.standard_normal(T).astype(np.float32)
    s = np.cumsum(sigma * eps).astype(np.float32)

    return s


def _generate_C(T: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    """Case C — GO: Shot noise with exponential relaxation.

    Latent state:
        x_t = lambda*x_{t-1} + A_t
      A_t = J_t with probability p_shot, else 0
      J_t ~ Exp(shot_scale)

    Observed:
    s_t = x_t + sigma_obs*epsilon_t,   epsilon_t ~ N(0, 1)

    Irreversibility mechanism:
      - Forward: sharp positive jumps followed by slow exponential decay
      - Backward (time-reversed): slow exponential GROWTH followed by
        sharp negative drops — physically impossible without energy input
      - This is the canonical example of thermodynamic irreversibility
        (energy dissipation creates a preferred arrow of time)

    Connection to fluctuation theorems:
      The asymmetry in the conditional distributions
        p(s_{t+1:t+m} | s_{t-n:t})  vs  p(s_{t-n:t} | s_{t+1:t+m})
      is directly related to the entropy production rate
      (Kawai-Parrondo-Van den Broeck, PRL 2007).
    """
    decay = params["decay"]
    shot_prob = params["shot_prob"]
    shot_scale = params["shot_scale"]
    sigma_obs = params["sigma_obs"]

    # Generate latent shot-noise process
    x = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        # Bernoulli x Exponential shot
        if rng.random() < shot_prob:
            shot = rng.exponential(scale=shot_scale)
        else:
            shot = 0.0
        x[t] = decay * x[t - 1] + shot

    # Add symmetric observation noise
    obs_noise = sigma_obs * rng.standard_normal(T).astype(np.float32)
    s = (x + obs_noise).astype(np.float32)

    return s


def _generate_D(T: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    """Case D — NO-GO: Pure sinusoid with symmetric i.i.d. noise.

    s_t = A sin(2*pi*t / P) + sigma*epsilon_t,   epsilon_t ~ N(0, 1)

    Time-reversibility:
            - The deterministic component sin(2*pi*t/P) reversed in time gives
                sin(2*pi*(T-1-t)/P) = sin(2*pi*T'/P - 2*pi*t/P), which is just a
                phase-shifted sinusoid — same distributional properties.
            - The noise is i.i.d. symmetric -> exchangeable.
            - Therefore Delta_arrow ~= 0.

    Expected behaviour:
      - Forward MLP should perform well (sinusoid is highly predictable).
      - Inverse MAP should also reconstruct, but NOT outperform forward.
    """
    amplitude = params["amplitude"]
    period = params["period"]
    sigma = params["sigma"]

    t = np.arange(T, dtype=np.float32)
    deterministic = amplitude * np.sin(2.0 * np.pi * t / period)
    noise = sigma * rng.standard_normal(T).astype(np.float32)
    s = (deterministic + noise).astype(np.float32)

    return s


def _load_era5_boxmean_w10_from_nc(nc_path: str) -> np.ndarray:
    """Load an ERA5 NetCDF and compute box-mean 10m wind speed series.

    Parameters
    ----------
    nc_path : str
        Path to NetCDF containing u10/v10 on a time axis.

    Returns
    -------
    np.ndarray
        1D float32 array (length T_data) of box-mean wind speed.
    """
    if xr is None:
        raise ImportError(
            "xarray is required for the ERA5 case. Install with: pip install xarray netcdf4"
        )

    ds = xr.open_dataset(nc_path)
    try:
        # Coordinate names vary across exports
        if "time" in ds.coords:
            time_name = "time"
        elif "valid_time" in ds.coords:
            time_name = "valid_time"
        else:
            datetime_coords = [
                k for k, v in ds.coords.items() if np.issubdtype(v.dtype, np.datetime64)
            ]
            if not datetime_coords:
                raise KeyError(
                    "Could not find a time coordinate (expected 'time' or 'valid_time'). "
                    f"Found coords: {list(ds.coords)}"
                )
            time_name = datetime_coords[0]

        # ERA5 naming is usually u10/v10, but some exports keep long names
        if "u10" in ds.data_vars and "v10" in ds.data_vars:
            u10 = ds["u10"]
            v10 = ds["v10"]
        else:
            alt_u = "10m_u_component_of_wind"
            alt_v = "10m_v_component_of_wind"
            if alt_u in ds.data_vars and alt_v in ds.data_vars:
                u10 = ds[alt_u]
                v10 = ds[alt_v]
            else:
                raise KeyError(
                    "Expected variables 'u10'/'v10' (or long-name equivalents). "
                    f"Found: {list(ds.data_vars)}"
                )

        w10 = np.sqrt(u10 ** 2 + v10 ** 2)

        spatial_dim_candidates = ("latitude", "longitude", "lat", "lon")
        spatial_dims = [d for d in w10.dims if d != time_name and d in spatial_dim_candidates]
        if spatial_dims:
            w10_mean = w10.mean(dim=tuple(spatial_dims))
        else:
            w10_mean = w10

        w_arr = np.asarray(w10_mean.to_numpy(), dtype=np.float32).ravel()

        # Handle missing values defensively
        if np.isnan(w_arr).any():
            mask = np.isfinite(w_arr)
            idx = np.arange(len(w_arr))
            w_arr = np.interp(idx, idx[mask], w_arr[mask]).astype(np.float32)

        # Ensure time dimension exists (sanity)
        _ = ds[time_name]

        return w_arr
    finally:
        try:
            ds.close()
        except Exception:
            pass

def _load_era5_boxmean_ssrd_wm2_from_nc(
    nc_path: str,
    daylight_mode: str = "utc_window",
    utc_start: int = 6,
    utc_end: int = 20,
    thr_wm2: float = 10.0,
    spatial_mode: str = "max_variability",     # {"mean","point","max_variability"}
    ilat: int = 0,
    ilon: int = 0,
) -> np.ndarray:
    """Load an ERA5 NetCDF and compute box-mean SSRD (solar irradiance) series in W/m².

    ERA5 variable `ssrd` (surface solar radiation downwards) is an accumulated energy per unit area
    (J/m²) over the time step (hourly product). For hourly data, the mean flux over the hour is:
        ssrd / 3600  (W/m²).

    Parameters
    ----------
    nc_path : str
        Path to NetCDF containing `ssrd` on a time axis.
    daylight_mode : {"utc_window","threshold","none"}
        - "utc_window": keep times with utc_start <= hour <= utc_end (UTC hours).
        - "threshold": keep times with irradiance > thr_wm2.
        - "none": no filtering.
    utc_start, utc_end : int
        UTC hour window bounds (inclusive) for "utc_window".
    thr_wm2 : float
        Threshold (W/m²) for "threshold".

    Returns
    -------
    np.ndarray
        1D float32 array (length T_filtered) of box-mean irradiance (W/m²).
    """
    if xr is None:
        raise ImportError(
            "xarray is required for the ERA5 case. Install with: pip install xarray netcdf4"
        )

    ds = xr.open_dataset(nc_path)
    try:
        # Coordinate names vary across exports
        if "time" in ds.coords:
            time_name = "time"
        elif "valid_time" in ds.coords:
            time_name = "valid_time"
        else:
            datetime_coords = [
                k for k, v in ds.coords.items() if np.issubdtype(v.dtype, np.datetime64)
            ]
            if not datetime_coords:
                raise KeyError(
                    "Could not find a time coordinate (expected 'time' or 'valid_time'). "
                    f"Found coords: {list(ds.coords)}"
                )
            time_name = datetime_coords[0]

        if "ssrd" not in ds.data_vars:
            raise KeyError(
                "Expected variable 'ssrd' in ERA5 NetCDF. "
                f"Found: {list(ds.data_vars)}"
            )

        ssrd = ds["ssrd"]

        # Convert accumulated energy (J/m²) to mean flux over the hour (W/m²)
        ghi = ssrd / 3600.0

        # Spatial mean if spatial dims exist
        spatial_dim_candidates = ("latitude", "longitude", "lat", "lon")
        spatial_dims = [d for d in ghi.dims if d != time_name and d in spatial_dim_candidates]
        
        if spatial_dims:
            # Identify coord names
            lat_name = "latitude" if "latitude" in ghi.coords else ("lat" if "lat" in ghi.coords else None)
            lon_name = "longitude" if "longitude" in ghi.coords else ("lon" if "lon" in ghi.coords else None)

            if spatial_mode == "mean":
                ghi_mean = ghi.mean(dim=tuple(spatial_dims))

            elif spatial_mode == "point":
                if lat_name is None or lon_name is None:
                    raise KeyError("Could not identify lat/lon coordinate names for point selection.")
                ghi_mean = ghi.isel({lat_name: int(ilat), lon_name: int(ilon)})

            elif spatial_mode == "max_variability":
                # Choose the grid point with highest rampiness (q95 of |Δ|) within the box.
                # This reduces "clear-sky like" smoothing caused by spatial averaging and
                # avoids manual cherry-picking of a single pixel.
                if lat_name is None or lon_name is None:
                    raise KeyError("Could not identify lat/lon coordinate names for max_variability.")
                absdiff = np.abs(ghi.diff(dim=time_name))
                score = absdiff.quantile(0.95, dim=time_name)
                flat_idx = int(np.asarray(score.values).reshape(-1).argmax())
                nlon = score.sizes[lon_name]
                best_ilat = flat_idx // nlon
                best_ilon = flat_idx % nlon
                ghi_mean = ghi.isel({lat_name: int(best_ilat), lon_name: int(best_ilon)})

            else:
                raise ValueError(
                    "spatial_mode must be one of {'mean','point','max_variability'}; "
                    f"got {spatial_mode!r}"
                )
        else:
            ghi_mean = ghi

        # Daylight filtering
        if daylight_mode not in ("utc_window", "threshold", "none"):
            raise ValueError(
                "daylight_mode must be one of {'utc_window','threshold','none'}; "
                f"got {daylight_mode!r}"
            )

        if daylight_mode == "utc_window":
            hours = ds[time_name].dt.hour
            mask = (hours >= int(utc_start)) & (hours <= int(utc_end))
            ghi_mean = ghi_mean.where(mask, drop=True)
        elif daylight_mode == "threshold":
            ghi_mean = ghi_mean.where(ghi_mean > float(thr_wm2), drop=True)
        else:
            pass

        ghi_arr = np.asarray(ghi_mean.to_numpy(), dtype=np.float32).ravel()

        # Fill missing values defensively
        if np.isnan(ghi_arr).any():
            mask = np.isfinite(ghi_arr)
            idx = np.arange(len(ghi_arr))
            ghi_arr = np.interp(idx, idx[mask], ghi_arr[mask]).astype(np.float32)

        # Sanity: ensure time exists
        _ = ds[time_name]

        return ghi_arr
    finally:
        try:
            ds.close()
        except Exception:
            pass



def _generate_ERA5(T: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    """Case ERA5 — load real data from NetCDF (deterministic).

    Notes
    -----
    - Ignores RNG.
    - If T is smaller than the available ERA5 length, the series is truncated
      to the first T points (useful for quick mode).
    - If T is larger, returns the full available series.
    """
    _ = rng
    nc_path = params.get("nc_path")
    if not nc_path:
        raise ValueError("ERA5 case requires params['nc_path']")

    path = str(nc_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "ERA5 NetCDF file not found. "
            f"Expected at: {path}\n"
            "Set CASES['ERA5'].params['nc_path'] to an existing .nc file."
        )

    series = _load_era5_boxmean_w10_from_nc(path)
    if T is not None and T > 0 and T < len(series):
        series = series[:T]
    return series.astype(np.float32, copy=False)

def _generate_ERA_ssrd(T: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    """Case ERA_ssrd — load ERA5 SSRD from NetCDF and return a 1D daylight-filtered irradiance series.

    Expected params
    ---------------
    nc_path : str
        Path to the ERA5 NetCDF file containing `ssrd`.
    daylight_mode : {"utc_window","threshold","none"}, optional
        Default "utc_window".
    utc_start, utc_end : int, optional
        Default 6 and 20 (inclusive), UTC hours.
    thr_wm2 : float, optional
        Default 10.0 W/m² for threshold mode.

    Notes
    -----
    - Ignores RNG.
    - Converts ssrd to W/m² by dividing by 3600.
    - Applies daylight filtering before truncation to T.
    """
    _ = rng
    nc_path = params.get("nc_path")
    if not nc_path:
        raise ValueError("ERA_ssrd case requires params['nc_path']")

    path = str(nc_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "ERA5 NetCDF file not found. "
            f"Expected at: {path}\n"
            "Set CASES['ERA_ssrd'].params['nc_path'] to an existing .nc file."
        )

    daylight_mode = params.get("daylight_mode", "utc_window")
    utc_start = int(params.get("utc_start", 6))
    utc_end = int(params.get("utc_end", 20))
    thr_wm2 = float(params.get("thr_wm2", 10.0))

    series = _load_era5_boxmean_ssrd_wm2_from_nc(
        path,
        daylight_mode=daylight_mode,
        utc_start=utc_start,
        utc_end=utc_end,
        thr_wm2=thr_wm2,
    )

    if T is not None and T > 0 and T < len(series):
        series = series[:T]
    return series.astype(np.float32, copy=False)



# Generator dispatch table
_GENERATORS = {
    "A": _generate_A,
    "B": _generate_B,
    "C": _generate_C,
    "D": _generate_D,
    "ERA5": _generate_ERA5,
    "ERA_ssrd": _generate_ERA_ssrd,
}


# ============================================================================
# 2. PUBLIC GENERATION API
# ============================================================================

def generate_series(case_key: str, T: int, seed: int) -> np.ndarray:
    """Generate a synthetic time series for the given case.

    Parameters
    ----------
    case_key : str
        One of "A", "B", "C", "D".
    T : int
        Length of the series.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        1D float32 array of shape (T,).
    """
    if case_key not in _GENERATORS:
        raise ValueError(
            f"Unknown case_key='{case_key}'. "
            f"Available: {sorted(_GENERATORS.keys())}"
        )

    spec = CASES[case_key]
    rng = np.random.default_rng(seed)
    series = _GENERATORS[case_key](T, rng, spec.params)

    # Synthetic cases have fixed length by construction.
    if case_key in {"A", "B", "C", "D"}:
        assert series.shape == (T,), f"Expected shape ({T},), got {series.shape}"

    assert series.ndim == 1, f"Expected 1D series, got shape {series.shape}"
    assert series.dtype == np.float32, f"Expected float32, got {series.dtype}"
    assert np.all(np.isfinite(series)), f"Series contains NaN/Inf for case {case_key}"

    return series


# ============================================================================
# 3. WINDOWING
# ============================================================================

def make_windows(series: np.ndarray, n: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build supervised (past, future) window pairs via sliding window.

    Parameters
    ----------
    series : np.ndarray
        1D time series of shape (T,).
    n : int
        Past window length.
    m : int
        Future window length.

    Returns
    -------
    X : np.ndarray, shape (N_windows, n)
        Past windows.
    Y : np.ndarray, shape (N_windows, m)
        Future windows (immediately following past).
    """
    T = len(series)
    assert T > n + m, f"Series too short: T={T}, n+m={n+m}"

    N = T - n - m + 1
    X = np.zeros((N, n), dtype=np.float32)
    Y = np.zeros((N, m), dtype=np.float32)

    for i in range(N):
        X[i] = series[i: i + n]
        Y[i] = series[i + n: i + n + m]

    return X, Y


# ============================================================================
# 4. STANDARDISATION
# ============================================================================

@dataclass
class Standardizer:
    """Per-position z-score standardiser fitted on training data.

    Stores mean and std per position within the window, enabling
    position-aware standardisation that preserves temporal structure.
    """
    mean: np.ndarray  # shape (1, window_len)
    std: np.ndarray   # shape (1, window_len)

    def transform(self, a: np.ndarray) -> np.ndarray:
        """Standardise: (a - mean) / std."""
        return ((a - self.mean) / (self.std + 1e-8)).astype(np.float32)

    def inverse(self, a: np.ndarray) -> np.ndarray:
        """De-standardise: a * std + mean."""
        return (a * (self.std + 1e-8) + self.mean).astype(np.float32)


def fit_standardizer(data: np.ndarray) -> Standardizer:
    """Fit a per-position standardiser on training data.

    Parameters
    ----------
    data : np.ndarray, shape (N, window_len)
        Training windows.

    Returns
    -------
    Standardizer
        Fitted standardiser with mean/std of shape (1, window_len).
    """
    mean = np.mean(data, axis=0, keepdims=True).astype(np.float32)
    std = np.std(data, axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)  # avoid division by zero
    return Standardizer(mean=mean, std=std)


# ============================================================================
# 5. PYTORCH DATASET
# ============================================================================

class XYDataset(Dataset):
    """Simple PyTorch Dataset wrapping (X, Y) numpy arrays."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.shape[0] == Y.shape[0], "X and Y must have same N"
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


# ============================================================================
# 6. COMPLETE DATA PREPARATION PIPELINE
# ============================================================================

@dataclass
class PreparedDataset:
    """All data needed for training and evaluation of one case.

    Contains raw and standardised windows, standardisers for
    inverse transform, and metadata.
    """
    # --- Metadata ---
    case_key: str
    case_name: str
    series_raw: np.ndarray           # shape (T,)

    # --- Raw windows ---
    X_train: np.ndarray              # shape (n_train, n)
    Y_train: np.ndarray              # shape (n_train, m)
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray

    # --- Standardised windows ---
    X_train_s: np.ndarray
    Y_train_s: np.ndarray
    X_val_s: np.ndarray
    Y_val_s: np.ndarray
    X_test_s: np.ndarray
    Y_test_s: np.ndarray

    # --- Standardisers (for inverse transform) ---
    sx: Standardizer
    sy: Standardizer

    # --- Naive baseline (unconditional mean of train futures) ---
    y_mean_s: np.ndarray             # shape (1, m), standardised

    # --- Split sizes ---
    n_train: int
    n_val: int
    n_test: int


def prepare_dataset(
    series: np.ndarray,
    cfg: FullConfig,
    case_key: str,
) -> PreparedDataset:
    """Full data preparation pipeline for one synthetic case.

    Steps:
      1. Build sliding windows (past x, future y)
      2. Chronological train/val/test split
      3. Fit standardisers on train only
      4. Standardise all splits
      5. Compute naive baseline (mean future over train)

    Parameters
    ----------
    series : np.ndarray
        Raw time series, shape (T,).
    cfg : FullConfig
        Full configuration bundle.
    case_key : str
        Case identifier (for metadata).

    Returns
    -------
    PreparedDataset
        Complete dataset ready for model training and evaluation.
    """
    exp = cfg.experiment
    spec = CASES[case_key]

    # 1. Windowing
    X, Y = make_windows(series, n=exp.n, m=exp.m)
    N = X.shape[0]

    # 2. Chronological split
    n_train = int(exp.train_frac * N)
    n_val = int(exp.val_frac * N)
    n_test = N - n_train - n_val
    assert n_test > 0, f"Not enough data for test: N={N}, train={n_train}, val={n_val}"

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]

    # 3. Fit standardisers on train
    if exp.standardize:
        sx = fit_standardizer(X_train)
        sy = fit_standardizer(Y_train)
    else:
        # Identity standardiser
        sx = Standardizer(
            mean=np.zeros((1, exp.n), dtype=np.float32),
            std=np.ones((1, exp.n), dtype=np.float32),
        )
        sy = Standardizer(
            mean=np.zeros((1, exp.m), dtype=np.float32),
            std=np.ones((1, exp.m), dtype=np.float32),
        )

    # 4. Standardise
    X_train_s = sx.transform(X_train)
    X_val_s = sx.transform(X_val)
    X_test_s = sx.transform(X_test)

    Y_train_s = sy.transform(Y_train)
    Y_val_s = sy.transform(Y_val)
    Y_test_s = sy.transform(Y_test)

    # 5. Naive baseline: mean future over training set (standardised)
    y_mean_s = np.mean(Y_train_s, axis=0, keepdims=True).astype(np.float32)

    return PreparedDataset(
        case_key=case_key,
        case_name=spec.name,
        series_raw=series,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test,
        X_train_s=X_train_s, Y_train_s=Y_train_s,
        X_val_s=X_val_s, Y_val_s=Y_val_s,
        X_test_s=X_test_s, Y_test_s=Y_test_s,
        sx=sx, sy=sy,
        y_mean_s=y_mean_s,
        n_train=n_train, n_val=n_val, n_test=n_test,
    )


def make_dataloaders(
    dataset: PreparedDataset,
    batch_size: int,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from a PreparedDataset.

    Parameters
    ----------
    dataset : PreparedDataset
        Prepared dataset.
    batch_size : int
        Batch size for both loaders.
    shuffle_train : bool
        Whether to shuffle training data (default True).

    Returns
    -------
    train_loader, val_loader : Tuple[DataLoader, DataLoader]
    """
    train_ds = XYDataset(dataset.X_train_s, dataset.Y_train_s)
    val_ds = XYDataset(dataset.X_val_s, dataset.Y_val_s)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


# ============================================================================
# 7. DIAGNOSTIC UTILITIES
# ============================================================================

def series_summary(series: np.ndarray, name: str = "") -> Dict[str, float]:
    """Compute basic descriptive statistics for a time series.

    Useful for sanity-checking generated series before training.

    Returns
    -------
    dict with keys: mean, std, min, max, skewness, kurtosis,
                    acf_lag1, acf_lag5
    """
    n = len(series)
    mean = float(np.mean(series))
    std = float(np.std(series))
    skew = float(np.mean(((series - mean) / (std + 1e-8)) ** 3))
    kurt = float(np.mean(((series - mean) / (std + 1e-8)) ** 4) - 3.0)

    # Autocorrelation at lag 1 and 5
    def _acf(s, lag):
        if lag >= len(s):
            return 0.0
        s_centered = s - np.mean(s)
        c0 = np.sum(s_centered ** 2)
        if c0 < 1e-12:
            return 0.0
        ck = np.sum(s_centered[lag:] * s_centered[:-lag])
        return float(ck / c0)

    return {
        "name": name,
        "length": n,
        "mean": mean,
        "std": std,
        "min": float(np.min(series)),
        "max": float(np.max(series)),
        "skewness": skew,
        "kurtosis_excess": kurt,
        "acf_lag1": _acf(series, 1),
        "acf_lag5": _acf(series, 5),
    }


def increments_summary(series: np.ndarray) -> Dict[str, float]:
    """Compute statistics on first differences Delta s_t = s_t - s_{t-1}.

    Particularly useful for random walk (B) where the increments
    carry the relevant distributional information.
    """
    inc = np.diff(series)
    return series_summary(inc, name="increments")


# ============================================================================
# 8. SELF-TEST (generates all 4 series and prints summaries)
# ============================================================================

if __name__ == "__main__":
    from config import SYNTH_CASE_ORDER, CASES, get_default_config

    cfg = get_default_config()
    exp = cfg.experiment

    print("=" * 70)
    print("Retrodictive Forecasting — Synthetic Data Generation Summary")
    print("=" * 70)

    for key in SYNTH_CASE_ORDER:
        spec = CASES[key]
        print(f"\n--- Case {key}: {spec.name} ({spec.verdict.value}) ---")
        print(f"    {spec.description}")

        series = generate_series(key, T=exp.T, seed=exp.seed)
        stats = series_summary(series, name=spec.name)

        print(f"    Length: {stats['length']}")
        print(f"    Mean: {stats['mean']:.4f},  Std: {stats['std']:.4f}")
        print(f"    Min: {stats['min']:.4f},  Max: {stats['max']:.4f}")
        print(f"    Skewness: {stats['skewness']:.4f},  "
              f"Kurtosis(excess): {stats['kurtosis_excess']:.4f}")
        print(f"    ACF(1): {stats['acf_lag1']:.4f},  "
              f"ACF(5): {stats['acf_lag5']:.4f}")

        # Prepare dataset
        ds = prepare_dataset(series, cfg, case_key=key)
        print(f"    Windows: train={ds.n_train}, val={ds.n_val}, test={ds.n_test}")
        print(f"    X shape: ({ds.n_train}, {exp.n}),  Y shape: ({ds.n_train}, {exp.m})")
        print(f"    Standardised X_train: mean~{np.mean(ds.X_train_s):.4f}, "
              f"std~{np.std(ds.X_train_s):.4f}")
        print(f"    Standardised Y_train: mean~{np.mean(ds.Y_train_s):.4f}, "
              f"std~{np.std(ds.Y_train_s):.4f}")

        # Increment stats (useful for B)
        inc_stats = increments_summary(series)
        print(f"    Increments — Skew: {inc_stats['skewness']:.4f}, "
              f"Kurt: {inc_stats['kurtosis_excess']:.4f}, "
              f"ACF(1): {inc_stats['acf_lag1']:.4f}")

    print("\n All 4 synthetic cases generated and prepared successfully.")
