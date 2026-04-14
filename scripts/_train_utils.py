"""
_train_utils.py — Shared utilities for train_era5.py and train_wn2.py
======================================================================
Provides build_tensors(), run_svar_inference(), and save_plots().
Both training scripts import from here — never duplicate these functions.

Coordinate convention
---------------------
DataHarmonizer.harmonize() uses 'latitude' and 'longitude' (not 'lat'/'lon').
All spatial indexing here must use these names.

SST unit convention
-------------------
WN2 and ERA5 SST are in Kelvin. accumulate_sdd() expects degrees C.
build_tensors() subtracts 273.15 before computing the SDD label.
The WN2/ERA5 tensor passed to the model retains original units (K) —
the model learns from whatever units it sees during training.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from src.analytics.mhw_detection import compute_mhw_mask
from src.analytics.sdd import accumulate_sdd

# ---------------------------------------------------------------------------
# Domain constants (shared between train_era5.py and train_wn2.py)
# ---------------------------------------------------------------------------

GoM_BBOX     = (-71.0, 41.0, -66.0, 45.0)   # (lon_min, lat_min, lon_max, lat_max)

# Shared training periods for ERA5 and WN2.
# Both use 2022/2023 for apples-to-apples XAI comparison. ERA5 covers 1979-present
# on GEE; WN2 covers 2022-present. HYCOM GLBy0.08/expt_93.0 covers through 2024-09-04.
TRAIN_PERIOD = ("2022-01-01", "2022-12-31")
VAL_PERIOD   = ("2023-01-01", "2023-12-31")

SEQ_LEN      = 90     # atmospheric sequence length [days] fed to TransformerEncoder
N_MEMBERS    = 64     # ensemble members (WN2) / synthetic members (ERA5 proxy)
N_LAT        = 17     # Gulf of Maine grid cells at 0.25-degree resolution
N_LON        = 21
HYCOM_VARS   = ["water_temp", "salinity", "water_u", "water_v"]
WN2_VARS     = [
    "sea_surface_temperature",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


# ---------------------------------------------------------------------------
# build_tensors
# ---------------------------------------------------------------------------

def build_tensors(
    merged: xr.Dataset,
    threshold: xr.DataArray,
    seq_len: int = SEQ_LEN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a harmonized xr.Dataset into (hycom_t, wn2_t, label_t) PyTorch tensors.

    Parameters
    ----------
    merged : xr.Dataset
        Output of DataHarmonizer.harmonize(). Must have:
        - WN2 vars: dims (member, time, latitude, longitude), SST in Kelvin
        - HYCOM vars: dims (member, time, depth, latitude, longitude)
        - Coordinate names: 'latitude' and 'longitude'
    threshold : xr.DataArray
        Climatological SST threshold [deg C], dims (dayofyear, latitude, longitude).
        Produced by mhw_detection.compute_climatology().
    seq_len : int
        Number of time steps to use from the atmospheric sequence.
        Uses the LAST seq_len days of the time axis — most recent atmospheric
        forcing is most predictive of current thermal stress accumulation.

    Returns
    -------
    hycom_t : torch.Tensor, shape (1, member, depth=11, channels=4)
        Time- and spatially-averaged HYCOM profile per member.
        All 64 members are identical (HYCOM is broadcast by DataHarmonizer).
    wn2_t : torch.Tensor, shape (1, member, seq_len, features=5)
        Last seq_len days of WN2/ERA5 atmospheric sequence, spatially averaged.
    label_t : torch.Tensor, shape (1, member)
        Physics-based SDD label [deg C * day] spatially averaged over the GoM domain.
        Computed from merged SST (converted to deg C) via MHW mask + accumulate_sdd.
    """
    # HYCOM: time-and-spatial mean -> (member, depth=11, channels=4)
    hycom_arr = np.stack(
        [merged[v].mean(dim=["time", "latitude", "longitude"]).values for v in HYCOM_VARS],
        axis=-1,
    ).astype(np.float32)  # (member, depth, 4)

    # WN2/ERA5: last seq_len days, spatial mean -> (member, seq_len, features=5)
    wn2_arr = np.stack(
        [merged[v].isel(time=slice(-seq_len, None)).mean(dim=["latitude", "longitude"]).values
         for v in WN2_VARS],
        axis=-1,
    ).astype(np.float32)  # (member, seq_len, 5)

    # SDD label: convert SST from K to deg C before physics computation
    sst_celsius = merged["sea_surface_temperature"] - 273.15  # (member, time, lat, lon)
    mhw_mask    = compute_mhw_mask(sst_celsius, threshold)
    sdd_phys    = accumulate_sdd(sst_celsius, threshold, mhw_mask)  # (member, lat, lon)
    label_arr   = sdd_phys.mean(dim=["latitude", "longitude"]).values.astype(np.float32)  # (member,)

    hycom_t = torch.from_numpy(hycom_arr).unsqueeze(0)   # (1, M, 11, 4)
    wn2_t   = torch.from_numpy(wn2_arr).unsqueeze(0)    # (1, M, seq_len, 5)
    label_t = torch.from_numpy(label_arr).unsqueeze(0)  # (1, M)

    return hycom_t, wn2_t, label_t


# ---------------------------------------------------------------------------
# run_svar_inference
# ---------------------------------------------------------------------------

def run_svar_inference(
    model: torch.nn.Module,
    merged_val: xr.Dataset,
    device: torch.device,
    prefix: str,
) -> xr.Dataset:
    """
    Per-grid-cell SVaR inference. Iterates over each (latitude, longitude) cell,
    runs a forward pass with batch=1 for that cell's HYCOM profile and WN2
    atmospheric sequence, and computes ensemble quantiles.

    Parameters
    ----------
    model : MHWRiskModel
        Trained model in eval mode.
    merged_val : xr.Dataset
        Harmonized validation Dataset. Same structure as harmonize() output.
    device : torch.device
        CPU or CUDA device to run inference on.
    prefix : str
        Either 'era5' or 'wn2'. Used for output file naming.

    Returns
    -------
    svar_ds : xr.Dataset
        Dimensions: (latitude, longitude). Saved to data/results/{prefix}_svar.zarr.
        SVaR_95 : float [deg C * day] — 95th quantile of per-member SDD predictions;
            used as the parametric insurance trigger.
        SVaR_50 : float [deg C * day] — median SDD prediction across ensemble members.
        SVaR_05 : float [deg C * day] — 5th quantile; lower bound of the ensemble.
        spread  : float [deg C * day] — SVaR_95 minus SVaR_05; confirms non-degenerate
            ensemble when positive.
    """
    model.eval()
    lats = merged_val["latitude"].values
    lons = merged_val["longitude"].values
    n_lat, n_lon = len(lats), len(lons)

    svar_95 = np.zeros((n_lat, n_lon), dtype=np.float32)
    svar_50 = np.zeros((n_lat, n_lon), dtype=np.float32)
    svar_05 = np.zeros((n_lat, n_lon), dtype=np.float32)
    spread  = np.zeros((n_lat, n_lon), dtype=np.float32)

    with torch.no_grad():
        for i in range(n_lat):
            for j in range(n_lon):
                cell = merged_val.isel(latitude=i, longitude=j)

                # HYCOM profile for this cell: time-mean -> (member, depth, channels)
                hycom_cell = np.stack(
                    [cell[v].mean(dim="time").values for v in HYCOM_VARS],
                    axis=-1,
                ).astype(np.float32)  # (member, depth, 4)

                # WN2 sequence for this cell: last SEQ_LEN days -> (member, seq_len, features)
                wn2_cell = np.stack(
                    [cell[v].isel(time=slice(-SEQ_LEN, None)).values for v in WN2_VARS],
                    axis=-1,
                ).astype(np.float32)  # (member, seq_len, 5)

                ht = torch.from_numpy(hycom_cell).unsqueeze(0).to(device)  # (1, M, 11, 4)
                wt = torch.from_numpy(wn2_cell).unsqueeze(0).to(device)    # (1, M, 90, 5)

                sdd_pred, _, _ = model(ht, wt)  # (1, M)
                sdd_1d = sdd_pred[0]             # (M,)

                svar_95[i, j] = sdd_1d.quantile(0.95).item()
                svar_50[i, j] = sdd_1d.quantile(0.50).item()
                svar_05[i, j] = sdd_1d.quantile(0.05).item()
                spread[i, j]  = svar_95[i, j] - svar_05[i, j]

    svar_ds = xr.Dataset(
        {
            "SVaR_95": xr.DataArray(svar_95, dims=["latitude", "longitude"],
                                    coords={"latitude": lats, "longitude": lons}),
            "SVaR_50": xr.DataArray(svar_50, dims=["latitude", "longitude"],
                                    coords={"latitude": lats, "longitude": lons}),
            "SVaR_05": xr.DataArray(svar_05, dims=["latitude", "longitude"],
                                    coords={"latitude": lats, "longitude": lons}),
            "spread":  xr.DataArray(spread,  dims=["latitude", "longitude"],
                                    coords={"latitude": lats, "longitude": lons}),
        }
    )
    out_path = f"data/results/{prefix}_svar.zarr"
    svar_ds.to_zarr(out_path, mode="w")
    print(f"SVaR saved -> {out_path}  (lat={n_lat}, lon={n_lon})")
    return svar_ds


# ---------------------------------------------------------------------------
# save_plots
# ---------------------------------------------------------------------------

def save_plots(
    log_rows: list[dict],
    model: torch.nn.Module,
    hycom_val: torch.Tensor,
    wn2_val: torch.Tensor,
    label_val: torch.Tensor,
    device: torch.device,
    prefix: str,
) -> None:
    """
    Generate and save 5 diagnostic training plots to data/results/plots/.

    Parameters
    ----------
    log_rows : list of dict
        Per-epoch log records. Required keys: epoch, train_loss, val_loss,
        SVaR_95, SVaR_50, SVaR_05, spread [all deg C * day except epoch and losses].
        The gate histogram is computed via a forward pass, not from log_rows.
    model : MHWRiskModel
        Trained model in eval mode for final-epoch forward pass.
    hycom_val, wn2_val, label_val : torch.Tensor
        Validation tensors for pred-vs-actual scatter.
    device : torch.device
        CPU or CUDA device.
    prefix : str
        'era5' or 'wn2'. Used for file naming and plot titles.

    Plots saved
    -----------
    {prefix}_loss_curve.png    — train + val loss vs epoch
    {prefix}_svar_curve.png   — SVaR_95/50/05 vs epoch
    {prefix}_spread_curve.png — ensemble spread vs epoch
    {prefix}_gate_hist.png    — gate value histogram at final epoch
    {prefix}_pred_vs_actual.png — predicted SDD vs physics SDD (val set)
    """
    plots_dir = Path("data/results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs       = [r["epoch"]      for r in log_rows]
    train_losses = [r["train_loss"] for r in log_rows]
    val_losses   = [r["val_loss"]   for r in log_rows]
    svar_95      = [r["SVaR_95"]    for r in log_rows]
    svar_50      = [r["SVaR_50"]    for r in log_rows]
    svar_05      = [r["SVaR_05"]    for r in log_rows]
    spreads      = [r["spread"]     for r in log_rows]

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train loss", color="steelblue")
    ax.plot(epochs, val_losses,   label="Val loss",   color="tomato", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss [deg C^2 * day^2]")
    ax.set_title(f"{prefix.upper()} — Loss curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_loss_curve.png", dpi=120)
    plt.close(fig)

    # --- SVaR evolution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, svar_95, label="SVaR_95", color="firebrick")
    ax.plot(epochs, svar_50, label="SVaR_50", color="orange")
    ax.plot(epochs, svar_05, label="SVaR_05", color="steelblue")
    ax.set_xlabel("Epoch"); ax.set_ylabel("SDD [deg C * day]")
    ax.set_title(f"{prefix.upper()} — SVaR evolution")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_svar_curve.png", dpi=120)
    plt.close(fig)

    # --- Ensemble spread evolution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, spreads, color="purple")
    ax.set_xlabel("Epoch"); ax.set_ylabel("SVaR_95 - SVaR_05 [deg C * day]")
    ax.set_title(f"{prefix.upper()} — Ensemble spread")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_spread_curve.png", dpi=120)
    plt.close(fig)

    # --- Gate histogram (final epoch) ---
    model.eval()
    with torch.no_grad():
        _, _, gate = model(hycom_val.to(device), wn2_val.to(device))
    gate_vals = gate[0].cpu().numpy()  # (member,)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(gate_vals, bins=20, color="teal", edgecolor="white")
    ax.axvline(gate_vals.mean(), color="black", linestyle="--", label=f"mean={gate_vals.mean():.3f}")
    ax.set_xlabel("Gate value (0=atm-dominant, 1=depth-dominant)")
    ax.set_ylabel("Count"); ax.set_title(f"{prefix.upper()} — Gate distribution (final epoch)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_gate_hist.png", dpi=120)
    plt.close(fig)

    # --- Pred vs actual (val set) ---
    with torch.no_grad():
        sdd_pred, _, _ = model(hycom_val.to(device), wn2_val.to(device))
    pred_vals   = sdd_pred[0].cpu().numpy()   # (member,)
    actual_vals = label_val[0].cpu().numpy()  # (member,)

    fig, ax = plt.subplots(figsize=(5, 5))
    lim = max(pred_vals.max(), actual_vals.max()) * 1.1
    ax.scatter(actual_vals, pred_vals, alpha=0.6, color="steelblue", s=20)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Physics SDD [deg C * day]"); ax.set_ylabel("Predicted SDD [deg C * day]")
    ax.set_title(f"{prefix.upper()} — Predicted vs actual (val set)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_pred_vs_actual.png", dpi=120)
    plt.close(fig)

    print(f"Plots saved -> data/results/plots/{prefix}_*.png  (5 files)")
