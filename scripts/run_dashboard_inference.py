"""
run_dashboard_inference.py — Generate full-grid SVaR for the dashboard.

Uses real trained model weights with spatially-structured synthetic inputs
to produce a risk map across all 17×21 GoM grid cells.

IMPORTANT: Inputs are SYNTHETIC (not observational data). Results show the
model's spatial response pattern, not real-data inference. The dashboard
labels this section "Illustrative — real model weights, synthetic inputs."

Run:
    cd mhw-risk-profiler
    /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/run_dashboard_inference.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import zarr

from src.models.ensemble_wrapper import MHWRiskModel

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "data", "results_dashboard")
os.makedirs(OUT_DIR, exist_ok=True)

# GoM grid (matches harmonize() output)
LAT = np.arange(41.0, 45.25, 0.25)   # 17 points
LON = np.arange(-71.0, -65.75, 0.25)  # 21 points

N_MEMBERS      = 64
SEQ_LEN        = 90
DEPTH          = 11   # per ensemble_wrapper.py docstring
HYCOM_FEATURES = 4    # water_temp, salinity, water_u, water_v
WN2_FEATURES   = 5    # SST, 2m_temp, U10, V10, MSLP
LABEL_NORM     = 250.0


def _make_synthetic_inputs(lat_idx: int, lon_idx: int, seed: int):
    """
    Create synthetic (lat, lon)-dependent inputs for one GoM cell.

    Structure:
    - HYCOM temperature decreases with latitude (cooler water in northern GoM)
    - WN2 shows seasonal-like variation (higher mid-sequence)
    - Both get small cell-specific random noise so no two cells are identical
    """
    rng = np.random.default_rng(seed)

    # Latitude-based temperature bias: southern GoM warmer
    lat_norm = (LAT[lat_idx] - 43.0) / 2.0   # ~0 at center, ±1 at edges
    temp_bias = -lat_norm * 0.8               # warmer at lower lat

    # Longitude-based bias: offshore (eastern) cells slightly warmer
    lon_norm = (LON[lon_idx] - (-68.5)) / 2.5
    lon_bias = lon_norm * 0.3

    bias = temp_bias + lon_bias

    # HYCOM: (M, depth, 4) — depth profiles with slight vertical gradient
    depth_idx = np.arange(DEPTH) / DEPTH
    hycom = rng.standard_normal((N_MEMBERS, DEPTH, HYCOM_FEATURES)) * 0.5
    hycom[:, :, 0] += bias - depth_idx * 0.4   # water_temp: surface warmer
    hycom = hycom.astype(np.float32)

    # WN2: (M, seq_len, 5) — seasonal-like arc (warmer mid-summer)
    t = np.linspace(0, np.pi, SEQ_LEN)
    seasonal = np.sin(t) * 0.6
    wn2 = rng.standard_normal((N_MEMBERS, SEQ_LEN, WN2_FEATURES)) * 0.4
    wn2[:, :, 0] += bias + seasonal   # SST-like channel
    wn2 = wn2.astype(np.float32)

    return hycom, wn2


def run_inference(weights_path: str, label: str) -> dict:
    """Load weights, run per-cell inference, return result arrays."""
    print(f"  Loading weights: {weights_path}")
    model = MHWRiskModel()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    n_lat, n_lon = len(LAT), len(LON)
    svar_95 = np.zeros((n_lat, n_lon), dtype=np.float32)
    svar_50 = np.zeros((n_lat, n_lon), dtype=np.float32)
    svar_05 = np.zeros((n_lat, n_lon), dtype=np.float32)
    spread  = np.zeros((n_lat, n_lon), dtype=np.float32)

    with torch.no_grad():
        for i in range(n_lat):
            for j in range(n_lon):
                seed = i * 100 + j
                hycom_np, wn2_np = _make_synthetic_inputs(i, j, seed)

                ht = torch.from_numpy(hycom_np).unsqueeze(0)   # (1, M, 11, 4)
                wt = torch.from_numpy(wn2_np).unsqueeze(0)     # (1, M, 90, 5)

                sdd_pred, _, _ = model(ht, wt)            # (1, M)
                sdd_1d = sdd_pred[0] * LABEL_NORM         # physical units

                svar_95[i, j] = float(sdd_1d.quantile(0.95))
                svar_50[i, j] = float(sdd_1d.quantile(0.50))
                svar_05[i, j] = float(sdd_1d.quantile(0.05))
                spread[i, j]  = svar_95[i, j] - svar_05[i, j]

            print(f"    lat={LAT[i]:.2f} done ({i+1}/{n_lat})", flush=True)

    return {"SVaR_95": svar_95, "SVaR_50": svar_50, "SVaR_05": svar_05, "spread": spread}


def save_zarr(result: dict, out_path: str):
    store = zarr.open(out_path, mode="w")
    store.create_array("latitude",  data=LAT.astype(np.float32))
    store.create_array("longitude", data=LON.astype(np.float32))
    store.create_array("SVaR_95",   data=result["SVaR_95"])
    store.create_array("SVaR_50",   data=result["SVaR_50"])
    store.create_array("SVaR_05",   data=result["SVaR_05"])
    store.create_array("spread",    data=result["spread"])
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    era5_w = os.path.join(BASE, "data/models/era5_best_weights.pt")
    wn2_w  = os.path.join(BASE, "data/models/wn2_best_weights.pt")

    print("Running ERA5 dashboard inference...")
    era5_result = run_inference(era5_w, "era5")
    save_zarr(era5_result, os.path.join(OUT_DIR, "era5_svar_dashboard.zarr"))

    print("\nRunning WN2 dashboard inference...")
    wn2_result = run_inference(wn2_w, "wn2")
    save_zarr(wn2_result, os.path.join(OUT_DIR, "wn2_svar_dashboard.zarr"))

    print("\nDone. Run export_dashboard_json.py next.")
