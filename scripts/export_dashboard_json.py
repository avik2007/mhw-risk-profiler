"""
Export dashboard data to JSON for the GitHub Pages static dashboard.

Reads Option B training results and SVaR zarr outputs.
Outputs three JSON files to docs/data/:
  training.json   — per-epoch loss, val_loss, gate_mean, spread for ERA5 + WN2
  svar_map.json   — lat/lon grid + SVaR_95 values (valid cells only)
  metadata.json   — model card, data sources, v1 limitations
"""
import json
import os
import math

import numpy as np
import pandas as pd
import zarr
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "docs", "data")
os.makedirs(OUT, exist_ok=True)

# Build land mask once — Natural Earth 10m resolution
_shp = shpreader.Reader(
    shpreader.natural_earth(resolution='10m', category='physical', name='land')
)
_LAND_GEOM = unary_union([rec.geometry for rec in _shp.records()])


def _is_ocean(lat: float, lon: float) -> bool:
    return not _LAND_GEOM.contains(sgeom.Point(lon, lat))


def _to_list(arr):
    """Numpy array → Python list; np.nan → None for valid JSON."""
    result = []
    for v in np.array(arr).flat:
        result.append(None if (isinstance(v, float) and math.isnan(v)) else float(v))
    return result


def _to_grid(arr_2d):
    """2-D numpy array → nested list with None for NaN."""
    return [[None if math.isnan(float(v)) else float(v) for v in row]
            for row in np.array(arr_2d)]


# ── Training curves ──────────────────────────────────────────────────────────
print("Reading training logs...")

era5_csv = os.path.join(BASE, "data/results_era5/results/era5_training_log.csv")
wn2_csv  = os.path.join(BASE, "data/results_wn2/results/wn2_training_log.csv")

era5 = pd.read_csv(era5_csv)
wn2  = pd.read_csv(wn2_csv)

era5_best_ep  = int(era5.loc[era5["val_loss"].idxmin(), "epoch"])
wn2_best_ep   = int(wn2.loc[wn2["val_loss"].idxmin(), "epoch"])

training = {
    "era5": {
        "label": "ERA5 (reanalysis + synthetic ensemble)",
        "epochs":      era5["epoch"].tolist(),
        "train_loss":  [round(v, 4) for v in era5["train_loss"]],
        "val_loss":    [round(v, 4) for v in era5["val_loss"]],
        "gate_mean":   [round(v, 4) for v in era5["gate_mean"]],
        "spread":      [round(v, 4) for v in era5["spread"]],
        "best_epoch":  era5_best_ep,
        "best_val_loss": round(float(era5["val_loss"].min()), 6),
        "final_gate":  round(float(era5["gate_mean"].iloc[-1]), 4),
        "n_epochs":    len(era5),
    },
    "wn2": {
        "label": "WeatherNext 2 (64-member physical ensemble)",
        "epochs":      wn2["epoch"].tolist(),
        "train_loss":  [round(v, 4) for v in wn2["train_loss"]],
        "val_loss":    [round(v, 4) for v in wn2["val_loss"]],
        "gate_mean":   [round(v, 4) for v in wn2["gate_mean"]],
        "spread":      [round(v, 4) for v in wn2["spread"]],
        "best_epoch":  wn2_best_ep,
        "best_val_loss": round(float(wn2["val_loss"].min()), 6),
        "final_gate":  round(float(wn2["gate_mean"].iloc[-1]), 4),
        "n_epochs":    len(wn2),
    }
}

path = os.path.join(OUT, "training.json")
with open(path, "w") as f:
    json.dump(training, f, separators=(",", ":"))
print(f"  wrote {path}")


# ── SVaR spatial maps ────────────────────────────────────────────────────────
print("Reading SVaR zarr files...")

def load_svar_cells(zarr_path):
    """Return list of {lat, lon, svar95, svar50, spread} for ocean cells only."""
    z    = zarr.open(zarr_path)
    lat  = np.array(z["latitude"])
    lon  = np.array(z["longitude"])
    s95  = np.array(z["SVaR_95"])
    s50  = np.array(z["SVaR_50"])
    sprd = np.array(z["spread"])
    cells = []
    for i in range(len(lat)):
        for j in range(len(lon)):
            la, lo = float(lat[i]), float(lon[j])
            if math.isnan(float(s95[i, j])) or not _is_ocean(la, lo):
                continue
            cells.append({
                "lat":    round(la, 2),
                "lon":    round(lo, 2),
                "svar95": round(float(s95[i, j]), 2),
                "svar50": round(float(s50[i, j]), 2),
                "spread": round(float(sprd[i, j]), 4),
            })
    return cells


def make_grid_points(zarr_path):
    """Return all lat/lon grid points (valid + land) for background grid display."""
    z   = zarr.open(zarr_path)
    lat = np.array(z["latitude"])
    lon = np.array(z["longitude"])
    return [{"lat": round(float(la), 2), "lon": round(float(lo), 2)}
            for la in lat for lo in lon]


era5_zarr = os.path.join(BASE, "data/results_dashboard/era5_svar_dashboard.zarr")
wn2_zarr  = os.path.join(BASE, "data/results_dashboard/wn2_svar_dashboard.zarr")

svar_map = {
    "domain": {"lat_min": 41.0, "lat_max": 45.0, "lon_min": -71.0, "lon_max": -66.0},
    "grid_points": make_grid_points(era5_zarr),
    "era5": load_svar_cells(era5_zarr),
    "wn2":  load_svar_cells(wn2_zarr),
    "units": "deg_C_day",
    "label_norm": 250.0,
    "note": "Illustrative SVaR — real model weights, spatially-structured synthetic inputs across full 17×21 GoM grid.",
}

path = os.path.join(OUT, "svar_map.json")
with open(path, "w") as f:
    json.dump(svar_map, f, separators=(",", ":"))
print(f"  wrote {path}")


# ── Metadata / model card ────────────────────────────────────────────────────
print("Writing metadata...")

with open(os.path.join(BASE, "data/results_era5/results/era5_config.json")) as f:
    cfg = json.load(f)

metadata = {
    "model": {
        "name": "MHWRiskModel",
        "architecture": "1D-CNN depth encoder + Transformer temporal encoder + LeakyGate",
        "n_members": cfg.get("n_members", 64),
        "seq_len": cfg.get("seq_len", 90),
        "domain_bbox": cfg.get("domain_bbox", [-71.0, 41.0, -66.0, 45.0]),
        "train_period": cfg.get("train_period"),
        "val_period": cfg.get("val_period"),
        "label_norm": cfg.get("label_norm", 250.0),
        "n_ocean_cells": 161,
        "depth_levels": 10,
    },
    "data_sources": [
        {"name": "HYCOM GLBv0.08", "role": "3D ocean (temperature, salinity, currents)",
         "resolution": "0.08°", "depth_levels": 10, "years": "2022–2023"},
        {"name": "NCEI OISST v2.1", "role": "SST baseline for MHW threshold",
         "resolution": "0.25°", "years": "1982–2011 climatology"},
        {"name": "ECMWF ERA5", "role": "Atmospheric reanalysis (5 vars)",
         "resolution": "0.25°", "ensemble": "64 synthetic members via Gaussian expansion"},
        {"name": "Google WeatherNext 2", "role": "Probabilistic atmospheric forecast (4 vars)",
         "resolution": "0.25°", "ensemble": "64 physical ensemble members"},
    ],
    "v1_limitations": [
        "2-year OISST baseline — Hobday (2016) requires ≥30 years; v2 uses 1982–2011",
        "HYCOM labels deterministic across all 64 members → spread underestimated",
        "Partial SVaR inference (5 of 161 cells) — full run requires GCS data",
        "Model converges to near-mean prediction; XAI attribution in v2 with stochastic labels",
    ],
    "v2_roadmap": [
        "30-year OISST climatology baseline (1982–2011)",
        "GLORYS12V1 ocean reanalysis + AR(1) perturbation for stochastic labels",
        "Full spatial SVaR inference across 161+ GoM ocean cells",
        "Extended training to 2022–2025 using GLORYS12V1 coverage",
    ],
    "results_summary": {
        "era5": {
            "best_epoch": training["era5"]["best_epoch"],
            "best_val_loss": training["era5"]["best_val_loss"],
            "final_gate": training["era5"]["final_gate"],
            "early_stop_epoch": training["era5"]["n_epochs"],
        },
        "wn2": {
            "best_epoch": training["wn2"]["best_epoch"],
            "best_val_loss": training["wn2"]["best_val_loss"],
            "final_gate": training["wn2"]["final_gate"],
            "early_stop_epoch": training["wn2"]["n_epochs"],
        },
    },
}

path = os.path.join(OUT, "metadata.json")
with open(path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  wrote {path}")

print("Done.")
