#!/usr/bin/env python3
"""
train_era5.py — Train MHWRiskModel on ERA5 proxy data
======================================================
Trains on 2018 ERA5 data (synthetic ensemble via expand_and_perturb),
validates on 2019 ERA5 data, saves all artifacts.

Usage:
    # Dry-run (no GEE/HYCOM calls — uses synthetic tensors):
    conda run -n mhw-risk python scripts/train_era5.py --dry-run

    # Real training (requires GEE auth and HYCOM connectivity):
    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
    conda run -n mhw-risk python scripts/train_era5.py --epochs 50

Artifacts saved
---------------
    data/models/era5_weights.pt           — final epoch weights
    data/models/era5_best_weights.pt      — weights at lowest val loss
    data/results/era5_training_log.csv    — per-epoch metrics
    data/results/era5_config.json         — hyperparameters used
    data/results/era5_svar.zarr           — per-grid-cell SVaR (real run only)
    data/results/plots/era5_*.png         — 5 diagnostic plots
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

import sys
# Ensure project root is on sys.path so that `src.*` imports resolve
# regardless of how the script is invoked (e.g. `python scripts/train_era5.py`
# from the project root, or via `conda run -n mhw-risk python scripts/...`).
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(Path(__file__).parent))

from src.models.ensemble_wrapper import MHWRiskModel
from _train_utils import (
    GoM_BBOX, ERA5_TRAIN_PERIOD, ERA5_VAL_PERIOD, N_MEMBERS, SEQ_LEN,
    build_tensors, run_svar_inference, save_plots,
)

PREFIX = "era5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MHWRiskModel on ERA5 proxy data.")
    p.add_argument("--dry-run", action="store_true",
                   help="Use synthetic tensors — skip GEE and HYCOM network calls.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr",     type=float, default=1e-4)
    return p.parse_args()


def load_real_data():
    """
    Fetch ERA5 + HYCOM for train (2018) and val (2019) periods.

    Returns
    -------
    Tuple of (hycom_t_train, wn2_t_train, label_t_train,
              hycom_t_val, wn2_t_val, label_t_val,
              merged_val, threshold)
        All tensors are float32. merged_val is the harmonized 2019 Dataset
        used for per-grid-cell SVaR inference after training.
    """
    from src.ingestion.era5_harvester import ERA5Harvester
    from src.ingestion.harvester import DataHarmonizer, HYCOMLoader

    threshold_path = Path("data/processed/hycom_sst_threshold.zarr")
    if not threshold_path.exists():
        raise FileNotFoundError(
            "ERROR: hycom_sst_threshold.zarr not found. "
            "Run scripts/compute_hycom_climatology.py first."
        )
    threshold = xr.open_zarr(str(threshold_path))["threshold"]

    harvester = ERA5Harvester()
    harvester.authenticate()
    loader    = HYCOMLoader()
    harmonizer = DataHarmonizer()

    print("Fetching ERA5 train (2018)...")
    wn2_train  = harvester.fetch(*ERA5_TRAIN_PERIOD, GoM_BBOX)
    hycom_train = loader.fetch_tile(*ERA5_TRAIN_PERIOD, GoM_BBOX)
    merged_train = harmonizer.harmonize(wn2_train, hycom_train)
    # harmonize() calls expand_and_perturb() automatically when member=1
    hycom_t_train, wn2_t_train, label_t_train = build_tensors(merged_train, threshold)

    print("Fetching ERA5 val (2019)...")
    wn2_val   = harvester.fetch(*ERA5_VAL_PERIOD, GoM_BBOX)
    hycom_val_ds = loader.fetch_tile(*ERA5_VAL_PERIOD, GoM_BBOX)
    merged_val = harmonizer.harmonize(wn2_val, hycom_val_ds)
    hycom_t_val, wn2_t_val, label_t_val = build_tensors(merged_val, threshold)

    return (hycom_t_train, wn2_t_train, label_t_train,
            hycom_t_val, wn2_t_val, label_t_val,
            merged_val, threshold)


def main():
    args = parse_args()
    Path("data/models").mkdir(parents=True, exist_ok=True)
    Path("data/results/plots").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dry_run:
        print("[dry-run] Synthetic tensors — no GEE or HYCOM calls.")
        M = N_MEMBERS
        hycom_t_train = torch.randn(1, M, 11, 4)
        wn2_t_train   = torch.randn(1, M, SEQ_LEN, 5)
        label_t_train = torch.rand(1, M) * 20.0
        hycom_t_val   = torch.randn(1, M, 11, 4)
        wn2_t_val     = torch.randn(1, M, SEQ_LEN, 5)
        label_t_val   = torch.rand(1, M) * 20.0
        merged_val    = None
        threshold     = None
    else:
        (hycom_t_train, wn2_t_train, label_t_train,
         hycom_t_val, wn2_t_val, label_t_val,
         merged_val, threshold) = load_real_data()

    # Save config before training starts
    config = {
        "prefix": PREFIX,
        "lr": args.lr,
        "epochs": args.epochs,
        "n_members": N_MEMBERS,
        "seq_len": SEQ_LEN,
        "domain_bbox": GoM_BBOX,
        "train_period": ERA5_TRAIN_PERIOD,
        "val_period": ERA5_VAL_PERIOD,
        "grad_clip_max_norm": 1.0,
        "dry_run": args.dry_run,
    }
    with open(f"data/results/{PREFIX}_config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    model     = MHWRiskModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    hycom_t_train = hycom_t_train.to(device)
    wn2_t_train   = wn2_t_train.to(device)
    label_t_train = label_t_train.to(device)
    hycom_t_val   = hycom_t_val.to(device)
    wn2_t_val     = wn2_t_val.to(device)
    label_t_val   = label_t_val.to(device)

    log_rows = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- Training step ---
        model.train()
        sdd_pred, _, gate = model(hycom_t_train, wn2_t_train)
        train_loss = F.mse_loss(sdd_pred, label_t_train)
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- Validation step (no gradient) ---
        model.eval()
        with torch.no_grad():
            sdd_val, _, _ = model(hycom_t_val, wn2_t_val)
            val_loss = F.mse_loss(sdd_val, label_t_val)

        v95   = sdd_val[0].quantile(0.95).item()  # batch=1: index 0 is the only sample
        v50   = sdd_val[0].quantile(0.50).item()
        v05   = sdd_val[0].quantile(0.05).item()
        sprd  = v95 - v05
        gm    = gate[0].mean().item()

        row = {
            "epoch": epoch, "train_loss": round(train_loss.item(), 6),
            "val_loss": round(val_loss.item(), 6),
            "SVaR_95": round(v95, 4), "SVaR_50": round(v50, 4),
            "SVaR_05": round(v05, 4), "spread": round(sprd, 4),
            "gate_mean": round(gm, 4),
        }
        log_rows.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss.item():.4f} | val={val_loss.item():.4f} | "
            f"SVaR_95={v95:.2f} | spread={sprd:.2f} | gate={gm:.3f}"
        )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), f"data/models/{PREFIX}_best_weights.pt")

    # Final weights
    torch.save(model.state_dict(), f"data/models/{PREFIX}_weights.pt")
    print(f"Weights -> data/models/{PREFIX}_weights.pt")

    # Training log CSV
    with open(f"data/results/{PREFIX}_training_log.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Log     -> data/results/{PREFIX}_training_log.csv")

    # Plots
    save_plots(log_rows, model, hycom_t_val, wn2_t_val, label_t_val, device, PREFIX)

    # Per-grid-cell SVaR (real run only — dry-run skips this)
    if merged_val is not None:
        run_svar_inference(model, merged_val, device, PREFIX)
    else:
        print("[dry-run] Skipping per-grid-cell SVaR — merged_val not available.")

    print("Done.")


if __name__ == "__main__":
    main()
