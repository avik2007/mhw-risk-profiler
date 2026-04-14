#!/usr/bin/env python3
"""
train_wn2.py — Train MHWRiskModel on WeatherNext 2 real FGN ensemble
=====================================================================
Trains on 2022 WN2 data (64 real FGN members), validates on 2023 WN2 data.
Architecture and hyperparameters are identical to train_era5.py — any
difference in learned weights or attributions is attributable to data alone.

Usage:
    # Dry-run (no GEE/HYCOM calls — uses synthetic tensors):
    conda run -n mhw-risk python scripts/train_wn2.py --dry-run

    # Real training:
    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
    conda run -n mhw-risk python scripts/train_wn2.py --epochs 50

Artifacts saved
---------------
    data/models/wn2_weights.pt            — final epoch weights
    data/models/wn2_best_weights.pt       — weights at lowest val loss
    data/results/wn2_training_log.csv     — per-epoch metrics
    data/results/wn2_config.json          — hyperparameters used
    data/results/wn2_svar.zarr            — per-grid-cell SVaR (real run only)
    data/results/plots/wn2_*.png          — 5 diagnostic plots
"""
import argparse
import csv
import json
from pathlib import Path

import sys
# Ensure project root is on sys.path so that `src.*` imports resolve
# regardless of how the script is invoked (e.g. `python scripts/train_wn2.py`
# from the project root, or via `conda run -n mhw-risk python scripts/...`).
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import xarray as xr

from src.models.ensemble_wrapper import MHWRiskModel
from _train_utils import (
    GoM_BBOX, TRAIN_PERIOD, VAL_PERIOD, N_MEMBERS, SEQ_LEN,
    build_tensors, run_svar_inference, save_plots,
)

PREFIX = "wn2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MHWRiskModel on WeatherNext 2 ensemble.")
    p.add_argument("--dry-run", action="store_true",
                   help="Use synthetic tensors — skip GEE and HYCOM network calls.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr",     type=float, default=1e-4)
    return p.parse_args()


def load_real_data():
    """
    Load WeatherNext 2 + HYCOM from GCS for train (2022) and val (2023) periods.

    All data was pre-fetched to GCS by scripts/run_data_prep.py.
    Requires env var MHW_GCS_BUCKET (e.g. "gs://my-bucket").
    No live OPeNDAP or GEE calls are made here.

    WN2 tiles live under the existing WeatherNext2Harvester cache path:
    gs://bucket/weathernext2/cache/wn2_YYYY-MM-DD_YYYY-MM-DD_m64.zarr
    HYCOM tiles are shared with the ERA5 training run.

    Returns
    -------
    Tuple of (hycom_t_train, wn2_t_train, label_t_train,
              hycom_t_val, wn2_t_val, label_t_val,
              merged_val, threshold)
    """
    import os
    from src.ingestion.harvester import DataHarmonizer

    bucket = os.environ.get("MHW_GCS_BUCKET", "").rstrip("/")
    if not bucket:
        raise RuntimeError(
            "MHW_GCS_BUCKET env var not set. "
            "Run scripts/run_data_prep.py on GCP first, then set this variable."
        )

    harmonizer = DataHarmonizer()
    threshold  = xr.open_zarr(f"{bucket}/hycom/climatology/")["sst_threshold_90"]

    print("Loading WN2 train (2022) from GCS...")
    wn2_train   = xr.open_zarr(
        f"{bucket}/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr", chunks="auto"
    )
    hycom_train = xr.open_zarr(f"{bucket}/hycom/tiles/2022/", chunks="auto")
    merged_train = harmonizer.harmonize(wn2_train, hycom_train)
    hycom_t_train, wn2_t_train, label_t_train = build_tensors(merged_train, threshold)

    print("Loading WN2 val (2023) from GCS...")
    wn2_val   = xr.open_zarr(
        f"{bucket}/weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr", chunks="auto"
    )
    hycom_val = xr.open_zarr(f"{bucket}/hycom/tiles/2023/", chunks="auto")
    merged_val = harmonizer.harmonize(wn2_val, hycom_val)
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

    config = {
        "prefix": PREFIX,
        "lr": args.lr,
        "epochs": args.epochs,
        "n_members": N_MEMBERS,
        "seq_len": SEQ_LEN,
        "domain_bbox": GoM_BBOX,
        "train_period": list(TRAIN_PERIOD),
        "val_period": list(VAL_PERIOD),
        "grad_clip_max_norm": 1.0,
        "dry_run": args.dry_run,
        "note": "Real FGN ensemble — no expand_and_perturb applied",
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
        model.train()
        sdd_pred, _, gate = model(hycom_t_train, wn2_t_train)
        train_loss = F.mse_loss(sdd_pred, label_t_train)
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            sdd_val, _, _ = model(hycom_t_val, wn2_t_val)
            val_loss = F.mse_loss(sdd_val, label_t_val)

        v95  = sdd_val[0].quantile(0.95).item()  # batch=1: index 0 is the only sample
        v50  = sdd_val[0].quantile(0.50).item()
        v05  = sdd_val[0].quantile(0.05).item()
        sprd = v95 - v05
        gm   = gate[0].mean().item()

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

    torch.save(model.state_dict(), f"data/models/{PREFIX}_weights.pt")
    print(f"Weights -> data/models/{PREFIX}_weights.pt")

    with open(f"data/results/{PREFIX}_training_log.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Log     -> data/results/{PREFIX}_training_log.csv")

    save_plots(log_rows, model, hycom_t_val, wn2_t_val, label_t_val, device, PREFIX)

    if merged_val is not None:
        run_svar_inference(model, merged_val, device, PREFIX)
    else:
        print("[dry-run] Skipping per-grid-cell SVaR — merged_val not available.")

    print("Done.")


if __name__ == "__main__":
    main()
