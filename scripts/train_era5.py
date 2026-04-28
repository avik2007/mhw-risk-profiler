#!/usr/bin/env python3
"""
train_era5.py — Train MHWRiskModel on ERA5 proxy data
======================================================
Trains on 2022 ERA5 data (synthetic ensemble via expand_and_perturb),
validates on 2023 ERA5 data, saves all artifacts.

Usage:
    # Dry-run (no GCS calls — uses synthetic tensors):
    conda run -n mhw-risk python scripts/train_era5.py --dry-run

    # Real training (requires MHW_GCS_BUCKET set; run run_data_prep.py first):
    export MHW_GCS_BUCKET=gs://your-bucket-name
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
    GoM_BBOX, TRAIN_PERIOD, VAL_PERIOD, N_MEMBERS, SEQ_LEN,
    HYCOM_VARS, WN2_VARS,
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
    Load ERA5 + HYCOM from GCS for train (2022) and val (2023) periods.

    All data was pre-fetched to GCS by scripts/run_data_prep.py.
    Requires env var MHW_GCS_BUCKET (e.g. "gs://my-bucket").
    No live OPeNDAP or GEE calls are made here.

    harmonize() detects member=1 (ERA5 is deterministic) and calls
    expand_and_perturb() automatically to produce 64 synthetic members.

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

    print("Loading ERA5 train (2022) from GCS...")
    era5_train  = xr.open_zarr(f"{bucket}/era5/2022/", chunks="auto")
    hycom_train = xr.open_zarr(f"{bucket}/hycom/tiles/2022/", chunks="auto")
    merged_train = harmonizer.harmonize(era5_train, hycom_train)
    hycom_t_train, wn2_t_train, label_t_train = build_tensors(merged_train, threshold)

    print("Loading ERA5 val (2023) from GCS...")
    era5_val   = xr.open_zarr(f"{bucket}/era5/2023/", chunks="auto")
    hycom_val  = xr.open_zarr(f"{bucket}/hycom/tiles/2023/", chunks="auto")
    merged_val = harmonizer.harmonize(era5_val, hycom_val)
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
        # Synthetic merged_val for dry-run SVaR — 3×4 GoM sub-grid, 120 days, M members
        import pandas as pd
        _times = pd.date_range("2022-01-01", periods=120, freq="D")
        _lats  = [42.0, 43.0, 44.0]
        _lons  = [-70.0, -69.0, -68.0, -67.0]
        _nl, _nlo, _nd, _M = len(_lats), len(_lons), 11, M
        _rng = np.random.default_rng(42)
        _hycom_data = {
            v: xr.DataArray(
                _rng.standard_normal((_M, len(_times), _nd, _nl, _nlo)).astype(np.float32),
                dims=["member", "time", "depth", "latitude", "longitude"],
                coords={"time": _times, "latitude": _lats, "longitude": _lons},
            )
            for v in HYCOM_VARS
        }
        _wn2_data = {
            v: xr.DataArray(
                (_rng.standard_normal((_M, len(_times), _nl, _nlo)) + 280.0).astype(np.float32),
                dims=["member", "time", "latitude", "longitude"],
                coords={"time": _times, "latitude": _lats, "longitude": _lons},
            )
            for v in WN2_VARS
        }
        merged_val = xr.Dataset({**_hycom_data, **_wn2_data})
        threshold  = None
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
        "train_period": TRAIN_PERIOD,
        "val_period": VAL_PERIOD,
        "grad_clip_max_norm": 10.0,
        "label_norm": 250.0,
        "dry_run": args.dry_run,
    }
    with open(f"data/results/{PREFIX}_config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    model     = MHWRiskModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Move tensors to CPU before Dataset creation to save GPU VRAM
    # We will move each mini-batch to GPU inside the loop
    from torch.utils.data import DataLoader, TensorDataset
    train_ds = TensorDataset(hycom_t_train, wn2_t_train, label_t_train)
    val_ds   = TensorDataset(hycom_t_val, wn2_t_val, label_t_val)
    
    # Batch size of 32 cells (each with 64 members) -> ~2,000 profiles per step
    # This fits well within 16GB VRAM even with Transformer attention.
    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    log_rows = []
    best_val_loss = float("inf")
    patience = 10
    trigger_times = 0

    for epoch in range(1, args.epochs + 1):
        # --- Training phase ---
        model.train()
        epoch_train_loss = 0.0
        for h_batch, w_batch, l_batch in train_loader:
            h_batch, w_batch, l_batch = h_batch.to(device), w_batch.to(device), l_batch.to(device)
            
            sdd_pred, _, _ = model(h_batch, w_batch)
            loss = F.mse_loss(sdd_pred, l_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            epoch_train_loss += loss.item() * h_batch.size(0)
        
        avg_train_loss = epoch_train_loss / len(train_ds)
        scheduler.step()

        # --- Validation phase ---
        model.eval()
        epoch_val_loss = 0.0
        val_preds = []
        val_gates = []
        with torch.no_grad():
            for h_batch, w_batch, l_batch in val_loader:
                h_batch, w_batch, l_batch = h_batch.to(device), w_batch.to(device), l_batch.to(device)

                sdd_val, _, gate = model(h_batch, w_batch)
                loss = F.mse_loss(sdd_val, l_batch)

                epoch_val_loss += loss.item() * h_batch.size(0)
                val_preds.append(sdd_val.cpu())
                val_gates.append(gate.cpu())

        avg_val_loss = epoch_val_loss / len(val_ds)

        # Flatten all cell predictions for quantile analysis
        all_preds = torch.cat(val_preds, dim=0)  # (N_cells, member)
        v95  = all_preds.quantile(0.95).item()
        v50  = all_preds.quantile(0.50).item()
        v05  = all_preds.quantile(0.05).item()
        sprd = v95 - v05
        # Aggregate gate across all val cells × members (not just last batch)
        gm   = torch.cat(val_gates, dim=0).mean().item()

        row = {
            "epoch": epoch, "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "SVaR_95": round(v95, 4), "SVaR_50": round(v50, 4),
            "SVaR_05": round(v05, 4), "spread": round(sprd, 4),
            "gate_mean": round(gm, 4),
        }
        log_rows.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={avg_train_loss:.4f} | val={avg_val_loss:.4f} | "
            f"SVaR_95={v95:.2f} | spread={sprd:.2f} | gate={gm:.3f}"
        )

        # Early Stopping & Best Model Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), f"data/models/{PREFIX}_best_weights.pt")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final weights
    torch.save(model.state_dict(), f"data/models/{PREFIX}_weights.pt")
    print(f"Weights -> data/models/{PREFIX}_weights.pt")

    # Reload best-val weights for downstream plots + SVaR inference, so early-stop
    # patience-degraded final state is not what gets reported.
    best_path = f"data/models/{PREFIX}_best_weights.pt"
    if Path(best_path).exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Reloaded best-val weights from {best_path} for SVaR + plots")

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
