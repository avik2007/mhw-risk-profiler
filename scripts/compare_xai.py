#!/usr/bin/env python3
"""
compare_xai.py — Per-season Captum IG attribution comparison: ERA5 vs WN2
==========================================================================
Loads era5_weights.pt and wn2_weights.pt, runs Integrated Gradients per
season (DJF, MAM, JJA, SON) on each model with its own inputs, and saves
a structured attribution comparison to data/results/xai_comparison.json.

Usage:
    # Dry-run (no GEE/HYCOM — uses synthetic tensors):
    conda run -n mhw-risk python scripts/compare_xai.py --dry-run

    # Real run (requires both weight files and merged Zarr datasets):
    conda run -n mhw-risk python scripts/compare_xai.py \\
        --era5-data data/processed/merged_era5_val.zarr \\
        --wn2-data  data/processed/merged_wn2_val.zarr

Output
------
    data/results/xai_comparison.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from captum.attr import IntegratedGradients

import sys
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(Path(__file__).parent))

from src.models.ensemble_wrapper import MHWRiskModel
from _train_utils import (
    N_MEMBERS, SEQ_LEN, SEASONS, HYCOM_VARS, WN2_VARS,
)

ATM_FEATURE_NAMES   = WN2_VARS    # 5 atmospheric variables
HYCOM_CHANNEL_NAMES = HYCOM_VARS  # 4 HYCOM channels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--use-gcs", action="store_true",
                   help="Fetch and harmonize validation data from GCS.")
    p.add_argument("--era5-weights", default="data/models/era5_weights.pt")
    p.add_argument("--wn2-weights",  default="data/models/wn2_weights.pt")
    p.add_argument("--era5-data",    default=None,
                   help="Path to merged ERA5 val Zarr (required for real local run).")
    p.add_argument("--wn2-data",     default=None,
                   help="Path to merged WN2 val Zarr (required for real local run).")
    p.add_argument("--n-steps",      type=int, default=50,
                   help="IG integration steps. More steps = more accurate but slower.")
    return p.parse_args()


def load_model(weights_path: str, device: torch.device) -> MHWRiskModel:
    """
    Load MHWRiskModel from a saved weights file.

    Parameters
    ----------
    weights_path : str
        Path to a .pt file produced by train_era5.py or train_wn2.py.
    device : torch.device
        CPU or CUDA device.

    Returns
    -------
    model : MHWRiskModel in eval mode.
    """
    model = MHWRiskModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def latent_forward(model: MHWRiskModel):
    """
    Return a Captum-compatible wrapper that maps (hycom, wn2) to a scalar per batch.

    Captum IG requires a function that takes the inputs and returns a scalar output.
    We use the model's latent representation (the gated fusion layer output) rather
    than the SDD prediction itself, because the latent captures both HYCOM and
    atmospheric contributions before the final linear projection.

    The latent has shape (batch, member, latent_dim); we collapse to (batch,) by
    taking the mean across member and latent dimensions.

    Parameters
    ----------
    model : MHWRiskModel
        Trained model. The second return value of model() is the latent tensor.

    Returns
    -------
    Callable: (hycom_tensor, wn2_tensor) -> scalar tensor of shape (batch,).
    """
    def _forward(hycom_in: torch.Tensor, wn2_in: torch.Tensor) -> torch.Tensor:
        _, lat, _ = model(hycom_in, wn2_in)
        return lat.mean(dim=[1, 2])  # (batch,)
    return _forward


def get_season_tensors(
    merged: xr.Dataset,
    season_months: list,
    device: torch.device,
    n_members: int = N_MEMBERS,
    seq_len: int = SEQ_LEN,
) -> tuple:
    """
    Extract per-cell HYCOM and WN2 tensors for the given season months.

    The model is trained on per-cell inputs (N_cells, member, ...). Spatial-mean
    inputs are out-of-distribution for the trained model and would yield
    misleading IG attributions. This function mirrors build_tensors() in
    _train_utils.py: it stacks every valid ocean cell into the batch dimension
    so IG runs on the same input distribution as training.

    Parameters
    ----------
    merged : xr.Dataset
        Harmonized validation dataset. WN2 vars have dims (member, time, lat, lon).
        HYCOM vars have dims (member, time, depth, lat, lon).
    season_months : list of int
        Months belonging to the season, e.g. [6, 7, 8] for JJA.
    device : torch.device
        Destination device for returned tensors.
    n_members : int
        Number of ensemble members. Used for zero-padding if T_season < seq_len.
    seq_len : int
        Target atmospheric sequence length [days]. If fewer season days are
        available (e.g. DJF in 2019 = Jan-Feb only = ~59 days), the sequence is
        zero-padded at the start to reach seq_len.

    Returns
    -------
    hycom_t : torch.Tensor, shape (N_cells, member, depth=11, channels=4)
    wn2_t   : torch.Tensor, shape (N_cells, member, seq_len, features=5)
    """
    # Select time steps matching this season's months
    time_mask = merged["time"].dt.month.isin(season_months).values  # numpy bool array
    merged_season = merged.isel(time=time_mask)

    # Identify valid ocean cells using HYCOM water_temp[depth=0] (matches build_tensors)
    sst_celsius = merged_season["water_temp"].isel(depth=0)  # (member, time, lat, lon)
    valid_mask = ~sst_celsius.isel(member=0, time=0).isnull()
    lats, lons = np.where(valid_mask.values)
    n_cells = len(lats)

    # HYCOM: time-mean -> (member, depth, lat, lon, 4)
    hycom_raw = np.stack(
        [merged_season[v].mean(dim="time").values for v in HYCOM_VARS],
        axis=-1,
    ).astype(np.float32)

    # WN2: full season -> (member, T_season, lat, lon, 5), pad/truncate to seq_len on time
    wn2_full = np.stack(
        [merged_season[v].values for v in WN2_VARS],
        axis=-1,
    ).astype(np.float32)
    T = wn2_full.shape[1]
    if T >= seq_len:
        wn2_seq = wn2_full[:, -seq_len:, ...]
    else:
        pad_shape = (n_members, seq_len - T) + wn2_full.shape[2:]
        pad = np.zeros(pad_shape, dtype=np.float32)
        wn2_seq = np.concatenate([pad, wn2_full], axis=1)

    hycom_list, wn2_list = [], []
    for i in range(n_cells):
        lat_idx, lon_idx = lats[i], lons[i]
        hycom_list.append(hycom_raw[:, :, lat_idx, lon_idx, :])  # (M, depth, 4)
        wn2_list.append(wn2_seq[:, :, lat_idx, lon_idx, :])      # (M, seq_len, 5)

    hycom_t = torch.from_numpy(np.stack(hycom_list)).to(torch.float32).contiguous().to(device)
    wn2_t   = torch.from_numpy(np.stack(wn2_list)).to(torch.float32).contiguous().to(device)
    return hycom_t, wn2_t


def run_season_ig(
    model: MHWRiskModel,
    hycom_t: torch.Tensor,
    wn2_t: torch.Tensor,
    n_steps: int = 50,
    cells_per_chunk: int = 1,
) -> tuple:
    """
    Run Captum Integrated Gradients for one season and aggregate attribution scores
    across all valid ocean cells.

    Parameters
    ----------
    model : MHWRiskModel
        Trained model in eval mode.
    hycom_t : torch.Tensor, shape (N_cells, member, depth=11, channels=4)
        Per-cell HYCOM input tensor for the season.
    wn2_t : torch.Tensor, shape (N_cells, member, seq_len, features=5)
        Per-cell WN2/ERA5 atmospheric input tensor for the season.
    n_steps : int
        Number of IG integration steps. Higher = more accurate, slower.
    cells_per_chunk : int
        How many cells to feed to IG per call. Default 1 keeps the effective
        per-step batch (cells_per_chunk × M=64 × internal_batch_size=5) at the
        same ~320 profile budget as the pre-spatial-batching version, avoiding
        swap thrash on a 16GB VM.

    Returns
    -------
    atm_scores : dict[str, float]
        Mean |IG| attribution per WN2 variable (5 entries) across all cells.
    hycom_scores : dict[str, float]
        Mean |IG| attribution per HYCOM channel (4 entries) across all cells.
    gate_mean : float
        Mean gate value across all cells × members (0=atmosphere-dominant,
        1=depth-dominant).
    """
    n_cells = hycom_t.shape[0]
    ig = IntegratedGradients(latent_forward(model))

    hycom_abs_sum: torch.Tensor | None = None
    wn2_abs_sum:   torch.Tensor | None = None
    n_processed = 0

    for start in range(0, n_cells, cells_per_chunk):
        end = min(start + cells_per_chunk, n_cells)
        h_chunk = hycom_t[start:end].clone().requires_grad_(True)
        w_chunk = wn2_t[start:end].clone().requires_grad_(True)

        attr = ig.attribute(
            (h_chunk, w_chunk),
            baselines=(torch.zeros_like(h_chunk), torch.zeros_like(w_chunk)),
            n_steps=n_steps,
            internal_batch_size=5,
        )
        h_abs = attr[0].detach().abs().sum(dim=0)  # (M, depth, 4)
        w_abs = attr[1].detach().abs().sum(dim=0)  # (M, seq_len, 5)

        if hycom_abs_sum is None:
            hycom_abs_sum = h_abs
            wn2_abs_sum   = w_abs
        else:
            hycom_abs_sum += h_abs
            wn2_abs_sum   += w_abs
        n_processed += (end - start)

    hycom_abs_mean = hycom_abs_sum / n_processed   # (M, depth, 4)
    wn2_abs_mean   = wn2_abs_sum   / n_processed   # (M, seq_len, 5)

    hycom_scores = {
        HYCOM_CHANNEL_NAMES[c]: float(hycom_abs_mean[..., c].mean())
        for c in range(len(HYCOM_CHANNEL_NAMES))
    }
    atm_scores = {
        ATM_FEATURE_NAMES[f]: float(wn2_abs_mean[..., f].mean())
        for f in range(len(ATM_FEATURE_NAMES))
    }

    # Gate aggregated across all cells × members (mini-batched to avoid OOM)
    gate_chunks: list[torch.Tensor] = []
    bs = 32
    with torch.no_grad():
        for start in range(0, n_cells, bs):
            end = start + bs
            _, _, g = model(hycom_t[start:end], wn2_t[start:end])
            gate_chunks.append(g)
    gate_mean = float(torch.cat(gate_chunks, dim=0).mean())

    return atm_scores, hycom_scores, gate_mean


def compute_delta(era5_season: dict, wn2_season: dict) -> dict:
    """
    Compute WN2 minus ERA5 attribution delta for atm and hycom scores.

    A positive delta for a variable means WN2 attributes more importance to it
    than ERA5 — indicating the real FGN ensemble sees that variable as more
    informative than the synthetic ERA5 ensemble.

    Parameters
    ----------
    era5_season : dict with 'atm' and 'hycom' sub-dicts
    wn2_season  : dict with 'atm' and 'hycom' sub-dicts

    Returns
    -------
    delta : dict with 'atm' and 'hycom' sub-dicts, values = wn2 - era5.
    """
    delta = {}
    for stream in ("atm", "hycom"):
        delta[stream] = {
            var: round(wn2_season[stream][var] - era5_season[stream][var], 6)
            for var in wn2_season[stream]
        }
    return delta


def save_attribution_plots(result: dict, out_dir: str = "data/results/xai") -> None:
    """
    Save per-season IG attribution bar charts for ERA5 and WN2 models.

    Parameters
    ----------
    result : dict
        XAI comparison result as returned by main() — keys 'era5', 'wn2', 'delta'.
        Each season entry has 'atm' (5 WN2 vars) and 'hycom' (4 HYCOM vars) sub-dicts
        mapping variable name → mean absolute IG attribution [dimensionless].
    out_dir : str
        Directory to write PNG files. Created if absent.

    Plots saved
    -----------
    ig_attribution_DJF.png — side-by-side ERA5 vs WN2 attribution bar chart for DJF
    ig_attribution_MAM.png — same for MAM
    ig_attribution_JJA.png — same for JJA
    ig_attribution_SON.png — same for SON
    Each chart shows atmospheric vars (top panel) and HYCOM vars (bottom panel).
    Bar colour: steelblue = ERA5, tomato = WN2.

    Notes
    -----
    Integrated Gradients (IG) measures the contribution of each input variable to the
    model's latent fusion output. Higher absolute IG = stronger influence on the learned
    representation of MHW risk patterns. The latent representation captures both HYCOM
    and atmospheric contributions before the final linear SDD projection, making it a
    pure measure of physical importance rather than artifact of final layer bias.

    Attribution values are dimensionless (normalized gradients multiplied by input
    magnitude). They are comparable across variables within a season but not across
    seasons without normalization.

    Atmospheric variables (WeatherNext 2):
    - sea surface temperature [K]: captures rapid SST forcing; key trigger for MHW onset
    - 2m air temperature [K]: measures atmospheric heat content; modulates air-sea flux
    - u-wind, v-wind [m/s]: wind-driven mixing and entrainment; controls SST cooling
    - mean sea level pressure [Pa]: geostrophic forcing; anticyclones suppress mixing

    HYCOM variables (subsurface):
    - water temperature [K]: vertical thermal structure; controls MHW penetration depth
    - salinity [PSU]: density stratification; can suppress mixing and prolong MHW duration
    - u-current, v-current [m/s]: advection of warm/cold water; determines regional
      anomaly persistence

    Financial interpretation:
    Comparing ERA5 and WN2 bar heights reveals whether the deterministic reanalysis and
    the stochastic ensemble emphasise the same physical drivers of MHW risk. Large deltas
    (WN2 - ERA5) for specific variables indicate the probabilistic forecast captures
    ensemble uncertainty differently, a key validity check before underwriting parametric
    insurance triggers derived from either dataset. Consensus high IG across both models
    strengthens confidence in risk factor selection.
    """
    import matplotlib.pyplot as plt
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    seasons = ["DJF", "MAM", "JJA", "SON"]
    for season in seasons:
        era5_atm   = result["era5"][season]["atm"]
        era5_hycom = result["era5"][season]["hycom"]
        wn2_atm    = result["wn2"][season]["atm"]
        wn2_hycom  = result["wn2"][season]["hycom"]

        atm_vars   = list(era5_atm.keys())
        hycom_vars = list(era5_hycom.keys())

        fig, (ax_atm, ax_hycom) = plt.subplots(2, 1, figsize=(10, 7))
        fig.suptitle(f"IG Attribution — {season}", fontsize=13)

        x_atm = range(len(atm_vars))
        ax_atm.bar([x - 0.2 for x in x_atm], [era5_atm[v] for v in atm_vars],
                   width=0.4, label="ERA5", color="steelblue")
        ax_atm.bar([x + 0.2 for x in x_atm], [wn2_atm[v] for v in atm_vars],
                   width=0.4, label="WN2", color="tomato")
        ax_atm.set_xticks(list(x_atm))
        ax_atm.set_xticklabels(atm_vars, rotation=20, ha="right", fontsize=8)
        ax_atm.set_ylabel("Mean |IG| attribution")
        ax_atm.set_title("Atmospheric variables")
        ax_atm.legend()
        ax_atm.grid(True, alpha=0.3, axis="y")

        x_hycom = range(len(hycom_vars))
        ax_hycom.bar([x - 0.2 for x in x_hycom], [era5_hycom[v] for v in hycom_vars],
                     width=0.4, label="ERA5", color="steelblue")
        ax_hycom.bar([x + 0.2 for x in x_hycom], [wn2_hycom[v] for v in hycom_vars],
                     width=0.4, label="WN2", color="tomato")
        ax_hycom.set_xticks(list(x_hycom))
        ax_hycom.set_xticklabels(hycom_vars, rotation=15, ha="right", fontsize=9)
        ax_hycom.set_ylabel("Mean |IG| attribution")
        ax_hycom.set_title("HYCOM variables")
        ax_hycom.legend()
        ax_hycom.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        out_path = Path(out_dir) / f"ig_attribution_{season}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"XAI plot saved -> {out_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path("data/results").mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        for p in [args.era5_weights, args.wn2_weights]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Weight file not found: {p}")

    if args.dry_run:
        print("[dry-run] Using synthetic tensors.")
        import pandas as pd
        M = N_MEMBERS
        times = pd.date_range("2019-01-01", "2019-12-31", freq="D")
        fake_data = {
            v: xr.DataArray(
                np.random.rand(M, len(times), 4, 5).astype(np.float32) + 274.0,
                dims=["member", "time", "latitude", "longitude"],
                coords={"time": times},
            )
            for v in WN2_VARS
        }
        fake_data.update({
            v: xr.DataArray(
                np.random.rand(M, len(times), 11, 4, 5).astype(np.float32),
                dims=["member", "time", "depth", "latitude", "longitude"],
                coords={"time": times},
            )
            for v in HYCOM_VARS
        })
        era5_merged = xr.Dataset(fake_data)
        wn2_merged  = xr.Dataset(fake_data)
        era5_model  = MHWRiskModel().to(device)
        wn2_model   = MHWRiskModel().to(device)
    elif args.use_gcs:
        import os
        from src.ingestion.harvester import DataHarmonizer
        bucket = os.environ.get("MHW_GCS_BUCKET", "").rstrip("/")
        if not bucket:
            raise RuntimeError("MHW_GCS_BUCKET env var not set for --use-gcs.")
        
        harmonizer = DataHarmonizer()
        print(f"Loading val data from {bucket}...")
        
        # 2023 is our validation year
        hycom_val = xr.open_zarr(f"{bucket}/hycom/tiles/2023/", chunks="auto")
        era5_val  = xr.open_zarr(f"{bucket}/era5/2023/", chunks="auto")
        wn2_val   = xr.open_zarr(f"{bucket}/weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr", chunks="auto")
        
        print("Harmonizing ERA5 val...")
        era5_merged = harmonizer.harmonize(era5_val, hycom_val)
        print("Harmonizing WN2 val...")
        wn2_merged  = harmonizer.harmonize(wn2_val, hycom_val)
        
        era5_model = load_model(args.era5_weights, device)
        wn2_model  = load_model(args.wn2_weights,  device)
    else:
        if not args.era5_data or not args.wn2_data:
            raise ValueError("Must provide --era5-data and --wn2-data or use --use-gcs.")
        era5_merged = xr.open_zarr(args.era5_data)
        wn2_merged  = xr.open_zarr(args.wn2_data)
        era5_model  = load_model(args.era5_weights, device)
        wn2_model   = load_model(args.wn2_weights,  device)

    result = {"era5": {}, "wn2": {}, "delta": {}}

    for season_name, months in SEASONS.items():
        print(f"Running IG for season {season_name}...")

        era5_hycom_t, era5_wn2_t = get_season_tensors(era5_merged, months, device)
        wn2_hycom_t,  wn2_wn2_t  = get_season_tensors(wn2_merged,  months, device)

        era5_atm, era5_hycom, era5_gate = run_season_ig(
            era5_model, era5_hycom_t, era5_wn2_t, n_steps=args.n_steps
        )
        wn2_atm,  wn2_hycom,  wn2_gate  = run_season_ig(
            wn2_model,  wn2_hycom_t,  wn2_wn2_t,  n_steps=args.n_steps
        )

        result["era5"][season_name] = {
            "atm": {k: round(v, 6) for k, v in era5_atm.items()},
            "hycom": {k: round(v, 6) for k, v in era5_hycom.items()},
            "gate_mean": round(era5_gate, 4),
        }
        result["wn2"][season_name] = {
            "atm": {k: round(v, 6) for k, v in wn2_atm.items()},
            "hycom": {k: round(v, 6) for k, v in wn2_hycom.items()},
            "gate_mean": round(wn2_gate, 4),
        }
        result["delta"][season_name] = compute_delta(
            result["era5"][season_name], result["wn2"][season_name]
        )

        print(f"  ERA5 gate={era5_gate:.3f} | WN2 gate={wn2_gate:.3f}")

    out_path = "data/results/xai_comparison.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"XAI comparison saved -> {out_path}")
    save_attribution_plots(result)


if __name__ == "__main__":
    main()
