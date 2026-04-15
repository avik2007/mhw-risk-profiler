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
    p.add_argument("--era5-weights", default="data/models/era5_weights.pt")
    p.add_argument("--wn2-weights",  default="data/models/wn2_weights.pt")
    p.add_argument("--era5-data",    default=None,
                   help="Path to merged ERA5 val Zarr (required for real run).")
    p.add_argument("--wn2-data",     default=None,
                   help="Path to merged WN2 val Zarr (required for real run).")
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
    Extract HYCOM and WN2 tensors for the given season months from a merged Dataset.

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
    hycom_t : torch.Tensor, shape (1, member, depth=11, channels=4)
    wn2_t   : torch.Tensor, shape (1, member, seq_len, features=5)
    """
    # Select time steps matching this season's months
    time_mask = merged["time"].dt.month.isin(season_months).values  # numpy bool array
    merged_season = merged.isel(time=time_mask)

    # HYCOM: time+spatial mean -> (member, depth, channels=4)
    hycom_arr = np.stack(
        [merged_season[v].mean(dim=["time", "latitude", "longitude"]).values
         for v in HYCOM_VARS],
        axis=-1,
    ).astype(np.float32)  # (member, depth, 4)

    # WN2: spatial mean -> (member, T_season, features=5)
    wn2_season_arr = np.stack(
        [merged_season[v].mean(dim=["latitude", "longitude"]).values
         for v in WN2_VARS],
        axis=-1,
    ).astype(np.float32)  # (member, T_season, 5)

    # Pad or truncate WN2 to seq_len on the time axis (left-pad with zeros)
    T = wn2_season_arr.shape[1]
    if T >= seq_len:
        wn2_seq = wn2_season_arr[:, -seq_len:, :]  # take last seq_len days
    else:
        pad = np.zeros((n_members, seq_len - T, len(WN2_VARS)), dtype=np.float32)
        wn2_seq = np.concatenate([pad, wn2_season_arr], axis=1)

    hycom_t = torch.from_numpy(hycom_arr).unsqueeze(0).to(device)  # (1, M, 11, 4)
    wn2_t   = torch.from_numpy(wn2_seq).unsqueeze(0).to(device)    # (1, M, seq_len, 5)
    return hycom_t, wn2_t


def run_season_ig(
    model: MHWRiskModel,
    hycom_t: torch.Tensor,
    wn2_t: torch.Tensor,
    n_steps: int = 50,
) -> tuple:
    """
    Run Captum Integrated Gradients for one season and aggregate attribution scores.

    Parameters
    ----------
    model : MHWRiskModel
        Trained model in eval mode.
    hycom_t : torch.Tensor, shape (1, member, depth=11, channels=4)
        HYCOM input tensor for the season.
    wn2_t : torch.Tensor, shape (1, member, seq_len, features=5)
        WN2/ERA5 atmospheric input tensor for the season.
    n_steps : int
        Number of IG integration steps. Higher = more accurate, slower.

    Returns
    -------
    atm_scores : dict[str, float]
        Mean |IG| attribution per WN2 variable (5 entries).
        Higher score = that variable contributed more to the latent representation.
    hycom_scores : dict[str, float]
        Mean |IG| attribution per HYCOM channel (4 entries).
    gate_mean : float
        Mean gate value across members (0=atmosphere-dominant, 1=depth-dominant).
    """
    hycom_ig = hycom_t.requires_grad_(True)
    wn2_ig   = wn2_t.requires_grad_(True)

    ig = IntegratedGradients(latent_forward(model))
    attr = ig.attribute(
        (hycom_ig, wn2_ig),
        baselines=(torch.zeros_like(hycom_ig), torch.zeros_like(wn2_ig)),
        n_steps=n_steps,
        internal_batch_size=5,  # process 5 alpha-steps × M=64 = 320 effective batch
        # instead of n_steps × M = 50 × 64 = 3200, which allocates ~3.3 GB of
        # Transformer attention weights simultaneously and thrashes swap to disk.
    )
    # attr[0]: (1, member, depth=11, channels=4) — HYCOM attribution
    # attr[1]: (1, member, seq_len,  features=5) — WN2 attribution

    # Detach before converting to scalar — avoids spurious autograd warnings
    # from Captum's internal attribution norm checks.
    hycom_abs = attr[0].detach().abs()
    wn2_abs   = attr[1].detach().abs()

    # HYCOM: mean |IG| over (batch=1, member, depth) -> one score per channel
    hycom_scores = {
        HYCOM_CHANNEL_NAMES[c]: float(hycom_abs[..., c].mean())
        for c in range(len(HYCOM_CHANNEL_NAMES))
    }

    # WN2: mean |IG| over (batch=1, member, time_steps) -> one score per feature
    atm_scores = {
        ATM_FEATURE_NAMES[f]: float(wn2_abs[..., f].mean())
        for f in range(len(ATM_FEATURE_NAMES))
    }

    # Gate value — forward pass to get gate without IG overhead
    with torch.no_grad():
        _, _, gate = model(hycom_t, wn2_t)
    gate_mean = float(gate[0].mean())

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
    else:
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
