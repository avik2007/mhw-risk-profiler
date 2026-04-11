"""
compute_hycom_climatology.py — Build a location-varying MHW threshold from HYCOM surface SST.

Fetches 2 years of HYCOM surface SST (depth=0 only — no full 3D profile) for the Gulf of
Maine region, runs compute_climatology() to compute the 90th-percentile SST per calendar day
at each grid cell, and saves the result to data/processed/hycom_sst_threshold.zarr.

The saved threshold replaces hardcoded constant thresholds (e.g. 18°C) in all downstream
compute_mhw_mask() and accumulate_sdd() calls. Each grid cell gets its own seasonal
baseline, which means MHW events are detected relative to what is locally anomalous —
not a single species-specific temperature cutoff.

HYCOM expt_93.0 coverage: 2018-01-01 to 2020-02-19.
Valid full years for this script: 2018 and 2019.

Usage:
    conda run -n mhw-risk python scripts/compute_hycom_climatology.py
    conda run -n mhw-risk python scripts/compute_hycom_climatology.py --output-dir data/processed

Expected output:
    Fetching HYCOM surface SST for 2018 (2018-01-01 to 2018-12-31) ...
    Fetching HYCOM surface SST for 2019 (2019-01-01 to 2019-12-31) ...
    Computing 90th-percentile climatology over 730 daily timesteps ...
    Saved: data/processed/hycom_sst_threshold.zarr  dims={'dayofyear': 365, 'lat': 26, 'lon': 13}
    Threshold range: 8.34°C – 22.17°C  (spatial + seasonal spread)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import xarray as xr

from src.analytics.mhw_detection import compute_climatology
from src.ingestion.harvester import HYCOMLoader

# Gulf of Maine bounding box — matches the training domain in _train_utils.py
BBOX = (-71.0, 41.0, -66.0, 45.0)

# Fetch full calendar years — avoids partial-year bias in the percentile estimate
YEARS = (2018, 2019)


def fetch_surface_sst_year(year: int) -> xr.DataArray:
    """
    Fetch one full year of daily HYCOM surface SST (depth=0 only).

    Parameters
    ----------
    year : int
        Calendar year. Must be within HYCOM expt_93.0 coverage (2018–2019).

    Returns
    -------
    sst : xr.DataArray
        Daily mean surface SST [°C], dims (time=365, lat, lon).
        Depth=0 corresponds to TARGET_DEPTHS_M[0] = 0 m (ocean surface).
    """
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    print(f"Fetching HYCOM surface SST for {year} ({start} to {end}) ...")

    loader = HYCOMLoader()
    ds = loader.fetch_tile(start, end, BBOX)

    # Resample 3-hourly → daily mean, then extract surface layer only
    sst = (
        ds["water_temp"]
        .isel(depth=0)               # surface (0 m)
        .resample(time="1D").mean()  # daily mean
    )
    sst.load()  # evaluate Dask graph before returning
    return sst  # (time=365, lat, lon)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute location-varying MHW threshold from HYCOM surface SST."
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to save the threshold Zarr (default: data/processed)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "hycom_sst_threshold.zarr"

    if out_path.exists():
        print(f"Cache hit — {out_path} already exists. Delete it to recompute.")
        return

    # Fetch both years and concatenate along time axis
    sst_years = [fetch_surface_sst_year(year) for year in YEARS]
    sst_all = xr.concat(sst_years, dim="time")  # (time=730, lat, lon)
    print(f"Computing 90th-percentile climatology over {len(sst_all.time)} daily timesteps ...")

    threshold = compute_climatology(sst_all, percentile=90.0)
    # threshold: (dayofyear=365, lat, lon)

    threshold.to_dataset(name="sst_threshold_90").to_zarr(str(out_path), mode="w")

    # Verification
    ds_check = xr.open_zarr(str(out_path))
    t = ds_check["sst_threshold_90"]
    print(
        f"Saved: {out_path}  dims={dict(ds_check.sizes)}\n"
        f"Threshold range: {float(t.min()):.2f}°C – {float(t.max()):.2f}°C  "
        f"(spatial + seasonal spread)"
    )


if __name__ == "__main__":
    main()
