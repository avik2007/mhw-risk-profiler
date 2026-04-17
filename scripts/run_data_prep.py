#!/usr/bin/env python3
"""
run_data_prep.py — Pre-fetch all training data to GCS.

Run once on a spot GCE VM (e2-standard-2, us-central1) before any real training run.
All downstream training scripts (train_era5.py, train_wn2.py) read from GCS only
after this job completes.

Idempotent: each step checks GCS for an existing Zarr store and skips if present.
Safe to re-run after spot VM preemption — only incomplete steps will re-execute.

Required environment variables:
    GOOGLE_APPLICATION_CREDENTIALS  — path to GCP service account JSON key
    MHW_GCS_BUCKET                  — GCS bucket URI, e.g. "gs://my-bucket"

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
    export MHW_GCS_BUCKET=gs://your-bucket-name
    python scripts/run_data_prep.py

Expected output:
    [1/7] HYCOM tiles 2022 -> gs://bucket/hycom/tiles/2022/  OK
    [2/7] HYCOM tiles 2023 -> gs://bucket/hycom/tiles/2023/  OK
    [3/7] HYCOM climatology -> gs://bucket/hycom/climatology/  OK
    [4/7] ERA5 tiles 2022 -> gs://bucket/era5/2022/  OK
    [5/7] ERA5 tiles 2023 -> gs://bucket/era5/2023/  OK
    [6/7] WN2 tiles 2022 -> gs://bucket/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr  OK
    [7/7] WN2 tiles 2023 -> gs://bucket/weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr  OK
    Data prep complete.

Total estimated runtime: 4-7 hours on e2-standard-4 (OPeNDAP + GEE sampleRectangle).
Estimated cost: ~$0.19/hr on-demand e2-standard-4.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gcsfs
import xarray as xr

from src.analytics.mhw_detection import compute_climatology
from src.ingestion.era5_harvester import ERA5Harvester
from src.ingestion.harvester import HYCOMLoader, WeatherNext2Harvester, _gcs_safe_write

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Gulf of Maine bbox — must match GoM_BBOX in scripts/_train_utils.py
BBOX  = (-71.0, 41.0, -66.0, 45.0)
YEARS = (2022, 2023)


def _gcs_complete(fs: gcsfs.GCSFileSystem, gcs_uri: str) -> bool:
    """Return True only if gcs_uri contains a complete consolidated Zarr store.

    Checks for .zmetadata, which to_zarr(consolidated=True) writes last.
    A directory with only partial chunk files (mid-write preemption) returns
    False, preventing corrupt stores from passing the idempotency check.
    """
    meta = gcs_uri.removeprefix("gs://").rstrip("/") + "/.zmetadata"
    return fs.exists(meta)  # Python 3.9+ removeprefix — enforced by Dockerfile (python:3.11-slim)


def main() -> None:
    bucket = os.environ.get("MHW_GCS_BUCKET", "").rstrip("/")
    if not bucket:
        raise RuntimeError("MHW_GCS_BUCKET env var not set.")

    fs     = gcsfs.GCSFileSystem()
    loader = HYCOMLoader()

    # ---- Steps 1 & 2: HYCOM full-year tiles ----
    # These are the longest fetches (~1-2 hr each over OPeNDAP). Run first so
    # the climatology step can read from GCS rather than fetching a second time.
    for step, year in enumerate(YEARS, start=1):
        gcs_uri = f"{bucket}/hycom/tiles/{year}/"
        print(f"[{step}/5] HYCOM tiles {year} -> {gcs_uri}", flush=True)
        loader.fetch_and_cache(year, BBOX, gcs_uri)
        print(f"[{step}/5] HYCOM tiles {year}  OK", flush=True)

    # ---- Step 3: HYCOM climatology ----
    # Reads the tiles just written to GCS — avoids a second OPeNDAP fetch.
    # Computes 90th-percentile SST per (dayofyear, lat, lon) over 2022/2023.
    clim_uri = f"{bucket}/hycom/climatology/"
    print(f"[3/5] HYCOM climatology -> {clim_uri}", flush=True)
    if not _gcs_complete(fs, clim_uri):
        sst_years = []
        for year in YEARS:
            tile_uri = f"{bucket}/hycom/tiles/{year}/"
            ds  = xr.open_zarr(tile_uri, chunks="auto")
            sst = ds["water_temp"].isel(depth=0).resample(time="1D").mean()
            sst.load()
            sst_years.append(sst)
        sst_all   = xr.concat(sst_years, dim="time")  # (time=730, lat, lon)
        threshold = compute_climatology(sst_all, percentile=90.0)
        _gcs_safe_write(threshold.to_dataset(name="sst_threshold_90"), clim_uri)
        logger.info("HYCOM climatology written to %s", clim_uri)
    else:
        logger.info("Cache hit — skipping climatology: %s", clim_uri)
    print("[3/5] HYCOM climatology  OK", flush=True)

    # ---- Steps 4 & 5: ERA5 full-year tiles ----
    # GEE fetches are fast (~5-10 min/year). Authenticates once, reuses for both years.
    era5 = ERA5Harvester()
    era5.authenticate()
    for step, year in enumerate(YEARS, start=4):
        gcs_uri = f"{bucket}/era5/{year}/"
        print(f"[{step}/5] ERA5 tiles {year} -> {gcs_uri}", flush=True)
        era5.fetch_and_cache(year, BBOX, gcs_uri)
        print(f"[{step}/5] ERA5 tiles {year}  OK", flush=True)

    # ---- Steps 6 & 7: WeatherNext 2 annual tiles ----
    # GEE sampleRectangle compute path: ~365 API calls per year (~30-90 min/year).
    # authenticate() is called once inside fetch_and_cache (lazy, idempotent).
    wn2 = WeatherNext2Harvester(
        gcs_bucket=bucket.removeprefix("gs://"),
        gcs_prefix="weathernext2/cache",
        service_account_key=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    )
    for step, year in enumerate(YEARS, start=6):
        gcs_uri = f"{bucket}/weathernext2/cache/wn2_{year}-01-01_{year}-12-31_m64.zarr"
        print(f"[{step}/7] WN2 tiles {year} -> {gcs_uri}", flush=True)
        wn2.fetch_and_cache(year, BBOX, gcs_uri)
        print(f"[{step}/7] WN2 tiles {year}  OK", flush=True)

    print("Data prep complete.", flush=True)


if __name__ == "__main__":
    main()
