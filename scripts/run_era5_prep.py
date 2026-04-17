#!/usr/bin/env python3
"""
run_era5_prep.py — Pre-fetch ERA5 2022 and 2023 tiles to GCS in parallel with run_data_prep.py.

Idempotent: skips any year whose .zmetadata already exists in GCS.
GEE-based fetch — both years complete in ~20 minutes total.

Required environment variables:
    GOOGLE_APPLICATION_CREDENTIALS  — path to GCP service account JSON key
    MHW_GCS_BUCKET                  — GCS bucket URI, e.g. "gs://my-bucket"

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
    export MHW_GCS_BUCKET=gs://your-bucket-name
    python scripts/run_era5_prep.py

Expected output:
    [1/2] ERA5 2022 -> gs://bucket/era5/2022/  OK
    [2/2] ERA5 2023 -> gs://bucket/era5/2023/  OK
    ERA5 data prep complete.

Estimated runtime: ~20 minutes on e2-standard-4 (GEE server-side, both years).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.era5_harvester import ERA5Harvester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

BBOX  = (-71.0, 41.0, -66.0, 45.0)
YEARS = (2022, 2023)


def main() -> None:
    bucket = os.environ.get("MHW_GCS_BUCKET", "").rstrip("/")
    if not bucket:
        raise RuntimeError("MHW_GCS_BUCKET env var not set.")

    era5 = ERA5Harvester(
        service_account_key=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    )

    for step, year in enumerate(YEARS, start=1):
        gcs_uri = f"{bucket}/era5/{year}/"
        print(f"[{step}/2] ERA5 {year} -> {gcs_uri}", flush=True)
        era5.fetch_and_cache(year, BBOX, gcs_uri)
        print(f"[{step}/2] ERA5 {year}  OK", flush=True)

    print("ERA5 data prep complete.", flush=True)


if __name__ == "__main__":
    main()
