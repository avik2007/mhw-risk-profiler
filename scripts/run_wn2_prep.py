#!/usr/bin/env python3
"""
run_wn2_prep.py — Pre-fetch WeatherNext 2 training data to GCS.

Run on a dedicated spot-free VM in parallel with run_data_prep.py.
Fetches WN2 ensemble data for 2022 and 2023 via GEE sampleRectangle compute
path and writes consolidated Zarr stores to GCS.

Idempotent: checks GCS for .zmetadata before fetching. Safe to re-run after
preemption — only incomplete years will re-execute.

Required environment variables:
    GOOGLE_APPLICATION_CREDENTIALS  — path to GCP service account JSON key
    MHW_GCS_BUCKET                  — GCS bucket URI, e.g. "gs://my-bucket"

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
    export MHW_GCS_BUCKET=gs://your-bucket-name
    python scripts/run_wn2_prep.py [--year 2022]   # omit to run both years

Expected output:
    [1/2] WN2 tiles 2022 -> gs://bucket/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr  OK
    [2/2] WN2 tiles 2023 -> gs://bucket/weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr  OK
    WN2 data prep complete.

Estimated runtime: ~55 min/year on e2-standard-4 (GEE sampleRectangle, 64 members/day).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.harvester import WeatherNext2Harvester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

BBOX  = (-71.0, 41.0, -66.0, 45.0)
YEARS = (2022, 2023)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None,
                        help="Single year to fetch (2022 or 2023). Omit to run both.")
    args = parser.parse_args()

    bucket = os.environ.get("MHW_GCS_BUCKET", "").rstrip("/")
    if not bucket:
        raise RuntimeError("MHW_GCS_BUCKET env var not set.")

    wn2 = WeatherNext2Harvester(
        gcs_bucket=bucket.removeprefix("gs://"),
        gcs_prefix="weathernext2/cache",
        service_account_key=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    )

    years = (args.year,) if args.year else YEARS
    for step, year in enumerate(years, start=1):
        gcs_uri = f"{bucket}/weathernext2/cache/wn2_{year}-01-01_{year}-12-31_m64.zarr"
        total = len(years)
        print(f"[{step}/{total}] WN2 tiles {year} -> {gcs_uri}", flush=True)
        wn2.fetch_and_cache(year, BBOX, gcs_uri)
        print(f"[{step}/{total}] WN2 tiles {year}  OK", flush=True)

    print("WN2 data prep complete.", flush=True)


if __name__ == "__main__":
    main()
