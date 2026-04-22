#!/usr/bin/env python3
"""
run_hycom_months_prep.py — Fetch a contiguous range of HYCOM monthly tiles to GCS.

Designed to run in parallel with other instances covering different month ranges.
Writes only the per-month intermediate tiles; the annual assembly is left to
fetch_and_cache() running on the main VM once all months are cached.

Idempotent: skips any month whose .zmetadata already exists in GCS.

Required environment variables:
    GOOGLE_APPLICATION_CREDENTIALS  — path to GCP service account JSON key
    MHW_GCS_BUCKET                  — GCS bucket URI, e.g. "gs://my-bucket"
    HYCOM_YEAR                      — calendar year, e.g. "2022"
    HYCOM_START_MONTH               — first month to fetch, e.g. "7"
    HYCOM_END_MONTH                 — last month to fetch (inclusive), e.g. "9"

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
    export MHW_GCS_BUCKET=gs://your-bucket-name
    export HYCOM_YEAR=2022
    export HYCOM_START_MONTH=7
    export HYCOM_END_MONTH=9
    python scripts/run_hycom_months_prep.py

Estimated runtime: ~2.5 hours per month on e2-standard-4 (OPeNDAP).
"""
from __future__ import annotations

import calendar
import logging
import os
import sys
from pathlib import Path

import gcsfs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.harvester import HYCOMLoader, _gcs_complete, _gcs_safe_write

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

BBOX = (-71.0, 41.0, -66.0, 45.0)


def main() -> None:
    bucket = os.environ.get("MHW_GCS_BUCKET", "").rstrip("/")
    if not bucket:
        raise RuntimeError("MHW_GCS_BUCKET env var not set.")
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS env var not set.")

    year = int(os.environ.get("HYCOM_YEAR", ""))
    start_month = int(os.environ.get("HYCOM_START_MONTH", ""))
    end_month = int(os.environ.get("HYCOM_END_MONTH", ""))

    if not (1 <= start_month <= end_month <= 12):
        raise RuntimeError(
            f"Invalid month range: HYCOM_START_MONTH={start_month}, "
            f"HYCOM_END_MONTH={end_month}. Must satisfy 1 <= start <= end <= 12."
        )

    # IMPORTANT: month ranges across parallel VMs must not overlap.
    # Concurrent writes to the same month_uri via _gcs_safe_write (rm + to_zarr)
    # can interleave, producing a corrupt Zarr store that passes the .zmetadata check.
    annual_base = f"{bucket}/hycom/tiles/{year}"
    logger.info(
        "HYCOM %d months %d-%d -> %s",
        year, start_month, end_month, annual_base,
    )

    loader = HYCOMLoader()
    fs = gcsfs.GCSFileSystem()
    n_months = end_month - start_month + 1

    for step, month in enumerate(range(start_month, end_month + 1), start=1):
        last_day = calendar.monthrange(year, month)[1]
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{year}-{month:02d}-{last_day:02d}"
        month_uri = f"{annual_base}/monthly/m{month:02d}/"

        print(f"[{step}/{n_months}] HYCOM {year}-{month:02d} -> {month_uri}", flush=True)

        if _gcs_complete(fs, month_uri):
            logger.info("Cache hit — skipping HYCOM %d-%02d", year, month)
            print(f"[{step}/{n_months}] HYCOM {year}-{month:02d}  CACHED", flush=True)
            continue

        logger.info(
            "Fetching HYCOM %d-%02d (%s to %s)...", year, month, start_date, end_date
        )
        ds_month = loader.fetch_tile(start_date, end_date, BBOX)
        _gcs_safe_write(ds_month, month_uri)
        logger.info("Written: HYCOM %d-%02d -> %s", year, month, month_uri)
        print(f"[{step}/{n_months}] HYCOM {year}-{month:02d}  OK", flush=True)

    print(
        f"HYCOM {year} months {start_month}-{end_month} complete.",
        flush=True,
    )


if __name__ == "__main__":
    main()
