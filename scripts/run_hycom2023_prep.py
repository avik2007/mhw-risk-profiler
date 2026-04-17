#!/usr/bin/env python3
"""
run_hycom2023_prep.py — Pre-fetch HYCOM 2023 tile to GCS in parallel with run_data_prep.py.

Idempotent: skips any month whose .zmetadata already exists in GCS.
Safe to restart after preemption — only incomplete months re-execute.

Required environment variables:
    GOOGLE_APPLICATION_CREDENTIALS  — path to GCP service account JSON key
    MHW_GCS_BUCKET                  — GCS bucket URI, e.g. "gs://my-bucket"

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
    export MHW_GCS_BUCKET=gs://your-bucket-name
    python scripts/run_hycom2023_prep.py

Expected output:
    HYCOM 2023 -> gs://bucket/hycom/tiles/2023/  OK
    HYCOM 2023 complete.

Estimated runtime: ~5-7 hours on e2-standard-4 (OPeNDAP, 12 months x ~30 min/month).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.harvester import HYCOMLoader

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

    gcs_uri = f"{bucket}/hycom/tiles/2023/"
    print(f"HYCOM 2023 -> {gcs_uri}", flush=True)
    loader = HYCOMLoader()
    loader.fetch_and_cache(2023, BBOX, gcs_uri)
    print("HYCOM 2023 -> OK", flush=True)
    print("HYCOM 2023 complete.", flush=True)


if __name__ == "__main__":
    main()
