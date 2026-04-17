#!/usr/bin/env python3
"""
wn2_smoke_test.py — 3-day WeatherNext 2 end-to-end smoke test.

Fetches Jan 1-3 2022 over the Gulf of Maine bbox, prints output shape,
and writes to a throwaway GCS path to verify the full pipeline:
  GEE auth → fetch_ensemble → _build_dataset → _gcs_safe_write → open_zarr verify

Expected output:
    Dataset shape: (member=64, time=3, latitude=17, longitude=21)
    Variables: 2m_temperature, 10m_u_component_of_wind, ...
    Sample sst[0,0,0,0]: <float>
    Written to gs://mhw-risk-cache/wn2_smoke_test/
    Read back OK: dims match.

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
    export MHW_GCS_BUCKET=gs://mhw-risk-cache
    python scripts/wn2_smoke_test.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import xarray as xr

from src.ingestion.harvester import WeatherNext2Harvester, _gcs_safe_write

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BBOX       = (-71.0, 41.0, -66.0, 45.0)
START_DATE = "2022-01-01"
END_DATE   = "2022-01-04"   # exclusive → fetches Jan 1, 2, 3
SMOKE_URI  = "gs://mhw-risk-cache/wn2_smoke_test/"


def main() -> None:
    key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    bucket = os.environ.get("MHW_GCS_BUCKET", "").rstrip("/")
    if not bucket:
        raise RuntimeError("MHW_GCS_BUCKET env var not set.")

    wn2 = WeatherNext2Harvester(
        gcs_bucket=bucket.removeprefix("gs://"),
        gcs_prefix="wn2_smoke_test",
        service_account_key=key,
    )
    wn2.authenticate()

    print(f"Fetching WN2 {START_DATE} to {END_DATE}, bbox={BBOX} ...", flush=True)
    ds = wn2.fetch_ensemble(START_DATE, END_DATE, BBOX)

    print(f"\n--- Dataset ---")
    print(ds)
    print(f"\nDimensions: { {k: v for k, v in ds.dims.items()} }")
    print(f"Variables:  {list(ds.data_vars)}")

    sst = ds.get("sea_surface_temperature")
    if sst is not None:
        print(f"Sample sst[member=0, time=0, lat=0, lon=0]: {float(sst.isel(member=0, time=0, latitude=0, longitude=0).values):.4f} K")

    print(f"\nWriting to {SMOKE_URI} ...", flush=True)
    _gcs_safe_write(ds, SMOKE_URI)
    print("Write OK.", flush=True)

    print(f"Reading back from {SMOKE_URI} ...", flush=True)
    ds2 = xr.open_zarr(SMOKE_URI, chunks="auto")
    assert dict(ds2.dims) == dict(ds.dims), f"Dim mismatch: {ds2.dims} vs {ds.dims}"
    print("Read-back OK — dims match.")
    print("\nSmoke test PASSED.")


if __name__ == "__main__":
    main()
