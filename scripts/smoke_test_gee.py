"""
smoke_test_gee.py — WeatherNext 2 + HYCOM Connectivity Smoke Test
=================================================================
Verifies the full GEE-to-GCS-Zarr ingestion path for a minimal region.

Purpose
-------
This script is the verification gate for todo step 2 (Ensemble Connectivity
Test). It runs each component of the ingestion engine independently so that
failures can be attributed to a specific stage:

  Stage 1: GEE auth + WeatherNext 2 asset inspection (metadata only, no data)
  Stage 2: WeatherNext 2 pixel fetch → GCS Zarr write → lazy re-open
  Stage 3: HYCOM OPeNDAP connectivity → fetch tile → print T/S profile

Evidence required for DONE status:
  - Stage 1: collection size printed, band names printed
  - Stage 2: GCS URI printed, xr.Dataset repr printed with correct dims
  - Stage 3: HYCOM Dataset repr printed; T/S values at TARGET_DEPTHS_M visible

Usage
-----
    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
    conda run -n mhw-risk python scripts/smoke_test_gee.py

Parameters (edit the SMOKE_* constants below before running)
-------------------------------------------------------------
SMOKE_BBOX      : 1°×1° Gulf of Maine patch — well-documented MHW region
SMOKE_START     : Short 3-day window — minimises GEE compute cost
SMOKE_N_MEMBERS : 2 — enough to prove the member dimension; avoids 64× cost
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Smoke test parameters — adjust if needed
# ---------------------------------------------------------------------------
SMOKE_BBOX = (-70.5, 43.0, -69.5, 44.0)   # Gulf of Maine, 1°×1°
# HYCOM expt_93.0/ts3z covers 2018-01-01 to 2020-02-19; dates must be within this range.
# WeatherNext 2 dates to be confirmed after data access form is approved.
SMOKE_START = "2019-08-01"
SMOKE_END = "2019-08-03"
SMOKE_N_MEMBERS = 2
GCS_BUCKET = "mhw-risk-cache"
KEY_PATH = os.path.expanduser("~/.config/gcp-keys/mhw-harvester.json")

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

DIVIDER = "=" * 70


def stage1_inspect_wn2_asset() -> None:
    """
    Stage 1: Authenticate with GEE and print WeatherNext 2 collection metadata.

    Physical purpose: confirms that the GEE asset is accessible under the
    service account and that the collection has images for the smoke-test
    date range. Prints band names so that the member-band naming pattern
    can be verified before any pixel data is fetched.
    """
    import ee
    import json

    print(f"\n{DIVIDER}")
    print("STAGE 1 — GEE Auth + WeatherNext 2 Asset Inspection")
    print(DIVIDER)

    with open(KEY_PATH) as fh:
        key_data = json.load(fh)
    email = key_data["client_email"]
    credentials = ee.ServiceAccountCredentials(email=email, key_file=KEY_PATH)
    ee.Initialize(credentials=credentials)
    logger.info("GEE authenticated as: %s", email)

    WN2_ASSET = "projects/gcp-public-data-weathernext/assets/weathernext_2_0_0"
    collection = (
        ee.ImageCollection(WN2_ASSET)
        .filter(ee.Filter.date(SMOKE_START, SMOKE_END))
    )

    size = collection.size().getInfo()
    print(f"  Collection size for {SMOKE_START} to {SMOKE_END}: {size} images")

    if size == 0:
        print("  WARNING: 0 images returned — date range may be outside dataset coverage.")
        print("  WeatherNext 2 forecast horizon is typically 0–10 days ahead;")
        print("  historical reanalysis coverage depends on the specific asset version.")
        return

    first_img = ee.Image(collection.first())
    band_names = first_img.bandNames().getInfo()
    img_date = first_img.date().format("YYYY-MM-dd").getInfo()
    img_props = first_img.toDictionary().getInfo()

    print(f"  First image date    : {img_date}")
    print(f"  Number of bands     : {len(band_names)}")
    print(f"  First 20 band names : {band_names[:20]}")
    print(f"  Image properties    : {img_props}")
    print(f"\n  STAGE 1 PASSED\n")


def stage2_fetch_wn2_to_gcs_zarr() -> None:
    """
    Stage 2: Fetch WeatherNext 2 pixels for the smoke bbox and write to GCS Zarr.

    Physical purpose: end-to-end test of the GEE-to-GCS-Zarr ingestion path.
    Uses WeatherNext2Harvester which now uses the synchronous sampleRectangle
    compute path for small regions (avoids 2–20 min batch export queue).

    Expected evidence: GCS URI printed + xr.Dataset repr with dims
    (member=2, time=3, latitude, longitude).
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))
    from ingestion.harvester import WeatherNext2Harvester

    print(f"\n{DIVIDER}")
    print("STAGE 2 — WeatherNext 2 Fetch → GCS Zarr Write → Lazy Re-open")
    print(DIVIDER)

    wn2 = WeatherNext2Harvester(
        gcs_bucket=GCS_BUCKET,
        gcs_prefix="weathernext2/cache",
        service_account_key=KEY_PATH,
    )
    wn2.authenticate()

    ds = wn2.fetch_ensemble(
        start_date=SMOKE_START,
        end_date=SMOKE_END,
        bbox=SMOKE_BBOX,
        n_members=SMOKE_N_MEMBERS,
    )

    print(f"\n  GCS Zarr URI: gs://{GCS_BUCKET}/weathernext2/cache/"
          f"wn2_{SMOKE_START}_{SMOKE_END}_m{SMOKE_N_MEMBERS}.zarr")
    print(f"\n  Dataset repr:\n{ds}")
    print(f"\n  Dims   : {dict(ds.dims)}")
    print(f"  Chunks : {ds.chunks}")
    print(f"\n  STAGE 2 PASSED\n")


def stage3_fetch_hycom() -> None:
    """
    Stage 3: Open HYCOM GLBv0.08 via OPeNDAP and fetch a 3-day tile.

    Physical purpose: confirms that the HYCOM THREDDS server is reachable,
    that the variable names and coordinate names in the code match the
    actual dataset, and that interpolation to TARGET_DEPTHS_M works.

    Expected evidence: Dataset repr + T/S profile at a single grid point
    showing temperature decrease with depth (thermocline visible between
    ~50–200 m in the Gulf of Maine in August).
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))
    from ingestion.harvester import HYCOMLoader

    print(f"\n{DIVIDER}")
    print("STAGE 3 — HYCOM OPeNDAP Connectivity + Vertical Profile")
    print(DIVIDER)

    loader = HYCOMLoader()
    ds = loader.fetch_tile(
        start_date=SMOKE_START,
        end_date=SMOKE_END,
        bbox=SMOKE_BBOX,
    )

    print(f"\n  HYCOM tile Dataset repr:\n{ds}")

    # Print T/S vertical profile at the centre of the bbox (single point, first day)
    mid_lat = (SMOKE_BBOX[1] + SMOKE_BBOX[3]) / 2
    mid_lon = (SMOKE_BBOX[0] + SMOKE_BBOX[2]) / 2
    profile = ds[["water_temp", "salinity"]].sel(
        lat=mid_lat, lon=mid_lon, time=SMOKE_START, method="nearest"
    )

    print(f"\n  T/S vertical profile at ({mid_lat}°N, {mid_lon}°E) on {SMOKE_START}:")
    print(f"  {'Depth (m)':>10}  {'Temp (°C)':>12}  {'Salinity (psu)':>15}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*15}")
    for d in profile.depth.values:
        t_val = float(profile["water_temp"].sel(depth=d).values)
        s_val = float(profile["salinity"].sel(depth=d).values)
        print(f"  {d:>10.0f}  {t_val:>12.3f}  {s_val:>15.3f}")

    print(f"\n  Expected: temperature drops sharply between ~50–200 m (thermocline).")
    print(f"\n  STAGE 3 PASSED\n")


if __name__ == "__main__":
    print(f"\n{'#' * 70}")
    print("  MHW Risk Profiler — Ensemble Connectivity Smoke Test")
    print(f"  Bbox    : {SMOKE_BBOX}")
    print(f"  Dates   : {SMOKE_START} → {SMOKE_END}")
    print(f"  Members : {SMOKE_N_MEMBERS}")
    print(f"{'#' * 70}")

    try:
        stage1_inspect_wn2_asset()
    except Exception as exc:
        logger.error("STAGE 1 FAILED: %s", exc, exc_info=True)
        sys.exit(1)

    try:
        stage2_fetch_wn2_to_gcs_zarr()
    except Exception as exc:
        logger.error("STAGE 2 FAILED: %s", exc, exc_info=True)
        sys.exit(1)

    try:
        stage3_fetch_hycom()
    except Exception as exc:
        logger.error("STAGE 3 FAILED: %s", exc, exc_info=True)
        sys.exit(1)

    print(f"\n{'#' * 70}")
    print("  ALL STAGES PASSED — Ensemble Connectivity Smoke Test COMPLETE")
    print(f"{'#' * 70}\n")
