"""
fetch_oisst_climatology.py — Fetch NOAA OISST v2.1, compute 30-year baseline, write to GCS.
==========================================================================
Fetches daily GoM SST for 1982-2011 from NOAA CoastWatch ERDDAP griddap
(server-side spatial+temporal subset), computes 90th-percentile threshold
per (dayofyear, lat, lon) with an 11-day centered rolling window (Hobday 2016),
and writes to GCS as sst_threshold_90, replacing the scientifically invalid
2-year HYCOM baseline.

ERDDAP approach: 30 annual requests (~500 KB each) vs 10,800 global-file
downloads (18 GB total). Avoids NCEI rate limiting and lon-range mismatch
(OISST native 0-360 lon vs our GoM bbox in -180 to 180 convention).

Usage (on VM):
    nohup env \\
      GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \\
      MHW_GCS_BUCKET=gs://mhw-risk-cache \\
      /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/fetch_oisst_climatology.py \\
      >> ~/nohup_oisst.log 2>&1 </dev/null & disown $!

ERDDAP reachability check:
    curl -s -o /dev/null -w "%{http_code}" \\
      "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.nc?sst[(1982-01-01T12:00:00Z):(1982-01-01T12:00:00Z)][(0.0)][(41.0):(45.0)][(-71.0):(-66.0)]"
    Expected: 200.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile

import gcsfs
import requests
import xarray as xr

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.analytics.mhw_detection import compute_climatology
from src.ingestion.harvester import _gcs_safe_write, _gcs_complete

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIM_START_YEAR = 1982
CLIM_END_YEAR   = 2011

# GoM bounding box — matches GoM_BBOX in _train_utils.py
GoM_LON_MIN = -71.0
GoM_LON_MAX = -66.0
GoM_LAT_MIN =  41.0
GoM_LAT_MAX =  45.0

# ERDDAP griddap dataset with -180/180 lon convention (matches our GoM bbox)
ERDDAP_BASE = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
    "/ncdcOisst21Agg_LonPM180.nc"
)


# ---------------------------------------------------------------------------
# Fetch — one request per year via ERDDAP griddap
# ---------------------------------------------------------------------------

def fetch_oisst_gom_year(year: int) -> xr.DataArray:
    """
    Fetch one calendar year of OISST v2.1 SST for the GoM bbox via ERDDAP.

    Uses NOAA CoastWatch ERDDAP griddap server-side subsetting:
    - Temporal range: Jan 1 – Dec 31 of *year* (daily, noon UTC)
    - Spatial range: GoM bbox (lat 41–45°N, lon -71–-66°E)
    - Vertical: surface only (zlev=0)

    This fetches ~500 KB per year vs 1.7 MB × 365 = 620 MB with per-day files.
    ERDDAP lon convention is -180/180 so our negative-lon GoM bbox works directly.

    Parameters
    ----------
    year : int
        Calendar year (1982–2011).

    Returns
    -------
    xr.DataArray
        SST [°C], dims (time, lat, lon), subset to GoM bbox.
        NaN fill values from the source file are preserved.
    """
    url = (
        f"{ERDDAP_BASE}?sst"
        f"[({year}-01-01T12:00:00Z):({year}-12-31T12:00:00Z)]"
        f"[(0.0)]"
        f"[({GoM_LAT_MIN}):({GoM_LAT_MAX})]"
        f"[({GoM_LON_MIN}):({GoM_LON_MAX})]"
    )
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    ct = resp.headers.get("Content-Type", "")
    if "netcdf" not in ct and "octet-stream" not in ct:
        raise RuntimeError(
            f"ERDDAP returned unexpected Content-Type '{ct}' for {year} — "
            "likely an error page; check ERDDAP dataset ID or time range."
        )

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(resp.content)
        tmppath = tmp.name
    try:
        ds = xr.open_dataset(tmppath, engine="netcdf4")
        return ds["sst"].squeeze("zlev", drop=True).compute()
    finally:
        os.unlink(tmppath)


# ---------------------------------------------------------------------------
# Climatology + GCS write
# ---------------------------------------------------------------------------

def compute_and_write_climatology(sst_all: xr.DataArray, clim_uri: str) -> None:
    """
    Compute 90th-pct threshold with 11-day rolling window and write to GCS.

    Skips GCS write if the .complete sentinel already exists.

    Parameters
    ----------
    sst_all : xr.DataArray
        Concatenated daily SST [°C], dims (time, lat, lon),
        covering the full 1982–2011 climatology period.
    clim_uri : str
        GCS URI for the climatology Zarr store
        (e.g. gs://mhw-risk-cache/hycom/climatology/).
        Writes variable sst_threshold_90; overwrites any existing store.
    """
    fs = gcsfs.GCSFileSystem()

    if _gcs_complete(fs, clim_uri):
        logger.info("Cache hit — skipping climatology: %s", clim_uri)
        return

    logger.info("Computing 30-year OISST climatology (window=11 days)...")
    threshold = compute_climatology(sst_all, percentile=90.0, window=11)

    ds_threshold = threshold.to_dataset(name="sst_threshold_90")
    ds_threshold.attrs.update({
        "source":               "NOAA OISST v2.1 (AVHRR-only)",
        "period":               f"{CLIM_START_YEAR}–{CLIM_END_YEAR}",
        "percentile":           90.0,
        "rolling_window_days":  11,
        "citation":             "Hobday et al. (2016), Prog. Oceanogr.",
        "units":                "degree_C",
    })

    _gcs_safe_write(ds_threshold, clim_uri)
    logger.info("Climatology written to %s", clim_uri)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    bucket = os.environ.get("MHW_GCS_BUCKET", "").rstrip("/")
    if not bucket:
        raise RuntimeError("MHW_GCS_BUCKET env var not set.")

    clim_uri = f"{bucket}/hycom/climatology/"

    fs = gcsfs.GCSFileSystem()
    if _gcs_complete(fs, clim_uri):
        logger.info("OISST climatology already complete at %s — nothing to do.", clim_uri)
        return

    years = list(range(CLIM_START_YEAR, CLIM_END_YEAR + 1))
    logger.info(
        "Fetching OISST v2.1 %d–%d for GoM bbox via ERDDAP — %d annual requests...",
        CLIM_START_YEAR, CLIM_END_YEAR, len(years),
    )

    annual_slabs: list[xr.DataArray] = []
    for year in years:
        try:
            da = fetch_oisst_gom_year(year)
            annual_slabs.append(da)
            logger.info("Fetched %d (%d/%d)", year, len(annual_slabs), len(years))
        except Exception as exc:
            logger.error("Failed %d: %s — skipping", year, exc)

    if not annual_slabs:
        raise RuntimeError("No OISST data fetched — check ERDDAP URL and network access.")

    logger.info("Concatenating %d annual slabs...", len(annual_slabs))
    sst_all = xr.concat(annual_slabs, dim="time").sortby("time")

    compute_and_write_climatology(sst_all, clim_uri)
    logger.info("Done. Climatology at %s", clim_uri)


if __name__ == "__main__":
    main()
