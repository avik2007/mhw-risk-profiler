"""
fetch_oisst_climatology.py — Fetch NOAA OISST v2.1, compute 30-year baseline, write to GCS.
==========================================================================
Fetches daily GoM SST for 1982-2011 from NOAA NCEI THREDDS OPeNDAP,
computes 90th-percentile threshold per (dayofyear, lat, lon) with an 11-day
centered rolling window (Hobday 2016), and writes to GCS as sst_threshold_90,
replacing the scientifically invalid 2-year HYCOM baseline.

Usage (on VM):
    nohup env \\
      GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \\
      MHW_GCS_BUCKET=gs://mhw-risk-cache \\
      /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/fetch_oisst_climatology.py \\
      >> ~/nohup_oisst.log 2>&1 </dev/null & disown $!

THREDDS URL note:
    Verify reachability before VM run:
        curl -s -o /dev/null -w "%{http_code}" \\
          "https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/V2.1/AVHRR/198201/oisst-avhrr-v02r01.19820101.nc"
    Expected: 200. If unavailable, check https://www.ncei.noaa.gov/thredds/catalog.html for
    current OISST v2.1 paths and update OISST_THREDDS_BASE below.
"""
from __future__ import annotations

import calendar
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gcsfs
import numpy as np
import requests
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
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

OISST_BASE_URL = (
    "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation"
    "/v2.1/access/avhrr"
)


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

def build_oisst_url(year: int, month: int, day: int) -> str:
    """
    Build NCEI THREDDS OPeNDAP URL for a single OISST v2.1 daily file.

    Parameters
    ----------
    year, month, day : int
        Calendar date.

    Returns
    -------
    str
        OPeNDAP URL for the daily netCDF4 file.
        Format: .../YYYYMM/oisst-avhrr-v02r01.YYYYMMDD.nc
    """
    ym  = f"{year}{month:02d}"
    ymd = f"{year}{month:02d}{day:02d}"
    return f"{OISST_BASE_URL}/{ym}/oisst-avhrr-v02r01.{ymd}.nc"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_oisst_gom(year: int, month: int) -> xr.DataArray:
    """
    Fetch one calendar month of OISST v2.1 SST for the GoM bbox via HTTPS.

    Downloads each daily NetCDF4 file via requests (plain HTTPS, not OPeNDAP),
    opens from BytesIO to avoid netcdf4 engine's OPeNDAP path, subsets to GoM,
    and concatenates into a (time, lat, lon) DataArray.

    Parameters
    ----------
    year : int
        Calendar year (1982–2011).
    month : int
        Calendar month (1–12).

    Returns
    -------
    xr.DataArray
        SST [°C], dims (time, lat, lon), subset to GoM bbox.
        NaN fill values from the source file are preserved.
    """
    n_days = calendar.monthrange(year, month)[1]
    slabs = []
    for day in range(1, n_days + 1):
        url = build_oisst_url(year, month, day)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        # netcdf4 engine rejects file-like objects; write to named temp file instead
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(resp.content)
            tmppath = tmp.name
        try:
            day_ds = xr.open_dataset(
                tmppath,
                engine="netcdf4",
                drop_variables=["ice", "anom", "err"],
            )
            sst = (
                day_ds["sst"]
                .sel(lat=slice(GoM_LAT_MIN, GoM_LAT_MAX), lon=slice(GoM_LON_MIN, GoM_LON_MAX))
                .squeeze("zlev", drop=True)
                .compute()
            )
        finally:
            os.unlink(tmppath)
        slabs.append(sst)
    return xr.concat(slabs, dim="time")


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

    all_months = [
        (y, m)
        for y in range(CLIM_START_YEAR, CLIM_END_YEAR + 1)
        for m in range(1, 13)
    ]
    logger.info(
        "Fetching OISST v2.1 %d–%d for GoM bbox — %d months, 6 parallel workers...",
        CLIM_START_YEAR, CLIM_END_YEAR, len(all_months),
    )

    # Fetch months in parallel; results keyed by (year, month) to preserve order.
    # max_workers=6: empirically safe for NCEI THREDDS without triggering rate limits.
    results: dict[tuple[int, int], xr.DataArray] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        future_to_ym = {
            pool.submit(fetch_oisst_gom, y, m): (y, m)
            for y, m in all_months
        }
        for future in as_completed(future_to_ym):
            ym = future_to_ym[future]
            try:
                results[ym] = future.result()
                logger.info("Fetched %d-%02d (%d/%d done)", ym[0], ym[1], len(results), len(all_months))
            except Exception as exc:
                logger.error("Failed %d-%02d: %s — skipping", ym[0], ym[1], exc)

    monthly_slabs = [results[ym] for ym in all_months if ym in results]

    if not monthly_slabs:
        raise RuntimeError("No OISST data fetched — check THREDDS URL and network access.")

    logger.info("Concatenating %d monthly slabs...", len(monthly_slabs))
    sst_all = xr.concat(monthly_slabs, dim="time").sortby("time")

    compute_and_write_climatology(sst_all, clim_uri)
    logger.info("Done. Climatology at %s", clim_uri)


if __name__ == "__main__":
    main()
