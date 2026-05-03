# OISST 30-Year Climatology + WN2 SST Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scientifically invalid 2-year HYCOM climatology baseline with a 30-year NOAA OISST v2.1 baseline (Hobday 2016 compliant), while in parallel fixing the WN2 SST bug so `train_wn2.py` no longer crashes.

**Architecture:** Two independent tracks (A = OISST climatology, B = WN2 SST fix) that can execute in parallel. Track A touches `mhw_detection.py` and a new `fetch_oisst_climatology.py` script. Track B touches only `harvester.py`. Both converge at the VM re-run step. ERA5 and WN2 must both be retrained after Track A completes (new threshold changes all SDD labels).

**Tech Stack:** xarray, numpy, scipy (uniform_filter1d), gcsfs, netCDF4 / OPeNDAP, PyTorch (train_wn2.py)

---

## File Map

| File | Action | Reason |
|------|--------|--------|
| `src/analytics/mhw_detection.py` | Modify `compute_climatology()` | Add 11-day rolling window smoothing |
| `scripts/fetch_oisst_climatology.py` | Create | Fetch OISST 1982-2011, compute threshold, write to GCS |
| `src/ingestion/harvester.py` | Modify `WN2_VARIABLES` + `_build_dataset()` | Add SST back; mask land pixels (0 K → NaN) |
| `tests/test_mhw_detection.py` | Modify | Test rolling window behaviour |
| `tests/test_oisst_climatology.py` | Create | Test OISST fetch + write logic |
| `tests/test_wn2_harvester.py` | Modify | Test NaN masking of SST=0 pixels |

---

## TRACK A — OISST 30-Year Climatology

### Task A1: Add 11-day rolling window to `compute_climatology()`

**Files:**
- Modify: `src/analytics/mhw_detection.py:33-63`
- Modify: `tests/test_mhw_detection.py`

Current `compute_climatology()` (lines 33-63) computes raw groupby quantile with no smoothing. Hobday (2016) requires an 11-day centered rolling window to prevent aliasing of noisy daily percentiles.

- [ ] **Step A1.1: Write failing test**

Add to `tests/test_mhw_detection.py`:

```python
def test_compute_climatology_rolling_window():
    """Smoothed threshold must differ from raw quantile at day boundaries."""
    import numpy as np
    import xarray as xr
    from src.analytics.mhw_detection import compute_climatology

    rng = np.random.default_rng(0)
    # 5-year daily SST (lat=3, lon=4), spike on Jan 1 only
    times = xr.cftime_range("2000-01-01", periods=365 * 5, freq="D")
    sst = xr.DataArray(
        rng.normal(20.0, 1.0, (365 * 5, 3, 4)).astype("float32"),
        dims=["time", "lat", "lon"],
        coords={"time": times},
    )
    # Inject extreme spike on every Jan 1 to create a sharp edge in raw quantile
    sst.values[::365] = 999.0

    raw = compute_climatology(sst, percentile=90.0, window=1)   # window=1 → no smoothing
    smooth = compute_climatology(sst, percentile=90.0, window=11)

    # Smoothed threshold should be lower on day 1 (spike spread out) and higher on day 6
    assert float(smooth.sel(dayofyear=1).mean()) < float(raw.sel(dayofyear=1).mean())
    assert float(smooth.sel(dayofyear=6).mean()) > float(raw.sel(dayofyear=6).mean())

def test_compute_climatology_window_wraps():
    """Rolling window must wrap at day 365/day 1 boundary (no edge NaN)."""
    import numpy as np
    import xarray as xr
    from src.analytics.mhw_detection import compute_climatology

    rng = np.random.default_rng(1)
    times = xr.cftime_range("2000-01-01", periods=365 * 5, freq="D")
    sst = xr.DataArray(
        rng.normal(20.0, 1.0, (365 * 5, 2, 2)).astype("float32"),
        dims=["time", "lat", "lon"],
        coords={"time": times},
    )
    threshold = compute_climatology(sst, percentile=90.0, window=11)
    assert not bool(np.isnan(threshold.values).any()), "No NaN allowed after wrap-around rolling"
```

- [ ] **Step A1.2: Run test — expect FAIL**

```bash
/home/avik2007/miniconda3/envs/mhw-risk/bin/python -m pytest tests/test_mhw_detection.py::test_compute_climatology_rolling_window tests/test_mhw_detection.py::test_compute_climatology_window_wraps -v
```

Expected: `TypeError` (unexpected keyword `window`) or `AssertionError`.

- [ ] **Step A1.3: Implement rolling window in `compute_climatology()`**

Replace `src/analytics/mhw_detection.py` lines 33-63:

```python
def compute_climatology(
    sst_historical: xr.DataArray,
    percentile: float = 90.0,
    window: int = 11,
) -> xr.DataArray:
    """
    Compute the climatological SST percentile threshold for each calendar day.

    Groups historical SST by day-of-year, computes the requested percentile,
    then applies a centered rolling window to smooth the daily threshold curve.
    The rolling window is applied with wrap-around (day 365 neighbours day 1)
    to avoid edge NaN at the start/end of the year.

    Parameters
    ----------
    sst_historical : xr.DataArray
        Historical daily SST [deg C], dims (time, lat, lon).
        Should cover ≥ 30 years for Hobday (2016) compliance.
    percentile : float
        Climatological exceedance level. 90.0 = Hobday Category I MHW.
    window : int
        Centered rolling window width [days] applied to the dayofyear axis.
        11 is the Hobday (2016) standard. Set to 1 to disable smoothing.

    Returns
    -------
    threshold : xr.DataArray
        Smoothed percentile SST [deg C] for each calendar day.
        Dims: (dayofyear=365, lat, lon).
        Index dayofyear runs 1–365; day 366 (leap) omitted by groupby convention.
    """
    from scipy.ndimage import uniform_filter1d

    grouped = sst_historical.groupby("time.dayofyear")
    raw = grouped.quantile(percentile / 100.0, dim="time")  # (dayofyear=365, lat, lon)

    if window <= 1:
        return raw

    # Apply centered rolling mean along dayofyear axis with circular wrap
    # scipy uniform_filter1d mode='wrap' treats axis as circular (365 → 1)
    smoothed_vals = uniform_filter1d(raw.values, size=window, axis=0, mode="wrap")

    return xr.DataArray(
        smoothed_vals,
        dims=raw.dims,
        coords=raw.coords,
        attrs={**raw.attrs, "rolling_window_days": window},
    )
```

- [ ] **Step A1.4: Run tests — expect PASS**

```bash
/home/avik2007/miniconda3/envs/mhw-risk/bin/python -m pytest tests/test_mhw_detection.py -v
```

Expected: all tests in file pass.

- [ ] **Step A1.5: Run full suite — no regressions**

```bash
/home/avik2007/miniconda3/envs/mhw-risk/bin/python -m pytest tests/ -v
```

Expected: all 68 tests pass.

- [ ] **Step A1.6: Commit**

```bash
git add src/analytics/mhw_detection.py tests/test_mhw_detection.py
git commit -m "feat: add 11-day rolling window to compute_climatology() per Hobday (2016)"
```

---

### Task A2: Write `scripts/fetch_oisst_climatology.py`

**Files:**
- Create: `scripts/fetch_oisst_climatology.py`
- Create: `tests/test_oisst_climatology.py`

NOAA OISST v2.1 is available via NCEI THREDDS OPeNDAP. Files follow the URL pattern:
`https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/V2.1/AVHRR/{year}{month:02d}/oisst-avhrr-v02r01.{year}{month:02d}{day:02d}.nc`

Each file contains `sst` [°C] with dims `(time=1, zlev=1, lat, lon)` at 0.25° global resolution.
We subset to GoM bbox and open month-by-month via `xr.open_mfdataset` to batch OPeNDAP calls.

**Note:** Verify the THREDDS URL is reachable before VM run:
```bash
curl -I "https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/V2.1/AVHRR/198201/oisst-avhrr-v02r01.19820101.nc"
```
If THREDDS returns 404 or times out, fall back to ERDDAP (see comment in script).

- [ ] **Step A2.1: Write failing test**

Create `tests/test_oisst_climatology.py`:

```python
"""
Tests for fetch_oisst_climatology.py — mocked at the network boundary.
"""
import numpy as np
import pytest
import xarray as xr
from unittest.mock import patch, MagicMock


def _make_fake_oisst(n_days: int = 10, n_lat: int = 17, n_lon: int = 21) -> xr.Dataset:
    """Minimal OISST-shaped dataset for mocking."""
    rng = np.random.default_rng(42)
    times = xr.cftime_range("1982-01-01", periods=n_days, freq="D")
    sst = xr.DataArray(
        rng.normal(20.0, 2.0, (n_days, 1, n_lat, n_lon)).astype("float32"),
        dims=["time", "zlev", "lat", "lon"],
        coords={
            "time": times,
            "zlev": [0.0],
            "lat": np.linspace(41.125, 44.875, n_lat),
            "lon": np.linspace(-70.875, -66.125, n_lon),
        },
        attrs={"units": "degree_C"},
    )
    return xr.Dataset({"sst": sst})


def test_build_oisst_url_format():
    from scripts.fetch_oisst_climatology import build_oisst_url
    url = build_oisst_url(1982, 1, 1)
    assert "1982" in url
    assert "198201" in url
    assert "19820101" in url
    assert url.startswith("https://")


def test_fetch_oisst_gom_squeezes_zlev():
    """fetch_oisst_gom must drop zlev dim and return (time, lat, lon)."""
    from scripts.fetch_oisst_climatology import fetch_oisst_gom

    fake_ds = _make_fake_oisst(n_days=31)
    with patch("scripts.fetch_oisst_climatology.xr.open_mfdataset", return_value=fake_ds):
        sst = fetch_oisst_gom(1982, 1)

    assert "zlev" not in sst.dims
    assert set(sst.dims) == {"time", "lat", "lon"}


def test_compute_and_write_climatology_calls_gcs_safe_write():
    """compute_and_write_climatology must call _gcs_safe_write with threshold dataset."""
    from scripts.fetch_oisst_climatology import compute_and_write_climatology

    # 5 years × 12 months of fake monthly SSTs
    rng = np.random.default_rng(0)
    times = xr.cftime_range("1982-01-01", periods=365 * 5, freq="D")
    fake_sst = xr.DataArray(
        rng.normal(20.0, 1.0, (365 * 5, 4, 5)).astype("float32"),
        dims=["time", "lat", "lon"],
        coords={"time": times,
                "lat": np.linspace(41.0, 45.0, 4),
                "lon": np.linspace(-71.0, -66.0, 5)},
    )

    with patch("scripts.fetch_oisst_climatology._gcs_safe_write") as mock_write, \
         patch("scripts.fetch_oisst_climatology._gcs_complete", return_value=False):
        compute_and_write_climatology(fake_sst, "gs://fake-bucket/hycom/climatology/")

    mock_write.assert_called_once()
    call_args = mock_write.call_args[0]
    ds_written = call_args[0]
    assert "sst_threshold_90" in ds_written


def test_compute_and_write_climatology_skips_if_complete():
    """Must skip if GCS sentinel already present."""
    from scripts.fetch_oisst_climatology import compute_and_write_climatology

    fake_sst = xr.DataArray(
        np.ones((10, 2, 2), dtype="float32"),
        dims=["time", "lat", "lon"],
        coords={"time": xr.cftime_range("1982-01-01", periods=10, freq="D"),
                "lat": [41.0, 42.0], "lon": [-71.0, -70.0]},
    )

    with patch("scripts.fetch_oisst_climatology._gcs_safe_write") as mock_write, \
         patch("scripts.fetch_oisst_climatology._gcs_complete", return_value=True):
        compute_and_write_climatology(fake_sst, "gs://fake-bucket/hycom/climatology/")

    mock_write.assert_not_called()
```

- [ ] **Step A2.2: Run tests — expect FAIL**

```bash
/home/avik2007/miniconda3/envs/mhw-risk/bin/python -m pytest tests/test_oisst_climatology.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.fetch_oisst_climatology'`

- [ ] **Step A2.3: Implement `scripts/fetch_oisst_climatology.py`**

```python
"""
fetch_oisst_climatology.py — Fetch NOAA OISST v2.1, compute 30-year baseline, write to GCS.
==========================================================================
Fetches daily SST for the GoM bbox (1982-2011) from NOAA NCEI THREDDS OPeNDAP.
Computes 90th-percentile threshold per (dayofyear, lat, lon) with 11-day rolling
window per Hobday (2016). Writes result to GCS as sst_threshold_90, replacing
the 2-year HYCOM baseline.

Usage (on VM):
    nohup env \\
      GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \\
      MHW_GCS_BUCKET=gs://mhw-risk-cache \\
      /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/fetch_oisst_climatology.py \\
      >> ~/nohup_oisst.log 2>&1 </dev/null & disown $!

THREDDS URL fallback:
    If NCEI THREDDS is unavailable, set env var OISST_USE_ERDDAP=1 to use ERDDAP:
    https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg
"""
from __future__ import annotations

import calendar
import logging
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Allow running as script or importing from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.analytics.mhw_detection import compute_climatology
from src.ingestion.harvester import _gcs_safe_write, _gcs_complete

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 30-year OISST climatology period — Hobday (2016) standard
CLIM_START_YEAR = 1982
CLIM_END_YEAR   = 2011

# GoM bounding box: (lon_min, lat_min, lon_max, lat_max)
# Matches GoM_BBOX in _train_utils.py
GoM_LON_MIN = -71.0
GoM_LON_MAX = -66.0
GoM_LAT_MIN =  41.0
GoM_LAT_MAX =  45.0

OISST_THREDDS_BASE = (
    "https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/V2.1/AVHRR"
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
        Calendar date of the OISST daily analysis.

    Returns
    -------
    str
        OPeNDAP URL for the netCDF4 file.
    """
    ym = f"{year}{month:02d}"
    ymd = f"{year}{month:02d}{day:02d}"
    return f"{OISST_THREDDS_BASE}/{ym}/oisst-avhrr-v02r01.{ymd}.nc"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_oisst_gom(year: int, month: int) -> xr.DataArray:
    """
    Fetch one month of OISST v2.1 SST for the GoM bbox via OPeNDAP.

    Builds a list of daily OPeNDAP URLs, opens them with xr.open_mfdataset
    (lazy), subsets to GoM bbox, drops the zlev dimension, and returns
    a (time, lat, lon) DataArray.

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
    """
    n_days = calendar.monthrange(year, month)[1]
    urls = [build_oisst_url(year, month, d) for d in range(1, n_days + 1)]

    ds = xr.open_mfdataset(
        urls,
        combine="by_coords",
        engine="netcdf4",
        # Only load sst — skip ice fraction, anom, err vars
        drop_variables=["ice", "anom", "err"],
    )

    # Subset to GoM bbox; OISST lat runs south→north, lon runs -180→180
    sst = (
        ds["sst"]
        .sel(
            lat=slice(GoM_LAT_MIN, GoM_LAT_MAX),
            lon=slice(GoM_LON_MIN, GoM_LON_MAX),
        )
        .squeeze("zlev", drop=True)  # drop the single depth level
    )
    return sst.compute()   # materialise the small GoM subset


# ---------------------------------------------------------------------------
# Climatology + GCS write
# ---------------------------------------------------------------------------

def compute_and_write_climatology(sst_all: xr.DataArray, clim_uri: str) -> None:
    """
    Compute 90th-pct threshold with 11-day rolling window and write to GCS.

    Parameters
    ----------
    sst_all : xr.DataArray
        Concatenated daily SST [°C], dims (time, lat, lon),
        covering the full climatology period.
    clim_uri : str
        GCS URI for the climatology Zarr store
        (e.g. gs://mhw-risk-cache/hycom/climatology/).
        Writes variable sst_threshold_90 to this path.
    """
    import gcsfs
    fs = gcsfs.GCSFileSystem()

    if _gcs_complete(fs, clim_uri):
        logger.info("Cache hit — skipping climatology: %s", clim_uri)
        return

    logger.info("Computing 30-year OISST climatology (window=11 days)...")
    threshold = compute_climatology(sst_all, percentile=90.0, window=11)

    ds_threshold = threshold.to_dataset(name="sst_threshold_90")
    ds_threshold.attrs.update({
        "source": "NOAA OISST v2.1 (AVHRR-only)",
        "period": f"{CLIM_START_YEAR}–{CLIM_END_YEAR}",
        "percentile": 90.0,
        "rolling_window_days": 11,
        "citation": "Hobday et al. (2016), Prog. Oceanogr.",
        "units": "degree_C",
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

    import gcsfs
    fs = gcsfs.GCSFileSystem()
    if _gcs_complete(fs, clim_uri):
        logger.info("OISST climatology already complete. Nothing to do.")
        return

    logger.info(
        "Fetching OISST v2.1 %d–%d for GoM bbox...",
        CLIM_START_YEAR, CLIM_END_YEAR,
    )
    monthly_slabs: list[xr.DataArray] = []

    for year in range(CLIM_START_YEAR, CLIM_END_YEAR + 1):
        for month in range(1, 13):
            logger.info("Fetching %d-%02d...", year, month)
            try:
                slab = fetch_oisst_gom(year, month)
                monthly_slabs.append(slab)
            except Exception as exc:
                logger.error("Failed %d-%02d: %s — skipping", year, month, exc)

    logger.info("Concatenating %d monthly slabs...", len(monthly_slabs))
    sst_all = xr.concat(monthly_slabs, dim="time").sortby("time")

    compute_and_write_climatology(sst_all, clim_uri)
    logger.info("Done. Climatology at %s", clim_uri)


if __name__ == "__main__":
    main()
```

- [ ] **Step A2.4: Run tests — expect PASS**

```bash
/home/avik2007/miniconda3/envs/mhw-risk/bin/python -m pytest tests/test_oisst_climatology.py -v
```

Expected: all 4 tests pass.

- [ ] **Step A2.5: Run full suite — no regressions**

```bash
/home/avik2007/miniconda3/envs/mhw-risk/bin/python -m pytest tests/ -v
```

Expected: all 68 tests pass (+ 4 new = 72 total).

- [ ] **Step A2.6: Commit**

```bash
git add scripts/fetch_oisst_climatology.py tests/test_oisst_climatology.py
git commit -m "feat: add fetch_oisst_climatology.py — 30-year OISST baseline per Hobday (2016)"
```

---

### Task A3: VM — Delete old climatology, run OISST fetch, verify

Execute on mhw-data-prep (or fresh VM) AFTER steps A2.6 and B1.5 are committed and pushed.

- [ ] **Step A3.1: Pull latest code on VM**

```bash
cd ~/mhw-risk-profiler && git pull
```

- [ ] **Step A3.2: Verify THREDDS URL is reachable**

```bash
curl -s -o /dev/null -w "%{http_code}" \
  "https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/V2.1/AVHRR/198201/oisst-avhrr-v02r01.19820101.nc"
```

Expected: `200`. If `404` or timeout, check NCEI THREDDS status at `https://www.ncei.noaa.gov/thredds/catalog.html` and adjust `OISST_THREDDS_BASE` in the script.

- [ ] **Step A3.3: Delete old 2-year climatology from GCS**

```bash
gsutil -m rm -r gs://mhw-risk-cache/hycom/climatology/
```

- [ ] **Step A3.4: Launch OISST fetch (background)**

```bash
nohup env \
  GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
  MHW_GCS_BUCKET=gs://mhw-risk-cache \
  /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/fetch_oisst_climatology.py \
  >> ~/nohup_oisst.log 2>&1 </dev/null & disown $!
```

ETA: ~30-90 min depending on NCEI THREDDS throughput (360 month-batches × ~28 OPeNDAP calls each; GoM subset is ~20 KB per day file → total ~220 MB raw transfer).

- [ ] **Step A3.5: Verify output**

```bash
# Poll until complete
tail -20 ~/nohup_oisst.log

# Verify GCS write
gsutil ls gs://mhw-risk-cache/hycom/climatology/.complete

# Spot-check threshold values (expect 15-25 °C range for GoM)
/home/avik2007/miniconda3/envs/mhw-risk/bin/python - <<'EOF'
import xarray as xr
ds = xr.open_zarr("gs://mhw-risk-cache/hycom/climatology/")
print(ds)
print("threshold min:", float(ds["sst_threshold_90"].min()))
print("threshold max:", float(ds["sst_threshold_90"].max()))
print("threshold mean:", float(ds["sst_threshold_90"].mean()))
EOF
```

Expected: `sst_threshold_90` dims `(dayofyear=365, lat, lon)`, values in range ~10–28 °C for GoM.

---

## TRACK B — WN2 SST Fix

### Task B1: Add SST to `WN2_VARIABLES` + land mask in `_build_dataset()`

**Files:**
- Modify: `src/ingestion/harvester.py:177-184` (WN2_VARIABLES)
- Modify: `src/ingestion/harvester.py:542-560` (_build_dataset inner loop)
- Modify: `tests/test_wn2_harvester.py`

**Key context:**
- `WN2_VARIABLES` in `harvester.py` currently has 4 vars (no SST). Adding SST → 5 vars.
- `WN2_VARS` in `_train_utils.py` already lists SST as element 0. So tensor shape stays `(member, seq_len, 5)` — model architecture unchanged.
- K→°C conversion already handled in `_train_utils.py:117` (`- 273.15`). No new conversion needed here.
- NaN masking: in `_build_dataset()` at line 548, after building `arr = np.array(member_arrays, dtype=np.float32)`, apply `arr[arr == 0.0] = np.nan` for the SST variable only.
- Land mask concern (Gemini #2): xarray's spatial `.mean(dim=["latitude","longitude"])` uses `skipna=True` by default, so NaN land pixels are naturally excluded from the SDD label and feature vectors. No explicit land mask propagation to other variables needed.
- Cache poisoning (Gemini #1): VM re-fetch step must delete `_daily/` dirs before re-running. See Task B2.

- [ ] **Step B1.1: Write failing test**

Add to `tests/test_wn2_harvester.py`:

```python
def test_wn2_variables_includes_sst():
    from src.ingestion.harvester import WN2_VARIABLES
    assert "sea_surface_temperature" in WN2_VARIABLES, \
        "SST must be in WN2_VARIABLES for build_tensors() to find merged['sea_surface_temperature']"


def test_build_dataset_masks_sst_zero_as_nan(mock_ee, mock_gcsfs_not_complete):
    """
    SST pixels with value 0.0 (GEE defaultValue for land) must be replaced with NaN
    in _build_dataset output. Non-SST variables must NOT be masked.
    """
    import numpy as np
    from unittest.mock import patch, MagicMock
    from src.ingestion.harvester import WeatherNext2Harvester, WN2_VARIABLES

    harvester = WeatherNext2Harvester(gcs_bucket="gs://fake-bucket")

    # Build fake GEE sampleRectangle response where SST=0 over land pixels
    fake_props = {}
    n_members, n_lat, n_lon = 2, 3, 4
    for m in range(n_members):
        for var in WN2_VARIABLES:
            key = f"20220101000020220102000{m}_{var}"
            if var == "sea_surface_temperature":
                arr = np.zeros((n_lat, n_lon), dtype=np.float32)
                arr[1, 1] = 280.0  # one valid ocean pixel
            else:
                arr = np.ones((n_lat, n_lon), dtype=np.float32) * 273.0
            fake_props[key] = arr.tolist()

    fake_sample = {"properties": fake_props}
    fake_region = MagicMock()
    fake_region.bounds.return_value.getInfo.return_value = {
        "coordinates": [[[-71.0, 41.0], [-71.0, 45.0], [-66.0, 45.0], [-66.0, 41.0]]]
    }

    fake_collection = MagicMock()
    fake_collection.aggregate_array.return_value.getInfo.return_value = [
        "2022-01-01T00:00:00Z_2022-01-02T00:00:00Z_0"
    ]

    with patch("src.ingestion.harvester.ee") as mock_ee_inner, \
         patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
         patch("src.ingestion.harvester._gcs_complete", return_value=False), \
         patch("src.ingestion.harvester._gcs_safe_write"):
        mock_fs_cls.return_value = MagicMock()
        mock_ee_inner.ImageCollection.return_value.filter.return_value.filter.return_value \
            .filter.return_value.map.return_value.sort.return_value.limit.return_value = fake_collection
        # patch sampleRectangle
        mock_combined = MagicMock()
        mock_combined.sampleRectangle.return_value.getInfo.return_value = fake_sample
        mock_ee_inner.ImageCollection.return_value.filter.return_value.filter.return_value \
            .filter.return_value.map.return_value.sort.return_value.limit.return_value \
            .select.return_value.toBands.return_value = mock_combined

        ds = harvester._build_dataset(fake_collection, fake_region, n_members, "gs://fake/test")

    sst = ds["sea_surface_temperature"].values  # (member, time, lat, lon)
    assert np.isnan(sst[:, :, 0, 0]).all(), "Land pixel (0.0) must be NaN"
    assert np.isfinite(sst[:, :, 1, 1]).all(), "Ocean pixel (280.0) must be valid"

    # Non-SST variable must NOT be masked
    wind_u = ds["10m_u_component_of_wind"].values
    assert not np.isnan(wind_u).any(), "Wind must not be NaN-masked"
```

- [ ] **Step B1.2: Run failing test**

```bash
/home/avik2007/miniconda3/envs/mhw-risk/bin/python -m pytest tests/test_wn2_harvester.py::test_wn2_variables_includes_sst -v
```

Expected: `AssertionError: SST must be in WN2_VARIABLES`.

- [ ] **Step B1.3: Add SST to `WN2_VARIABLES`**

In `src/ingestion/harvester.py`, replace lines 177-184:

```python
WN2_VARIABLES = [
    "sea_surface_temperature",  # [K] Surface ocean temp — primary MHW driver; land pixels masked to NaN
    "2m_temperature",          # [K] Near-surface air temperature — proxy for atmospheric forcing
    "10m_u_component_of_wind", # [m/s] Zonal wind — drives Ekman pumping and vertical mixing
    "10m_v_component_of_wind", # [m/s] Meridional wind — completes horizontal wind vector
    "mean_sea_level_pressure",  # [Pa] MSLP — identifies anticyclonic blocking that suppresses mixing
]
```

- [ ] **Step B1.4: Add NaN masking in `_build_dataset()`**

In `src/ingestion/harvester.py` at the inner variable loop (currently around line 542-549), add NaN masking after arr is built:

```python
            for var in target_bands:
                member_arrays = [
                    np.array(props[f"{start_compact}_{end_compact}_{m}_{var}"])
                    for m in range(n_members)
                ]
                arr = np.array(member_arrays, dtype=np.float32)  # (n_members, n_lat, n_lon)
                # GEE uses defaultValue=0 for land pixels. SST=0 K is physically impossible
                # (ocean never reaches absolute zero); treat as land mask sentinel.
                if var == "sea_surface_temperature":
                    arr[arr == 0.0] = np.nan
                data_vars_day[var] = (["member", "latitude", "longitude"], arr)
```

- [ ] **Step B1.5: Run full test suite**

```bash
/home/avik2007/miniconda3/envs/mhw-risk/bin/python -m pytest tests/ -v
```

Expected: all 68+ tests pass (the complex `test_build_dataset_masks_sst_zero_as_nan` may need fixture adjustment — if it requires too much mock plumbing, simplify to testing only `WN2_VARIABLES` and the NaN logic in isolation).

- [ ] **Step B1.6: Commit + push**

```bash
git add src/ingestion/harvester.py tests/test_wn2_harvester.py
git commit -m "fix: add SST back to WN2_VARIABLES with land mask (0 K → NaN) in _build_dataset"
git push
```

---

### Task B2: VM — Delete daily cache dirs + re-fetch WN2 2022 + 2023 with SST

Execute on VM AFTER B1.6 is pushed.

- [ ] **Step B2.1: Pull latest code on VM**

```bash
cd ~/mhw-risk-profiler && git pull
```

- [ ] **Step B2.2: Delete stale 2022 WN2 tile AND daily cache**

```bash
# Delete annual zarr (no SST)
gsutil -m rm -r gs://mhw-risk-cache/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr
gsutil rm -f gs://mhw-risk-cache/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr.complete

# Delete per-day cache (CRITICAL — without this, _build_dataset re-uses old days without SST)
gsutil -m rm -r gs://mhw-risk-cache/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr_daily/
```

- [ ] **Step B2.3: Delete stale 2023 WN2 tile AND daily cache**

```bash
gsutil -m rm -r gs://mhw-risk-cache/weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr
gsutil rm -f gs://mhw-risk-cache/weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr.complete
gsutil -m rm -r gs://mhw-risk-cache/weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr_daily/
```

- [ ] **Step B2.4: Launch 2022 re-fetch (can run in parallel with 2023 on second VM)**

```bash
nohup env \
  GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
  MHW_GCS_BUCKET=gs://mhw-risk-cache \
  /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/run_wn2_prep.py --year 2022 \
  >> ~/nohup_wn2_2022.log 2>&1 </dev/null & disown $!
```

- [ ] **Step B2.5: Launch 2023 re-fetch (on second VM or sequential after 2022)**

```bash
nohup env \
  GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
  MHW_GCS_BUCKET=gs://mhw-risk-cache \
  /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/run_wn2_prep.py --year 2023 \
  >> ~/nohup_wn2_2023.log 2>&1 </dev/null & disown $!
```

ETA: ~55 min per year (9 sec/day × 365 days). Run both in parallel on separate VMs to cut total to ~55 min.

- [ ] **Step B2.6: Verify SST present in new WN2 tiles**

```bash
/home/avik2007/miniconda3/envs/mhw-risk/bin/python - <<'EOF'
import xarray as xr
ds = xr.open_zarr("gs://mhw-risk-cache/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr")
print(list(ds.data_vars))          # must include sea_surface_temperature
sst = ds["sea_surface_temperature"].isel(time=0, member=0)
print("SST min:", float(sst.min()))  # must be > 0 (NaN land pixels excluded from min with skipna)
print("SST max:", float(sst.max()))  # expect ~275-291 K over GoM in winter
EOF
```

---

### Task B3: VM — Run `train_wn2.py` after both tracks complete

Execute AFTER:
- Track A Task A3 complete (new 30-year climatology in GCS)
- Track B Task B2 complete (WN2 2022+2023 tiles have SST)

- [ ] **Step B3.1: Verify all prerequisites**

```bash
for path in \
  "hycom/climatology/.complete" \
  "weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr.complete" \
  "weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr.complete" \
  "hycom/tiles/2022/.complete" \
  "hycom/tiles/2023/.complete"; do
  gsutil ls "gs://mhw-risk-cache/${path}" 2>/dev/null && echo "OK: $path" || echo "MISSING: $path"
done
```

All 5 must show OK before proceeding.

- [ ] **Step B3.2: Also retrain ERA5 (new threshold changes SDD labels)**

```bash
nohup env \
  GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
  MHW_GCS_BUCKET=gs://mhw-risk-cache \
  /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/train_era5.py --epochs 50 \
  >> ~/train_era5_v2.log 2>&1 </dev/null & disown $!
```

- [ ] **Step B3.3: Train WN2**

```bash
nohup env \
  GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
  MHW_GCS_BUCKET=gs://mhw-risk-cache \
  /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/train_wn2.py --epochs 50 \
  >> ~/train_wn2.log 2>&1 </dev/null & disown $!
```

- [ ] **Step B3.4: Verify training metrics**

```bash
tail -5 ~/train_wn2.log
```

Expected:
- `spread > 0` (real 64-member WN2 ensemble with genuine SST spread)
- `SVaR_95 > SVaR_50 > SVaR_05` (non-degenerate quantile ordering)
- Loss decreasing over epochs

---

## Success Criteria Checklist

- [ ] `compute_climatology()` accepts `window` param; 11-day rolling applied with wrap-around
- [ ] `fetch_oisst_climatology.py` writes `sst_threshold_90` to GCS; threshold in range 10-28 °C for GoM
- [ ] All 72 tests pass
- [ ] `WN2_VARIABLES` contains `sea_surface_temperature`
- [ ] WN2 2022/2023 GCS tiles have `sea_surface_temperature` variable
- [ ] `train_wn2.py` completes 50 epochs without crash
- [ ] `spread > 0` in WN2 training log
- [ ] ERA5 re-trained with new 30-year threshold
- [ ] CLAUDE.md Lessons Applied updated with OISST approach
