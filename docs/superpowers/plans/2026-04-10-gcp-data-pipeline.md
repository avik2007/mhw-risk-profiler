# GCP Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace live OPeNDAP/GEE calls in training scripts with GCS-cached Zarr reads, pre-fetched by a spot GCE data prep job.

**Architecture:** `HYCOMLoader` and `ERA5Harvester` gain `fetch_and_cache(year, bbox, gcs_uri)` methods. A new `run_data_prep.py` orchestrator writes all training data to GCS once. Training scripts (`load_real_data()` in `train_era5.py` and `train_wn2.py`) replace live fetches with `xr.open_zarr()` calls. ERA5 period is aligned to 2022/2023 (same as WN2) for apples-to-apples XAI comparison.

**Tech Stack:** `xarray>=2024.2.0`, `gcsfs>=2024.2.0`, `earthengine-api>=0.1.390`, `zarr>=2.17`, `pytest>=8.0.0`. No new dependencies.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/ingestion/harvester.py` | Add `HYCOMLoader.fetch_and_cache()` after `fetch_tile()` (line ~553) |
| Modify | `src/ingestion/era5_harvester.py` | Add `ERA5Harvester.fetch_and_cache()` after `fetch()` (line ~186) |
| Create | `tests/test_harvester_cache.py` | Unit tests for both `fetch_and_cache()` methods — mocked, no network |
| Modify | `scripts/_train_utils.py` | Replace 4 period constants with `TRAIN_PERIOD`/`VAL_PERIOD` (2022/2023) |
| Modify | `scripts/train_era5.py` | GCS-only `load_real_data()`, fix `ds["threshold"]` bug, update imports |
| Modify | `scripts/train_wn2.py` | Same as train_era5.py |
| Create | `scripts/run_data_prep.py` | Data prep orchestrator — idempotent, runs on spot GCE VM |
| Create | `docs/gcp-data-prep-runbook.md` | gcloud commands for VM setup and job execution |

---

## Task 1: `HYCOMLoader.fetch_and_cache()`

**Files:**
- Modify: `src/ingestion/harvester.py` — add method to `HYCOMLoader` after `fetch_tile()` (~line 553)
- Create: `tests/test_harvester_cache.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_harvester_cache.py`:

```python
"""
Unit tests for HYCOMLoader.fetch_and_cache() and ERA5Harvester.fetch_and_cache().
No network calls — gcsfs and fetch methods are fully mocked.
"""
from unittest.mock import MagicMock, call, patch
import pytest

from src.ingestion.harvester import HYCOMLoader


class TestHYCOMLoaderFetchAndCache:
    def test_cache_hit_skips_fetch(self):
        """If gcs_uri already exists, fetch_tile() is never called."""
        loader = HYCOMLoader()
        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(loader, "fetch_tile") as mock_fetch:
            mock_fs_cls.return_value.exists.return_value = True
            loader.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/hycom/tiles/2022/")
            mock_fetch.assert_not_called()

    def test_cache_miss_calls_fetch_and_writes(self):
        """If gcs_uri does not exist, fetch_tile() is called and result written to GCS."""
        import xarray as xr
        import numpy as np

        loader = HYCOMLoader()
        fake_ds = xr.Dataset({"water_temp": xr.DataArray(np.zeros((2, 3, 4, 5)),
                              dims=["time", "depth", "lat", "lon"])})

        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(loader, "fetch_tile", return_value=fake_ds) as mock_fetch, \
             patch.object(fake_ds, "to_zarr") as mock_to_zarr:
            mock_fs_cls.return_value.exists.return_value = False
            loader.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/hycom/tiles/2022/")
            mock_fetch.assert_called_once_with("2022-01-01", "2022-12-31", (-71.0, 41.0, -66.0, 45.0))
            mock_to_zarr.assert_called_once_with("gs://bucket/hycom/tiles/2022/", mode="w", consolidated=True)

    def test_cache_hit_check_strips_gs_prefix(self):
        """gcsfs.exists() is called with the path without 'gs://' prefix."""
        loader = HYCOMLoader()
        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(loader, "fetch_tile"):
            mock_fs_cls.return_value.exists.return_value = True
            loader.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/hycom/tiles/2022/")
            mock_fs_cls.return_value.exists.assert_called_once_with("bucket/hycom/tiles/2022/")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n mhw-risk pytest tests/test_harvester_cache.py -v 2>&1 | tail -20
```

Expected: `AttributeError: HYCOMLoader has no attribute 'fetch_and_cache'`

- [ ] **Step 3: Add `import gcsfs` to harvester.py module-level imports**

In `src/ingestion/harvester.py`, add to the existing imports block (after `import dask.array as da`):

```python
import gcsfs
```

- [ ] **Step 4: Add `fetch_and_cache()` to `HYCOMLoader` after `fetch_tile()` (~line 553)**

```python
    def fetch_and_cache(
        self,
        year: int,
        bbox: tuple[float, float, float, float],
        gcs_uri: str,
    ) -> None:
        """
        Fetch one full calendar year of HYCOM data and write to GCS as Zarr.

        Parameters
        ----------
        year : int
            Calendar year to fetch (Jan 1 – Dec 31).
            Must be within HYCOM GLBy0.08/expt_93.0 coverage (2018-12-04 to 2024-09-04).
        bbox : tuple of float
            (lon_min, lat_min, lon_max, lat_max) in decimal degrees WGS84.
        gcs_uri : str
            GCS destination, e.g. "gs://bucket/hycom/tiles/2022/".
            If the URI already exists, returns immediately — idempotent, safe to re-run
            after a spot VM preemption.

        Side effects
        ------------
        Writes a Zarr store to gcs_uri with dims (time, depth, lat, lon)
        and variables: water_temp [°C], salinity [psu], water_u [m/s], water_v [m/s].
        Credentials are read from GOOGLE_APPLICATION_CREDENTIALS automatically by gcsfs.
        """
        fs = gcsfs.GCSFileSystem()
        path = gcs_uri.removeprefix("gs://")
        if fs.exists(path):
            logger.info("Cache hit — skipping HYCOM fetch for %d: %s", year, gcs_uri)
            return

        start_date = f"{year}-01-01"
        end_date   = f"{year}-12-31"
        logger.info("Fetching HYCOM year %d (%s to %s)...", year, start_date, end_date)

        ds = self.fetch_tile(start_date, end_date, bbox)
        ds.to_zarr(gcs_uri, mode="w", consolidated=True)
        logger.info("HYCOM year %d written to %s", year, gcs_uri)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
conda run -n mhw-risk pytest tests/test_harvester_cache.py::TestHYCOMLoaderFetchAndCache -v
```

Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/harvester.py tests/test_harvester_cache.py
git commit -m "feat: add HYCOMLoader.fetch_and_cache() with GCS idempotent caching"
```

---

## Task 2: `ERA5Harvester.fetch_and_cache()`

**Files:**
- Modify: `src/ingestion/era5_harvester.py` — add method after `fetch()` (~line 186)
- Modify: `tests/test_harvester_cache.py` — add ERA5 test class

- [ ] **Step 1: Add ERA5 tests to `tests/test_harvester_cache.py`**

Append to the existing file:

```python
from src.ingestion.era5_harvester import ERA5Harvester


class TestERA5HarvesterFetchAndCache:
    def test_cache_hit_skips_fetch(self):
        """If gcs_uri already exists, fetch() is never called."""
        harvester = ERA5Harvester()
        harvester._initialized = True  # skip auth
        with patch("src.ingestion.era5_harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(harvester, "fetch") as mock_fetch:
            mock_fs_cls.return_value.exists.return_value = True
            harvester.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/era5/2022/")
            mock_fetch.assert_not_called()

    def test_cache_miss_calls_fetch_and_writes(self):
        """If gcs_uri does not exist, fetch() is called and result written to GCS."""
        import xarray as xr
        import numpy as np

        harvester = ERA5Harvester()
        harvester._initialized = True
        fake_ds = xr.Dataset({"sea_surface_temperature": xr.DataArray(
            np.zeros((1, 10, 4, 5)), dims=["member", "time", "latitude", "longitude"])})

        with patch("src.ingestion.era5_harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(harvester, "fetch", return_value=fake_ds) as mock_fetch, \
             patch.object(fake_ds, "to_zarr") as mock_to_zarr:
            mock_fs_cls.return_value.exists.return_value = False
            harvester.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/era5/2022/")
            mock_fetch.assert_called_once_with("2022-01-01", "2022-12-31", (-71.0, 41.0, -66.0, 45.0))
            mock_to_zarr.assert_called_once_with("gs://bucket/era5/2022/", mode="w", consolidated=True)

    def test_raises_if_not_authenticated(self):
        """fetch_and_cache() raises RuntimeError if authenticate() was not called."""
        harvester = ERA5Harvester()
        # _initialized is False by default
        with patch("src.ingestion.era5_harvester.gcsfs.GCSFileSystem") as mock_fs_cls:
            mock_fs_cls.return_value.exists.return_value = False
            with pytest.raises(RuntimeError, match="authenticate()"):
                harvester.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/era5/2022/")
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
conda run -n mhw-risk pytest tests/test_harvester_cache.py::TestERA5HarvesterFetchAndCache -v 2>&1 | tail -10
```

Expected: `AttributeError: ERA5Harvester has no attribute 'fetch_and_cache'`

- [ ] **Step 3: Add `import gcsfs` to era5_harvester.py**

In `src/ingestion/era5_harvester.py`, add after `import numpy as np`:

```python
import gcsfs
```

- [ ] **Step 4: Add `fetch_and_cache()` to `ERA5Harvester` after `fetch()` (~line 186)**

```python
    def fetch_and_cache(
        self,
        year: int,
        bbox: tuple[float, float, float, float],
        gcs_uri: str,
    ) -> None:
        """
        Fetch one full calendar year of ERA5 data from GEE and write to GCS as Zarr.

        Parameters
        ----------
        year : int
            Calendar year to fetch (Jan 1 – Dec 31).
            ECMWF/ERA5/DAILY on GEE covers 1979–present.
        bbox : tuple of float
            (lon_min, lat_min, lon_max, lat_max) in decimal degrees WGS84.
        gcs_uri : str
            GCS destination, e.g. "gs://bucket/era5/2022/".
            If the URI already exists, returns immediately — idempotent, safe to re-run
            after a spot VM preemption.

        Side effects
        ------------
        Writes a Zarr store to gcs_uri with dims (member=1, time, latitude, longitude)
        and WN2-compatible variable names. Downstream callers must invoke
        DataHarmonizer.expand_and_perturb() (via harmonize()) to expand to 64 members.
        Credentials are read from GOOGLE_APPLICATION_CREDENTIALS automatically by gcsfs.
        """
        if not self._initialized:
            raise RuntimeError("Call authenticate() before fetch_and_cache().")

        fs = gcsfs.GCSFileSystem()
        path = gcs_uri.removeprefix("gs://")
        if fs.exists(path):
            logger.info("Cache hit — skipping ERA5 fetch for %d: %s", year, gcs_uri)
            return

        start_date = f"{year}-01-01"
        end_date   = f"{year}-12-31"
        logger.info("Fetching ERA5 year %d (%s to %s)...", year, start_date, end_date)

        ds = self.fetch(start_date, end_date, bbox)
        ds.to_zarr(gcs_uri, mode="w", consolidated=True)
        logger.info("ERA5 year %d written to %s", year, gcs_uri)
```

- [ ] **Step 5: Run all cache tests**

```bash
conda run -n mhw-risk pytest tests/test_harvester_cache.py -v
```

Expected: `6 passed`

- [ ] **Step 6: Run full test suite to confirm no regressions**

```bash
conda run -n mhw-risk pytest tests/ -v 2>&1 | tail -10
```

Expected: all prior tests still passing

- [ ] **Step 7: Commit**

```bash
git add src/ingestion/era5_harvester.py tests/test_harvester_cache.py
git commit -m "feat: add ERA5Harvester.fetch_and_cache() with GCS idempotent caching"
```

---

## Task 3: Period constant alignment in `_train_utils.py`

**Files:**
- Modify: `scripts/_train_utils.py` — replace 4 period constants with shared `TRAIN_PERIOD`/`VAL_PERIOD`

- [ ] **Step 1: Verify which period constants are exported and consumed**

```bash
conda run -n mhw-risk grep -n "ERA5_TRAIN_PERIOD\|ERA5_VAL_PERIOD\|WN2_TRAIN_PERIOD\|WN2_VAL_PERIOD\|TRAIN_PERIOD\|VAL_PERIOD" scripts/train_era5.py scripts/train_wn2.py scripts/_train_utils.py
```

Expected output confirms:
- `train_era5.py` imports `ERA5_TRAIN_PERIOD`, `ERA5_VAL_PERIOD`
- `train_wn2.py` imports `WN2_TRAIN_PERIOD`, `WN2_VAL_PERIOD`
- `_train_utils.py` defines all four plus `TRAIN_PERIOD = ERA5_TRAIN_PERIOD`

- [ ] **Step 2: Replace the period constants block in `scripts/_train_utils.py`**

Find and replace lines 39–51 (the period constants block):

Old:
```python
# ERA5 training periods: 2018 reanalysis train, 2019 reanalysis val
ERA5_TRAIN_PERIOD = ("2018-01-01", "2018-12-31")
ERA5_VAL_PERIOD   = ("2019-01-01", "2019-12-31")

# WN2 training periods: 2022 forecast runs train, 2023 forecast runs val
# WN2 uses 2022+ because GLBy0.08/expt_93.0 covers 2018-12-04 to 2024-09-04
# and WN2 forecast run structure starts from 2022-present in the GEE asset.
WN2_TRAIN_PERIOD  = ("2022-01-01", "2022-12-31")
WN2_VAL_PERIOD    = ("2023-01-01", "2023-12-31")

# Backward-compat aliases (ERA5 periods) — used by existing test code
TRAIN_PERIOD = ERA5_TRAIN_PERIOD
VAL_PERIOD   = ERA5_VAL_PERIOD
```

New:
```python
# Shared training periods for ERA5 and WN2.
# Both use 2022/2023 for apples-to-apples XAI comparison. ERA5 covers 1979-present
# on GEE; WN2 covers 2022-present. HYCOM GLBy0.08/expt_93.0 covers through 2024-09-04.
TRAIN_PERIOD = ("2022-01-01", "2022-12-31")
VAL_PERIOD   = ("2023-01-01", "2023-12-31")
```

- [ ] **Step 3: Run full test suite to confirm no regressions**

```bash
conda run -n mhw-risk pytest tests/ -v 2>&1 | tail -15
```

Expected: all tests still passing (no test imports the removed constants)

- [ ] **Step 4: Commit**

```bash
git add scripts/_train_utils.py
git commit -m "refactor: align ERA5 and WN2 training periods to 2022/2023 shared constants"
```

---

## Task 4: GCS-only `load_real_data()` in `train_era5.py`

**Files:**
- Modify: `scripts/train_era5.py` — replace `load_real_data()` body, fix imports, fix threshold var name

- [ ] **Step 1: Update the import line at the top of `train_era5.py`**

Old (line ~45-48):
```python
from _train_utils import (
    GoM_BBOX, ERA5_TRAIN_PERIOD, ERA5_VAL_PERIOD, N_MEMBERS, SEQ_LEN,
    build_tensors, run_svar_inference, save_plots,
)
```

New:
```python
from _train_utils import (
    GoM_BBOX, TRAIN_PERIOD, VAL_PERIOD, N_MEMBERS, SEQ_LEN,
    build_tensors, run_svar_inference, save_plots,
)
```

- [ ] **Step 2: Replace the `load_real_data()` function body in `train_era5.py`**

Old (lines ~62-105):
```python
def load_real_data():
    """
    Fetch ERA5 + HYCOM for train (2018) and val (2019) periods.
    ...
    """
    from src.ingestion.era5_harvester import ERA5Harvester
    from src.ingestion.harvester import DataHarmonizer, HYCOMLoader

    threshold_path = Path("data/processed/hycom_sst_threshold.zarr")
    if not threshold_path.exists():
        raise FileNotFoundError(
            "ERROR: hycom_sst_threshold.zarr not found. "
            "Run scripts/compute_hycom_climatology.py first."
        )
    threshold = xr.open_zarr(str(threshold_path))["threshold"]

    harvester = ERA5Harvester()
    harvester.authenticate()
    loader    = HYCOMLoader()
    harmonizer = DataHarmonizer()

    print("Fetching ERA5 train (2018)...")
    wn2_train  = harvester.fetch(*ERA5_TRAIN_PERIOD, GoM_BBOX)
    hycom_train = loader.fetch_tile(*ERA5_TRAIN_PERIOD, GoM_BBOX)
    merged_train = harmonizer.harmonize(wn2_train, hycom_train)
    # harmonize() calls expand_and_perturb() automatically when member=1
    hycom_t_train, wn2_t_train, label_t_train = build_tensors(merged_train, threshold)

    print("Fetching ERA5 val (2019)...")
    wn2_val   = harvester.fetch(*ERA5_VAL_PERIOD, GoM_BBOX)
    hycom_val_ds = loader.fetch_tile(*ERA5_VAL_PERIOD, GoM_BBOX)
    merged_val = harmonizer.harmonize(wn2_val, hycom_val_ds)
    hycom_t_val, wn2_t_val, label_t_val = build_tensors(merged_val, threshold)

    return (hycom_t_train, wn2_t_train, label_t_train,
            hycom_t_val, wn2_t_val, label_t_val,
            merged_val, threshold)
```

New:
```python
def load_real_data():
    """
    Load ERA5 + HYCOM from GCS for train (2022) and val (2023) periods.

    All data was pre-fetched to GCS by scripts/run_data_prep.py.
    Requires env var MHW_GCS_BUCKET (e.g. "gs://my-bucket").
    No live OPeNDAP or GEE calls are made here.

    harmonize() detects member=1 (ERA5 is deterministic) and calls
    expand_and_perturb() automatically to produce 64 synthetic members.

    Returns
    -------
    Tuple of (hycom_t_train, wn2_t_train, label_t_train,
              hycom_t_val, wn2_t_val, label_t_val,
              merged_val, threshold)
    """
    import os
    from src.ingestion.harvester import DataHarmonizer

    bucket = os.environ.get("MHW_GCS_BUCKET", "").rstrip("/")
    if not bucket:
        raise RuntimeError(
            "MHW_GCS_BUCKET env var not set. "
            "Run scripts/run_data_prep.py on GCP first, then set this variable."
        )

    harmonizer = DataHarmonizer()
    threshold  = xr.open_zarr(f"{bucket}/hycom/climatology/")["sst_threshold_90"]

    print("Loading ERA5 train (2022) from GCS...")
    era5_train  = xr.open_zarr(f"{bucket}/era5/2022/", chunks="auto")
    hycom_train = xr.open_zarr(f"{bucket}/hycom/tiles/2022/", chunks="auto")
    merged_train = harmonizer.harmonize(era5_train, hycom_train)
    hycom_t_train, wn2_t_train, label_t_train = build_tensors(merged_train, threshold)

    print("Loading ERA5 val (2023) from GCS...")
    era5_val   = xr.open_zarr(f"{bucket}/era5/2023/", chunks="auto")
    hycom_val  = xr.open_zarr(f"{bucket}/hycom/tiles/2023/", chunks="auto")
    merged_val = harmonizer.harmonize(era5_val, hycom_val)
    hycom_t_val, wn2_t_val, label_t_val = build_tensors(merged_val, threshold)

    return (hycom_t_train, wn2_t_train, label_t_train,
            hycom_t_val, wn2_t_val, label_t_val,
            merged_val, threshold)
```

- [ ] **Step 3: Update `config` dict in `train_era5.py` `main()` to use new constant names**

Old (~line 140-143):
```python
        "train_period": ERA5_TRAIN_PERIOD,
        "val_period": ERA5_VAL_PERIOD,
```

New:
```python
        "train_period": TRAIN_PERIOD,
        "val_period": VAL_PERIOD,
```

- [ ] **Step 4: Verify dry-run still works (no GCS calls in dry-run path)**

```bash
conda run -n mhw-risk python scripts/train_era5.py --dry-run --epochs 2 2>&1 | tail -10
```

Expected: 2 epochs complete, artifacts saved, no errors

- [ ] **Step 5: Commit**

```bash
git add scripts/train_era5.py
git commit -m "feat: train_era5.py load_real_data() reads from GCS; align to 2022/2023 periods"
```

---

## Task 5: GCS-only `load_real_data()` in `train_wn2.py`

**Files:**
- Modify: `scripts/train_wn2.py` — same pattern as Task 4

- [ ] **Step 1: Update the import line in `train_wn2.py`**

Old (line ~45-48):
```python
from _train_utils import (
    GoM_BBOX, WN2_TRAIN_PERIOD, WN2_VAL_PERIOD, N_MEMBERS, SEQ_LEN,
    build_tensors, run_svar_inference, save_plots,
)
```

New:
```python
from _train_utils import (
    GoM_BBOX, TRAIN_PERIOD, VAL_PERIOD, N_MEMBERS, SEQ_LEN,
    build_tensors, run_svar_inference, save_plots,
)
```

- [ ] **Step 2: Replace `load_real_data()` in `train_wn2.py`**

Old (lines ~62-112):
```python
def load_real_data():
    """
    Fetch WeatherNext 2 + HYCOM for train (2022) and val (2023) periods.
    ...
    """
    import os
    from src.ingestion.harvester import WeatherNext2Harvester, DataHarmonizer, HYCOMLoader

    threshold_path = Path("data/processed/hycom_sst_threshold.zarr")
    if not threshold_path.exists():
        raise FileNotFoundError(
            "ERROR: hycom_sst_threshold.zarr not found. "
            "Run scripts/compute_hycom_climatology.py first."
        )
    threshold = xr.open_zarr(str(threshold_path))["threshold"]

    gcs_bucket = os.environ["GCS_BUCKET"]  # set in environment; see mondal-mhw-gcp-info.md
    key        = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    harvester  = WeatherNext2Harvester(gcs_bucket=gcs_bucket, service_account_key=key)
    harvester.authenticate()
    loader     = HYCOMLoader()
    harmonizer = DataHarmonizer()

    print("Fetching WeatherNext 2 train (2022)...")
    wn2_train   = harvester.fetch_ensemble(*WN2_TRAIN_PERIOD, GoM_BBOX)
    hycom_train = loader.fetch_tile(*WN2_TRAIN_PERIOD, GoM_BBOX)
    merged_train = harmonizer.harmonize(wn2_train, hycom_train)
    # WN2 returns member=64 — harmonize() skips expand_and_perturb automatically
    hycom_t_train, wn2_t_train, label_t_train = build_tensors(merged_train, threshold)

    print("Fetching WeatherNext 2 val (2023)...")
    wn2_val      = harvester.fetch_ensemble(*WN2_VAL_PERIOD, GoM_BBOX)
    hycom_val_ds = loader.fetch_tile(*WN2_VAL_PERIOD, GoM_BBOX)
    merged_val   = harmonizer.harmonize(wn2_val, hycom_val_ds)
    hycom_t_val, wn2_t_val, label_t_val = build_tensors(merged_val, threshold)

    return (hycom_t_train, wn2_t_train, label_t_train,
            hycom_t_val, wn2_t_val, label_t_val,
            merged_val, threshold)
```

New:
```python
def load_real_data():
    """
    Load WeatherNext 2 + HYCOM from GCS for train (2022) and val (2023) periods.

    All data was pre-fetched to GCS by scripts/run_data_prep.py.
    Requires env var MHW_GCS_BUCKET (e.g. "gs://my-bucket").
    No live OPeNDAP or GEE calls are made here.

    WN2 tiles live under the existing WeatherNext2Harvester cache path:
    gs://bucket/weathernext2/cache/wn2_YYYY-MM-DD_YYYY-MM-DD_m64.zarr
    HYCOM tiles are shared with the ERA5 training run.

    Returns
    -------
    Tuple of (hycom_t_train, wn2_t_train, label_t_train,
              hycom_t_val, wn2_t_val, label_t_val,
              merged_val, threshold)
    """
    import os
    from src.ingestion.harvester import DataHarmonizer

    bucket = os.environ.get("MHW_GCS_BUCKET", "").rstrip("/")
    if not bucket:
        raise RuntimeError(
            "MHW_GCS_BUCKET env var not set. "
            "Run scripts/run_data_prep.py on GCP first, then set this variable."
        )

    harmonizer = DataHarmonizer()
    threshold  = xr.open_zarr(f"{bucket}/hycom/climatology/")["sst_threshold_90"]

    print("Loading WN2 train (2022) from GCS...")
    wn2_train   = xr.open_zarr(
        f"{bucket}/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr", chunks="auto"
    )
    hycom_train = xr.open_zarr(f"{bucket}/hycom/tiles/2022/", chunks="auto")
    merged_train = harmonizer.harmonize(wn2_train, hycom_train)
    hycom_t_train, wn2_t_train, label_t_train = build_tensors(merged_train, threshold)

    print("Loading WN2 val (2023) from GCS...")
    wn2_val   = xr.open_zarr(
        f"{bucket}/weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr", chunks="auto"
    )
    hycom_val = xr.open_zarr(f"{bucket}/hycom/tiles/2023/", chunks="auto")
    merged_val = harmonizer.harmonize(wn2_val, hycom_val)
    hycom_t_val, wn2_t_val, label_t_val = build_tensors(merged_val, threshold)

    return (hycom_t_train, wn2_t_train, label_t_train,
            hycom_t_val, wn2_t_val, label_t_val,
            merged_val, threshold)
```

- [ ] **Step 3: Update `config` dict in `train_wn2.py` `main()` to use new constant names**

Find the config dict (~lines 130-145) and replace:
```python
        "train_period": WN2_TRAIN_PERIOD,
        "val_period": WN2_VAL_PERIOD,
```
with:
```python
        "train_period": TRAIN_PERIOD,
        "val_period": VAL_PERIOD,
```

- [ ] **Step 4: Verify dry-run still works**

```bash
conda run -n mhw-risk python scripts/train_wn2.py --dry-run --epochs 2 2>&1 | tail -10
```

Expected: 2 epochs complete, no errors

- [ ] **Step 5: Run full test suite**

```bash
conda run -n mhw-risk pytest tests/ -v 2>&1 | tail -10
```

Expected: all tests passing

- [ ] **Step 6: Commit**

```bash
git add scripts/train_wn2.py
git commit -m "feat: train_wn2.py load_real_data() reads from GCS; remove GCS_BUCKET dependency"
```

---

## Task 6: `scripts/run_data_prep.py`

**Files:**
- Create: `scripts/run_data_prep.py`

- [ ] **Step 1: Create `scripts/run_data_prep.py`**

```python
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
    [1/5] HYCOM tiles 2022 -> gs://bucket/hycom/tiles/2022/  OK
    [2/5] HYCOM tiles 2023 -> gs://bucket/hycom/tiles/2023/  OK
    [3/5] HYCOM climatology -> gs://bucket/hycom/climatology/  OK
    [4/5] ERA5 tiles 2022 -> gs://bucket/era5/2022/  OK
    [5/5] ERA5 tiles 2023 -> gs://bucket/era5/2023/  OK
    [WN2] gs://bucket/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr  OK
    [WN2] gs://bucket/weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr  OK
    Data prep complete.

Total estimated runtime: 3-5 hours on a spot e2-standard-2 (mostly OPeNDAP network I/O).
Estimated cost: ~$0.05-0.09 on spot e2-standard-2 at $0.017/hr.
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
from src.ingestion.harvester import HYCOMLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Gulf of Maine bbox — must match GoM_BBOX in scripts/_train_utils.py
BBOX  = (-71.0, 41.0, -66.0, 45.0)
YEARS = (2022, 2023)


def _gcs_exists(fs: gcsfs.GCSFileSystem, gcs_uri: str) -> bool:
    """Return True if gcs_uri exists, stripping gs:// prefix for gcsfs."""
    return fs.exists(gcs_uri.removeprefix("gs://"))


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
    if not _gcs_exists(fs, clim_uri):
        sst_years = []
        for year in YEARS:
            tile_uri = f"{bucket}/hycom/tiles/{year}/"
            ds  = xr.open_zarr(tile_uri, chunks="auto")
            sst = ds["water_temp"].isel(depth=0).resample(time="1D").mean()
            sst.load()
            sst_years.append(sst)
        sst_all   = xr.concat(sst_years, dim="time")  # (time=730, lat, lon)
        threshold = compute_climatology(sst_all, percentile=90.0)
        threshold.to_dataset(name="sst_threshold_90").to_zarr(clim_uri, mode="w")
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

    # ---- WN2 verification ----
    # WN2 is fetched separately by WeatherNext2Harvester.fetch_ensemble() (existing behavior).
    # This step only checks that expected cache paths exist and warns if not.
    for year in YEARS:
        wn2_path = f"{bucket}/weathernext2/cache/wn2_{year}-01-01_{year}-12-31_m64.zarr"
        exists = _gcs_exists(fs, wn2_path)
        status = "OK" if exists else "WARNING — not found; run WeatherNext2Harvester.fetch_ensemble() for this year"
        print(f"[WN2] {wn2_path}  {status}", flush=True)

    print("Data prep complete.", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script is importable (syntax check)**

```bash
conda run -n mhw-risk python -c "import scripts.run_data_prep" 2>&1 || \
conda run -n mhw-risk python -c "
import sys; sys.path.insert(0, 'scripts')
import importlib.util
spec = importlib.util.spec_from_file_location('run_data_prep', 'scripts/run_data_prep.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('Import OK')
"
```

Expected: `Import OK`

- [ ] **Step 3: Verify `--help` equivalent (error on missing env var)**

```bash
conda run -n mhw-risk python scripts/run_data_prep.py 2>&1 | head -5
```

Expected: `RuntimeError: MHW_GCS_BUCKET env var not set.`

- [ ] **Step 4: Commit**

```bash
git add scripts/run_data_prep.py
git commit -m "feat: add run_data_prep.py — idempotent GCS data prep orchestrator for spot GCE"
```

---

## Task 7: GCE runbook and final cleanup

**Files:**
- Create: `docs/gcp-data-prep-runbook.md`

- [ ] **Step 1: Create `docs/gcp-data-prep-runbook.md`**

```markdown
# GCP Data Prep Runbook

One-time job to pre-fetch all training data to GCS. Run before any real training run.

## Prerequisites

- `gcloud` CLI authenticated (`gcloud auth login` or service account)
- `GOOGLE_APPLICATION_CREDENTIALS` pointing to the service account key
- `MHW_GCS_BUCKET` set to your GCS bucket URI (see `mondal-mhw-gcp-info.md`)
- GEE whitelist approved for WeatherNext 2 (already done — see recent actions log)

## 1. Create the spot VM

```bash
gcloud compute instances create mhw-data-prep \
  --zone=us-central1-a \
  --machine-type=e2-standard-2 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-standard \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --scopes=cloud-platform
```

## 2. SSH into the VM

```bash
gcloud compute ssh mhw-data-prep --zone=us-central1-a
```

## 3. Set up the environment on the VM

```bash
# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh

# Clone repo and create env
git clone <your-repo-url> mhw-risk-profiler
cd mhw-risk-profiler
conda env create -f environment.yml  # or: conda create -n mhw-risk python=3.11 && pip install -r requirements.txt

# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
export MHW_GCS_BUCKET=gs://your-bucket-name   # see mondal-mhw-gcp-info.md

# Copy credentials to VM (from local machine, in a separate terminal):
# gcloud compute scp ~/.config/gcp-keys/mhw-harvester.json mhw-data-prep:~/.config/gcp-keys/ --zone=us-central1-a
```

## 4. Run the data prep job

```bash
conda run -n mhw-risk python scripts/run_data_prep.py 2>&1 | tee data_prep.log
```

Estimated runtime: 3-5 hours. The job is idempotent — if the VM is preempted, re-run the
same command and it will skip completed steps.

## 5. Verify outputs

```bash
conda run -n mhw-risk python -c "
import os, xarray as xr
b = os.environ['MHW_GCS_BUCKET']
for path in [
    f'{b}/hycom/tiles/2022/', f'{b}/hycom/tiles/2023/',
    f'{b}/hycom/climatology/',
    f'{b}/era5/2022/', f'{b}/era5/2023/',
]:
    ds = xr.open_zarr(path)
    print(path, '->', dict(ds.sizes))
"
```

Expected: each path prints non-empty dims.

## 6. Delete the VM

```bash
gcloud compute instances delete mhw-data-prep --zone=us-central1-a
```

## 7. Run real training (from local machine or separate GCE GPU VM)

```bash
export MHW_GCS_BUCKET=gs://your-bucket-name
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
conda run -n mhw-risk python scripts/train_era5.py --epochs 50
conda run -n mhw-risk python scripts/train_wn2.py --epochs 50
```

## Notes

- WN2 tiles (`weathernext2/cache/wn2_2022-*.zarr`, `wn2_2023-*.zarr`) must be
  pre-fetched separately by running a `WeatherNext2Harvester.fetch_ensemble()` call
  before training. The data prep job warns if these are missing but does not fetch them
  (WN2 GEE access uses a different auth flow).
- `scripts/compute_hycom_climatology.py` is superseded by this pipeline. Delete it
  after verifying the GCS climatology path is correct.
```

- [ ] **Step 2: Run full test suite one final time**

```bash
conda run -n mhw-risk pytest tests/ -v 2>&1 | tail -15
```

Expected: all tests passing

- [ ] **Step 3: Commit**

```bash
git add docs/gcp-data-prep-runbook.md
git commit -m "docs: add GCP data prep runbook with VM setup and verification commands"
```
