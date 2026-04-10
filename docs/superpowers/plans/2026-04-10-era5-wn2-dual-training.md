# ERA5 / WeatherNext 2 Dual Training & XAI Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train two independent `MHWRiskModel` instances — one on ERA5 proxy (synthetic ensemble), one on WeatherNext 2 real FGN ensemble — producing per-grid-cell SVaR Zarr outputs and a per-season Captum IG attribution comparison JSON.

**Architecture:** Sequential execution (ERA5 first, WN2 second, XAI last). Shared `scripts/_train_utils.py` module holds `build_tensors()`, `run_svar_inference()`, and `save_plots()` used by both training scripts. `compare_xai.py` loads both weight files and runs Captum IG per season. No changes to `MHWRiskModel`, `svar.py`, `mhw_detection.py`, `cnn1d.py`, or `transformer.py`.

**Tech Stack:** `earthengine-api`, `xarray>=2024.2.0`, `torch>=2.2.0`, `captum>=0.7.0`, `matplotlib>=3.8.0`, `numpy>=1.26.0`, `pytest>=8.0.0`.

**Supersedes:** `docs/superpowers/plans/2026-04-03-era5-proxy-training.md` — that plan is replaced in full by this one.

**Spec:** `docs/superpowers/specs/2026-04-10-era5-wn2-xai-comparison-design.md`

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `src/ingestion/era5_harvester.py` | `ERA5Harvester`: fetch ECMWF/ERA5/DAILY from GEE |
| Modify | `src/ingestion/harvester.py` | Add `DataHarmonizer.expand_and_perturb()` static method |
| Modify | `src/ingestion/__init__.py` | Export `ERA5Harvester` |
| Modify | `requirements.txt` | Add `matplotlib>=3.8.0` |
| Create | `scripts/_train_utils.py` | `build_tensors()`, `run_svar_inference()`, `save_plots()` |
| Create | `scripts/train_era5.py` | ERA5 training loop, all artifacts |
| Create | `scripts/train_wn2.py` | WN2 training loop, all artifacts (symmetric) |
| Create | `scripts/compare_xai.py` | Per-season Captum IG comparison |
| Create | `scripts/scope_wn2_asset.py` | One-shot WN2 GEE schema inspection |
| Create | `tests/test_era5_harvester.py` | Offline unit tests: band mapping, shape, noise |
| Create | `tests/test_train_utils.py` | Unit tests for build_tensors and shared utilities |

---

## Domain Constants (used throughout)

```python
GoM_BBOX     = (-71.0, 41.0, -66.0, 45.0)   # (lon_min, lat_min, lon_max, lat_max)
TRAIN_PERIOD = ("2018-01-01", "2018-12-31")
VAL_PERIOD   = ("2019-01-01", "2019-12-31")
SEQ_LEN      = 90     # atmospheric sequence length [days]
N_MEMBERS    = 64     # ensemble members
N_LAT, N_LON = 17, 21 # Gulf of Maine grid cells at 0.25-degree
HYCOM_VARS   = ["water_temp", "salinity", "water_u", "water_v"]
WN2_VARS     = ["sea_surface_temperature", "2m_temperature",
                "10m_u_component_of_wind", "10m_v_component_of_wind",
                "mean_sea_level_pressure"]
SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
```

---

## Task 0: WN2 GEE Asset Scoping

**Files:**
- Create: `scripts/scope_wn2_asset.py`

This task is about gathering information, not writing production code. The output is a printed report that answers: is WN2 organized as a daily time series (like ERA5) or as forecast runs with init dates and lead times? This determines how `train_wn2.py` structures its data.

- [ ] **Step 1: Write `scripts/scope_wn2_asset.py`**

```python
#!/usr/bin/env python3
"""
scope_wn2_asset.py — Inspect WeatherNext 2 GEE asset schema.

Run once to understand the time axis structure before implementing train_wn2.py.
Requires GEE authentication: export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json

Usage:
    conda run -n mhw-risk python scripts/scope_wn2_asset.py
"""
import os
import ee

KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if KEY:
    import json
    with open(KEY) as fh:
        _kd = json.load(fh)
    creds = ee.ServiceAccountCredentials(email=_kd["client_email"], key_file=KEY)
    ee.Initialize(credentials=creds)
else:
    ee.Initialize()

ASSET = "projects/gcp-public-data-weathernext/assets/weathernext_2_0_0"
col = ee.ImageCollection(ASSET)

print("=== WeatherNext 2 Asset Schema ===\n")
print(f"Total images: {col.size().getInfo()}")

# Inspect first image
first = ee.Image(col.first())
print(f"\nFirst image date:       {first.date().format('YYYY-MM-dd').getInfo()}")
print(f"First image band count: {first.bandNames().size().getInfo()}")
print(f"First 20 band names:    {first.bandNames().slice(0, 20).getInfo()}")
print(f"First image properties: {first.propertyNames().getInfo()}")

# Check if 'system:time_start' is present (daily time series) or init+lead structure
props = first.toDictionary(first.propertyNames()).getInfo()
for k in ["system:time_start", "initialization_time", "lead_time", "forecast_hour",
          "number_of_members", "forecast_reference_time"]:
    print(f"  {k}: {props.get(k, '<not present>')}")

# Inspect last image to determine coverage end date
last = ee.Image(col.sort("system:time_start", False).first())
print(f"\nLast image date:  {last.date().format('YYYY-MM-dd').getInfo()}")

# Sample 5 consecutive images to check time spacing
print("\nFirst 5 image dates (time axis structure):")
imgs = col.sort("system:time_start").toList(5)
for i in range(5):
    img = ee.Image(imgs.get(i))
    print(f"  [{i}] {img.date().format('YYYY-MM-dd HH:mm').getInfo()}")

print("\n=== Interpretation guide ===")
print("If dates are daily (YYYY-MM-DD 00:00): daily time series → treat like ERA5.")
print("If dates are 6-hourly or include HH:MM: sub-daily → must resample to daily.")
print("If 'initialization_time' exists: forecast run structure → need lead-time handling.")
```

- [ ] **Step 2: Run the scoping script**

```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
conda run -n mhw-risk python scripts/scope_wn2_asset.py 2>&1 | tee docs/superpowers/specs/wn2_asset_schema.txt
```

Expected: printed schema with time axis structure, band count, property names.

- [ ] **Step 3: Update the spec with findings**

Open `docs/superpowers/specs/2026-04-10-era5-wn2-xai-comparison-design.md` and add a "Phase 0 Findings" section recording:
- Time axis type (daily vs forecast runs)
- Coverage dates (first and last image dates)
- Band naming pattern (e.g. `sea_surface_temperature_member_0` or `sea_surface_temperature_00`)
- Whether `train_wn2.py` needs any date filtering or resampling beyond what `WeatherNext2Harvester.fetch_ensemble()` already handles

- [ ] **Step 4: Commit**

```bash
git add scripts/scope_wn2_asset.py docs/superpowers/specs/wn2_asset_schema.txt docs/superpowers/specs/2026-04-10-era5-wn2-xai-comparison-design.md
git commit -m "chore: scope WN2 GEE asset schema, document time axis structure"
```

---

## Task 1: Add matplotlib to requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add matplotlib**

Add this line to `requirements.txt` under `# -- Scientific utilities`:

```
matplotlib>=3.8.0
```

- [ ] **Step 2: Install in the conda environment**

```bash
conda run -n mhw-risk pip install "matplotlib>=3.8.0"
```

Expected: `Successfully installed matplotlib-3.x.x` (or "already satisfied").

- [ ] **Step 3: Verify import**

```bash
conda run -n mhw-risk python -c "import matplotlib; print(matplotlib.__version__)"
```

Expected: version string printed, no ImportError.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add matplotlib>=3.8.0 to requirements"
```

---

## Task 2: ERA5Harvester + Unit Tests

**Files:**
- Create: `src/ingestion/era5_harvester.py`
- Create: `tests/test_era5_harvester.py`
- Modify: `src/ingestion/__init__.py`

ERA5 band → WN2-compatible variable name mapping:

| ERA5 Band | Output Variable | Units |
|-----------|----------------|-------|
| `mean_2m_air_temperature` | `2m_temperature` | K |
| `u_component_of_wind_10m` | `10m_u_component_of_wind` | m/s |
| `v_component_of_wind_10m` | `10m_v_component_of_wind` | m/s |
| `mean_sea_level_pressure` | `mean_sea_level_pressure` | Pa |
| `sea_surface_temperature` | `sea_surface_temperature` | K |

- [ ] **Step 1: Write failing tests**

```python
# tests/test_era5_harvester.py
"""Offline unit tests for ERA5Harvester — no GEE calls."""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import xarray as xr

from src.ingestion.era5_harvester import ERA5Harvester, ERA5_BANDS


def _make_fake_ds(n_time=10, n_lat=4, n_lon=5):
    """Build a synthetic xr.Dataset with ERA5 band names and member=1."""
    data = {
        band: xr.DataArray(
            np.random.rand(1, n_time, n_lat, n_lon),
            dims=["member", "time", "latitude", "longitude"],
        )
        for band in ERA5_BANDS
    }
    return xr.Dataset(data)


def test_band_mapping():
    """All 5 WN2-compatible variable names appear in the output Dataset."""
    expected = set(ERA5_BANDS.values())
    ds = _make_fake_ds()
    ds = ds.rename(ERA5_BANDS)
    assert set(ds.data_vars) == expected


def test_output_shape():
    """Output dims are (member=1, time, latitude, longitude)."""
    ds = _make_fake_ds(n_time=10, n_lat=4, n_lon=5)
    assert ds.dims["member"] == 1
    assert ds.dims["time"] == 10
    assert ds.dims["latitude"] == 4
    assert ds.dims["longitude"] == 5


def test_noise_spread():
    """After expand_and_perturb(), ensemble spread is non-degenerate."""
    from src.ingestion.harvester import DataHarmonizer

    ds = _make_fake_ds(n_time=10, n_lat=4, n_lon=5)
    ds = ds.rename(ERA5_BANDS)

    perturbed = DataHarmonizer.expand_and_perturb(ds, n_members=64, seed=42)

    assert perturbed.dims["member"] == 64

    sst_std = float(perturbed["sea_surface_temperature"].std("member").mean())
    assert sst_std == pytest.approx(0.5, rel=0.5), (
        f"SST std {sst_std:.4f} not within 50% of target 0.5 K"
    )

    sdd_vals = perturbed["sea_surface_temperature"].values.ravel()
    assert np.quantile(sdd_vals, 0.95) > np.quantile(sdd_vals, 0.50), (
        "Ensemble is degenerate: Q95 ≤ Q50"
    )
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n mhw-risk pytest tests/test_era5_harvester.py -v 2>&1 | head -30
```

Expected: `ImportError` or `ModuleNotFoundError` — `era5_harvester` does not exist yet.

- [ ] **Step 3: Implement `src/ingestion/era5_harvester.py`**

```python
"""
era5_harvester.py — ERA5 daily data harvester for MHW proxy training
=====================================================================
Fetches the ECMWF/ERA5/DAILY ImageCollection from Google Earth Engine
and returns an xr.Dataset with variable names matching WeatherNext 2,
making it a drop-in replacement for WeatherNext2Harvester in the training pipeline.

ERA5 is a deterministic reanalysis (1 member). The single-member output feeds
DataHarmonizer.expand_and_perturb() which broadcasts to 64 synthetic members,
preserving the (member, ...) tensor contract required by MHWRiskModel.

Physical note
-------------
ERA5 and WeatherNext 2 share the same 5 atmospheric variables at 0.25-degree
resolution in matching SI units. ERA5 is the ECMWF reanalysis of historical
atmospheric state — physically consistent and freely available on GEE without
a whitelist. The variable naming difference between the two products is the only
incompatibility; this module resolves it via ERA5_BANDS renaming.

Dependencies
------------
    earthengine-api>=0.1.390, xarray>=2024.2.0, numpy>=1.26.0
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Mapping: ERA5 band name → WN2-compatible variable name
# Source for band names: GEE catalog — ECMWF/ERA5/DAILY
ERA5_BANDS: dict[str, str] = {
    "mean_2m_air_temperature":    "2m_temperature",
    "u_component_of_wind_10m":    "10m_u_component_of_wind",
    "v_component_of_wind_10m":    "10m_v_component_of_wind",
    "mean_sea_level_pressure":    "mean_sea_level_pressure",
    "sea_surface_temperature":    "sea_surface_temperature",
}

GEE_COLLECTION = "ECMWF/ERA5/DAILY"


class ERA5Harvester:
    """
    Fetches ECMWF ERA5 daily reanalysis from Google Earth Engine for a spatial
    bounding box and date range, renaming variables to match WeatherNext 2 output.

    Output is a single-member (member=1) xr.Dataset. Downstream code must call
    DataHarmonizer.expand_and_perturb() to broadcast to 64 synthetic members
    before passing to MHWRiskModel.

    Parameters
    ----------
    service_account_key : str, optional
        Path to a GCP service account JSON key file.
        If None, falls back to Application Default Credentials.
    """

    def __init__(self, service_account_key: Optional[str] = None) -> None:
        self._key = service_account_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self._initialized = False

    def authenticate(self) -> None:
        """
        Authenticate with Google Earth Engine.

        Uses service account credentials if _key is set, otherwise falls back
        to Application Default Credentials (ADC). Idempotent — safe to call
        multiple times.
        """
        import ee

        if not self._initialized:
            if self._key:
                import json
                with open(self._key) as fh:
                    kd = json.load(fh)
                creds = ee.ServiceAccountCredentials(
                    email=kd["client_email"], key_file=self._key
                )
                ee.Initialize(credentials=creds)
            else:
                ee.Initialize()
            self._initialized = True
            logger.info("ERA5Harvester: GEE authentication successful.")

    def fetch(
        self,
        start_date: str,
        end_date: str,
        bbox: tuple[float, float, float, float],
    ) -> xr.Dataset:
        """
        Query ECMWF/ERA5/DAILY from GEE for the specified date range and bounding box.

        Parameters
        ----------
        start_date : str
            ISO 8601 start date, e.g. "2018-01-01".
        end_date : str
            ISO 8601 end date, inclusive, e.g. "2018-12-31".
        bbox : tuple of float
            (lon_min, lat_min, lon_max, lat_max) in decimal degrees WGS84.

        Returns
        -------
        ds : xr.Dataset
            Dimensions: (member=1, time, latitude, longitude)
            Variables: WN2_VARIABLES — same names as WeatherNext2Harvester.fetch_ensemble()
            Units: SI (K for temperature, Pa for pressure, m/s for wind)
            The member=1 dimension is intentional — call DataHarmonizer.expand_and_perturb()
            downstream to broadcast to 64 synthetic ensemble members.
        """
        import ee

        if not self._initialized:
            raise RuntimeError("Call authenticate() before fetch().")

        lon_min, lat_min, lon_max, lat_max = bbox
        region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

        logger.info(
            "Fetching ERA5 daily: %s to %s, bbox=%s", start_date, end_date, bbox
        )

        collection = (
            ee.ImageCollection(GEE_COLLECTION)
            .filterDate(start_date, end_date)
            .select(list(ERA5_BANDS.keys()))
        )

        n_images = collection.size().getInfo()
        if n_images == 0:
            raise ValueError(
                f"No ERA5 images found for {start_date} to {end_date}. "
                "Check GEE asset availability for this period."
            )

        logger.info("ERA5: %d daily images found.", n_images)

        image_list = collection.sort("system:time_start").toList(n_images)
        time_coords: list = []
        data_by_var: dict[str, list] = {band: [] for band in ERA5_BANDS}

        for t_idx in range(n_images):
            img = ee.Image(image_list.get(t_idx))
            sample = img.sampleRectangle(
                region=region, defaultValue=0
            ).getInfo()
            props = sample["properties"]
            for band in ERA5_BANDS:
                data_by_var[band].append(np.array(props[band]))  # (lat, lon)
            date_str = img.date().format("YYYY-MM-dd").getInfo()
            time_coords.append(np.datetime64(date_str))

        # Stack into (time, lat, lon) arrays, then rename and add member dim
        ds_vars = {}
        for era5_band, wn2_name in ERA5_BANDS.items():
            arr = np.stack(data_by_var[era5_band], axis=0)  # (time, lat, lon)
            arr = arr[np.newaxis, ...]                       # (member=1, time, lat, lon)
            first_img_arr = data_by_var[era5_band][0]
            n_lat, n_lon = first_img_arr.shape
            ds_vars[wn2_name] = xr.DataArray(
                arr,
                dims=["member", "time", "latitude", "longitude"],
            )

        ds = xr.Dataset(ds_vars).assign_coords(
            member=[0],
            time=time_coords,
        )

        logger.info("ERA5 dataset built: %s", ds)
        return ds
```

- [ ] **Step 4: Add `expand_and_perturb()` to `DataHarmonizer` in `src/ingestion/harvester.py`**

Add `NOISE_SIGMAS` constant after the existing constants block, then add the static method inside `DataHarmonizer`:

```python
# Add after the existing module constants (after TARGET_DEPTHS_M)
NOISE_SIGMAS: dict[str, float] = {
    "sea_surface_temperature":    0.5,   # K  — primary MHW driver
    "2m_temperature":             0.5,   # K  — coherent with SST
    "10m_u_component_of_wind":    0.3,   # m/s
    "10m_v_component_of_wind":    0.3,   # m/s
    "mean_sea_level_pressure":    50.0,  # Pa — ~0.5 hPa synoptic noise
}
```

Add this static method inside `DataHarmonizer` (before `harmonize()`):

```python
@staticmethod
def expand_and_perturb(
    ds: xr.Dataset,
    n_members: int = 64,
    seed: int = 42,
) -> xr.Dataset:
    """
    Broadcast a single-member ERA5 Dataset to n_members synthetic members
    by injecting independent Gaussian noise into each atmospheric variable.

    Parameters
    ----------
    ds : xr.Dataset
        ERA5 dataset with member=1 dimension.
        Must contain variables matching WN2_VARIABLES naming convention.
    n_members : int
        Target number of synthetic ensemble members. Default 64 matches WN2.
    seed : int
        Base random seed. Member i uses seed + i for reproducibility.
        Changing seed produces a different but equally valid synthetic ensemble.

    Returns
    -------
    ds_perturbed : xr.Dataset
        Same variables and spatial/temporal dimensions but member=n_members.
        Each member has independent Gaussian noise added to each variable.
        Noise σ values calibrated to match published WN2 intra-ensemble spread.

    Physical rationale
    ------------------
    ERA5 is deterministic. To use it with SVaR (which requires ensemble spread),
    we inject per-member Gaussian noise. The σ values are chosen to approximate
    the intra-ensemble spread documented for WeatherNext 2. This is a proxy:
    real WN2 spread is non-Gaussian and temporally correlated (FGN), but
    Gaussian noise is sufficient for training because the model learns from
    the physics SDD label, not from the ensemble structure itself.
    """
    # NOISE_SIGMAS is defined at module level in harvester.py — reference directly,
    # no import needed (this static method lives in the same module).

    # Broadcast single member across n_members dimension
    ds_broadcast = ds.isel(member=0).expand_dims(member=range(n_members))

    # Build list of per-member Datasets with independent noise
    member_datasets = []
    for i in range(n_members):
        rng = np.random.default_rng(seed + i)
        ds_m = ds_broadcast.isel(member=i).copy(deep=True)
        for var, sigma in NOISE_SIGMAS.items():
            if var in ds_m:
                noise = rng.normal(0.0, sigma, ds_m[var].shape).astype(np.float32)
                ds_m[var] = ds_m[var] + noise
        member_datasets.append(ds_m)

    ds_perturbed = xr.concat(member_datasets, dim="member")
    ds_perturbed["member"] = np.arange(n_members)
    return ds_perturbed
```

Update `harmonize()` to call `expand_and_perturb()` automatically when member=1. Add this block **after** the `wn2_regridded = wn2_ds.interp(...)` line and **before** the HYCOM regridding:

```python
# Auto-expand single-member input (ERA5 proxy path)
if wn2_regridded.dims.get("member", 1) == 1:
    logger.info("member=1 detected — expanding to 64 synthetic members.")
    wn2_regridded = DataHarmonizer.expand_and_perturb(wn2_regridded)
```

- [ ] **Step 5: Export `ERA5Harvester` from `src/ingestion/__init__.py`**

Open `src/ingestion/__init__.py` and add:

```python
from .era5_harvester import ERA5Harvester
```

- [ ] **Step 6: Run the three tests — all must pass**

```bash
conda run -n mhw-risk pytest tests/test_era5_harvester.py -v
```

Expected output:
```
PASSED tests/test_era5_harvester.py::test_band_mapping
PASSED tests/test_era5_harvester.py::test_output_shape
PASSED tests/test_era5_harvester.py::test_noise_spread
3 passed in X.Xs
```

- [ ] **Step 7: Commit**

```bash
git add src/ingestion/era5_harvester.py src/ingestion/harvester.py src/ingestion/__init__.py tests/test_era5_harvester.py
git commit -m "feat: add ERA5Harvester and DataHarmonizer.expand_and_perturb"
```

---

## Task 3: Shared Training Utilities

**Files:**
- Create: `scripts/_train_utils.py`
- Create: `tests/test_train_utils.py`

`_train_utils.py` holds the three functions used by both `train_era5.py` and `train_wn2.py`: `build_tensors()`, `save_plots()`, and `run_svar_inference()`. Keeping them here avoids duplication and ensures both training scripts produce identical artifact formats.

**Key data model:**

After `DataHarmonizer.harmonize()`, the merged Dataset has:
- WN2 variables: dims `(member, time, latitude, longitude)` — no depth
- HYCOM variables: dims `(member, time, depth, latitude, longitude)` — HYCOM broadcast across members (all 64 slices are identical; HYCOM is deterministic)
- Coordinate names: `latitude` and `longitude` (not `lat`/`lon`) — set by `harmonize()`'s `rename()` call

SST from WN2/ERA5 is in Kelvin. `accumulate_sdd()` expects °C. Always subtract 273.15 before passing to `compute_mhw_mask()` and `accumulate_sdd()`.

- [ ] **Step 1: Write failing tests for `build_tensors`**

```python
# tests/test_train_utils.py
"""Unit tests for scripts/_train_utils.py — no GEE/HYCOM calls."""
import sys
sys.path.insert(0, "scripts")  # allow import of scripts/_train_utils

import numpy as np
import pytest
import torch
import xarray as xr

from _train_utils import build_tensors, HYCOM_VARS, WN2_VARS, SEQ_LEN, N_MEMBERS


def _make_merged(n_members=4, n_time=100, n_depth=11, n_lat=4, n_lon=5):
    """Synthetic merged Dataset matching DataHarmonizer.harmonize() output."""
    times = np.array([np.datetime64("2018-01-01") + np.timedelta64(i, "D")
                      for i in range(n_time)])
    wn2_data = {
        v: xr.DataArray(
            np.random.rand(n_members, n_time, n_lat, n_lon).astype(np.float32) + 274.0,
            dims=["member", "time", "latitude", "longitude"],
            coords={"time": times},
        )
        for v in WN2_VARS
    }
    hycom_data = {
        v: xr.DataArray(
            np.random.rand(n_members, n_time, n_depth, n_lat, n_lon).astype(np.float32),
            dims=["member", "time", "depth", "latitude", "longitude"],
            coords={"time": times},
        )
        for v in HYCOM_VARS
    }
    return xr.Dataset({**wn2_data, **hycom_data})


def _make_threshold(n_lat=4, n_lon=5):
    """Synthetic climatological threshold with dims (dayofyear, latitude, longitude)."""
    return xr.DataArray(
        np.full((366, n_lat, n_lon), 273.5, dtype=np.float32),
        dims=["dayofyear", "latitude", "longitude"],
        coords={"dayofyear": np.arange(1, 367)},
    )


def test_build_tensors_shapes():
    """build_tensors() returns tensors with correct shapes."""
    merged = _make_merged(n_members=4, n_time=100)
    threshold = _make_threshold()
    hycom_t, wn2_t, label_t = build_tensors(merged, threshold, seq_len=SEQ_LEN)

    assert hycom_t.shape == (1, 4, 11, 4), f"HYCOM shape mismatch: {hycom_t.shape}"
    assert wn2_t.shape   == (1, 4, SEQ_LEN, 5), f"WN2 shape mismatch: {wn2_t.shape}"
    assert label_t.shape == (1, 4), f"label shape mismatch: {label_t.shape}"


def test_build_tensors_dtype():
    """All output tensors are float32."""
    merged = _make_merged(n_members=4, n_time=100)
    threshold = _make_threshold()
    hycom_t, wn2_t, label_t = build_tensors(merged, threshold, seq_len=SEQ_LEN)

    assert hycom_t.dtype  == torch.float32
    assert wn2_t.dtype    == torch.float32
    assert label_t.dtype  == torch.float32


def test_build_tensors_label_nonneg():
    """SDD label is always ≥ 0 (Stress Degree Days cannot be negative)."""
    merged = _make_merged(n_members=4, n_time=100)
    threshold = _make_threshold()
    _, _, label_t = build_tensors(merged, threshold, seq_len=SEQ_LEN)
    assert (label_t >= 0).all(), f"Negative SDD values: min={label_t.min():.4f}"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n mhw-risk pytest tests/test_train_utils.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named '_train_utils'`

- [ ] **Step 3: Implement `scripts/_train_utils.py`**

```python
"""
_train_utils.py — Shared utilities for train_era5.py and train_wn2.py
======================================================================
Provides build_tensors(), run_svar_inference(), and save_plots().
Both training scripts import from here — never duplicate these functions.

Coordinate convention
---------------------
DataHarmonizer.harmonize() uses 'latitude' and 'longitude' (not 'lat'/'lon').
All spatial indexing here must use these names.

SST unit convention
-------------------
WN2 and ERA5 SST are in Kelvin. accumulate_sdd() expects °C.
build_tensors() subtracts 273.15 before computing the SDD label.
The WN2/ERA5 tensor passed to the model retains original units (K) —
the model learns from whatever units it sees during training.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from src.analytics.mhw_detection import compute_mhw_mask
from src.analytics.sdd import accumulate_sdd

# ---------------------------------------------------------------------------
# Domain constants (shared between train_era5.py and train_wn2.py)
# ---------------------------------------------------------------------------

GoM_BBOX     = (-71.0, 41.0, -66.0, 45.0)   # (lon_min, lat_min, lon_max, lat_max)
TRAIN_PERIOD = ("2018-01-01", "2018-12-31")
VAL_PERIOD   = ("2019-01-01", "2019-12-31")
SEQ_LEN      = 90     # atmospheric sequence length [days] fed to TransformerEncoder
N_MEMBERS    = 64     # ensemble members (WN2) / synthetic members (ERA5 proxy)
N_LAT        = 17     # Gulf of Maine grid cells at 0.25-degree resolution
N_LON        = 21
HYCOM_VARS   = ["water_temp", "salinity", "water_u", "water_v"]
WN2_VARS     = [
    "sea_surface_temperature",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


# ---------------------------------------------------------------------------
# build_tensors
# ---------------------------------------------------------------------------

def build_tensors(
    merged: xr.Dataset,
    threshold: xr.DataArray,
    seq_len: int = SEQ_LEN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a harmonized xr.Dataset into (hycom_t, wn2_t, label_t) PyTorch tensors.

    Parameters
    ----------
    merged : xr.Dataset
        Output of DataHarmonizer.harmonize(). Must have:
        - WN2 vars: dims (member, time, latitude, longitude), SST in Kelvin
        - HYCOM vars: dims (member, time, depth, latitude, longitude)
        - Coordinate names: 'latitude' and 'longitude'
    threshold : xr.DataArray
        Climatological SST threshold [°C], dims (dayofyear, latitude, longitude).
        Produced by mhw_detection.compute_climatology().
    seq_len : int
        Number of time steps to use from the atmospheric sequence.
        Uses the LAST seq_len days of the time axis.

    Returns
    -------
    hycom_t : torch.Tensor, shape (1, member, depth=11, channels=4)
        Time- and spatially-averaged HYCOM profile per member.
        All 64 members are identical (HYCOM is broadcast by DataHarmonizer).
    wn2_t : torch.Tensor, shape (1, member, seq_len, features=5)
        Last seq_len days of WN2/ERA5 atmospheric sequence, spatially averaged.
    label_t : torch.Tensor, shape (1, member)
        Physics-based SDD label [°C·day] spatially averaged over the GoM domain.
        Computed from merged SST (converted to °C) via MHW mask + accumulate_sdd.
    """
    # HYCOM: time-and-spatial mean → (member, depth=11, channels=4)
    hycom_arr = np.stack(
        [merged[v].mean(dim=["time", "latitude", "longitude"]).values for v in HYCOM_VARS],
        axis=-1,
    ).astype(np.float32)  # (member, depth, 4)

    # WN2/ERA5: last seq_len days, spatial mean → (member, seq_len, features=5)
    wn2_arr = np.stack(
        [merged[v].isel(time=slice(-seq_len, None)).mean(dim=["latitude", "longitude"]).values
         for v in WN2_VARS],
        axis=-1,
    ).astype(np.float32)  # (member, seq_len, 5)

    # SDD label: convert SST from K to °C before physics computation
    sst_celsius = merged["sea_surface_temperature"] - 273.15  # (member, time, lat, lon)
    mhw_mask    = compute_mhw_mask(sst_celsius, threshold)
    sdd_phys    = accumulate_sdd(sst_celsius, threshold, mhw_mask)  # (member, lat, lon)
    label_arr   = sdd_phys.mean(dim=["latitude", "longitude"]).values.astype(np.float32)  # (member,)

    hycom_t = torch.from_numpy(hycom_arr).unsqueeze(0)   # (1, M, 11, 4)
    wn2_t   = torch.from_numpy(wn2_arr).unsqueeze(0)    # (1, M, seq_len, 5)
    label_t = torch.from_numpy(label_arr).unsqueeze(0)  # (1, M)

    return hycom_t, wn2_t, label_t


# ---------------------------------------------------------------------------
# run_svar_inference
# ---------------------------------------------------------------------------

def run_svar_inference(
    model: torch.nn.Module,
    merged_val: xr.Dataset,
    device: torch.device,
    prefix: str,
) -> xr.Dataset:
    """
    Per-grid-cell SVaR inference. Iterates over each (latitude, longitude) cell,
    runs a forward pass with batch=1 for that cell's HYCOM profile and WN2
    atmospheric sequence, and computes ensemble quantiles.

    Parameters
    ----------
    model : MHWRiskModel
        Trained model in eval mode.
    merged_val : xr.Dataset
        Harmonized validation Dataset (2019). Same structure as harmonize() output.
    device : torch.device
        CPU or CUDA device to run inference on.
    prefix : str
        Either 'era5' or 'wn2'. Used for output file naming.

    Returns
    -------
    svar_ds : xr.Dataset
        Dimensions: (latitude, longitude)
        Variables: SVaR_95, SVaR_50, SVaR_05, spread [°C·day]
        Saved to: data/results/{prefix}_svar.zarr
    """
    model.eval()
    lats = merged_val["latitude"].values
    lons = merged_val["longitude"].values
    n_lat, n_lon = len(lats), len(lons)
    M = merged_val.dims["member"]

    svar_95 = np.zeros((n_lat, n_lon), dtype=np.float32)
    svar_50 = np.zeros((n_lat, n_lon), dtype=np.float32)
    svar_05 = np.zeros((n_lat, n_lon), dtype=np.float32)
    spread  = np.zeros((n_lat, n_lon), dtype=np.float32)

    with torch.no_grad():
        for i in range(n_lat):
            for j in range(n_lon):
                cell = merged_val.isel(latitude=i, longitude=j)

                # HYCOM profile for this cell: time-mean → (member, depth, channels)
                hycom_cell = np.stack(
                    [cell[v].mean(dim="time").values for v in HYCOM_VARS],
                    axis=-1,
                ).astype(np.float32)  # (member, depth, 4)

                # WN2 sequence for this cell: last SEQ_LEN days → (member, seq_len, features)
                wn2_cell = np.stack(
                    [cell[v].isel(time=slice(-SEQ_LEN, None)).values for v in WN2_VARS],
                    axis=-1,
                ).astype(np.float32)  # (member, seq_len, 5)

                ht = torch.from_numpy(hycom_cell).unsqueeze(0).to(device)  # (1, M, 11, 4)
                wt = torch.from_numpy(wn2_cell).unsqueeze(0).to(device)    # (1, M, 90, 5)

                sdd_pred, _, _ = model(ht, wt)  # (1, M)
                sdd_1d = sdd_pred[0]             # (M,)

                svar_95[i, j] = sdd_1d.quantile(0.95).item()
                svar_50[i, j] = sdd_1d.quantile(0.50).item()
                svar_05[i, j] = sdd_1d.quantile(0.05).item()
                spread[i, j]  = svar_95[i, j] - svar_05[i, j]

    svar_ds = xr.Dataset(
        {
            "SVaR_95": xr.DataArray(svar_95, dims=["latitude", "longitude"],
                                    coords={"latitude": lats, "longitude": lons}),
            "SVaR_50": xr.DataArray(svar_50, dims=["latitude", "longitude"],
                                    coords={"latitude": lats, "longitude": lons}),
            "SVaR_05": xr.DataArray(svar_05, dims=["latitude", "longitude"],
                                    coords={"latitude": lats, "longitude": lons}),
            "spread":  xr.DataArray(spread,  dims=["latitude", "longitude"],
                                    coords={"latitude": lats, "longitude": lons}),
        }
    )
    out_path = f"data/results/{prefix}_svar.zarr"
    svar_ds.to_zarr(out_path, mode="w")
    print(f"SVaR saved → {out_path}  (lat={n_lat}, lon={n_lon})")
    return svar_ds


# ---------------------------------------------------------------------------
# save_plots
# ---------------------------------------------------------------------------

def save_plots(
    log_rows: list[dict],
    model: torch.nn.Module,
    hycom_val: torch.Tensor,
    wn2_val: torch.Tensor,
    label_val: torch.Tensor,
    device: torch.device,
    prefix: str,
) -> None:
    """
    Generate and save 5 diagnostic training plots to data/results/plots/.

    Parameters
    ----------
    log_rows : list of dict
        Per-epoch log records with keys: epoch, train_loss, val_loss,
        SVaR_95, SVaR_50, SVaR_05, spread, gate_mean.
    model : MHWRiskModel
        Trained model in eval mode for final-epoch forward pass.
    hycom_val, wn2_val, label_val : torch.Tensor
        Validation tensors for pred-vs-actual scatter.
    device : torch.device
    prefix : str
        'era5' or 'wn2'. Used for file naming and plot titles.

    Plots saved
    -----------
    {prefix}_loss_curve.png    — train + val loss vs epoch
    {prefix}_svar_curve.png   — SVaR_95/50/05 vs epoch
    {prefix}_spread_curve.png — ensemble spread vs epoch
    {prefix}_gate_hist.png    — gate value histogram at final epoch
    {prefix}_pred_vs_actual.png — predicted SDD vs physics SDD (val set)
    """
    plots_dir = Path("data/results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs      = [r["epoch"]      for r in log_rows]
    train_losses = [r["train_loss"] for r in log_rows]
    val_losses   = [r["val_loss"]   for r in log_rows]
    svar_95      = [r["SVaR_95"]    for r in log_rows]
    svar_50      = [r["SVaR_50"]    for r in log_rows]
    svar_05      = [r["SVaR_05"]    for r in log_rows]
    spreads      = [r["spread"]     for r in log_rows]

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train loss", color="steelblue")
    ax.plot(epochs, val_losses,   label="Val loss",   color="tomato", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss [°C²·day²]")
    ax.set_title(f"{prefix.upper()} — Loss curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_loss_curve.png", dpi=120)
    plt.close(fig)

    # --- SVaR evolution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, svar_95, label="SVaR_95", color="firebrick")
    ax.plot(epochs, svar_50, label="SVaR_50", color="orange")
    ax.plot(epochs, svar_05, label="SVaR_05", color="steelblue")
    ax.set_xlabel("Epoch"); ax.set_ylabel("SDD [°C·day]")
    ax.set_title(f"{prefix.upper()} — SVaR evolution")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_svar_curve.png", dpi=120)
    plt.close(fig)

    # --- Ensemble spread evolution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, spreads, color="purple")
    ax.set_xlabel("Epoch"); ax.set_ylabel("SVaR_95 − SVaR_05 [°C·day]")
    ax.set_title(f"{prefix.upper()} — Ensemble spread")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_spread_curve.png", dpi=120)
    plt.close(fig)

    # --- Gate histogram (final epoch) ---
    model.eval()
    with torch.no_grad():
        _, _, gate = model(hycom_val.to(device), wn2_val.to(device))
    gate_vals = gate[0].cpu().numpy()  # (member,)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(gate_vals, bins=20, color="teal", edgecolor="white")
    ax.axvline(gate_vals.mean(), color="black", linestyle="--", label=f"mean={gate_vals.mean():.3f}")
    ax.set_xlabel("Gate value (0=atm-dominant, 1=depth-dominant)")
    ax.set_ylabel("Count"); ax.set_title(f"{prefix.upper()} — Gate distribution (final epoch)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_gate_hist.png", dpi=120)
    plt.close(fig)

    # --- Pred vs actual (val set) ---
    with torch.no_grad():
        sdd_pred, _, _ = model(hycom_val.to(device), wn2_val.to(device))
    pred_vals   = sdd_pred[0].cpu().numpy()   # (member,)
    actual_vals = label_val[0].cpu().numpy()  # (member,)

    fig, ax = plt.subplots(figsize=(5, 5))
    lim = max(pred_vals.max(), actual_vals.max()) * 1.1
    ax.scatter(actual_vals, pred_vals, alpha=0.6, color="steelblue", s=20)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Physics SDD [°C·day]"); ax.set_ylabel("Predicted SDD [°C·day]")
    ax.set_title(f"{prefix.upper()} — Predicted vs actual (val set)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_pred_vs_actual.png", dpi=120)
    plt.close(fig)

    print(f"Plots saved → data/results/plots/{prefix}_*.png  (5 files)")
```

- [ ] **Step 4: Run the three unit tests — all must pass**

```bash
conda run -n mhw-risk pytest tests/test_train_utils.py -v
```

Expected:
```
PASSED tests/test_train_utils.py::test_build_tensors_shapes
PASSED tests/test_train_utils.py::test_build_tensors_dtype
PASSED tests/test_train_utils.py::test_build_tensors_label_nonneg
3 passed in X.Xs
```

- [ ] **Step 5: Commit**

```bash
git add scripts/_train_utils.py tests/test_train_utils.py
git commit -m "feat: add shared training utilities — build_tensors, run_svar_inference, save_plots"
```

---

## Task 4: train_era5.py

**Files:**
- Create: `scripts/train_era5.py`

- [ ] **Step 1: Write `scripts/train_era5.py`**

```python
#!/usr/bin/env python3
"""
train_era5.py — Train MHWRiskModel on ERA5 proxy data
======================================================
Trains on 2018 ERA5 data (synthetic ensemble via expand_and_perturb),
validates on 2019 ERA5 data, saves all artifacts.

Usage:
    # Dry-run (no GEE/HYCOM calls — uses synthetic tensors):
    conda run -n mhw-risk python scripts/train_era5.py --dry-run

    # Real training (requires GEE auth and HYCOM connectivity):
    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
    conda run -n mhw-risk python scripts/train_era5.py --epochs 50

Artifacts saved
---------------
    data/models/era5_weights.pt           — final epoch weights
    data/models/era5_best_weights.pt      — weights at lowest val loss
    data/results/era5_training_log.csv    — per-epoch metrics
    data/results/era5_config.json         — hyperparameters used
    data/results/era5_svar.zarr           — per-grid-cell SVaR (real run only)
    data/results/plots/era5_*.png         — 5 diagnostic plots
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from src.models.ensemble_wrapper import MHWRiskModel

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _train_utils import (
    GoM_BBOX, TRAIN_PERIOD, VAL_PERIOD, N_MEMBERS, SEQ_LEN,
    build_tensors, run_svar_inference, save_plots,
)

PREFIX = "era5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MHWRiskModel on ERA5 proxy data.")
    p.add_argument("--dry-run", action="store_true",
                   help="Use synthetic tensors — skip GEE and HYCOM network calls.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr",     type=float, default=1e-4)
    return p.parse_args()


def load_real_data(args):
    """Fetch ERA5 + HYCOM for train (2018) and val (2019) periods."""
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
    wn2_train  = harvester.fetch(*TRAIN_PERIOD, GoM_BBOX)
    hycom_train = loader.fetch_tile(*TRAIN_PERIOD, GoM_BBOX)
    merged_train = harmonizer.harmonize(wn2_train, hycom_train)
    # harmonize() calls expand_and_perturb() automatically when member=1
    hycom_t_train, wn2_t_train, label_t_train = build_tensors(merged_train, threshold)

    print("Fetching ERA5 val (2019)...")
    wn2_val   = harvester.fetch(*VAL_PERIOD, GoM_BBOX)
    hycom_val_ds = loader.fetch_tile(*VAL_PERIOD, GoM_BBOX)
    merged_val = harmonizer.harmonize(wn2_val, hycom_val_ds)
    hycom_t_val, wn2_t_val, label_t_val = build_tensors(merged_val, threshold)

    return (hycom_t_train, wn2_t_train, label_t_train,
            hycom_t_val, wn2_t_val, label_t_val,
            merged_val, threshold)


def main():
    args = parse_args()
    Path("data/models").mkdir(parents=True, exist_ok=True)
    Path("data/results/plots").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dry_run:
        print("[dry-run] Synthetic tensors — no GEE or HYCOM calls.")
        M = N_MEMBERS
        hycom_t_train = torch.randn(1, M, 11, 4)
        wn2_t_train   = torch.randn(1, M, SEQ_LEN, 5)
        label_t_train = torch.rand(1, M) * 20.0
        hycom_t_val   = torch.randn(1, M, 11, 4)
        wn2_t_val     = torch.randn(1, M, SEQ_LEN, 5)
        label_t_val   = torch.rand(1, M) * 20.0
        merged_val    = None
        threshold     = None
    else:
        (hycom_t_train, wn2_t_train, label_t_train,
         hycom_t_val, wn2_t_val, label_t_val,
         merged_val, threshold) = load_real_data(args)

    # Save config before training starts
    config = {
        "prefix": PREFIX,
        "lr": args.lr,
        "epochs": args.epochs,
        "n_members": N_MEMBERS,
        "seq_len": SEQ_LEN,
        "domain_bbox": GoM_BBOX,
        "train_period": TRAIN_PERIOD,
        "val_period": VAL_PERIOD,
        "grad_clip_max_norm": 1.0,
        "dry_run": args.dry_run,
    }
    with open(f"data/results/{PREFIX}_config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    model     = MHWRiskModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    hycom_t_train = hycom_t_train.to(device)
    wn2_t_train   = wn2_t_train.to(device)
    label_t_train = label_t_train.to(device)
    hycom_t_val   = hycom_t_val.to(device)
    wn2_t_val     = wn2_t_val.to(device)
    label_t_val   = label_t_val.to(device)

    log_rows = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- Training step ---
        model.train()
        sdd_pred, _, gate = model(hycom_t_train, wn2_t_train)
        train_loss = F.mse_loss(sdd_pred, label_t_train)
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- Validation step (no gradient) ---
        model.eval()
        with torch.no_grad():
            sdd_val, _, _ = model(hycom_t_val, wn2_t_val)
            val_loss = F.mse_loss(sdd_val, label_t_val)

        v95   = sdd_val[0].quantile(0.95).item()
        v50   = sdd_val[0].quantile(0.50).item()
        v05   = sdd_val[0].quantile(0.05).item()
        sprd  = v95 - v05
        gm    = gate[0].mean().item()

        row = {
            "epoch": epoch, "train_loss": round(train_loss.item(), 6),
            "val_loss": round(val_loss.item(), 6),
            "SVaR_95": round(v95, 4), "SVaR_50": round(v50, 4),
            "SVaR_05": round(v05, 4), "spread": round(sprd, 4),
            "gate_mean": round(gm, 4),
        }
        log_rows.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss.item():.4f} | val={val_loss.item():.4f} | "
            f"SVaR_95={v95:.2f} | spread={sprd:.2f} | gate={gm:.3f}"
        )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), f"data/models/{PREFIX}_best_weights.pt")

    # Final weights
    torch.save(model.state_dict(), f"data/models/{PREFIX}_weights.pt")
    print(f"Weights → data/models/{PREFIX}_weights.pt")

    # Training log CSV
    with open(f"data/results/{PREFIX}_training_log.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Log     → data/results/{PREFIX}_training_log.csv")

    # Plots
    save_plots(log_rows, model, hycom_t_val, wn2_t_val, label_t_val, device, PREFIX)

    # Per-grid-cell SVaR (real run only — dry-run skips this)
    if merged_val is not None:
        run_svar_inference(model, merged_val, device, PREFIX)
    else:
        print("[dry-run] Skipping per-grid-cell SVaR — merged_val not available.")

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run dry-run — verify all artifacts are produced**

```bash
conda run -n mhw-risk python scripts/train_era5.py --dry-run --epochs 3
```

Expected output (last lines):
```
Epoch 001/3 | train=X.XXXX | val=X.XXXX | SVaR_95=X.XX | spread=X.XX | gate=X.XXX
Epoch 002/3 | ...
Epoch 003/3 | ...
Weights → data/models/era5_weights.pt
Log     → data/results/era5_training_log.csv
Plots saved → data/results/plots/era5_*.png  (5 files)
[dry-run] Skipping per-grid-cell SVaR — merged_val not available.
Done.
```

- [ ] **Step 3: Verify artifact files exist**

```bash
ls -lh data/models/era5_weights.pt data/models/era5_best_weights.pt \
        data/results/era5_training_log.csv data/results/era5_config.json \
        data/results/plots/era5_loss_curve.png \
        data/results/plots/era5_svar_curve.png \
        data/results/plots/era5_spread_curve.png \
        data/results/plots/era5_gate_hist.png \
        data/results/plots/era5_pred_vs_actual.png
```

Expected: all 9 files present, non-zero size.

- [ ] **Step 4: Verify training log CSV has correct columns**

```bash
conda run -n mhw-risk python -c "
import csv
with open('data/results/era5_training_log.csv') as f:
    rows = list(csv.DictReader(f))
print(f'Rows: {len(rows)}')
print(f'Columns: {list(rows[0].keys())}')
assert len(rows) == 3, f'Expected 3 rows (--epochs 3), got {len(rows)}'
assert 'val_loss' in rows[0], 'val_loss column missing'
assert float(rows[0][\"spread\"]) > 0, 'spread is zero — ensemble is degenerate'
print('CSV validation passed.')
"
```

Expected: `Rows: 3`, all 8 columns present, `CSV validation passed.`

- [ ] **Step 5: Commit**

```bash
git add scripts/train_era5.py
git commit -m "feat: add train_era5.py — ERA5 proxy training with val split and all artifacts"
```

---

## Task 5: train_wn2.py

**Files:**
- Create: `scripts/train_wn2.py`

Symmetric to `train_era5.py`. Only differences: uses `WeatherNext2Harvester.fetch_ensemble()`, does not call `expand_and_perturb()` (WN2 already returns 64 real members), and saves artifacts with the `wn2_` prefix.

- [ ] **Step 1: Write `scripts/train_wn2.py`**

```python
#!/usr/bin/env python3
"""
train_wn2.py — Train MHWRiskModel on WeatherNext 2 real FGN ensemble
=====================================================================
Trains on 2018 WN2 data (64 real FGN members), validates on 2019 WN2 data.
Architecture and hyperparameters are identical to train_era5.py — any
difference in learned weights or attributions is attributable to data alone.

Usage:
    # Dry-run (no GEE/HYCOM calls — uses synthetic tensors):
    conda run -n mhw-risk python scripts/train_wn2.py --dry-run

    # Real training:
    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
    conda run -n mhw-risk python scripts/train_wn2.py --epochs 50

Artifacts saved
---------------
    data/models/wn2_weights.pt            — final epoch weights
    data/models/wn2_best_weights.pt       — weights at lowest val loss
    data/results/wn2_training_log.csv     — per-epoch metrics
    data/results/wn2_config.json          — hyperparameters used
    data/results/wn2_svar.zarr            — per-grid-cell SVaR (real run only)
    data/results/plots/wn2_*.png          — 5 diagnostic plots
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import xarray as xr

from src.models.ensemble_wrapper import MHWRiskModel

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _train_utils import (
    GoM_BBOX, TRAIN_PERIOD, VAL_PERIOD, N_MEMBERS, SEQ_LEN,
    build_tensors, run_svar_inference, save_plots,
)

PREFIX = "wn2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MHWRiskModel on WeatherNext 2 ensemble.")
    p.add_argument("--dry-run", action="store_true",
                   help="Use synthetic tensors — skip GEE and HYCOM network calls.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr",     type=float, default=1e-4)
    return p.parse_args()


def load_real_data(args):
    """Fetch WN2 + HYCOM for train (2018) and val (2019) periods."""
    from src.ingestion.harvester import WeatherNext2Harvester, DataHarmonizer, HYCOMLoader
    import os

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

    print("Fetching WeatherNext 2 train (2018)...")
    wn2_train   = harvester.fetch_ensemble(*TRAIN_PERIOD, GoM_BBOX)
    hycom_train = loader.fetch_tile(*TRAIN_PERIOD, GoM_BBOX)
    merged_train = harmonizer.harmonize(wn2_train, hycom_train)
    # WN2 returns member=64 — harmonize() skips expand_and_perturb automatically
    hycom_t_train, wn2_t_train, label_t_train = build_tensors(merged_train, threshold)

    print("Fetching WeatherNext 2 val (2019)...")
    wn2_val      = harvester.fetch_ensemble(*VAL_PERIOD, GoM_BBOX)
    hycom_val_ds = loader.fetch_tile(*VAL_PERIOD, GoM_BBOX)
    merged_val   = harmonizer.harmonize(wn2_val, hycom_val_ds)
    hycom_t_val, wn2_t_val, label_t_val = build_tensors(merged_val, threshold)

    return (hycom_t_train, wn2_t_train, label_t_train,
            hycom_t_val, wn2_t_val, label_t_val,
            merged_val, threshold)


def main():
    args = parse_args()
    Path("data/models").mkdir(parents=True, exist_ok=True)
    Path("data/results/plots").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dry_run:
        print("[dry-run] Synthetic tensors — no GEE or HYCOM calls.")
        M = N_MEMBERS
        hycom_t_train = torch.randn(1, M, 11, 4)
        wn2_t_train   = torch.randn(1, M, SEQ_LEN, 5)
        label_t_train = torch.rand(1, M) * 20.0
        hycom_t_val   = torch.randn(1, M, 11, 4)
        wn2_t_val     = torch.randn(1, M, SEQ_LEN, 5)
        label_t_val   = torch.rand(1, M) * 20.0
        merged_val    = None
        threshold     = None
    else:
        (hycom_t_train, wn2_t_train, label_t_train,
         hycom_t_val, wn2_t_val, label_t_val,
         merged_val, threshold) = load_real_data(args)

    config = {
        "prefix": PREFIX,
        "lr": args.lr,
        "epochs": args.epochs,
        "n_members": N_MEMBERS,
        "seq_len": SEQ_LEN,
        "domain_bbox": GoM_BBOX,
        "train_period": TRAIN_PERIOD,
        "val_period": VAL_PERIOD,
        "grad_clip_max_norm": 1.0,
        "dry_run": args.dry_run,
        "note": "Real FGN ensemble — no expand_and_perturb applied",
    }
    with open(f"data/results/{PREFIX}_config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    model     = MHWRiskModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    hycom_t_train = hycom_t_train.to(device)
    wn2_t_train   = wn2_t_train.to(device)
    label_t_train = label_t_train.to(device)
    hycom_t_val   = hycom_t_val.to(device)
    wn2_t_val     = wn2_t_val.to(device)
    label_t_val   = label_t_val.to(device)

    log_rows = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        sdd_pred, _, gate = model(hycom_t_train, wn2_t_train)
        train_loss = F.mse_loss(sdd_pred, label_t_train)
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            sdd_val, _, _ = model(hycom_t_val, wn2_t_val)
            val_loss = F.mse_loss(sdd_val, label_t_val)

        v95  = sdd_val[0].quantile(0.95).item()
        v50  = sdd_val[0].quantile(0.50).item()
        v05  = sdd_val[0].quantile(0.05).item()
        sprd = v95 - v05
        gm   = gate[0].mean().item()

        row = {
            "epoch": epoch, "train_loss": round(train_loss.item(), 6),
            "val_loss": round(val_loss.item(), 6),
            "SVaR_95": round(v95, 4), "SVaR_50": round(v50, 4),
            "SVaR_05": round(v05, 4), "spread": round(sprd, 4),
            "gate_mean": round(gm, 4),
        }
        log_rows.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss.item():.4f} | val={val_loss.item():.4f} | "
            f"SVaR_95={v95:.2f} | spread={sprd:.2f} | gate={gm:.3f}"
        )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), f"data/models/{PREFIX}_best_weights.pt")

    torch.save(model.state_dict(), f"data/models/{PREFIX}_weights.pt")
    print(f"Weights → data/models/{PREFIX}_weights.pt")

    with open(f"data/results/{PREFIX}_training_log.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Log     → data/results/{PREFIX}_training_log.csv")

    save_plots(log_rows, model, hycom_t_val, wn2_t_val, label_t_val, device, PREFIX)

    if merged_val is not None:
        run_svar_inference(model, merged_val, device, PREFIX)
    else:
        print("[dry-run] Skipping per-grid-cell SVaR — merged_val not available.")

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run dry-run**

```bash
conda run -n mhw-risk python scripts/train_wn2.py --dry-run --epochs 3
```

Expected: same output pattern as train_era5.py with `wn2_` prefix. No errors.

- [ ] **Step 3: Verify artifacts**

```bash
ls -lh data/models/wn2_weights.pt data/models/wn2_best_weights.pt \
        data/results/wn2_training_log.csv data/results/wn2_config.json \
        data/results/plots/wn2_loss_curve.png \
        data/results/plots/wn2_svar_curve.png \
        data/results/plots/wn2_spread_curve.png \
        data/results/plots/wn2_gate_hist.png \
        data/results/plots/wn2_pred_vs_actual.png
```

Expected: all 9 files present, non-zero size.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_wn2.py
git commit -m "feat: add train_wn2.py — WeatherNext 2 training symmetric to train_era5.py"
```

---

## Task 6: compare_xai.py

**Files:**
- Create: `scripts/compare_xai.py`

Loads both weight files, runs Captum IG per season on each model with its own inputs, saves `xai_comparison.json`.

**Seasonal input handling:** The atmospheric sequence fed to the Transformer is `seq_len=90` days. For each season, we extract the days matching that season's months from the merged dataset's time axis. If fewer than 90 days are available (DJF in 2018 = Jan-Feb only = ~59 days), we use all available days and pad with zeros at the start to reach seq_len.

**IG target:** The `latent_forward` wrapper (identical to the smoke test in `ensemble_wrapper.py`) returns `latent.mean(dim=[1,2])` — scalar per batch item. IG attributes this scalar back to both input tensors.

**Attribution aggregation:** For each season, compute `attr[0].abs().mean(dim=[0,1,3])` over (batch, member, time) → one score per HYCOM channel (depth=11 collapsed too: `.mean(-1)` or `.sum(-1)`). For WN2: `attr[1].abs().mean(dim=[0,1,2])` over (batch, member, time_steps) → one score per feature.

- [ ] **Step 1: Write `scripts/compare_xai.py`**

```python
#!/usr/bin/env python3
"""
compare_xai.py — Per-season Captum IG attribution comparison: ERA5 vs WN2
==========================================================================
Loads era5_weights.pt and wn2_weights.pt, runs Integrated Gradients per
season (DJF, MAM, JJA, SON) on each model with its own inputs, and saves
a structured attribution comparison to data/results/xai_comparison.json.

Usage:
    # Dry-run (no GEE/HYCOM — uses synthetic tensors):
    conda run -n mhw-risk python scripts/compare_xai.py --dry-run

    # Real run (requires both weight files and merged Zarr datasets):
    conda run -n mhw-risk python scripts/compare_xai.py \\
        --era5-data data/processed/merged_era5_val.zarr \\
        --wn2-data  data/processed/merged_wn2_val.zarr

Output
------
    data/results/xai_comparison.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from captum.attr import IntegratedGradients

from src.models.ensemble_wrapper import MHWRiskModel

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _train_utils import (
    N_MEMBERS, SEQ_LEN, SEASONS, HYCOM_VARS, WN2_VARS,
)

ATM_FEATURE_NAMES  = WN2_VARS   # 5 atmospheric variables
HYCOM_CHANNEL_NAMES = HYCOM_VARS  # 4 HYCOM channels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--era5-weights", default="data/models/era5_weights.pt")
    p.add_argument("--wn2-weights",  default="data/models/wn2_weights.pt")
    p.add_argument("--era5-data",    default=None,
                   help="Path to merged ERA5 val Zarr (required for real run).")
    p.add_argument("--wn2-data",     default=None,
                   help="Path to merged WN2 val Zarr (required for real run).")
    p.add_argument("--n-steps",      type=int, default=50,
                   help="IG integration steps. More steps = more accurate but slower.")
    return p.parse_args()


def load_model(weights_path: str, device: torch.device) -> MHWRiskModel:
    """Load MHWRiskModel from a weights file."""
    model = MHWRiskModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def latent_forward(model: MHWRiskModel):
    """Return a Captum-compatible wrapper that maps (hycom, wn2) → scalar per batch."""
    def _forward(hycom_in: torch.Tensor, wn2_in: torch.Tensor) -> torch.Tensor:
        _, lat, _ = model(hycom_in, wn2_in)
        return lat.mean(dim=[1, 2])  # (batch,)
    return _forward


def get_season_tensors(
    merged: xr.Dataset,
    season_months: list[int],
    device: torch.device,
    n_members: int = N_MEMBERS,
    seq_len: int = SEQ_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract HYCOM and WN2 tensors for the given season months.

    Parameters
    ----------
    merged : xr.Dataset
        Harmonized validation dataset. WN2 vars: (member, time, lat, lon).
        HYCOM vars: (member, time, depth, lat, lon).
    season_months : list of int
        Months belonging to the season, e.g. [6, 7, 8] for JJA.
    seq_len : int
        Target sequence length for WN2. If fewer season days available, zero-pad.

    Returns
    -------
    hycom_t : torch.Tensor, shape (1, member, depth=11, channels=4)
    wn2_t   : torch.Tensor, shape (1, member, seq_len, features=5)
    """
    # Select time steps matching this season's months
    # Use .isin() for month filtering, then .values for boolean numpy index
    time_mask = merged["time"].dt.month.isin(season_months).values  # numpy bool array
    merged_season = merged.isel(time=time_mask)

    # HYCOM: time+spatial mean → (member, depth, channels)
    hycom_arr = np.stack(
        [merged_season[v].mean(dim=["time", "latitude", "longitude"]).values
         for v in HYCOM_VARS],
        axis=-1,
    ).astype(np.float32)  # (member, depth, 4)

    # WN2: spatial mean → (member, T_season, features)
    wn2_season_arr = np.stack(
        [merged_season[v].mean(dim=["latitude", "longitude"]).values
         for v in WN2_VARS],
        axis=-1,
    ).astype(np.float32)  # (member, T_season, 5)

    # Pad or truncate WN2 to seq_len on the time axis (left-pad with zeros)
    T = wn2_season_arr.shape[1]
    if T >= seq_len:
        wn2_seq = wn2_season_arr[:, -seq_len:, :]  # take last seq_len days
    else:
        pad = np.zeros((n_members, seq_len - T, len(WN2_VARS)), dtype=np.float32)
        wn2_seq = np.concatenate([pad, wn2_season_arr], axis=1)

    hycom_t = torch.from_numpy(hycom_arr).unsqueeze(0).to(device)  # (1, M, 11, 4)
    wn2_t   = torch.from_numpy(wn2_seq).unsqueeze(0).to(device)    # (1, M, seq_len, 5)
    return hycom_t, wn2_t


def run_season_ig(
    model: MHWRiskModel,
    hycom_t: torch.Tensor,
    wn2_t: torch.Tensor,
    n_steps: int = 50,
) -> tuple[dict[str, float], dict[str, float], float]:
    """
    Run Captum IG for one season and aggregate attribution scores.

    Returns
    -------
    atm_scores : dict  — mean |IG| per WN2 variable (5 entries)
    hycom_scores : dict — mean |IG| per HYCOM channel (4 entries)
    gate_mean : float  — mean gate value across members
    """
    hycom_ig = hycom_t.requires_grad_(True)
    wn2_ig   = wn2_t.requires_grad_(True)

    ig = IntegratedGradients(latent_forward(model))
    attr = ig.attribute(
        (hycom_ig, wn2_ig),
        baselines=(torch.zeros_like(hycom_ig), torch.zeros_like(wn2_ig)),
        n_steps=n_steps,
    )
    # attr[0]: (1, member, depth=11, channels=4) — HYCOM attribution
    # attr[1]: (1, member, seq_len,  features=5) — WN2 attribution

    # HYCOM: mean |IG| over (batch=1, member, depth) → one score per channel
    hycom_abs = attr[0].abs()                              # (1, M, 11, 4)
    hycom_scores = {
        HYCOM_CHANNEL_NAMES[c]: float(hycom_abs[..., c].mean())
        for c in range(len(HYCOM_CHANNEL_NAMES))
    }

    # WN2: mean |IG| over (batch=1, member, time_steps) → one score per feature
    wn2_abs = attr[1].abs()                                # (1, M, seq_len, 5)
    atm_scores = {
        ATM_FEATURE_NAMES[f]: float(wn2_abs[..., f].mean())
        for f in range(len(ATM_FEATURE_NAMES))
    }

    # Gate value (no IG needed — just a forward pass)
    with torch.no_grad():
        _, _, gate = model(hycom_t, wn2_t)
    gate_mean = float(gate[0].mean())

    return atm_scores, hycom_scores, gate_mean


def compute_delta(era5_season: dict, wn2_season: dict) -> dict:
    """Compute WN2 − ERA5 attribution delta for atm and hycom scores."""
    delta = {}
    for stream in ("atm", "hycom"):
        delta[stream] = {
            var: round(wn2_season[stream][var] - era5_season[stream][var], 6)
            for var in wn2_season[stream]
        }
    return delta


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path("data/results").mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        for p in [args.era5_weights, args.wn2_weights]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Weight file not found: {p}")

    # Build synthetic data for dry-run
    if args.dry_run:
        print("[dry-run] Using synthetic tensors.")
        M = N_MEMBERS

        # Fake merged dataset with all required variables and time axis
        import pandas as pd
        times = pd.date_range("2019-01-01", "2019-12-31", freq="D")
        fake_data = {
            v: xr.DataArray(
                np.random.rand(M, len(times), 4, 5).astype(np.float32) + 274.0,
                dims=["member", "time", "latitude", "longitude"],
                coords={"time": times},
            )
            for v in WN2_VARS
        }
        fake_data.update({
            v: xr.DataArray(
                np.random.rand(M, len(times), 11, 4, 5).astype(np.float32),
                dims=["member", "time", "depth", "latitude", "longitude"],
                coords={"time": times},
            )
            for v in HYCOM_VARS
        })
        era5_merged = xr.Dataset(fake_data)
        wn2_merged  = xr.Dataset(fake_data)
        era5_model  = MHWRiskModel().to(device)
        wn2_model   = MHWRiskModel().to(device)
    else:
        era5_merged = xr.open_zarr(args.era5_data)
        wn2_merged  = xr.open_zarr(args.wn2_data)
        era5_model  = load_model(args.era5_weights, device)
        wn2_model   = load_model(args.wn2_weights,  device)

    result = {"era5": {}, "wn2": {}, "delta": {}}

    for season_name, months in SEASONS.items():
        print(f"Running IG for season {season_name}...")

        era5_hycom_t, era5_wn2_t = get_season_tensors(era5_merged, months, device)
        wn2_hycom_t,  wn2_wn2_t  = get_season_tensors(wn2_merged,  months, device)

        era5_atm, era5_hycom, era5_gate = run_season_ig(
            era5_model, era5_hycom_t, era5_wn2_t, n_steps=args.n_steps
        )
        wn2_atm,  wn2_hycom,  wn2_gate  = run_season_ig(
            wn2_model,  wn2_hycom_t,  wn2_wn2_t,  n_steps=args.n_steps
        )

        result["era5"][season_name] = {
            "atm": {k: round(v, 6) for k, v in era5_atm.items()},
            "hycom": {k: round(v, 6) for k, v in era5_hycom.items()},
            "gate_mean": round(era5_gate, 4),
        }
        result["wn2"][season_name] = {
            "atm": {k: round(v, 6) for k, v in wn2_atm.items()},
            "hycom": {k: round(v, 6) for k, v in wn2_hycom.items()},
            "gate_mean": round(wn2_gate, 4),
        }
        result["delta"][season_name] = compute_delta(
            result["era5"][season_name], result["wn2"][season_name]
        )

        print(f"  ERA5 gate={era5_gate:.3f} | WN2 gate={wn2_gate:.3f}")

    out_path = "data/results/xai_comparison.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"XAI comparison saved → {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run dry-run**

```bash
conda run -n mhw-risk python scripts/compare_xai.py --dry-run
```

Expected output:
```
[dry-run] Using synthetic tensors.
Running IG for season DJF...
  ERA5 gate=X.XXX | WN2 gate=X.XXX
Running IG for season MAM...
  ...
Running IG for season SON...
  ERA5 gate=X.XXX | WN2 gate=X.XXX
XAI comparison saved → data/results/xai_comparison.json
```

- [ ] **Step 3: Validate JSON structure**

```bash
conda run -n mhw-risk python -c "
import json
with open('data/results/xai_comparison.json') as f:
    r = json.load(f)

assert set(r.keys()) == {'era5', 'wn2', 'delta'}, 'Top-level keys wrong'
for top in ('era5', 'wn2', 'delta'):
    assert set(r[top].keys()) == {'DJF', 'MAM', 'JJA', 'SON'}, f'{top}: missing seasons'
    for season in r[top]:
        if top != 'delta':
            assert 'gate_mean' in r[top][season], f'{top}/{season}: missing gate_mean'
        assert len(r[top][season]['atm'])   == 5, f'{top}/{season}: expected 5 atm vars'
        assert len(r[top][season]['hycom']) == 4, f'{top}/{season}: expected 4 hycom channels'
print('JSON structure validated — all 4 seasons, all variables present.')
"
```

Expected: `JSON structure validated — all 4 seasons, all variables present.`

- [ ] **Step 4: Commit**

```bash
git add scripts/compare_xai.py
git commit -m "feat: add compare_xai.py — per-season Captum IG attribution comparison"
```

---

## Task 7: Full Test Suite

**Files:**
- No new files — runs existing tests

- [ ] **Step 1: Run full test suite**

```bash
conda run -n mhw-risk pytest tests/ -v
```

Expected: all existing tests pass plus the new ones from Tasks 2 and 3.

- [ ] **Step 2: Run all three dry-runs in sequence**

```bash
conda run -n mhw-risk python scripts/train_era5.py  --dry-run --epochs 5
conda run -n mhw-risk python scripts/train_wn2.py   --dry-run --epochs 5
conda run -n mhw-risk python scripts/compare_xai.py --dry-run
```

Expected: no errors, all artifacts present under `data/`.

- [ ] **Step 3: Final artifact check**

```bash
ls -lh data/models/ data/results/*.csv data/results/*.json data/results/plots/
```

Expected: 4 weight files (era5/wn2 × final/best), 2 CSVs, 3 JSONs (2 configs + xai_comparison), 10 PNGs.

- [ ] **Step 4: Commit**

```bash
git add data/models/.gitkeep data/results/.gitkeep data/results/plots/.gitkeep 2>/dev/null || true
git commit -m "test: verify full dry-run pipeline — ERA5, WN2, XAI comparison"
```

---

## Execution Order Summary

| Order | Task | Blocker |
|-------|------|---------|
| 1 | Task 0 — WN2 scoping | None |
| 2 | Task 1 — matplotlib | None |
| 3 | Task 2 — ERA5Harvester + tests | None |
| 4 | Task 3 — _train_utils + tests | Task 2 (expand_and_perturb needed) |
| 5 | Task 4 — train_era5.py | Task 3 |
| 6 | Task 5 — train_wn2.py | Task 3 |
| 7 | Task 6 — compare_xai.py | Tasks 4 + 5 (weight files needed for real run) |
| 8 | Task 7 — full suite | All above |
