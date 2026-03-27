# MHW Detection & SVaR Analytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/analytics/` — MHW event detection (Hobday et al. 2016), Stress Degree Day accumulation, and Stochastic Value-at-Risk estimation from the 64-member ensemble SDD distribution.

**Architecture:** Three focused modules with clean interfaces. `mhw_detection.py` and `sdd.py` operate on xarray DataArrays from the harmonized dataset; `svar.py` operates on the `(batch, member)` torch tensor produced by `MHWRiskModel`. The analytics layer is fully testable without WeatherNext 2 access: detection and SDD can use HYCOM surface temperature as an SST proxy; SVaR only needs synthetic SDD tensors.

**Tech Stack:** `xarray>=2024.2.0`, `numpy>=1.26.0`, `torch>=2.2.0`, `pytest` (tests only). No new dependencies.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `tests/__init__.py` | Makes tests/ a package |
| Create | `tests/conftest.py` | Shared fixtures: synthetic SST DataArrays, threshold arrays, SDD tensors |
| Create | `tests/test_mhw_detection.py` | Unit tests for `mhw_detection.py` |
| Create | `tests/test_sdd.py` | Unit tests for `sdd.py` |
| Create | `tests/test_svar.py` | Unit tests for `svar.py` |
| Create | `src/analytics/mhw_detection.py` | `compute_climatology()`, `compute_mhw_mask()` |
| Create | `src/analytics/sdd.py` | `accumulate_sdd()` |
| Create | `src/analytics/svar.py` | `compute_svar()`, `compute_ensemble_stats()` |
| Modify | `src/analytics/__init__.py` | Export public API |
| Modify | `mhw_claude_actions/mhw_claude_todo.md` | Mark ACTIVE complete, update QUEUED |

---

## Physical Reference (read before touching any analytics code)

**MHW definition — Hobday et al. (2016) Category I:**
- SST exceeds the local 90th percentile of historical daily SST (computed from a 1982–2011 climatology)
- Exceedance lasts ≥ 5 consecutive days
- The anomaly above the threshold is the "intensity" at each time step

**Stress Degree Days (SDD):**
- `SDD = Σ max(SST_t − threshold_t, 0)` over all days where the MHW mask is True
- Units: °C·day (thermal load)
- Used as the payout trigger variable for parametric insurance

**SVaR (Stochastic Value-at-Risk):**
- The 64-member ensemble produces 64 SDD realisations per location per season
- `SVaR_p = quantile(SDD_members, p)` — the p-th percentile of the ensemble distribution
- `SVaR_95` represents the worst-case SDD that is exceeded in only 5% of ensemble members
- Used by the financial model to price the insurance payout

**Development proxy (no WeatherNext 2 needed):**
- Use `hycom_ds['water_temp'].isel(depth=0)` (HYCOM surface temperature, °C)
  as the SST input. HYCOM `depth=0` corresponds to TARGET_DEPTHS_M[0] = 0 m.
- Broadcast to a synthetic `member` dimension of size 4 for development.
- Use a constant threshold of 18°C (Gulf of Maine salmon stress onset) for testing.

---

## Task 1: Test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `tests/__init__.py`** (empty file)

```bash
touch tests/__init__.py
```

- [ ] **Step 2: Create `tests/conftest.py`** with shared fixtures

```python
"""
conftest.py — shared pytest fixtures for analytics unit tests.

All fixtures use synthetic data so tests run without HYCOM/WeatherNext 2 access.
Sizes are kept tiny (member=4, time=10, lat=3, lon=3) for fast execution.
"""
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr


@pytest.fixture
def time_coord():
    """10-day time axis starting 2019-07-01 using pandas DatetimeIndex."""
    return pd.date_range("2019-07-01", periods=10, freq="D")


@pytest.fixture
def sst_above(time_coord):
    """
    SST DataArray where every value is 2°C above a 20°C baseline.
    shape: (member=4, time=10, lat=3, lon=3)
    Every day of every member should trigger MHW detection with threshold=20°C.
    """
    data = np.full((4, 10, 3, 3), 22.0, dtype=np.float32)  # 2°C above threshold
    return xr.DataArray(
        data,
        dims=["member", "time", "lat", "lon"],
        coords={
            "member": np.arange(4),
            "time": time_coord,
            "lat": [42.0, 42.25, 42.5],
            "lon": [-70.0, -69.75, -69.5],
        },
    )


@pytest.fixture
def sst_below(time_coord):
    """
    SST DataArray where every value is 1°C below threshold.
    shape: (member=4, time=10, lat=3, lon=3)
    No MHW should be detected with threshold=20°C.
    """
    data = np.full((4, 10, 3, 3), 19.0, dtype=np.float32)
    return xr.DataArray(
        data,
        dims=["member", "time", "lat", "lon"],
        coords={
            "member": np.arange(4),
            "time": time_coord,
            "lat": [42.0, 42.25, 42.5],
            "lon": [-70.0, -69.75, -69.5],
        },
    )


@pytest.fixture
def sst_mixed(time_coord):
    """
    SST DataArray: first 4 days below threshold, last 6 days above.
    With min_duration=5, only the last 6 days should register as MHW.
    shape: (member=1, time=10, lat=1, lon=1)
    """
    data = np.array([19.0, 19.0, 19.0, 19.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0],
                    dtype=np.float32).reshape(1, 10, 1, 1)
    return xr.DataArray(
        data,
        dims=["member", "time", "lat", "lon"],
        coords={
            "member": [0],
            "time": time_coord,
            "lat": [42.0],
            "lon": [-70.0],
        },
    )


@pytest.fixture
def threshold_20(time_coord):
    """
    Constant 20°C threshold broadcast across dayofyear=365, lat=3, lon=3.
    shape: (dayofyear=365, lat=3, lon=3)
    """
    data = np.full((365, 3, 3), 20.0, dtype=np.float32)
    return xr.DataArray(
        data,
        dims=["dayofyear", "lat", "lon"],
        coords={
            "dayofyear": np.arange(1, 366),
            "lat": [42.0, 42.25, 42.5],
            "lon": [-70.0, -69.75, -69.5],
        },
    )


@pytest.fixture
def threshold_20_small(time_coord):
    """Threshold for sst_mixed fixture (lat=1, lon=1)."""
    data = np.full((365, 1, 1), 20.0, dtype=np.float32)
    return xr.DataArray(
        data,
        dims=["dayofyear", "lat", "lon"],
        coords={
            "dayofyear": np.arange(1, 366),
            "lat": [42.0],
            "lon": [-70.0],
        },
    )


@pytest.fixture
def sdd_tensor():
    """
    Synthetic (batch=2, member=8) SDD tensor.
    Member 7 of batch 0 has a very high SDD (severe MHW member).
    """
    t = torch.zeros(2, 8)
    t[0, :] = torch.tensor([1.0, 1.5, 2.0, 1.2, 1.8, 1.3, 1.6, 10.0])
    t[1, :] = torch.tensor([0.5, 0.8, 0.6, 0.9, 0.7, 0.8, 0.6, 0.7])
    return t
```

- [ ] **Step 3: Verify conftest loads without error**

```bash
conda run -n mhw-risk pytest tests/ --collect-only 2>&1 | head -20
```

Expected: `no tests ran` or collection summary with 0 tests — no import errors.

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add analytics test infrastructure and shared fixtures"
```

---

## Task 2: `mhw_detection.py` — tests first

**Files:**
- Create: `tests/test_mhw_detection.py`
- Create: `src/analytics/mhw_detection.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mhw_detection.py
"""Unit tests for MHW event detection (Hobday et al. 2016 Category I)."""
import numpy as np
import pytest
import xarray as xr

from src.analytics.mhw_detection import compute_climatology, compute_mhw_mask


class TestComputeMhwMask:
    def test_all_above_threshold_all_days_flagged(self, sst_above, threshold_20):
        """All SST values 2°C above threshold for 10 days → all days flagged."""
        mask = compute_mhw_mask(sst_above, threshold_20, min_duration=5)
        assert mask.dims == ("member", "time", "lat", "lon")
        assert mask.dtype == bool
        assert bool(mask.all())

    def test_all_below_threshold_no_days_flagged(self, sst_below, threshold_20):
        """All SST values below threshold → no MHW days."""
        mask = compute_mhw_mask(sst_below, threshold_20, min_duration=5)
        assert not bool(mask.any())

    def test_consecutive_day_filter(self, sst_mixed, threshold_20_small):
        """First 4 days below, last 6 days above → only last 6 days flagged."""
        mask = compute_mhw_mask(sst_mixed, threshold_20_small, min_duration=5)
        values = mask.values[0, :, 0, 0]  # member=0, lat=0, lon=0
        assert not values[:4].any(), "First 4 days (below threshold) must not be flagged"
        assert values[4:].all(), "Last 6 days (above threshold, ≥5 consecutive) must be flagged"

    def test_short_exceedance_not_flagged(self, time_coord):
        """4 consecutive days above threshold (< min_duration=5) → not flagged."""
        # 4 days above, then 6 below
        data = np.array([21.0, 21.0, 21.0, 21.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0],
                        dtype=np.float32).reshape(1, 10, 1, 1)
        sst = xr.DataArray(data, dims=["member", "time", "lat", "lon"],
                           coords={"member": [0],
                                   "time": pd.date_range("2019-07-01", periods=10, freq="D"),
                                   "lat": [42.0], "lon": [-70.0]})
        threshold = xr.DataArray(np.full((365, 1, 1), 20.0, dtype=np.float32),
                                 dims=["dayofyear", "lat", "lon"],
                                 coords={"dayofyear": np.arange(1, 366),
                                         "lat": [42.0], "lon": [-70.0]})
        mask = compute_mhw_mask(sst, threshold, min_duration=5)
        assert not bool(mask.any()), "Run of 4 days must not reach min_duration=5 threshold"

    def test_output_shape_matches_input(self, sst_above, threshold_20):
        """Output mask has same shape as input SST."""
        mask = compute_mhw_mask(sst_above, threshold_20)
        assert mask.shape == sst_above.shape

    def test_member_independence(self, time_coord, threshold_20_small):
        """Each member's mask is computed independently — member 0 above, member 1 below."""
        import pandas as pd
        data = np.ones((2, 10, 1, 1), dtype=np.float32) * 19.0
        data[0, :, :, :] = 22.0  # member 0 all above
        sst = xr.DataArray(data, dims=["member", "time", "lat", "lon"],
                           coords={"member": [0, 1], "time": time_coord,
                                   "lat": [42.0], "lon": [-70.0]})
        mask = compute_mhw_mask(sst, threshold_20_small, min_duration=5)
        assert bool(mask.sel(member=0).all()), "member 0 should be all True"
        assert not bool(mask.sel(member=1).any()), "member 1 should be all False"


class TestComputeClimatology:
    def test_returns_dayofyear_lat_lon(self, time_coord):
        """compute_climatology returns (dayofyear, lat, lon) DataArray."""
        # Build 3 years of synthetic SST data: random values around 20°C
        times = xr.cftime_range("2017-01-01", periods=365 * 3, freq="D")
        data = np.random.uniform(15.0, 25.0, (365 * 3, 2, 2)).astype(np.float32)
        sst_hist = xr.DataArray(data, dims=["time", "lat", "lon"],
                                coords={"time": times, "lat": [42.0, 42.25],
                                        "lon": [-70.0, -69.75]})
        clim = compute_climatology(sst_hist, percentile=90)
        assert "dayofyear" in clim.dims
        assert clim.dims == ("dayofyear", "lat", "lon")
        assert clim.shape == (365, 2, 2)

    def test_constant_sst_gives_constant_threshold(self):
        """If historical SST is always 20°C, 90th percentile must be 20°C everywhere."""
        times = xr.cftime_range("2017-01-01", periods=365 * 3, freq="D")
        data = np.full((365 * 3, 1, 1), 20.0, dtype=np.float32)
        sst_hist = xr.DataArray(data, dims=["time", "lat", "lon"],
                                coords={"time": times, "lat": [42.0], "lon": [-70.0]})
        clim = compute_climatology(sst_hist, percentile=90)
        np.testing.assert_allclose(clim.values, 20.0, atol=1e-4)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
conda run -n mhw-risk pytest tests/test_mhw_detection.py -v 2>&1 | tail -20
```

Expected: `ModuleNotFoundError: No module named 'src.analytics.mhw_detection'`

- [ ] **Step 3: Implement `src/analytics/mhw_detection.py`**

```python
"""
mhw_detection.py — Marine Heatwave event detection (Hobday et al. 2016, Category I)
=====================================================================================
Detects MHW events from SST DataArrays by comparing against a climatological
90th-percentile threshold and requiring ≥ 5 consecutive days of exceedance.

Physical note
-------------
A Marine Heatwave (Hobday et al. 2016) is defined as a discrete, prolonged, anomalously
warm water event where SST exceeds the local 90th percentile climatological threshold
for five or more consecutive days. This threshold is computed from a historical baseline
period (typically 1982–2011) to avoid conflating warming trends with events.

The 90th percentile is used (not the mean or 95th) because:
  - It is sensitive enough to detect events that cause biological stress before mortality.
  - Salmon begin showing physiological stress at SSTs 2-4°C above their local seasonal
    average, which typically aligns with the 90th percentile in temperate aquaculture zones.

Development proxy
-----------------
When WeatherNext 2 access is unavailable, use HYCOM surface temperature:
    sst_proxy = hycom_ds['water_temp'].isel(depth=0)  # °C, depth index 0 = 0 m
Broadcast to a member dimension before calling compute_mhw_mask().

Dependencies: xarray>=2024.2.0, numpy>=1.26.0
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def compute_climatology(
    sst_historical: xr.DataArray,
    percentile: float = 90.0,
) -> xr.DataArray:
    """
    Compute the climatological SST percentile threshold for each calendar day.

    Groups historical SST by day-of-year and computes the requested percentile
    at each grid point, producing the Hobday et al. baseline threshold array.

    Parameters
    ----------
    sst_historical : xr.DataArray
        Historical daily SST [°C], dims (time, lat, lon).
        Should cover ≥ 3 years for stable percentile estimates.
        Recommended: 1982–2011 (30-year WMO standard period).
    percentile : float
        Percentile to compute. Default 90 per Hobday et al. (2016) Category I.

    Returns
    -------
    threshold : xr.DataArray
        90th-percentile SST [°C] for each calendar day.
        Dims: (dayofyear=365, lat, lon).
        Index dayofyear runs 1–365; day 366 (leap) is omitted by groupby convention.
    """
    grouped = sst_historical.groupby("time.dayofyear")
    threshold = grouped.reduce(np.percentile, dim="time", q=percentile)
    return threshold  # dims: (dayofyear, lat, lon)


def compute_mhw_mask(
    sst: xr.DataArray,
    threshold: xr.DataArray,
    min_duration: int = 5,
) -> xr.DataArray:
    """
    Detect Marine Heatwave events per Hobday et al. (2016) Category I.

    A grid point and time step is flagged as an MHW event if:
      1. SST > threshold at that day-of-year and location.
      2. The exceedance is part of a run of ≥ min_duration consecutive days.

    Parameters
    ----------
    sst : xr.DataArray
        Daily SST [°C], dims (member, time, lat, lon).
        Each member is processed independently — no information sharing.
        When WeatherNext 2 is unavailable, use HYCOM water_temp[depth=0]
        broadcast across a synthetic member dimension.
    threshold : xr.DataArray
        Climatological SST threshold [°C], dims (dayofyear, lat, lon).
        Produced by compute_climatology() or loaded from a precomputed file.
        Must cover all day-of-year values present in sst.time.
    min_duration : int
        Minimum consecutive days above threshold to qualify as an MHW event.
        Default 5 per Hobday et al. (2016). Reduce to 1 for pointwise exceedance.

    Returns
    -------
    mhw_mask : xr.DataArray
        Boolean, same shape as sst: (member, time, lat, lon).
        True where an MHW event is active. Use this mask to gate SDD accumulation.
    """
    # Align threshold to the SST time axis by day-of-year
    doy = sst.time.dt.dayofyear
    thresh_aligned = threshold.sel(dayofyear=doy)      # (time, lat, lon)

    # Pointwise exceedance: SST strictly above threshold
    above = sst > thresh_aligned                        # (member, time, lat, lon), bool

    # Apply consecutive-day filter along time axis per member per grid point
    above_np = above.values.astype(np.int8)             # (member, time, lat, lon)
    mask_np  = _apply_min_duration(above_np, min_duration, time_axis=1)

    return xr.DataArray(
        mask_np.astype(bool),
        dims=sst.dims,
        coords=sst.coords,
        attrs={"min_duration_days": min_duration, "method": "Hobday_2016_Cat1"},
    )


def _apply_min_duration(
    above: np.ndarray,
    min_duration: int,
    time_axis: int = 1,
) -> np.ndarray:
    """
    Zero out runs shorter than min_duration along the time axis.

    Uses a forward cumsum pass to compute run lengths ending at each time step,
    then a backward pass to propagate the valid-run flag back to the run start.
    Both passes are fully vectorized over (member, lat, lon) simultaneously.

    Parameters
    ----------
    above : np.ndarray
        Integer array (0/1) indicating pointwise threshold exceedance.
        Expected shape: (member, time, lat, lon).
    min_duration : int
        Minimum run length to retain. Runs shorter than this are zeroed out.
    time_axis : int
        Axis index corresponding to time. Default 1.

    Returns
    -------
    filtered : np.ndarray
        Same shape as above; short runs zeroed out, long runs preserved.

    Example
    -------
    above = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  min_duration=5
    filtered → [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  (run of 6 ≥ 5: kept)

    above = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  min_duration=5
    filtered → [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  (run of 4 < 5: zeroed)
    """
    if min_duration <= 1:
        return above

    above_int = above.astype(np.int32)
    n_time    = above_int.shape[time_axis]

    # -- Forward pass: compute run length ending at each time step --
    # run_len[t] = consecutive 1s ending at t (resets to 0 on a 0)
    run_len = np.zeros_like(above_int)
    for t in range(n_time):
        if t == 0:
            run_len[:, t] = above_int[:, t]
        else:
            run_len[:, t] = np.where(above_int[:, t] == 1,
                                     run_len[:, t - 1] + 1,
                                     0)

    # -- Backward pass: propagate "part of a long run" flag back to run start --
    # A time step t is in a valid run if:
    #   (a) the run ending at t has length >= min_duration, OR
    #   (b) t is True and t+1 is already marked as part of a valid run
    filtered = np.zeros_like(above_int)
    for t in range(n_time - 1, -1, -1):
        long_run_ends_here = (run_len[:, t] >= min_duration)
        if t < n_time - 1:
            carries = (above_int[:, t] == 1) & (filtered[:, t + 1] == 1)
            in_valid_run = long_run_ends_here | carries
        else:
            in_valid_run = long_run_ends_here
        filtered[:, t] = np.where(in_valid_run & (above_int[:, t] == 1), 1, 0)

    return filtered
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n mhw-risk pytest tests/test_mhw_detection.py -v 2>&1
```

Expected: All 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/analytics/mhw_detection.py tests/test_mhw_detection.py
git commit -m "feat: implement MHW detection (Hobday 2016 Category I) with consecutive-day filter"
```

---

## Task 3: `sdd.py` — tests first

**Files:**
- Create: `tests/test_sdd.py`
- Create: `src/analytics/sdd.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sdd.py
"""Unit tests for Stress Degree Day accumulation."""
import numpy as np
import pytest
import xarray as xr

from src.analytics.sdd import accumulate_sdd


class TestAccumulateSdd:
    def test_all_above_accumulates_correctly(self, sst_above, threshold_20, time_coord):
        """
        SST = 22°C, threshold = 20°C for 10 days, all days flagged.
        SDD = (22 - 20) * 10 = 20 °C·day per grid point.
        """
        mhw_mask = xr.ones_like(sst_above, dtype=bool)  # all days flagged
        sdd = accumulate_sdd(sst_above, threshold_20, mhw_mask)
        assert sdd.dims == ("member", "lat", "lon")
        np.testing.assert_allclose(sdd.values, 20.0, atol=1e-4)

    def test_no_mhw_gives_zero_sdd(self, sst_above, threshold_20):
        """No MHW days → SDD = 0 everywhere, even if SST is above threshold."""
        mhw_mask = xr.zeros_like(sst_above, dtype=bool)  # no days flagged
        sdd = accumulate_sdd(sst_above, threshold_20, mhw_mask)
        np.testing.assert_allclose(sdd.values, 0.0, atol=1e-4)

    def test_partial_days_accumulate_correctly(self, sst_mixed, threshold_20_small):
        """
        sst_mixed: 4 days at 19°C, 6 days at 21°C.
        mhw_mask: last 6 days True.
        SDD = (21 - 20) * 6 = 6 °C·day.
        """
        mhw_mask_data = np.zeros((1, 10, 1, 1), dtype=bool)
        mhw_mask_data[0, 4:, 0, 0] = True
        mhw_mask = xr.DataArray(mhw_mask_data, dims=sst_mixed.dims,
                                coords=sst_mixed.coords)
        sdd = accumulate_sdd(sst_mixed, threshold_20_small, mhw_mask)
        np.testing.assert_allclose(sdd.values[0, 0, 0], 6.0, atol=1e-4)

    def test_output_shape_drops_time_dim(self, sst_above, threshold_20):
        """Output has member, lat, lon dims — time dimension is summed out."""
        mhw_mask = xr.ones_like(sst_above, dtype=bool)
        sdd = accumulate_sdd(sst_above, threshold_20, mhw_mask)
        assert "time" not in sdd.dims
        assert set(sdd.dims) == {"member", "lat", "lon"}

    def test_sdd_always_nonnegative(self, sst_below, threshold_20):
        """
        Even if SST dips below threshold during a flagged day (shouldn't happen
        in practice but must not produce negative SDD).
        """
        # Force all days flagged but SST below threshold
        mhw_mask = xr.ones_like(sst_below, dtype=bool)
        sdd = accumulate_sdd(sst_below, threshold_20, mhw_mask)
        assert (sdd.values >= 0).all(), "SDD must never be negative"

    def test_member_values_are_independent(self, time_coord, threshold_20_small):
        """Each member accumulates its own SDD — no cross-member averaging."""
        data = np.zeros((2, 10, 1, 1), dtype=np.float32)
        data[0, :, :, :] = 22.0  # member 0: +2°C above threshold every day
        data[1, :, :, :] = 21.0  # member 1: +1°C above threshold every day
        sst = xr.DataArray(data, dims=["member", "time", "lat", "lon"],
                           coords={"member": [0, 1], "time": time_coord,
                                   "lat": [42.0], "lon": [-70.0]})
        mhw_mask = xr.ones_like(sst, dtype=bool)
        sdd = accumulate_sdd(sst, threshold_20_small, mhw_mask)
        np.testing.assert_allclose(sdd.sel(member=0).values, 20.0, atol=1e-4)
        np.testing.assert_allclose(sdd.sel(member=1).values, 10.0, atol=1e-4)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
conda run -n mhw-risk pytest tests/test_sdd.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'src.analytics.sdd'`

- [ ] **Step 3: Implement `src/analytics/sdd.py`**

```python
"""
sdd.py — Stress Degree Day accumulation for MHW risk quantification
=====================================================================
Computes Stress Degree Days (SDD) from SST anomalies during MHW events.

Physical note
-------------
Stress Degree Days are the thermal integral of SST above the MHW threshold,
accumulated only during active MHW event days (where mhw_mask is True):

    SDD = Σ_t max(SST_t − threshold_t, 0)   for all t where mhw_mask_t = True

Units: °C·day (thermal load, analogous to heating degree days in building energy).

SDD is the primary trigger variable for parametric insurance payout:
  - Low SDD  (<5 °C·day): no significant biological stress, no payout
  - Medium SDD (5-15 °C·day): sublethal stress, partial payout tier
  - High SDD  (>15 °C·day): mortality risk, full payout triggered

The 64-member ensemble produces 64 SDD realisations, from which SVaR is estimated.

Dependencies: xarray>=2024.2.0, numpy>=1.26.0
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def accumulate_sdd(
    sst: xr.DataArray,
    threshold: xr.DataArray,
    mhw_mask: xr.DataArray,
) -> xr.DataArray:
    """
    Accumulate Stress Degree Days above the MHW threshold during flagged event days.

    Parameters
    ----------
    sst : xr.DataArray
        Daily SST [°C], dims (member, time, lat, lon).
        HYCOM surface proxy: hycom_ds['water_temp'].isel(depth=0) broadcast to member dim.
        WeatherNext 2 production: sea_surface_temperature [converted from K to °C].
    threshold : xr.DataArray
        Climatological SST threshold [°C], dims (dayofyear, lat, lon).
        Produced by mhw_detection.compute_climatology().
        Must cover all day-of-year values present in sst.time.
    mhw_mask : xr.DataArray
        Boolean MHW event mask, same dims as sst: (member, time, lat, lon).
        Produced by mhw_detection.compute_mhw_mask().
        Only time steps where mask is True contribute to SDD.

    Returns
    -------
    sdd : xr.DataArray
        Stress Degree Days [°C·day], dims (member, lat, lon).
        All values ≥ 0. Time dimension is summed out.
        Pass sdd.values to torch.tensor() to feed into compute_svar() in svar.py.
    """
    # Align threshold to the SST time axis by day-of-year
    doy = sst.time.dt.dayofyear
    thresh_aligned = threshold.sel(dayofyear=doy)      # (time, lat, lon)

    # SST anomaly above threshold — clipped to 0 (no negative contributions)
    anomaly = (sst - thresh_aligned).clip(min=0.0)     # (member, time, lat, lon)

    # Mask out non-MHW days before summing
    masked_anomaly = anomaly.where(mhw_mask, other=0.0)

    # Sum over time axis to produce cumulative thermal load
    sdd = masked_anomaly.sum(dim="time")               # (member, lat, lon)

    sdd.attrs.update({
        "units": "degC day",
        "long_name": "Stress Degree Days above MHW threshold",
        "description": (
            "Cumulative thermal load above the 90th-percentile climatological SST "
            "threshold, accumulated only during active MHW event days. "
            "Primary trigger variable for parametric insurance payout."
        ),
    })
    return sdd
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n mhw-risk pytest tests/test_sdd.py -v 2>&1
```

Expected: All 6 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/analytics/sdd.py tests/test_sdd.py
git commit -m "feat: implement Stress Degree Day accumulation with MHW mask gating"
```

---

## Task 4: `svar.py` — tests first

**Files:**
- Create: `tests/test_svar.py`
- Create: `src/analytics/svar.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_svar.py
"""Unit tests for Stochastic Value-at-Risk estimation."""
import pytest
import torch

from src.analytics.svar import compute_svar, compute_ensemble_stats


class TestComputeSvar:
    def test_output_shape_is_batch(self, sdd_tensor):
        """SVaR output has shape (batch,) — one value per location/season."""
        svar = compute_svar(sdd_tensor, quantile=0.95)
        assert svar.shape == (2,)

    def test_svar_95_captures_high_member(self, sdd_tensor):
        """
        sdd_tensor batch 0: 7 members near 1-2 °C·day, member 7 = 10 °C·day.
        SVaR_95 of 8 members: top 5% = top 0.4 members → member 7 (10.0) dominates.
        """
        svar = compute_svar(sdd_tensor, quantile=0.95)
        assert svar[0].item() > 5.0, \
            f"SVaR_95 must reflect extreme member; got {svar[0].item():.3f}"

    def test_svar_50_is_median(self):
        """SVaR at quantile=0.50 is the ensemble median."""
        sdd = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])  # (1, 8)
        svar = compute_svar(sdd, quantile=0.50)
        assert abs(svar[0].item() - 4.5) < 0.01

    def test_uniform_ensemble_all_quantiles_equal(self):
        """All members identical → all quantiles equal."""
        sdd = torch.full((3, 64), 7.5)
        for q in [0.05, 0.50, 0.95, 0.99]:
            svar = compute_svar(sdd, quantile=q)
            torch.testing.assert_close(svar, torch.full((3,), 7.5))

    def test_quantile_monotone(self, sdd_tensor):
        """SVaR_99 >= SVaR_95 >= SVaR_50 for the same batch."""
        s50 = compute_svar(sdd_tensor, quantile=0.50)
        s95 = compute_svar(sdd_tensor, quantile=0.95)
        s99 = compute_svar(sdd_tensor, quantile=0.99)
        assert (s99 >= s95).all()
        assert (s95 >= s50).all()

    def test_invalid_quantile_raises(self, sdd_tensor):
        """Quantile outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="quantile"):
            compute_svar(sdd_tensor, quantile=1.5)
        with pytest.raises(ValueError, match="quantile"):
            compute_svar(sdd_tensor, quantile=0.0)

    def test_nonnegative_sdd_gives_nonnegative_svar(self):
        """All-zero SDD ensemble → SVaR = 0."""
        sdd = torch.zeros(4, 64)
        svar = compute_svar(sdd, quantile=0.95)
        assert (svar >= 0).all()
        torch.testing.assert_close(svar, torch.zeros(4))


class TestComputeEnsembleStats:
    def test_output_keys(self, sdd_tensor):
        """Returns dict with expected keys."""
        stats = compute_ensemble_stats(sdd_tensor)
        for key in ("mean", "std", "svar_50", "svar_90", "svar_95", "svar_99"):
            assert key in stats, f"Missing key: {key}"

    def test_shapes(self, sdd_tensor):
        """All stat tensors have shape (batch,)."""
        stats = compute_ensemble_stats(sdd_tensor)
        for key, val in stats.items():
            assert val.shape == (2,), f"{key} shape mismatch: {val.shape}"

    def test_mean_is_correct(self):
        """Mean computed correctly for a known input."""
        sdd = torch.tensor([[2.0, 4.0, 6.0, 8.0]])  # mean = 5.0
        stats = compute_ensemble_stats(sdd)
        torch.testing.assert_close(stats["mean"], torch.tensor([5.0]))
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
conda run -n mhw-risk pytest tests/test_svar.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'src.analytics.svar'`

- [ ] **Step 3: Implement `src/analytics/svar.py`**

```python
"""
svar.py — Stochastic Value-at-Risk estimation from ensemble SDD distributions
=============================================================================
Computes SVaR from the 64-member WeatherNext 2 ensemble SDD distribution
produced by MHWRiskModel or by the physics-based SDD accumulation pipeline.

Financial motivation
--------------------
SVaR (Stochastic Value-at-Risk) quantifies the tail risk of the ensemble SDD
distribution for parametric insurance pricing:

    SVaR_p = quantile(SDD_members, p)

Interpretation:
  SVaR_95 = 15 °C·day → in the worst 5% of atmospheric scenarios (as represented
  by the 64-member FGN ensemble), the aquaculture site accumulates 15 °C·day of
  thermal stress — sufficient to trigger the insurance payout threshold.

The spread between SVaR_50 and SVaR_95 reflects ensemble forecast uncertainty:
  Tight spread → low model uncertainty → high confidence in risk estimate
  Wide spread  → high model uncertainty → larger insurance risk premium required

This module is the final step of the 'Science-to-Insight' pipeline:
  WeatherNext 2 ensemble → MHWRiskModel → SDD per member → SVaR → payout probability

Dependencies: torch>=2.2.0
"""
from __future__ import annotations

import torch


def compute_svar(
    sdd: torch.Tensor,
    quantile: float = 0.95,
) -> torch.Tensor:
    """
    Estimate Stochastic Value-at-Risk from the ensemble SDD distribution.

    Parameters
    ----------
    sdd : torch.Tensor, shape (batch, member)
        Per-member Stress Degree Days [°C·day].
        Produced by MHWRiskModel.forward() or by accumulate_sdd() converted to tensor.
        All values should be ≥ 0 (guaranteed by MHWRiskModel Softplus head).
    quantile : float
        Probability level for VaR. Must be in (0, 1].
        Common choices:
          0.50 — median (expected SDD under the ensemble)
          0.90 — moderate tail risk (1-in-10 scenario)
          0.95 — standard VaR level used in insurance pricing
          0.99 — extreme tail risk (1-in-100 scenario)

    Returns
    -------
    svar : torch.Tensor, shape (batch,)
        SVaR at the requested quantile level [°C·day].
        One value per batch item (location × season combination).
    """
    if not (0.0 < quantile <= 1.0):
        raise ValueError(
            f"quantile must be in (0, 1], got {quantile}. "
            "Common values: 0.50 (median), 0.95 (standard VaR), 0.99 (extreme tail)."
        )
    return torch.quantile(sdd.float(), q=quantile, dim=1)   # (batch,)


def compute_ensemble_stats(
    sdd: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute a full set of ensemble statistics for reporting and model monitoring.

    Parameters
    ----------
    sdd : torch.Tensor, shape (batch, member)
        Per-member SDD [°C·day] from MHWRiskModel or physics-based accumulation.

    Returns
    -------
    stats : dict[str, torch.Tensor]
        Keys and shapes (all shape (batch,)):
          "mean"    — ensemble mean SDD [°C·day]
          "std"     — ensemble standard deviation (uncertainty spread)
          "svar_50" — median SVaR (expected scenario)
          "svar_90" — 90th-percentile SVaR (moderate tail)
          "svar_95" — 95th-percentile SVaR (standard insurance VaR level)
          "svar_99" — 99th-percentile SVaR (extreme tail, stress-test scenario)

        Financial interpretation of std:
          High std → large spread between worst and median member → higher risk premium
          Low std  → ensemble agrees → lower uncertainty loading on the insurance price
    """
    sdd_f = sdd.float()
    return {
        "mean":    sdd_f.mean(dim=1),
        "std":     sdd_f.std(dim=1),
        "svar_50": torch.quantile(sdd_f, 0.50, dim=1),
        "svar_90": torch.quantile(sdd_f, 0.90, dim=1),
        "svar_95": torch.quantile(sdd_f, 0.95, dim=1),
        "svar_99": torch.quantile(sdd_f, 0.99, dim=1),
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n mhw-risk pytest tests/test_svar.py -v 2>&1
```

Expected: All 10 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/analytics/svar.py tests/test_svar.py
git commit -m "feat: implement SVaR estimation with ensemble quantile statistics"
```

---

## Task 5: Wire up `src/analytics/__init__.py` and run full test suite

**Files:**
- Modify: `src/analytics/__init__.py`

- [ ] **Step 1: Update `__init__.py`**

```python
# src/analytics/__init__.py
from .mhw_detection import compute_climatology, compute_mhw_mask
from .sdd import accumulate_sdd
from .svar import compute_svar, compute_ensemble_stats

__all__ = [
    "compute_climatology",
    "compute_mhw_mask",
    "accumulate_sdd",
    "compute_svar",
    "compute_ensemble_stats",
]
```

- [ ] **Step 2: Run the full test suite**

```bash
conda run -n mhw-risk pytest tests/ -v 2>&1
```

Expected: All tests PASSED. Output should show test counts for all three test files.

- [ ] **Step 3: Commit**

```bash
git add src/analytics/__init__.py
git commit -m "feat: export analytics public API from src/analytics/__init__.py"
```

---

## Task 6: Integration smoke test

This validates the end-to-end analytics pipeline using HYCOM data (no WeatherNext 2 required).

- [ ] **Step 1: Run the smoke test**

```bash
conda run -n mhw-risk python -c "
import numpy as np
import torch
import xarray as xr

from src.analytics.mhw_detection import compute_climatology, compute_mhw_mask
from src.analytics.sdd import accumulate_sdd
from src.analytics.svar import compute_svar, compute_ensemble_stats

# --- Load HYCOM sample data ---
hycom = xr.open_zarr('data/processed/hycom_2019-08-01_2019-08-03.zarr', consolidated=False)
print('HYCOM loaded:', hycom)

# Surface SST proxy: water_temp at depth index 0 (0 m), shape (time, lat, lon)
sst_surface = hycom['water_temp'].isel(depth=0).load()
print('SST surface shape:', sst_surface.shape, 'dims:', sst_surface.dims)

# Broadcast to a synthetic member dimension (size 4)
sst_ensemble = sst_surface.expand_dims(dim={'member': 4}).assign_coords(member=np.arange(4))
print('SST ensemble shape:', sst_ensemble.shape)

# Synthetic climatological threshold: constant 18°C (Gulf of Maine salmon stress onset)
n_lat, n_lon = sst_surface.sizes['lat'], sst_surface.sizes['lon']
threshold = xr.DataArray(
    np.full((365, n_lat, n_lon), 18.0, dtype=np.float32),
    dims=['dayofyear', 'lat', 'lon'],
    coords={'dayofyear': np.arange(1, 366),
            'lat': sst_surface['lat'].values,
            'lon': sst_surface['lon'].values},
)
print('Threshold shape:', threshold.shape)

# MHW detection
mhw_mask = compute_mhw_mask(sst_ensemble, threshold, min_duration=2)
n_mhw = int(mhw_mask.sum())
print(f'MHW days detected: {n_mhw} of {mhw_mask.size} total cell-days')

# SDD accumulation
sdd_xr = accumulate_sdd(sst_ensemble, threshold, mhw_mask)
print('SDD shape:', sdd_xr.shape, '| range:', float(sdd_xr.min()), '-', float(sdd_xr.max()), 'degC.day')

# SVaR from physics-based SDD (flatten lat/lon into batch dim)
sdd_flat = sdd_xr.values.reshape(4, -1).T   # (lat*lon, member)
sdd_tensor = torch.tensor(sdd_flat, dtype=torch.float32)
svar = compute_svar(sdd_tensor, quantile=0.95)
stats = compute_ensemble_stats(sdd_tensor)

print('SVaR_95 shape:', svar.shape, '| mean:', float(svar.mean()), 'degC.day')
print('Ensemble stats keys:', list(stats.keys()))
print('mean SDD across locations:', float(stats['mean'].mean()))
print()
print('=== Integration smoke test PASSED ===')
" 2>&1
```

Expected output: No errors, shapes printed, `Integration smoke test PASSED`.

- [ ] **Step 2: Commit nothing** — smoke test is a verification step, not a code change.

---

## Task 7: Update `mhw_claude_todo.md`

- [ ] **Step 1: Update the todo file**

Move "Implement MHW Detection & SVaR Analytics" from QUEUED to COMPLETED. Append to the COMPLETED section:

```
- [2026-03-27] MHW Detection & SVaR Analytics implemented and tested (src/analytics/)
  - mhw_detection.py: compute_climatology(), compute_mhw_mask() — Hobday 2016 Category I
  - sdd.py: accumulate_sdd() — thermal load above threshold, MHW-mask gated
  - svar.py: compute_svar(), compute_ensemble_stats() — ensemble quantile VaR
  - Full test suite: tests/test_mhw_detection.py, test_sdd.py, test_svar.py
  - Integration smoke test passed with HYCOM proxy (18°C constant threshold, member=4)
  - End-to-end with real WeatherNext 2 blocked until Google whitelist (see PENDING)
```

- [ ] **Step 2: Commit**

```bash
git add mhw_claude_actions/mhw_claude_todo.md
git commit -m "docs: mark MHW analytics task complete in todo"
```

---

## Verification Gate (CLAUDE.md)

Task is complete when:
1. `pytest tests/ -v` exits 0 with all tests PASSED
2. Integration smoke test prints `Integration smoke test PASSED`
3. Both outputs are saved as evidence in `mhw_claude_recentactions.md`

---

## Known Limitation

The end-to-end pipeline (WeatherNext 2 SST → 64-member SDD → SVaR) requires WeatherNext 2
GEE access (currently PENDING whitelist). The analytics layer is fully testable and
production-ready; only the data feed is blocked. No code changes are required once
the whitelist is granted — the harmonizer already outputs the correct shapes.
