# Pre-WeatherNext Analytics: Payout Engine + HYCOM Climatological Threshold

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the two analytics components that are fully independent of WeatherNext 2: (1) the parametric insurance payout engine; (2) a location-varying MHW threshold derived from HYCOM historical surface SST via `compute_climatology()`.

**Architecture:** Two independent tasks. Task 1 is pure math (no data). Task 2 fetches 2 years of HYCOM surface SST (depth=0 only — much lighter than the full 3D tile), runs the existing `compute_climatology()` function, and saves a `(dayofyear=365, lat, lon)` threshold Zarr that all downstream MHW detection calls can load instead of using a hardcoded constant.

**Tech Stack:** `xarray>=2024.2.0`, `numpy>=1.26.0`, `torch>=2.2.0`, `pytest>=8.0.0`. No new dependencies.

---

## File Map

| Action  | Path                                       | Responsibility                                                   |
|---------|--------------------------------------------|------------------------------------------------------------------|
| Create  | `src/analytics/payout.py`                  | Parametric insurance payout from SVaR                            |
| Create  | `tests/test_payout.py`                     | Unit tests for payout.py                                         |
| Modify  | `src/analytics/__init__.py`                | Export `compute_payout`, `compute_expected_loss_ratio`           |
| Create  | `scripts/compute_hycom_climatology.py`     | Fetch 2yr HYCOM surface SST, compute 90th-pct threshold, save Zarr |
| Create  | `tests/test_climatology_script.py`         | Unit test for threshold shape/values using a synthetic fixture   |

---

## Physical Reference

**Location-varying threshold:**
`compute_climatology(sst_historical)` already exists in `src/analytics/mhw_detection.py`.
It groups SST by `time.dayofyear` and computes the 90th percentile at each grid cell,
returning a `(dayofyear=365, lat, lon)` DataArray. The script fetches 2 years of
HYCOM surface SST (2018 + 2019, depth=0 only) and passes it directly to this function.
Two years is a short baseline — production will use the full 1982–2011 climatology once
WeatherNext 2 access is confirmed and a longer HYCOM or satellite SST record is fetched.
The output Zarr at `data/processed/hycom_sst_threshold.zarr` replaces every hardcoded
`18.0` threshold constant in the analytics pipeline.

**Insurance payout structure:**
- Attachment point `a` [°C·day]: below this, zero payout.
- Cap `c` [°C·day]: above this, maximum payout.
- Coverage `V` [USD]: total insured value.
- `payout = V × clip((SVaR − a) / (c − a), 0, 1)`

---

## Task 1: Parametric payout engine

**Files:**
- Create: `src/analytics/payout.py`
- Create: `tests/test_payout.py`
- Modify: `src/analytics/__init__.py`

---

- [ ] **Step 1: Write failing tests**

```python
# tests/test_payout.py
"""Unit tests for the parametric insurance payout engine."""
import pytest
import torch

from src.analytics.payout import compute_payout, compute_expected_loss_ratio


class TestComputePayout:
    def test_below_attachment_zero_payout(self):
        """SVaR below attachment → zero payout."""
        svar = torch.tensor([10.0, 0.0, 19.9])
        payout = compute_payout(svar, attachment=20.0, cap=60.0, coverage=1_000_000.0)
        assert torch.all(payout == 0.0), f"expected all zeros, got {payout}"

    def test_above_cap_full_payout(self):
        """SVaR at or above cap → full coverage payout."""
        svar = torch.tensor([60.0, 100.0, 999.0])
        payout = compute_payout(svar, attachment=20.0, cap=60.0, coverage=1_000_000.0)
        assert torch.allclose(payout, torch.full((3,), 1_000_000.0)), f"expected 1M, got {payout}"

    def test_midpoint_half_payout(self):
        """SVaR at midpoint between attachment and cap → half coverage."""
        svar = torch.tensor([40.0])  # midpoint of [20, 60]
        payout = compute_payout(svar, attachment=20.0, cap=60.0, coverage=1_000_000.0)
        assert payout.item() == pytest.approx(500_000.0, abs=1.0)

    def test_batch_shape_preserved(self):
        """Output shape matches input shape."""
        svar = torch.zeros(3, 5)  # batch=3, member=5
        payout = compute_payout(svar, attachment=20.0, cap=60.0, coverage=1_000_000.0)
        assert payout.shape == (3, 5)

    def test_attachment_equals_cap_raises(self):
        """attachment == cap is undefined (division by zero)."""
        with pytest.raises(ValueError, match="cap.*must be.*greater than.*attachment"):
            compute_payout(torch.tensor([30.0]), attachment=20.0, cap=20.0, coverage=1e6)

    def test_cap_less_than_attachment_raises(self):
        """cap < attachment is nonsensical."""
        with pytest.raises(ValueError, match="cap.*must be.*greater than.*attachment"):
            compute_payout(torch.tensor([30.0]), attachment=60.0, cap=20.0, coverage=1e6)

    def test_exact_attachment_boundary(self):
        """SVaR exactly at attachment → zero payout (not partial)."""
        svar = torch.tensor([20.0])
        payout = compute_payout(svar, attachment=20.0, cap=60.0, coverage=1_000_000.0)
        assert payout.item() == pytest.approx(0.0, abs=1e-5)


class TestComputeExpectedLossRatio:
    def test_full_loss_ratio_at_cap(self):
        """All members at cap SDD → loss_ratio_95 = 1.0."""
        sdd = torch.full((2, 8), 60.0)  # all members at cap
        result = compute_expected_loss_ratio(sdd, attachment=20.0, cap=60.0, coverage=1e6)
        assert torch.allclose(result["loss_ratio_95"], torch.ones(2), atol=1e-4)

    def test_zero_loss_ratio_below_attachment(self):
        """All members below attachment → loss_ratio_95 = 0.0."""
        sdd = torch.full((2, 8), 5.0)
        result = compute_expected_loss_ratio(sdd, attachment=20.0, cap=60.0, coverage=1e6)
        assert torch.all(result["loss_ratio_95"] == 0.0)

    def test_output_keys_present(self):
        """Result dict contains the four required keys."""
        sdd = torch.rand(2, 8) * 80.0
        result = compute_expected_loss_ratio(sdd, attachment=20.0, cap=60.0, coverage=1e6)
        for key in ("svar_95", "payout_95", "loss_ratio_95", "payout_50"):
            assert key in result, f"missing key: {key}"

    def test_loss_ratio_in_unit_interval(self):
        """Loss ratio is always in [0, 1]."""
        sdd = torch.rand(10, 8) * 100.0  # random SDDs spanning [0, 100]
        result = compute_expected_loss_ratio(sdd, attachment=20.0, cap=60.0, coverage=1e6)
        assert (result["loss_ratio_95"] >= 0.0).all()
        assert (result["loss_ratio_95"] <= 1.0).all()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n mhw-risk pytest tests/test_payout.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'compute_payout' from 'src.analytics.payout'`

---

- [ ] **Step 3: Implement `src/analytics/payout.py`**

```python
"""
payout.py — Parametric insurance payout engine for MHW aquaculture risk
========================================================================
Translates ensemble SVaR estimates into insurance payout amounts for
parametric aquaculture insurance contracts triggered by Stress Degree Days.

Parametric insurance structure
-------------------------------
A parametric contract pays out automatically when a measurable index (here,
SVaR — the 95th-percentile SDD across ensemble members) crosses a pre-agreed
attachment point. There is no claims adjustment; payment is triggered purely
by the index value.

  Attachment point a [°C·day]:  SVaR must exceed this before any payout.
  Cap c [°C·day]:               SVaR at or above this triggers full payout.
  Coverage V [USD]:             Maximum insurance payout (sum insured).

  payout = V × clip((SVaR − a) / (c − a), 0, 1)

Financial motivation
---------------------
SVaR_95 = 40 °C·day with attachment = 20, cap = 60, coverage = $1M →
payout = $1M × (40−20)/(60−20) = $500,000.

The spread between SVaR_50 and SVaR_95 guides reinsurance pricing:
  tight spread  → low model uncertainty → lower uncertainty loading
  wide spread   → high model uncertainty → higher risk premium required

Dependencies: torch>=2.2.0
"""
from __future__ import annotations

import torch


def compute_payout(
    svar: torch.Tensor,
    attachment: float,
    cap: float,
    coverage: float,
) -> torch.Tensor:
    """
    Compute parametric insurance payout from SVaR estimates.

    Parameters
    ----------
    svar : torch.Tensor
        Stochastic Value-at-Risk [°C·day].  Any shape — the function is
        element-wise.  Typically shape (batch,) from compute_svar(), or
        shape (batch, member) for per-member analysis.
    attachment : float
        Attachment point [°C·day].  SVaR below this triggers zero payout.
        Typical range: 15–25 °C·day for Gulf of Maine salmon aquaculture.
    cap : float
        Cap [°C·day].  SVaR at or above this triggers full coverage payout.
        Must be strictly greater than attachment.
        Typical range: 40–80 °C·day depending on species and season length.
    coverage : float
        Sum insured [USD].  Maximum payout per contract.
        Set to the replacement cost of at-risk biomass plus operating losses.

    Returns
    -------
    payout : torch.Tensor
        Insurance payout [USD], same shape as svar.
        All values in [0, coverage].
    """
    if cap <= attachment:
        raise ValueError(
            f"cap ({cap}) must be greater than attachment ({attachment}). "
            "A contract where cap ≤ attachment is undefined (denominator ≤ 0)."
        )
    denom = cap - attachment
    rate = torch.clamp((svar - attachment) / denom, min=0.0, max=1.0)
    return coverage * rate


def compute_expected_loss_ratio(
    sdd: torch.Tensor,
    attachment: float,
    cap: float,
    coverage: float,
) -> dict[str, torch.Tensor]:
    """
    Compute SVaR-based payout and loss ratio statistics from the full ensemble.

    Parameters
    ----------
    sdd : torch.Tensor, shape (batch, member)
        Per-member Stress Degree Days [°C·day] from MHWRiskModel or
        from accumulate_sdd() converted to a tensor.
    attachment : float
        Attachment point [°C·day].  See compute_payout() for details.
    cap : float
        Cap [°C·day].  See compute_payout() for details.
    coverage : float
        Sum insured [USD].  See compute_payout() for details.

    Returns
    -------
    result : dict[str, torch.Tensor]
        All tensors have shape (batch,):
          "svar_95"      — SVaR at 95th percentile [°C·day]
          "payout_95"    — insurance payout triggered by SVaR_95 [USD]
          "loss_ratio_95"— payout_95 / coverage ∈ [0, 1]
          "payout_50"    — payout triggered by SVaR_50 (median scenario) [USD]
    """
    from .svar import compute_ensemble_stats

    stats = compute_ensemble_stats(sdd)
    svar_95  = stats["svar_95"]
    svar_50  = stats["svar_50"]
    payout_95 = compute_payout(svar_95, attachment, cap, coverage)
    payout_50 = compute_payout(svar_50, attachment, cap, coverage)
    return {
        "svar_95":       svar_95,
        "payout_95":     payout_95,
        "loss_ratio_95": payout_95 / coverage,
        "payout_50":     payout_50,
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
conda run -n mhw-risk pytest tests/test_payout.py -v
```

Expected output (truncated):
```
PASSED tests/test_payout.py::TestComputePayout::test_below_attachment_zero_payout
PASSED tests/test_payout.py::TestComputePayout::test_above_cap_full_payout
...
11 passed in 0.XXs
```

---

- [ ] **Step 5: Export from `src/analytics/__init__.py`**

Read the current file first, then add the two new names. Current content ends with the existing exports. Append:

```python
from .payout import compute_expected_loss_ratio, compute_payout
```

- [ ] **Step 6: Commit**

```bash
git add src/analytics/payout.py src/analytics/__init__.py tests/test_payout.py
git commit -m "feat: add parametric insurance payout engine with SVaR loss-ratio stats"
```

---

## Task 2: HYCOM location-varying climatological threshold

**Files:**
- Create: `scripts/compute_hycom_climatology.py`
- Create: `tests/test_climatology_script.py`

This script fetches HYCOM surface SST (depth=0 only — no full 3D profile) for 2018 and 2019
via the existing `HYCOMLoader.fetch_tile()`, concatenates both years, and passes the result
to the existing `compute_climatology()` to produce the per-cell 90th-percentile threshold.
Fetching surface-only is ~10× lighter than the full 3D tile used in training.

> **Runtime note:** Surface-only fetch (depth=0, all lat/lon, daily) for 2 years × 365 days
> via OPeNDAP. Expect 3–8 minutes total. Both years are fetched independently and concatenated
> before computing the climatology.

---

- [ ] **Step 1: Write failing test**

```python
# tests/test_climatology_script.py
"""
Unit test for compute_hycom_climatology.py logic.
Uses a synthetic 2-year SST DataArray — no OPeNDAP access required.
"""
import numpy as np
import pytest
import xarray as xr

from src.analytics.mhw_detection import compute_climatology


def make_synthetic_sst(n_years: int = 2) -> xr.DataArray:
    """
    Two years of daily SST at a 3×3 grid, starting 2018-01-01.
    Values are latitude-dependent so the threshold varies by location.
    """
    times = np.array(
        [np.datetime64("2018-01-01") + np.timedelta64(i, "D") for i in range(365 * n_years)]
    )
    lats = np.array([42.0, 43.0, 44.0])
    lons = np.array([-70.0, -69.5, -69.0])

    rng = np.random.default_rng(0)
    # Base temperature decreases with latitude (warmer south)
    base = 20.0 - (lats[:, np.newaxis] - 42.0) * 2.0   # (lat, lon) broadcast
    data = (
        base[np.newaxis, :, :]
        + rng.normal(0, 1.5, (len(times), len(lats), len(lons)))
    ).astype(np.float32)

    return xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
    )


class TestComputeClimatology:
    def test_output_shape(self):
        """Threshold has shape (dayofyear=365, lat, lon)."""
        sst = make_synthetic_sst()
        threshold = compute_climatology(sst, percentile=90.0)
        assert threshold.dims == ("dayofyear", "lat", "lon")
        assert threshold.sizes["dayofyear"] == 365
        assert threshold.sizes["lat"] == 3
        assert threshold.sizes["lon"] == 3

    def test_location_varying(self):
        """Southern cells (higher base SST) have higher threshold than northern cells."""
        sst = make_synthetic_sst()
        threshold = compute_climatology(sst, percentile=90.0)
        south_mean = float(threshold.sel(lat=42.0).mean())
        north_mean = float(threshold.sel(lat=44.0).mean())
        assert south_mean > north_mean, (
            f"south threshold {south_mean:.2f} should exceed north {north_mean:.2f}"
        )

    def test_no_nan_in_threshold(self):
        """No NaN values — all grid cells have enough data for 90th percentile."""
        sst = make_synthetic_sst()
        threshold = compute_climatology(sst, percentile=90.0)
        assert not threshold.isnull().any(), "unexpected NaN in climatological threshold"

    def test_threshold_above_median(self):
        """90th percentile must be above the median at every grid cell."""
        sst = make_synthetic_sst()
        p90 = compute_climatology(sst, percentile=90.0)
        p50 = compute_climatology(sst, percentile=50.0)
        assert (p90 >= p50).all(), "90th percentile must be ≥ 50th everywhere"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
conda run -n mhw-risk pytest tests/test_climatology_script.py -v 2>&1 | head -15
```

Expected: all 4 tests pass immediately — `compute_climatology` already exists.
(This step confirms the existing function satisfies the contract before we build the script around it.)

---

- [ ] **Step 3: Create `scripts/compute_hycom_climatology.py`**

```python
"""
compute_hycom_climatology.py — Build a location-varying MHW threshold from HYCOM surface SST.

Fetches 2 years of HYCOM surface SST (depth=0 only — no full 3D profile) for the Gulf of
Maine region, runs compute_climatology() to compute the 90th-percentile SST per calendar day
at each grid cell, and saves the result to data/processed/hycom_sst_threshold.zarr.

The saved threshold replaces hardcoded constant thresholds (e.g. 18°C) in all downstream
compute_mhw_mask() and accumulate_sdd() calls. Each grid cell gets its own seasonal
baseline, which means MHW events are detected relative to what is locally anomalous —
not a single species-specific temperature cutoff.

HYCOM expt_93.0 coverage: 2018-01-01 to 2020-02-19.
Valid full years for this script: 2018 and 2019.

Usage:
    conda run -n mhw-risk python scripts/compute_hycom_climatology.py
    conda run -n mhw-risk python scripts/compute_hycom_climatology.py --output-dir data/processed

Expected output:
    Fetching HYCOM surface SST for 2018 (2018-01-01 to 2018-12-31) ...
    Fetching HYCOM surface SST for 2019 (2019-01-01 to 2019-12-31) ...
    Computing 90th-percentile climatology over 730 daily timesteps ...
    Saved: data/processed/hycom_sst_threshold.zarr  dims={'dayofyear': 365, 'lat': 26, 'lon': 13}
    Threshold range: 8.34°C – 22.17°C  (spatial + seasonal spread)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import xarray as xr

from src.analytics.mhw_detection import compute_climatology
from src.ingestion.harvester import HYCOMLoader

# Gulf of Maine bounding box — matches the existing processed Zarr tile
BBOX = (-71.0, 43.0, -69.5, 44.0)

# Fetch full calendar years — avoids partial-year bias in the percentile estimate
YEARS = (2018, 2019)


def fetch_surface_sst_year(year: int) -> xr.DataArray:
    """
    Fetch one full year of daily HYCOM surface SST (depth=0 only).

    Parameters
    ----------
    year : int
        Calendar year. Must be within HYCOM expt_93.0 coverage (2018–2019).

    Returns
    -------
    sst : xr.DataArray
        Daily mean surface SST [°C], dims (time=365, lat, lon).
        Depth=0 corresponds to TARGET_DEPTHS_M[0] = 0 m (ocean surface).
    """
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    print(f"Fetching HYCOM surface SST for {year} ({start} to {end}) ...")

    loader = HYCOMLoader()
    ds = loader.fetch_tile(start, end, BBOX)

    # Resample 3-hourly → daily mean, then extract surface layer only
    sst = (
        ds["water_temp"]
        .isel(depth=0)              # surface (0 m)
        .resample(time="1D").mean() # daily mean
    )
    sst.load()  # evaluate Dask graph before returning
    return sst  # (time=365, lat, lon)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute location-varying MHW threshold from HYCOM surface SST."
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to save the threshold Zarr (default: data/processed)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "hycom_sst_threshold.zarr"

    if out_path.exists():
        print(f"Cache hit — {out_path} already exists. Delete it to recompute.")
        return

    # Fetch both years and concatenate along time axis
    sst_years = [fetch_surface_sst_year(year) for year in YEARS]
    sst_all = xr.concat(sst_years, dim="time")  # (time=730, lat, lon)
    print(f"Computing 90th-percentile climatology over {len(sst_all.time)} daily timesteps ...")

    threshold = compute_climatology(sst_all, percentile=90.0)
    # threshold: (dayofyear=365, lat, lon)

    threshold.to_dataset(name="sst_threshold_90").to_zarr(str(out_path), mode="w")

    # Verification
    ds_check = xr.open_zarr(str(out_path))
    t = ds_check["sst_threshold_90"]
    print(
        f"Saved: {out_path}  dims={dict(ds_check.dims)}\n"
        f"Threshold range: {float(t.min()):.2f}°C – {float(t.max()):.2f}°C  "
        f"(spatial + seasonal spread)"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify the script parses without error**

```bash
conda run -n mhw-risk python scripts/compute_hycom_climatology.py --help
```

Expected:
```
usage: compute_hycom_climatology.py [-h] [--output-dir OUTPUT_DIR]
```

- [ ] **Step 5: Run the full fetch (network required, ~3–8 minutes)**

```bash
conda run -n mhw-risk python scripts/compute_hycom_climatology.py 2>&1 | tee /tmp/climatology.log
```

Expected final lines:
```
Computing 90th-percentile climatology over 730 daily timesteps ...
Saved: data/processed/hycom_sst_threshold.zarr  dims={'dayofyear': 365, 'lat': ..., 'lon': ...}
Threshold range: X.XX°C – XX.XX°C  (spatial + seasonal spread)
```

- [ ] **Step 6: Spot-check the threshold in Python**

```bash
conda run -n mhw-risk python -c "
import xarray as xr
ds = xr.open_zarr('data/processed/hycom_sst_threshold.zarr')
t = ds['sst_threshold_90']
print('Shape:', dict(t.sizes))
print('Summer peak (doy=213 = Aug 1):', t.sel(dayofyear=213).values.mean().round(2), 'degC mean across grid')
print('Winter trough (doy=15 = Jan 15):', t.sel(dayofyear=15).values.mean().round(2), 'degC mean across grid')
print('Spatial std at summer peak (location-varying signal):', t.sel(dayofyear=213).values.std().round(3))
"
```

Expected: summer mean noticeably higher than winter mean; spatial std > 0 (confirms location-varying, not flat).

- [ ] **Step 7: Commit**

```bash
git add scripts/compute_hycom_climatology.py tests/test_climatology_script.py
git commit -m "feat: add HYCOM location-varying climatological MHW threshold script"
```

---

## Self-Review

**Spec coverage:**
- Parametric payout engine → Task 1 ✓
- Location-varying MHW threshold → Task 2 ✓
- No WN2 proxy training — deferred until GEE whitelist ✓
- No hardcoded 18°C constant in Task 2 — threshold is data-derived per grid cell ✓

**Placeholder scan:** No TBD, TODO, or "similar to" references. All code blocks complete.

**Type consistency:**
- `compute_climatology(sst_all, percentile=90.0)` signature matches `mhw_detection.py` definition ✓
- `threshold.to_dataset(name="sst_threshold_90")` → loaded as `ds["sst_threshold_90"]` in spot-check ✓
- `compute_payout` / `compute_expected_loss_ratio` names consistent between Task 1 impl and `__init__.py` export ✓

