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
