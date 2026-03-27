# tests/test_mhw_detection.py
"""Unit tests for MHW event detection (Hobday et al. 2016 Category I)."""
import numpy as np
import pandas as pd
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
        times = xr.date_range("2017-01-01", periods=365 * 3, freq="D", use_cftime=True)
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
        times = xr.date_range("2017-01-01", periods=365 * 3, freq="D", use_cftime=True)
        data = np.full((365 * 3, 1, 1), 20.0, dtype=np.float32)
        sst_hist = xr.DataArray(data, dims=["time", "lat", "lon"],
                                coords={"time": times, "lat": [42.0], "lon": [-70.0]})
        clim = compute_climatology(sst_hist, percentile=90)
        np.testing.assert_allclose(clim.values, 20.0, atol=1e-4)
