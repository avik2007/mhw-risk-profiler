# tests/test_sdd.py
"""Unit tests for Stress Degree Day accumulation."""
import numpy as np
import pytest
import xarray as xr

from src.analytics.sdd import accumulate_sdd


class TestAccumulateSdd:
    def test_all_above_accumulates_correctly(self, sst_above, threshold_20):
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
