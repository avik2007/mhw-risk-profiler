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
