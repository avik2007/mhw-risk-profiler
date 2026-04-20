"""
tests/test_oisst_climatology.py — Unit tests for fetch_oisst_climatology.py.
All network calls mocked at the xr.open_mfdataset boundary.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from unittest.mock import patch, MagicMock


def _make_fake_oisst(n_days: int = 28, n_lat: int = 17, n_lon: int = 21) -> xr.Dataset:
    """Minimal OISST-shaped dataset (time, zlev, lat, lon) for mocking."""
    rng = np.random.default_rng(42)
    times = xr.cftime_range("1982-01-01", periods=n_days, freq="D")
    sst = xr.DataArray(
        rng.normal(20.0, 2.0, (n_days, 1, n_lat, n_lon)).astype("float32"),
        dims=["time", "zlev", "lat", "lon"],
        coords={
            "time": times,
            "zlev": [0.0],
            "lat":  np.linspace(41.125, 44.875, n_lat),
            "lon":  np.linspace(-70.875, -66.125, n_lon),
        },
        attrs={"units": "degree_C"},
    )
    return xr.Dataset({"sst": sst})


def test_build_oisst_url_format():
    """URL must embed year, YYYYMM, and YYYYMMDD and start with https."""
    from scripts.fetch_oisst_climatology import build_oisst_url
    url = build_oisst_url(1982, 1, 1)
    assert url.startswith("https://")
    assert "1982" in url
    assert "198201" in url
    assert "19820101" in url


def test_build_oisst_url_zero_pads_month_and_day():
    """Month and day must be zero-padded to 2 digits."""
    from scripts.fetch_oisst_climatology import build_oisst_url
    url = build_oisst_url(1990, 3, 5)
    assert "199003" in url
    assert "19900305" in url


def _make_fake_response(day_ds: xr.Dataset) -> MagicMock:
    """Fake requests.Response whose .content is ignored; open_dataset is mocked separately."""
    resp = MagicMock()
    resp.content = b"fake-netcdf-bytes"
    resp.raise_for_status = MagicMock()
    return resp


def test_fetch_oisst_gom_drops_zlev():
    """fetch_oisst_gom must squeeze zlev and return (time, lat, lon) DataArray."""
    from scripts.fetch_oisst_climatology import fetch_oisst_gom

    single_day = _make_fake_oisst(n_days=1)
    fake_resp = _make_fake_response(single_day)

    with patch("scripts.fetch_oisst_climatology.requests.get", return_value=fake_resp), \
         patch("scripts.fetch_oisst_climatology.xr.open_dataset", return_value=single_day):
        sst = fetch_oisst_gom(1982, 1)

    assert "zlev" not in sst.dims
    assert set(sst.dims) == {"time", "lat", "lon"}
    assert sst.dtype == np.float32


def test_fetch_oisst_gom_subsets_to_gom_bbox():
    """fetch_oisst_gom must subset lat/lon to the GoM bbox."""
    from scripts.fetch_oisst_climatology import fetch_oisst_gom, GoM_LAT_MIN, GoM_LAT_MAX, GoM_LON_MIN, GoM_LON_MAX

    n_lat_wide, n_lon_wide = 40, 60
    rng = np.random.default_rng(0)
    times = xr.cftime_range("1982-01-01", periods=1, freq="D")
    sst = xr.DataArray(
        rng.normal(20.0, 1.0, (1, 1, n_lat_wide, n_lon_wide)).astype("float32"),
        dims=["time", "zlev", "lat", "lon"],
        coords={
            "time": times,
            "zlev": [0.0],
            "lat":  np.linspace(30.0, 55.0, n_lat_wide),
            "lon":  np.linspace(-90.0, -55.0, n_lon_wide),
        },
    )
    wide_ds = xr.Dataset({"sst": sst})
    fake_resp = _make_fake_response(wide_ds)

    with patch("scripts.fetch_oisst_climatology.requests.get", return_value=fake_resp), \
         patch("scripts.fetch_oisst_climatology.xr.open_dataset", return_value=wide_ds):
        result = fetch_oisst_gom(1982, 1)

    assert float(result.lat.min()) >= GoM_LAT_MIN - 0.5
    assert float(result.lat.max()) <= GoM_LAT_MAX + 0.5
    assert float(result.lon.min()) >= GoM_LON_MIN - 0.5
    assert float(result.lon.max()) <= GoM_LON_MAX + 0.5


def test_compute_and_write_climatology_writes_sst_threshold_90():
    """Must call _gcs_safe_write with a dataset containing sst_threshold_90."""
    from scripts.fetch_oisst_climatology import compute_and_write_climatology

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
         patch("scripts.fetch_oisst_climatology._gcs_complete", return_value=False), \
         patch("scripts.fetch_oisst_climatology.gcsfs.GCSFileSystem"):
        compute_and_write_climatology(fake_sst, "gs://fake-bucket/hycom/climatology/")

    mock_write.assert_called_once()
    ds_written = mock_write.call_args[0][0]
    assert "sst_threshold_90" in ds_written


def test_compute_and_write_climatology_skips_if_complete():
    """Must skip GCS write if .complete sentinel already present."""
    from scripts.fetch_oisst_climatology import compute_and_write_climatology

    fake_sst = xr.DataArray(
        np.ones((10, 2, 2), dtype="float32"),
        dims=["time", "lat", "lon"],
        coords={"time": xr.cftime_range("1982-01-01", periods=10, freq="D"),
                "lat": [41.0, 42.0], "lon": [-71.0, -70.0]},
    )

    with patch("scripts.fetch_oisst_climatology._gcs_safe_write") as mock_write, \
         patch("scripts.fetch_oisst_climatology._gcs_complete", return_value=True), \
         patch("scripts.fetch_oisst_climatology.gcsfs.GCSFileSystem"):
        compute_and_write_climatology(fake_sst, "gs://fake-bucket/hycom/climatology/")

    mock_write.assert_not_called()
