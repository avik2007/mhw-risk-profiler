"""
Unit tests for HYCOMLoader.fetch_and_cache() and ERA5Harvester.fetch_and_cache().
No network calls — gcsfs and fetch methods are fully mocked.
"""
import calendar
from unittest.mock import call, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.ingestion.harvester import HYCOMLoader


def _make_monthly_ds(start: str, end: str, bbox=None) -> xr.Dataset:
    """Minimal HYCOM-shaped Dataset covering [start, end] at daily frequency."""
    times = pd.date_range(start, end, freq="D")
    return xr.Dataset(
        {"water_temp": xr.DataArray(
            np.zeros((len(times), 1, 1, 1), dtype=np.float32),
            dims=["time", "depth", "lat", "lon"],
            coords={"time": times},
        )}
    )


def _fake_open_zarr_2022(uri: str, **kwargs) -> xr.Dataset:
    """Return a fake monthly Dataset based on the mMM fragment in the URI."""
    month = int(uri.rstrip("/").split("/m")[-1])
    _, last = calendar.monthrange(2022, month)
    return _make_monthly_ds(f"2022-{month:02d}-01", f"2022-{month:02d}-{last:02d}")


class TestHYCOMLoaderFetchAndCache:
    def test_cache_hit_skips_fetch(self):
        """If the annual .zmetadata exists, fetch_tile() is never called."""
        loader = HYCOMLoader()
        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(loader, "fetch_tile") as mock_fetch:
            mock_fs_cls.return_value.exists.return_value = True
            loader.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/hycom/tiles/2022/")
            mock_fetch.assert_not_called()

    def test_cache_hit_checks_complete_sentinel(self):
        """Cache hit check uses .complete sentinel, not bare directory existence."""
        loader = HYCOMLoader()
        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(loader, "fetch_tile"):
            mock_fs_cls.return_value.exists.return_value = True
            loader.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/hycom/tiles/2022/")
            # First exists() call must check the annual .complete sentinel
            first_checked = mock_fs_cls.return_value.exists.call_args_list[0].args[0]
            assert first_checked == "bucket/hycom/tiles/2022/.complete"

    def test_cache_miss_fetches_all_twelve_months(self):
        """On a full cache miss, fetch_tile() is called once per month with correct date ranges."""
        loader = HYCOMLoader()
        bbox = (-71.0, 41.0, -66.0, 45.0)

        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(loader, "fetch_tile", side_effect=_make_monthly_ds) as mock_fetch, \
             patch("xarray.Dataset.to_zarr"), \
             patch("xarray.open_zarr", side_effect=_fake_open_zarr_2022):
            mock_fs_cls.return_value.exists.return_value = False
            loader.fetch_and_cache(2022, bbox, "gs://bucket/hycom/tiles/2022/")

        assert mock_fetch.call_count == 12
        assert mock_fetch.call_args_list[0].args == ("2022-01-01", "2022-01-31", bbox)
        assert mock_fetch.call_args_list[1].args == ("2022-02-01", "2022-02-28", bbox)
        assert mock_fetch.call_args_list[11].args == ("2022-12-01", "2022-12-31", bbox)

    def test_cache_miss_writes_monthly_and_annual_zarrs(self):
        """On a full cache miss, to_zarr is called 12 times for monthly stores and once for annual."""
        loader = HYCOMLoader()
        bbox = (-71.0, 41.0, -66.0, 45.0)
        base = "gs://bucket/hycom/tiles/2022"

        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(loader, "fetch_tile", side_effect=_make_monthly_ds), \
             patch("xarray.Dataset.to_zarr") as mock_to_zarr, \
             patch("xarray.open_zarr", side_effect=_fake_open_zarr_2022):
            mock_fs_cls.return_value.exists.return_value = False
            loader.fetch_and_cache(2022, bbox, f"{base}/")

        written_uris = [c.args[0] for c in mock_to_zarr.call_args_list]
        assert f"{base}/monthly/m01/" in written_uris
        assert f"{base}/monthly/m12/" in written_uris
        assert f"{base}/" in written_uris
        assert mock_to_zarr.call_count == 13  # 12 monthly + 1 annual

    def test_partial_resume_skips_completed_months(self):
        """Months whose .zmetadata already exists are skipped; only missing months are fetched."""
        loader = HYCOMLoader()
        bbox = (-71.0, 41.0, -66.0, 45.0)

        def exists_side_effect(path: str) -> bool:
            # Annual store is incomplete; months 1-6 are already done.
            if path.endswith("tiles/2022/.complete"):
                return False
            if "monthly/m" in path and path.endswith(".complete"):
                month = int(path.rstrip("/.complete").split("/m")[-1])
                return month <= 6
            return False

        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(loader, "fetch_tile", side_effect=_make_monthly_ds) as mock_fetch, \
             patch("xarray.Dataset.to_zarr"), \
             patch("xarray.open_zarr", side_effect=_fake_open_zarr_2022):
            mock_fs_cls.return_value.exists.side_effect = exists_side_effect
            loader.fetch_and_cache(2022, bbox, "gs://bucket/hycom/tiles/2022/")

        assert mock_fetch.call_count == 6  # only months 7-12
        assert mock_fetch.call_args_list[0].args == ("2022-07-01", "2022-07-31", bbox)

    def test_fetch_raises_propagates_without_writing_annual(self):
        """If fetch_tile raises on a month, the exception propagates and the annual to_zarr is never called."""
        loader = HYCOMLoader()
        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(loader, "fetch_tile", side_effect=RuntimeError("OPeNDAP timeout")), \
             patch("xarray.Dataset.to_zarr") as mock_to_zarr:
            mock_fs_cls.return_value.exists.return_value = False
            with pytest.raises(RuntimeError, match="OPeNDAP timeout"):
                loader.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/hycom/tiles/2022/")
            annual_writes = [c for c in mock_to_zarr.call_args_list
                             if c.args and c.args[0] == "gs://bucket/hycom/tiles/2022/"]
            assert len(annual_writes) == 0


from src.ingestion.era5_harvester import ERA5Harvester


class TestERA5HarvesterFetchAndCache:
    def test_cache_hit_skips_fetch(self):
        """If .zmetadata exists at gcs_uri, fetch() is never called."""
        harvester = ERA5Harvester()
        harvester._initialized = True
        with patch("src.ingestion.era5_harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(harvester, "fetch") as mock_fetch:
            mock_fs_cls.return_value.exists.return_value = True
            harvester.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/era5/2022/")
            mock_fetch.assert_not_called()

    def test_cache_hit_checks_complete_sentinel(self):
        """Cache hit check uses .complete sentinel, not bare directory existence."""
        harvester = ERA5Harvester()
        harvester._initialized = True
        with patch("src.ingestion.era5_harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(harvester, "fetch"):
            mock_fs_cls.return_value.exists.return_value = True
            harvester.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/era5/2022/")
            checked_path = mock_fs_cls.return_value.exists.call_args_list[0].args[0]
            assert checked_path == "bucket/era5/2022/.complete"

    def test_cache_miss_calls_fetch_and_writes(self):
        """On a cache miss, fetch() is called with inclusive year range and result written to GCS."""
        harvester = ERA5Harvester()
        harvester._initialized = True
        fake_ds = xr.Dataset({"sea_surface_temperature": xr.DataArray(
            np.zeros((1, 10, 4, 5)), dims=["member", "time", "latitude", "longitude"])})

        with patch("src.ingestion.era5_harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(harvester, "fetch", return_value=fake_ds) as mock_fetch, \
             patch("src.ingestion.era5_harvester._gcs_safe_write") as mock_write:
            mock_fs_cls.return_value.exists.return_value = False
            harvester.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/era5/2022/")
            # end_date must be "2023-01-01" so GEE filterDate (exclusive end) includes Dec 31
            mock_fetch.assert_called_once_with("2022-01-01", "2023-01-01", (-71.0, 41.0, -66.0, 45.0))
            mock_write.assert_called_once_with(fake_ds, "gs://bucket/era5/2022/")

    def test_raises_if_not_authenticated(self):
        """fetch_and_cache() raises RuntimeError if authenticate() was not called."""
        harvester = ERA5Harvester()
        with patch("src.ingestion.era5_harvester.gcsfs.GCSFileSystem") as mock_fs_cls:
            mock_fs_cls.return_value.exists.return_value = False
            with pytest.raises(RuntimeError, match="authenticate()"):
                harvester.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/era5/2022/")

    def test_fetch_raises_propagates_without_writing(self):
        """If fetch() raises, the exception propagates and to_zarr is never called."""
        harvester = ERA5Harvester()
        harvester._initialized = True
        with patch("src.ingestion.era5_harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(harvester, "fetch", side_effect=RuntimeError("GEE timeout")), \
             patch("xarray.Dataset.to_zarr") as mock_to_zarr:
            mock_fs_cls.return_value.exists.return_value = False
            with pytest.raises(RuntimeError, match="GEE timeout"):
                harvester.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/era5/2022/")
            mock_to_zarr.assert_not_called()
