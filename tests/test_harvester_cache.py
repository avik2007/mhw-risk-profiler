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
             patch("xarray.Dataset.to_zarr") as mock_to_zarr:
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

    def test_fetch_raises_propagates_without_writing(self):
        """If fetch_tile raises, the exception propagates and to_zarr is never called."""
        loader = HYCOMLoader()
        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(loader, "fetch_tile", side_effect=RuntimeError("OPeNDAP timeout")), \
             patch("xarray.Dataset.to_zarr") as mock_to_zarr:
            mock_fs_cls.return_value.exists.return_value = False
            with pytest.raises(RuntimeError, match="OPeNDAP timeout"):
                loader.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://bucket/hycom/tiles/2022/")
            mock_to_zarr.assert_not_called()


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
             patch("xarray.Dataset.to_zarr") as mock_to_zarr:
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
