"""
Unit tests for WeatherNext2Harvester.fetch_and_cache() and related methods.
No network calls — ee and gcsfs are fully mocked.
"""
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import xarray as xr

from src.ingestion.harvester import WeatherNext2Harvester


def _make_fake_ds(n_members: int = 64, n_days: int = 5) -> xr.Dataset:
    """Minimal WN2-shaped Dataset for mocking _build_dataset return values."""
    return xr.Dataset(
        {"sea_surface_temperature": xr.DataArray(
            np.zeros((n_members, n_days, 17, 21), dtype=np.float32),
            dims=["member", "time", "latitude", "longitude"],
        )}
    )


class TestFetchAndCacheCacheHit:
    def test_cache_hit_skips_build_dataset(self):
        """If .zmetadata exists at gcs_uri, _build_dataset is never called."""
        h = WeatherNext2Harvester(gcs_bucket="my-bucket")
        h._ee_initialized = True
        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(h, "_build_dataset") as mock_build:
            mock_fs_cls.return_value.exists.return_value = True
            h.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://my-bucket/wn2/2022/")
            mock_build.assert_not_called()

    def test_cache_hit_checks_zmetadata_path(self):
        """Cache check uses .zmetadata (completeness marker), not bare directory."""
        h = WeatherNext2Harvester(gcs_bucket="my-bucket")
        h._ee_initialized = True
        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(h, "_build_dataset"):
            mock_fs_cls.return_value.exists.return_value = True
            h.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://my-bucket/wn2/2022/")
            checked = mock_fs_cls.return_value.exists.call_args_list[0].args[0]
            assert checked == "my-bucket/wn2/2022/.zmetadata"


class TestFetchAndCacheCacheMiss:
    def test_cache_miss_writes_to_supplied_gcs_uri(self):
        """On cache miss, the Dataset is written to the exact gcs_uri passed by the caller."""
        h = WeatherNext2Harvester(gcs_bucket="my-bucket")
        h._ee_initialized = True
        gcs_uri = "gs://my-bucket/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr"

        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch("src.ingestion.harvester.ee") as mock_ee, \
             patch.object(h, "_build_dataset", return_value=_make_fake_ds()) as mock_build, \
             patch("src.ingestion.harvester._gcs_safe_write") as mock_write:
            mock_fs_cls.return_value.exists.return_value = False
            mock_ee.ImageCollection.return_value.filter.return_value = MagicMock()
            mock_ee.Filter.date.return_value = MagicMock()
            mock_ee.Filter.stringEndsWith.return_value = MagicMock()
            mock_ee.Filter.eq.return_value = MagicMock()
            mock_ee.Geometry.Rectangle.return_value = MagicMock()

            h.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), gcs_uri)

            mock_write.assert_called_once_with(mock_build.return_value, gcs_uri)

    def test_cache_miss_passes_exclusive_end_date(self):
        """fetch_and_cache for year=2022 passes end_date='2023-01-01' (exclusive) to the collection filter."""
        h = WeatherNext2Harvester(gcs_bucket="my-bucket")
        h._ee_initialized = True

        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch("src.ingestion.harvester.ee") as mock_ee, \
             patch.object(h, "_build_dataset", return_value=_make_fake_ds()), \
             patch("xarray.Dataset.to_zarr"):
            mock_fs_cls.return_value.exists.return_value = False
            mock_ee.ImageCollection.return_value.filter.return_value = MagicMock()

            h.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://my-bucket/wn2/2022/")

            # ee.Filter.date must have been called with "2022-01-01" and "2023-01-01"
            mock_ee.Filter.date.assert_called_once_with("2022-01-01", "2023-01-01")


class TestFetchAndCacheLazyAuth:
    def test_authenticate_called_if_not_initialized(self):
        """fetch_and_cache calls authenticate() lazily if ee is not yet initialized."""
        h = WeatherNext2Harvester(gcs_bucket="my-bucket")
        # _ee_initialized is False — do NOT set it manually

        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(h, "authenticate") as mock_auth, \
             patch("src.ingestion.harvester.ee") as mock_ee, \
             patch.object(h, "_build_dataset", return_value=_make_fake_ds()), \
             patch("xarray.Dataset.to_zarr"):
            mock_fs_cls.return_value.exists.return_value = False
            mock_auth.side_effect = lambda: setattr(h, "_ee_initialized", True)
            mock_ee.ImageCollection.return_value.filter.return_value = MagicMock()

            h.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://my-bucket/wn2/2022/")

            mock_auth.assert_called_once()

    def test_authenticate_not_called_if_already_initialized(self):
        """fetch_and_cache does not call authenticate() again if already initialized."""
        h = WeatherNext2Harvester(gcs_bucket="my-bucket")
        h._ee_initialized = True

        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch.object(h, "authenticate") as mock_auth, \
             patch("src.ingestion.harvester.ee") as mock_ee, \
             patch.object(h, "_build_dataset", return_value=_make_fake_ds()), \
             patch("xarray.Dataset.to_zarr"):
            mock_fs_cls.return_value.exists.return_value = False
            mock_ee.ImageCollection.return_value.filter.return_value = MagicMock()

            h.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://my-bucket/wn2/2022/")

            mock_auth.assert_not_called()


class TestFetchAndCacheExceptionPropagation:
    def test_build_dataset_exception_does_not_write_to_gcs(self):
        """If _build_dataset raises, to_zarr is never called — no partial write to GCS."""
        h = WeatherNext2Harvester(gcs_bucket="my-bucket")
        h._ee_initialized = True

        with patch("src.ingestion.harvester.gcsfs.GCSFileSystem") as mock_fs_cls, \
             patch("src.ingestion.harvester.ee") as mock_ee, \
             patch.object(h, "_build_dataset", side_effect=RuntimeError("GEE timeout")), \
             patch("xarray.Dataset.to_zarr") as mock_to_zarr:
            mock_fs_cls.return_value.exists.return_value = False
            mock_ee.ImageCollection.return_value.filter.return_value = MagicMock()

            with pytest.raises(RuntimeError, match="GEE timeout"):
                h.fetch_and_cache(2022, (-71.0, 41.0, -66.0, 45.0), "gs://my-bucket/wn2/2022/")

            mock_to_zarr.assert_not_called()


class TestFetchEnsembleFilters:
    def test_applies_00z_init_filter(self):
        """fetch_ensemble filters start_time to T00:00:00Z (00Z init only)."""
        h = WeatherNext2Harvester(gcs_bucket="my-bucket")
        h._ee_initialized = True

        with patch("src.ingestion.harvester.ee") as mock_ee, \
             patch.object(h, "_fetch_and_write_zarr", return_value="gs://bucket/wn2.zarr"), \
             patch("xarray.open_zarr", return_value=_make_fake_ds()):
            mock_ee.Filter.date.return_value = MagicMock()
            mock_ee.Filter.stringEndsWith.return_value = MagicMock()
            mock_ee.Filter.eq.return_value = MagicMock()
            mock_ee.ImageCollection.return_value.filter.return_value = MagicMock()
            mock_ee.Geometry.Rectangle.return_value = MagicMock()

            h.fetch_ensemble("2022-01-01", "2023-01-01", (-71.0, 41.0, -66.0, 45.0))

            mock_ee.Filter.stringEndsWith.assert_called_once_with("start_time", "T00:00:00Z")

    def test_applies_forecast_hour_24_filter(self):
        """fetch_ensemble filters forecast_hour to 24 (next-day forecast only)."""
        h = WeatherNext2Harvester(gcs_bucket="my-bucket")
        h._ee_initialized = True

        with patch("src.ingestion.harvester.ee") as mock_ee, \
             patch.object(h, "_fetch_and_write_zarr", return_value="gs://bucket/wn2.zarr"), \
             patch("xarray.open_zarr", return_value=_make_fake_ds()):
            mock_ee.Filter.date.return_value = MagicMock()
            mock_ee.Filter.stringEndsWith.return_value = MagicMock()
            mock_ee.Filter.eq.return_value = MagicMock()
            mock_ee.ImageCollection.return_value.filter.return_value = MagicMock()
            mock_ee.Geometry.Rectangle.return_value = MagicMock()

            h.fetch_ensemble("2022-01-01", "2023-01-01", (-71.0, 41.0, -66.0, 45.0))

            mock_ee.Filter.eq.assert_called_once_with("forecast_hour", 24)

    def test_raises_if_not_authenticated(self):
        """fetch_ensemble raises RuntimeError if authenticate() was not called."""
        h = WeatherNext2Harvester(gcs_bucket="my-bucket")
        with pytest.raises(RuntimeError, match="authenticate()"):
            h.fetch_ensemble("2022-01-01", "2023-01-01", (-71.0, 41.0, -66.0, 45.0))
