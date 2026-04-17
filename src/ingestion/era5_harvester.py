"""
era5_harvester.py — ERA5 daily data harvester for MHW proxy training
=====================================================================
Fetches the ECMWF/ERA5/DAILY ImageCollection from Google Earth Engine
and returns an xr.Dataset with variable names matching WeatherNext 2,
making it a drop-in replacement for WeatherNext2Harvester in the training pipeline.

ERA5 is a deterministic reanalysis (1 member). The single-member output feeds
DataHarmonizer.expand_and_perturb() which broadcasts to 64 synthetic members,
preserving the (member, ...) tensor contract required by MHWRiskModel.

Physical note
-------------
ERA5 and WeatherNext 2 share the same 5 atmospheric variables at 0.25-degree
resolution in matching SI units. ERA5 is the ECMWF reanalysis of historical
atmospheric state — physically consistent and freely available on GEE without
a whitelist. The variable naming difference between the two products is the only
incompatibility; this module resolves it via ERA5_BANDS renaming.

Dependencies
------------
    earthengine-api>=0.1.390, xarray>=2024.2.0, numpy>=1.26.0
"""
from __future__ import annotations

import logging
import os
from datetime import date as _date, timedelta
from typing import Optional

import gcsfs
import numpy as np
import xarray as xr

from src.ingestion.harvester import _gcs_safe_write

logger = logging.getLogger(__name__)

# Mapping: ERA5 band name → WN2-compatible variable name
# Source for band names: GEE catalog — ECMWF/ERA5/DAILY
ERA5_BANDS: dict[str, str] = {
    "temperature_2m":              "2m_temperature",
    "u_component_of_wind_10m":    "10m_u_component_of_wind",
    "v_component_of_wind_10m":    "10m_v_component_of_wind",
    "mean_sea_level_pressure":    "mean_sea_level_pressure",
    "sea_surface_temperature":    "sea_surface_temperature",
}

GEE_COLLECTION = "ECMWF/ERA5/HOURLY"


class ERA5Harvester:
    """
    Fetches ECMWF ERA5 daily reanalysis from Google Earth Engine for a spatial
    bounding box and date range, renaming variables to match WeatherNext 2 output.

    Output is a single-member (member=1) xr.Dataset. Downstream code must call
    DataHarmonizer.expand_and_perturb() to broadcast to 64 synthetic members
    before passing to MHWRiskModel.

    Parameters
    ----------
    service_account_key : str, optional
        Path to a GCP service account JSON key file.
        If None, falls back to Application Default Credentials.
    """

    def __init__(self, service_account_key: Optional[str] = None) -> None:
        self._key = service_account_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self._initialized = False

    def authenticate(self) -> None:
        """
        Authenticate with Google Earth Engine.

        Uses service account credentials if _key is set, otherwise falls back
        to Application Default Credentials (ADC). Idempotent — safe to call
        multiple times.
        """
        import ee

        if not self._initialized:
            if self._key:
                import json
                with open(self._key) as fh:
                    kd = json.load(fh)
                creds = ee.ServiceAccountCredentials(
                    email=kd["client_email"], key_file=self._key
                )
                ee.Initialize(credentials=creds)
            else:
                ee.Initialize()
            self._initialized = True
            logger.info("ERA5Harvester: GEE authentication successful.")

    def fetch(
        self,
        start_date: str,
        end_date: str,
        bbox: tuple[float, float, float, float],
    ) -> xr.Dataset:
        """
        Query ECMWF/ERA5/HOURLY from GEE for the specified date range and bounding box,
        aggregating 24 hourly images per calendar day to a daily mean server-side.

        Parameters
        ----------
        start_date : str
            ISO 8601 start date, e.g. "2018-01-01".
        end_date : str
            ISO 8601 end date, inclusive, e.g. "2018-12-31".
        bbox : tuple of float
            (lon_min, lat_min, lon_max, lat_max) in decimal degrees WGS84.

        Returns
        -------
        ds : xr.Dataset
            Dimensions: (member=1, time, latitude, longitude)
            Variables: WN2_VARIABLES — same names as WeatherNext2Harvester.fetch_ensemble()
            Units: SI (K for temperature, Pa for pressure, m/s for wind)
            The member=1 dimension is intentional — call DataHarmonizer.expand_and_perturb()
            downstream to broadcast to 64 synthetic ensemble members.
        """
        import ee

        if not self._initialized:
            raise RuntimeError("Call authenticate() before fetch().")

        lon_min, lat_min, lon_max, lat_max = bbox
        region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

        logger.info(
            "Fetching ERA5 daily: %s to %s, bbox=%s", start_date, end_date, bbox
        )

        collection = (
            ee.ImageCollection(GEE_COLLECTION)
            .filterDate(start_date, end_date)
            .select(list(ERA5_BANDS.keys()))
        )

        n_images = collection.size().getInfo()
        if n_images == 0:
            raise ValueError(
                f"No ERA5 images found for {start_date} to {end_date}. "
                "Check GEE asset availability for this period."
            )

        logger.info("ERA5 hourly: %d images found, aggregating to daily means.", n_images)

        # Build calendar-day list. end_date is exclusive (GEE filterDate convention).
        start_d = _date.fromisoformat(start_date)
        end_d   = _date.fromisoformat(end_date)
        unique_dates = []
        d = start_d
        while d < end_d:
            unique_dates.append(d)
            d += timedelta(days=1)

        time_coords: list = []
        data_by_var: dict[str, list] = {band: [] for band in ERA5_BANDS}

        for i, d in enumerate(unique_dates):
            date_str      = d.isoformat()
            next_date_str = (d + timedelta(days=1)).isoformat()
            # Average 24 hourly images into one daily mean image server-side.
            daily_img = collection.filterDate(date_str, next_date_str).mean()
            sample = daily_img.sampleRectangle(region=region, defaultValue=0).getInfo()
            props = sample["properties"]
            for band in ERA5_BANDS:
                data_by_var[band].append(np.array(props[band]))  # (lat, lon)
            time_coords.append(np.datetime64(date_str))
            logger.info("ERA5 fetched %s (%d/%d)", date_str, i + 1, len(unique_dates))

        # Stack into (time, lat, lon) arrays, then rename and add member dim
        ds_vars = {}
        for era5_band, wn2_name in ERA5_BANDS.items():
            arr = np.stack(data_by_var[era5_band], axis=0)  # (time, lat, lon)
            arr = arr[np.newaxis, ...]                       # (member=1, time, lat, lon)
            ds_vars[wn2_name] = xr.DataArray(
                arr,
                dims=["member", "time", "latitude", "longitude"],
            )

        # Reconstruct lat/lon coordinate arrays from bbox and array shape.
        # GEE sampleRectangle returns rows north-to-south, so lat is descending.
        first_band = next(iter(ERA5_BANDS))
        n_lat, n_lon = data_by_var[first_band][0].shape
        lat_coords = np.linspace(lat_max, lat_min, n_lat)
        lon_coords = np.linspace(lon_min, lon_max, n_lon)

        ds = xr.Dataset(ds_vars).assign_coords(
            member=[0],
            time=time_coords,
            latitude=lat_coords,
            longitude=lon_coords,
        )

        logger.info("ERA5 dataset built: %s", ds)
        return ds

    def fetch_and_cache(
        self,
        year: int,
        bbox: tuple[float, float, float, float],
        gcs_uri: str,
    ) -> None:
        """
        Fetch one full calendar year of ERA5 data from GEE and write to GCS as Zarr.

        Parameters
        ----------
        year : int
            Calendar year to fetch (Jan 1 – Dec 31).
            ECMWF/ERA5/DAILY on GEE covers 1979–present.
        bbox : tuple of float
            (lon_min, lat_min, lon_max, lat_max) in decimal degrees WGS84, -180–180 convention.
            Passed directly to fetch() without transformation.
            Source: project-standard Gulf of Maine region, e.g. (-71.0, 41.0, -66.0, 45.0).
        gcs_uri : str
            GCS destination, e.g. "gs://bucket/era5/2022/".
            If the URI already exists, returns immediately — idempotent, safe to re-run
            after a spot VM preemption.

        Side effects
        ------------
        Writes a Zarr store to gcs_uri with dims (member=1, time, latitude, longitude)
        and WN2-compatible variable names. This Zarr store is the canonical ERA5 input tile
        for the MHW risk training pipeline; read by run_data_prep.py and train_era5.py
        via xr.open_zarr(). Downstream callers must invoke DataHarmonizer.expand_and_perturb()
        (via harmonize()) to expand to 64 synthetic members.
        Credentials are read from GOOGLE_APPLICATION_CREDENTIALS automatically by gcsfs.
        """
        if not self._initialized:
            raise RuntimeError("Call authenticate() before fetch_and_cache().")

        fs = gcsfs.GCSFileSystem()
        meta = gcs_uri.removeprefix("gs://").rstrip("/") + "/.zmetadata"
        if fs.exists(meta):
            logger.info("Cache hit — skipping ERA5 fetch for %d: %s", year, gcs_uri)
            return

        start_date = f"{year}-01-01"
        # GEE filterDate end is exclusive — use Jan 1 of the next year to include Dec 31.
        end_date   = f"{year + 1}-01-01"
        logger.info("Fetching ERA5 year %d (%s to %s)...", year, start_date, end_date)

        ds = self.fetch(start_date, end_date, bbox)
        _gcs_safe_write(ds, gcs_uri)
        logger.info("ERA5 year %d written to %s", year, gcs_uri)
