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
from typing import Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Mapping: ERA5 band name → WN2-compatible variable name
# Source for band names: GEE catalog — ECMWF/ERA5/DAILY
ERA5_BANDS: dict[str, str] = {
    "mean_2m_air_temperature":    "2m_temperature",
    "u_component_of_wind_10m":    "10m_u_component_of_wind",
    "v_component_of_wind_10m":    "10m_v_component_of_wind",
    "mean_sea_level_pressure":    "mean_sea_level_pressure",
    "sea_surface_temperature":    "sea_surface_temperature",
}

GEE_COLLECTION = "ECMWF/ERA5/DAILY"


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
        Query ECMWF/ERA5/DAILY from GEE for the specified date range and bounding box.

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

        logger.info("ERA5: %d daily images found.", n_images)

        image_list = collection.sort("system:time_start").toList(n_images)
        time_coords: list = []
        data_by_var: dict[str, list] = {band: [] for band in ERA5_BANDS}

        for t_idx in range(n_images):
            img = ee.Image(image_list.get(t_idx))
            sample = img.sampleRectangle(
                region=region, defaultValue=0
            ).getInfo()
            props = sample["properties"]
            for band in ERA5_BANDS:
                data_by_var[band].append(np.array(props[band]))  # (lat, lon)
            date_str = img.date().format("YYYY-MM-dd").getInfo()
            time_coords.append(np.datetime64(date_str))

        # Stack into (time, lat, lon) arrays, then rename and add member dim
        ds_vars = {}
        for era5_band, wn2_name in ERA5_BANDS.items():
            arr = np.stack(data_by_var[era5_band], axis=0)  # (time, lat, lon)
            arr = arr[np.newaxis, ...]                       # (member=1, time, lat, lon)
            first_img_arr = data_by_var[era5_band][0]
            n_lat, n_lon = first_img_arr.shape
            ds_vars[wn2_name] = xr.DataArray(
                arr,
                dims=["member", "time", "latitude", "longitude"],
            )

        ds = xr.Dataset(ds_vars).assign_coords(
            member=[0],
            time=time_coords,
        )

        logger.info("ERA5 dataset built: %s", ds)
        return ds
