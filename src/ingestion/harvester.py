"""
harvester.py — MHW Risk Profiler Data Ingestion Engine
=======================================================
Harmonizes two independent Earth observation data streams:

  1. Google WeatherNext 2 (FGN ensemble, 64 members) accessed via the GEE Python API
     and cached to Google Cloud Storage (GCS) as Zarr.
  2. HYCOM (Hybrid Coordinate Ocean Model) daily ocean reanalysis accessed as NetCDF
     via the THREDDS Data Server.

Both streams are regridded to a shared 0.25-degree, CF-compliant daily xarray Dataset
that retains the 64-member ensemble dimension from WeatherNext 2. This harmonized output
is the sole input to the MHW detection, SDD accumulation, and SVaR analytics layers.

Physical motivation
-------------------
Marine Heatwaves are driven by a coupled atmosphere-ocean mechanism:
  - Atmospheric forcing (reduced winds, anomalous anticyclonic circulation) suppresses
    vertical mixing, allowing surface heat to accumulate. WeatherNext 2 captures this
    through 10 m wind speed, mean sea-level pressure, and net surface heat flux.
  - Oceanic memory (subsurface warm anomalies, salinity-driven stratification) modulates
    how quickly heat is mixed down or retained. HYCOM provides this via 3D T/S fields.

Neither source alone is sufficient for accurate MHW prediction. This module fuses them.

Dependencies
------------
  earthengine-api, google-cloud-storage, xarray, dask, netCDF4, zarr, numpy, scipy
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import dask
import dask.array as da
import gcsfs
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target harmonized grid — 0.25-degree global, daily
TARGET_LON = np.arange(-180.0, 180.0, 0.25)
TARGET_LAT = np.arange(-90.0, 90.25, 0.25)

# WeatherNext 2 GEE asset path.
# Source: developers.google.com/weathernext/guides/earth-engine
# Access requires submitting the WeatherNext Data Request form before the service
# account is whitelisted. Historical data (>48 h old) is CC BY 4.0.
WN2_GEE_ASSET = "projects/gcp-public-data-weathernext/assets/weathernext_2_0_0"

# HYCOM THREDDS URLs for GLBy0.08 expt_93.0 (3-hourly, 2018-12-04 to 2024-09-04).
# GLBy0.08 supersedes GLBv0.08: same variables and structure, extended coverage.
# Covers 2022-2023 WN2 training/validation periods (GLBv0.08 ended 2020-02-19).
# Temperature and salinity are in a separate dataset from currents (OPeNDAP convention).
# Longitude coordinate is 0–360; bboxes in -180–180 must be converted before slicing.
HYCOM_THREDDS_TS = (
    "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z"
)
HYCOM_THREDDS_UV = (
    "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
)
# Back-compat alias used by __init__ default argument
HYCOM_THREDDS_BASE = HYCOM_THREDDS_TS

# Variables to extract from each source
WN2_VARIABLES = [
    "2m_temperature",          # [K] Near-surface air temperature — proxy for atmospheric forcing
    "10m_u_component_of_wind", # [m/s] Zonal wind — drives Ekman pumping and vertical mixing
    "10m_v_component_of_wind", # [m/s] Meridional wind — completes horizontal wind vector
    "mean_sea_level_pressure",  # [Pa] MSLP — identifies anticyclonic blocking that suppresses mixing
    "sea_surface_temperature",  # [K] SST — primary MHW detection variable
]

HYCOM_VARIABLES = [
    "water_temp",   # [deg C] Ocean potential temperature on hybrid levels
    "salinity",     # [psu] Practical salinity — controls density stratification
    "water_u",      # [m/s] Zonal ocean current — lateral heat advection
    "water_v",      # [m/s] Meridional ocean current — lateral heat advection
]

# Standard output depth levels [m] — interpolated from HYCOM hybrid coordinate
# Chosen to capture the mixed layer (0-100 m) and seasonal thermocline (100-300 m)
TARGET_DEPTHS_M = np.array([0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300])

# Per-variable Gaussian noise σ for expand_and_perturb() — calibrated to
# approximate WeatherNext 2 intra-ensemble spread.
NOISE_SIGMAS: dict[str, float] = {
    "sea_surface_temperature":    0.5,   # K  — primary MHW driver
    "2m_temperature":             0.5,   # K  — coherent with SST
    "10m_u_component_of_wind":    0.3,   # m/s
    "10m_v_component_of_wind":    0.3,   # m/s
    "mean_sea_level_pressure":    50.0,  # Pa — ~0.5 hPa synoptic noise
}


# ---------------------------------------------------------------------------
# GEE / WeatherNext 2 Harvester
# ---------------------------------------------------------------------------

class WeatherNext2Harvester:
    """
    Queries the WeatherNext 2 ensemble dataset from Google Earth Engine and
    caches outputs to Google Cloud Storage as Zarr for Dask-parallelised access.

    WeatherNext 2 uses a Functional Generative Network (FGN) architecture that
    produces ensemble members as draws from a learned atmospheric state distribution.
    This means tail-risk events (e.g., extreme MHW drivers) are represented more
    faithfully than in perturbation-based ensembles.

    The dataset is exposed in GEE under the `gcp-public-data-weathernext` project
    as an ImageCollection where each Image corresponds to one ensemble member for
    one initialisation time.
    """

    def __init__(
        self,
        gcs_bucket: str,
        gcs_prefix: str = "weathernext2/cache",
        service_account_key: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        gcs_bucket : str
            GCS bucket name for caching Zarr outputs.
            Must already exist; this class does not create buckets.
        gcs_prefix : str
            Object prefix (folder) within the bucket for Zarr stores.
        service_account_key : str, optional
            Path to a GCP service account JSON key file.
            If None, falls back to Application Default Credentials (ADC).
            ADC is preferred in production; key file is for local development.
        """
        self._gcs_bucket = gcs_bucket
        self._gcs_prefix = gcs_prefix
        self._key = service_account_key
        self._ee_initialized = False
        self._gcs_client = None

    def authenticate(self) -> None:
        """
        Authenticate with both Google Earth Engine and Google Cloud Storage.

        GEE authentication uses the earthengine-api flow (service account or OAuth).
        GCS authentication uses google-cloud-storage with the same credential source.

        This method is idempotent — safe to call multiple times.
        """
        import ee
        from google.cloud import storage
        from google.oauth2 import service_account

        if not self._ee_initialized:
            if self._key:
                import json
                with open(self._key) as fh:
                    _key_data = json.load(fh)
                credentials = ee.ServiceAccountCredentials(
                    email=_key_data["client_email"], key_file=self._key
                )
                ee.Initialize(credentials=credentials)
                gcs_creds = service_account.Credentials.from_service_account_file(
                    self._key,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                self._gcs_client = storage.Client(credentials=gcs_creds)
            else:
                # Application Default Credentials — standard in Cloud Run / Vertex AI
                ee.Initialize()
                self._gcs_client = storage.Client()
            self._ee_initialized = True
            logger.info("GEE and GCS authentication successful.")

    def fetch_ensemble(
        self,
        start_date: str,
        end_date: str,
        bbox: tuple[float, float, float, float],
        n_members: int = 64,
    ) -> xr.Dataset:
        """
        Query WeatherNext 2 for a date range and spatial bounding box, returning
        n_members ensemble members as a single xr.Dataset with a 'member' dimension.

        Parameters
        ----------
        start_date : str
            ISO 8601 start date, e.g. "2023-06-01".
        end_date : str
            ISO 8601 end date, inclusive, e.g. "2023-08-31".
        bbox : tuple of float
            (lon_min, lat_min, lon_max, lat_max) in decimal degrees WGS84.
            Defines the spatial region of interest — typically an aquaculture lease
            plus a 2-degree buffer to capture upstream forcing.
        n_members : int
            Number of ensemble members to retrieve (1–64, max 64 for WeatherNext 2).
            In the GEE ImageCollection each image stores all 64 members as separate
            band groups. This parameter slices the first n_members from those groups.
            Reducing this speeds up development; use all 64 for production SVaR.

        Returns
        -------
        ds : xr.Dataset
            Dimensions: (member, time, latitude, longitude)
            Variables: WN2_VARIABLES (see module constants)
            Units: SI (K for temperature, Pa for pressure, m/s for wind)
            The 'member' dimension is essential for downstream SVaR calculation —
            each member represents a plausible atmospheric trajectory, and the
            spread across members quantifies forecast uncertainty.
        """
        import ee

        if not self._ee_initialized:
            raise RuntimeError("Call authenticate() before fetch_ensemble().")

        lon_min, lat_min, lon_max, lat_max = bbox
        region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
        date_filter = ee.Filter.date(start_date, end_date)

        logger.info(
            "Querying WeatherNext 2: %s to %s, bbox=%s, members=%d",
            start_date, end_date, bbox, n_members,
        )

        # Filter to complete-ensemble images only (number_of_members == 64).
        # The collection always stores all 64 members per image; we select a subset
        # of member bands after filtering, NOT filter by member count.
        collection = (
            ee.ImageCollection(WN2_GEE_ASSET)
            .filter(date_filter)
            .filter(ee.Filter.eq("number_of_members", 64))
        )

        zarr_uri = self._fetch_and_write_zarr(
            collection, region, start_date, end_date, n_members
        )
        ds = xr.open_zarr(zarr_uri, chunks="auto", consolidated=True)

        logger.info("WeatherNext 2 dataset loaded: %s", ds)
        return ds

    def _fetch_and_write_zarr(
        self,
        collection,
        region,
        start_date: str,
        end_date: str,
        n_members: int,
    ) -> str:
        """
        Compute WeatherNext 2 pixel values for a spatial region, build an xr.Dataset,
        and write it to GCS as a consolidated Zarr store.

        Uses the GEE synchronous compute path (`sampleRectangle`) rather than
        asynchronous batch Export. This is appropriate for small regions (≤ 5°×5°)
        and avoids the 2–20 minute GEE task queue. For large-area production runs
        the batch export path with GeoTIFF conversion is required.

        Parameters
        ----------
        collection : ee.ImageCollection
            WeatherNext 2 collection filtered to the target date range.
            Each image in the collection represents one forecast timestep and
            contains all 64 ensemble members as separate band groups.
        region : ee.Geometry
            Spatial extent for pixel extraction (must be small for compute path).
        start_date, end_date : str
            ISO 8601 dates used to construct a deterministic GCS cache key.
        n_members : int
            Number of ensemble members to extract from the full 64-member set.

        Returns
        -------
        gcs_uri : str
            GCS URI of the written Zarr store: "gs://<bucket>/<prefix>/<key>.zarr"
        """
        import ee
        import json

        store_key = f"{self._gcs_prefix}/wn2_{start_date}_{end_date}_m{n_members}.zarr"
        gcs_uri = f"gs://{self._gcs_bucket}/{store_key}"

        # Cache check: if the store already exists skip re-fetching
        bucket = self._gcs_client.bucket(self._gcs_bucket)
        if any(bucket.list_blobs(prefix=store_key)):
            logger.info("Cache hit for WeatherNext 2 Zarr: %s", gcs_uri)
            return gcs_uri

        logger.info("Fetching WeatherNext 2 via compute path (sampleRectangle).")

        # Inspect the first image to discover band naming convention.
        # WeatherNext 2 encodes each variable × member combination as a separate
        # band, typically named "{variable}_member_{i}" or "{variable}_{i:02d}".
        first_image = ee.Image(collection.first())
        band_names = first_image.bandNames().getInfo()
        logger.info("WeatherNext 2 band names (first image): %s", band_names)

        # Build a mapping: for each WN2_VARIABLE, find the bands for members 0…n_members-1.
        # Accepts two common naming patterns used by GEE public weather datasets:
        #   Pattern A: "{variable}_member_{i}"  (e.g. "sea_surface_temperature_member_0")
        #   Pattern B: "{variable}_{i:02d}"     (e.g. "sea_surface_temperature_00")
        member_bands: dict[str, list[str]] = {}
        for var in WN2_VARIABLES:
            matched: list[str] = []
            for m in range(n_members):
                for band in [
                    f"{var}_member_{m}",
                    f"{var}_{m:02d}",
                    f"{var}_{m}",
                ]:
                    if band in band_names:
                        matched.append(band)
                        break
                else:
                    # Fall back: treat the variable name itself as a single-member band
                    if var in band_names and m == 0:
                        matched.append(var)
            if matched:
                member_bands[var] = matched

        if not member_bands:
            raise RuntimeError(
                f"Could not match WN2_VARIABLES to bands in the collection. "
                f"Available bands: {band_names}"
            )

        # Iterate over each image (one timestep) and extract pixel values
        n_images = collection.size().getInfo()
        image_list = collection.toList(n_images)

        # Collect arrays per variable: list of (n_lat, n_lon) arrays per (member, time)
        time_coords: list = []
        data_by_var: dict[str, list] = {var: [] for var in member_bands}

        for t_idx in range(n_images):
            img = ee.Image(image_list.get(t_idx))
            # Extract all required bands in one call
            all_bands = [b for bands in member_bands.values() for b in bands]
            sample = img.select(all_bands).sampleRectangle(
                region=region, defaultValue=0
            ).getInfo()

            props = sample["properties"]
            for var, bands in member_bands.items():
                # Each band's pixel grid is a list-of-lists (lat × lon)
                member_arrays = [np.array(props[b]) for b in bands]
                data_by_var[var].append(member_arrays)  # (n_members, lat, lon)

            # Capture the image date
            date_str = img.date().format("YYYY-MM-dd").getInfo()
            time_coords.append(np.datetime64(date_str))

        # Assemble into an xr.Dataset: dims = (member, time, latitude, longitude)
        # Use the pixel grid from the first variable/member to infer spatial coords.
        first_var = next(iter(data_by_var))
        first_grid = np.array(data_by_var[first_var][0][0])  # (lat, lon)
        n_lat, n_lon = first_grid.shape
        lon_min_v, lat_min_v, lon_max_v, lat_max_v = (
            region.bounds().getInfo()["coordinates"][0][0][0],
            region.bounds().getInfo()["coordinates"][0][0][1],
            region.bounds().getInfo()["coordinates"][0][2][0],
            region.bounds().getInfo()["coordinates"][0][2][1],
        )
        lat_coords = np.linspace(lat_min_v, lat_max_v, n_lat)
        lon_coords = np.linspace(lon_min_v, lon_max_v, n_lon)

        data_vars = {}
        for var, time_list in data_by_var.items():
            # time_list: list of length n_images, each element: list of n_members arrays (lat, lon)
            arr = np.array(time_list)  # (time, member, lat, lon)
            arr = np.transpose(arr, (1, 0, 2, 3))  # (member, time, lat, lon)
            data_vars[var] = (["member", "time", "latitude", "longitude"], arr)

        ds = xr.Dataset(
            data_vars,
            coords={
                "member": np.arange(n_members),
                "time": time_coords,
                "latitude": lat_coords,
                "longitude": lon_coords,
            },
        )

        # Write to GCS via the direct gs:// URI.
        # gcsfs (registered as an fsspec backend) resolves credentials from
        # GOOGLE_APPLICATION_CREDENTIALS automatically, so no manual token
        # construction is needed. This is compatible with zarr v2 and v3.
        logger.info("Writing WeatherNext 2 Dataset to GCS Zarr: %s", gcs_uri)
        if self._key:
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", self._key)
        ds.to_zarr(gcs_uri, mode="w", consolidated=True)
        logger.info("WeatherNext 2 Zarr written: %s", gcs_uri)

        return gcs_uri


# ---------------------------------------------------------------------------
# HYCOM Loader
# ---------------------------------------------------------------------------

class HYCOMLoader:
    """
    Fetches HYCOM GLBv0.08 daily ocean reanalysis (temperature, salinity, currents)
    and extracts depth-resolved profiles on a consistent set of standard depth levels.

    HYCOM uses a Hybrid Coordinate system:
      - Z-levels (fixed depth) in the upper ocean and near coastlines.
      - Sigma (terrain-following) levels over the continental shelf.
      - Isopycnal (density-following) layers in the open ocean interior.

    This heterogeneous vertical coordinate means raw HYCOM output has varying
    layer thicknesses and depths across space and time. The loader interpolates
    all profiles onto TARGET_DEPTHS_M (see module constants) before returning,
    yielding a spatially and temporally consistent vertical grid suitable for
    the 1D-CNN architecture.
    """

    def __init__(
        self,
        thredds_url: str = HYCOM_THREDDS_TS,
        thredds_uv_url: str = HYCOM_THREDDS_UV,
    ) -> None:
        """
        Parameters
        ----------
        thredds_url : str
            OPeNDAP URL for the HYCOM ts3z dataset (temperature + salinity).
            Default: HYCOM GLBv0.08 expt_93.0 ts3z (2018-01-01 to 2020-02-19, 3-hourly).
        thredds_uv_url : str
            OPeNDAP URL for the HYCOM uv3z dataset (horizontal currents).
            Must be the same experiment and time range as thredds_url.
        """
        self._url = thredds_url
        self._url_uv = thredds_uv_url

    def fetch_tile(
        self,
        start_date: str,
        end_date: str,
        bbox: tuple[float, float, float, float],
    ) -> xr.Dataset:
        """
        Download a spatiotemporal tile of HYCOM 3D temperature, salinity, and
        horizontal currents, then interpolate from native hybrid levels to TARGET_DEPTHS_M.

        Parameters
        ----------
        start_date : str
            ISO 8601 start date, e.g. "2019-06-01".
        end_date : str
            ISO 8601 end date, inclusive, e.g. "2019-08-31".
        bbox : tuple of float
            (lon_min, lat_min, lon_max, lat_max) in decimal degrees WGS84.
            Longitude may be in -180..180 convention; this method converts internally
            to the 0..360 convention used by HYCOM.

        Returns
        -------
        ds_std : xr.Dataset
            Dimensions: (time, depth, lat, lon)
            Variables: water_temp [deg C], salinity [psu], water_u [m/s], water_v [m/s]
            Depth coordinate: TARGET_DEPTHS_M [m], positive downward.
            Longitude coordinate: -180..180 (converted back from HYCOM 0..360).

            Physical note: The mixed layer depth (MLD) is implicitly encoded here —
            near-isothermal profiles indicate deep mixing (MHW-suppressing), while
            sharp thermoclines indicate strong stratification (MHW-amplifying).
            The 1D-CNN uses this vertical structure as its primary feature.
        """
        lon_min, lat_min, lon_max, lat_max = bbox

        # HYCOM stores longitude in 0..360 convention. Convert the bbox.
        # This simple modulo conversion is valid for regions that do not straddle
        # the prime meridian (lon_min and lon_max on the same side of 0°).
        lon_min_360 = lon_min % 360
        lon_max_360 = lon_max % 360

        logger.info("Fetching HYCOM ts3z tile: %s to %s, bbox=%s", start_date, end_date, bbox)

        def _open_and_slice(url: str, variables: list[str]) -> xr.Dataset:
            """
            Open a HYCOM OPeNDAP dataset, decode the non-standard time axis, and
            slice to the requested spatiotemporal bounding box.

            HYCOM uses 'hours since 2000-01-01 00:00:00' as its time unit, which
            xarray cannot decode with CF conventions ('hours since analysis' is
            what xarray sees). We open with decode_times=False and decode manually.

            Strategy: filter time by raw float values BEFORE decoding so that the
            full 6000+ step time axis is never sorted or loaded into memory.
            """
            ds = xr.open_dataset(
                url,
                engine="netcdf4",
                decode_times=False,
                chunks={"time": 10, "depth": -1, "lat": 100, "lon": 100},
            )

            # Compute the time window bounds in the raw float unit
            # (hours since 2000-01-01 00:00:00, matching HYCOM's actual reference epoch).
            ref = np.datetime64("2000-01-01T00:00:00", "ns")
            start_h = float(
                (np.datetime64(start_date, "ns") - ref) / np.timedelta64(1, "h")
            )
            end_h = float(
                (np.datetime64(end_date, "ns") + np.timedelta64(1, "D") - ref)
                / np.timedelta64(1, "h")
            )

            t_raw = ds["time"].values                        # 1D float array, cheap OPeNDAP read
            t_indices = np.where((t_raw >= start_h) & (t_raw < end_h))[0]

            if len(t_indices) == 0:
                raise ValueError(
                    f"No HYCOM timesteps found between {start_date} and {end_date}. "
                    f"Dataset covers {t_raw[0]:.0f}–{t_raw[-1]:.0f} hours since 2000-01-01. "
                    f"Requested: {start_h:.0f}–{end_h:.0f}."
                )

            # Spatial + time slice — isel(time=...) avoids any index sort requirement
            ds_sliced = ds.isel(time=t_indices).sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min_360, lon_max_360),
            )[variables]

            # Decode the selected time steps and replace the raw float coordinate
            decoded = ref + (t_raw[t_indices] * 3.6e12).astype("timedelta64[ns]")
            ds_sliced = ds_sliced.assign_coords(time=decoded)

            return ds_sliced

        ds_ts = _open_and_slice(self._url, ["water_temp", "salinity"])
        ds_uv = _open_and_slice(self._url_uv, ["water_u", "water_v"])

        ds_raw = xr.merge([ds_ts, ds_uv])

        # Convert longitude back to -180..180 for downstream harmonization
        ds_raw = ds_raw.assign_coords(
            lon=xr.where(ds_raw["lon"] > 180, ds_raw["lon"] - 360, ds_raw["lon"])
        )

        ds_std = self._interpolate_to_standard_depths(ds_raw)
        logger.info("HYCOM tile loaded and interpolated: %s", ds_std)
        return ds_std

    def fetch_and_cache(
        self,
        year: int,
        bbox: tuple[float, float, float, float],
        gcs_uri: str,
    ) -> None:
        """
        Fetch one full calendar year of HYCOM data and write to GCS as Zarr.

        Parameters
        ----------
        year : int
            Calendar year to fetch (Jan 1 – Dec 31).
            Must be within HYCOM GLBy0.08/expt_93.0 coverage (2018-12-04 to 2024-09-04).
        bbox : tuple of float
            (lon_min, lat_min, lon_max, lat_max) in decimal degrees WGS84, -180–180 convention.
            Passed directly to fetch_tile() without transformation. Note: HYCOM source uses
            0–360 longitude internally; fetch_tile() handles the conversion transparently.
            Source: project-standard Gulf of Maine region, e.g. (-71.0, 41.0, -66.0, 45.0).
        gcs_uri : str
            GCS destination, e.g. "gs://bucket/hycom/tiles/2022/".
            If the URI already exists, returns immediately — idempotent, safe to re-run
            after a spot VM preemption.

        Side effects
        ------------
        Writes a Zarr store to gcs_uri with dims (time, depth, lat, lon)
        and variables: water_temp [°C], salinity [psu], water_u [m/s], water_v [m/s].
        This Zarr store is the canonical HYCOM input tile for the MHW detection and SVaR
        analytics pipeline for this year/region; it is read by run_data_prep.py and
        train_era5.py / train_wn2.py via xr.open_zarr().
        Credentials are read from GOOGLE_APPLICATION_CREDENTIALS automatically by gcsfs.
        """
        fs = gcsfs.GCSFileSystem()
        path = gcs_uri.removeprefix("gs://")  # Python 3.9+ — enforced by Dockerfile (python:3.11-slim)
        if fs.exists(path):
            logger.info("Cache hit — skipping HYCOM fetch for %d: %s", year, gcs_uri)
            return

        start_date = f"{year}-01-01"
        end_date   = f"{year}-12-31"
        logger.info("Fetching HYCOM year %d (%s to %s)...", year, start_date, end_date)

        ds = self.fetch_tile(start_date, end_date, bbox)
        ds.to_zarr(gcs_uri, mode="w", consolidated=True)
        logger.info("HYCOM year %d written to %s", year, gcs_uri)

    def _interpolate_to_standard_depths(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Interpolate HYCOM data from native hybrid vertical levels to TARGET_DEPTHS_M.

        This step is critical because the hybrid coordinate produces variable layer
        thicknesses: the top 10 levels may span 0-200 m in deep water but only
        0-30 m on the shelf. Without interpolation, the 1D-CNN would see inputs of
        inconsistent physical scale across spatial locations.

        Parameters
        ----------
        ds : xr.Dataset
            Raw HYCOM dataset with native 'depth' coordinate [m].
            The 'depth' values represent layer mid-depths, positive downward.

        Returns
        -------
        ds_interp : xr.Dataset
            Same variables but with 'depth' replaced by TARGET_DEPTHS_M.
            Points above the seafloor but below the native grid are filled with NaN
            to preserve the distinction between "no data" and "value is zero".
        """
        native_depths = ds["depth"].values.astype(float)

        arrays = {}
        for var in HYCOM_VARIABLES:
            arr = ds[var].values  # (time, depth, lat, lon)
            interp_layers = []
            for d_target in TARGET_DEPTHS_M:
                # Linear interpolation along depth axis at each (t, lat, lon)
                interp_layers.append(
                    np.interp(
                        d_target,
                        native_depths,
                        arr,
                        left=np.nan,
                        right=np.nan,
                    )
                    if arr.ndim == 1
                    else self._interp_depth_axis(arr, native_depths, d_target)
                )
            arrays[var] = (
                ["time", "depth", "lat", "lon"],
                np.stack(interp_layers, axis=1),
            )

        return xr.Dataset(
            arrays,
            coords={
                "time": ds["time"],
                "depth": ("depth", TARGET_DEPTHS_M, {"units": "m", "positive": "down"}),
                "lat": ds["lat"],
                "lon": ds["lon"],
            },
        )

    @staticmethod
    def _interp_depth_axis(
        arr: np.ndarray,
        native_depths: np.ndarray,
        target_depth: float,
    ) -> np.ndarray:
        """
        Vectorised linear interpolation along axis=1 (depth) for a 3D or 4D array.

        Parameters
        ----------
        arr : np.ndarray
            Shape (time, native_depth, lat, lon) — raw HYCOM variable.
        native_depths : np.ndarray
            Native HYCOM layer mid-depths [m], shape (native_depth,).
        target_depth : float
            Target depth [m] at which to evaluate the interpolant.

        Returns
        -------
        out : np.ndarray
            Shape (time, lat, lon) — the variable interpolated to target_depth.
        """
        idx = np.searchsorted(native_depths, target_depth)
        if idx == 0:
            return arr[:, 0, :, :]
        if idx >= len(native_depths):
            return np.full(arr[:, 0, :, :].shape, np.nan)
        d0, d1 = native_depths[idx - 1], native_depths[idx]
        w = (target_depth - d0) / (d1 - d0)
        return (1 - w) * arr[:, idx - 1, :, :] + w * arr[:, idx, :, :]


# ---------------------------------------------------------------------------
# Harmonizer
# ---------------------------------------------------------------------------

class DataHarmonizer:
    """
    Combines WeatherNext 2 atmospheric fields and HYCOM oceanic fields into a
    single CF-compliant xr.Dataset on a shared 0.25-degree daily grid.

    The harmonized dataset retains the 64-member 'member' dimension from
    WeatherNext 2. HYCOM fields, being deterministic reanalysis, are broadcast
    across all ensemble members — each member therefore represents one plausible
    atmosphere paired with the observed ocean state.

    This design encodes the physical assumption that atmospheric uncertainty
    (captured by the FGN ensemble) is the primary source of MHW forecast spread,
    while the oceanic initial condition (from HYCOM) is treated as known.
    """

    @staticmethod
    def expand_and_perturb(
        ds: xr.Dataset,
        n_members: int = 64,
        seed: int = 42,
    ) -> xr.Dataset:
        """
        Broadcast a single-member ERA5 Dataset to n_members synthetic members
        by injecting independent Gaussian noise into each atmospheric variable.

        Parameters
        ----------
        ds : xr.Dataset
            ERA5 dataset with member=1 dimension.
            Must contain variables matching WN2_VARIABLES naming convention.
        n_members : int
            Target number of synthetic ensemble members. Default 64 matches WN2.
        seed : int
            Base random seed. Member i uses seed + i for reproducibility.
            Changing seed produces a different but equally valid synthetic ensemble.

        Returns
        -------
        ds_perturbed : xr.Dataset
            Same variables and spatial/temporal dimensions but member=n_members.
            Each member has independent Gaussian noise added to each variable.
            Noise σ values calibrated to match published WN2 intra-ensemble spread.

        Physical rationale
        ------------------
        ERA5 is deterministic. To use it with SVaR (which requires ensemble spread),
        we inject per-member Gaussian noise. The σ values are chosen to approximate
        the intra-ensemble spread documented for WeatherNext 2. This is a proxy:
        real WN2 spread is non-Gaussian and temporally correlated (FGN), but
        Gaussian noise is sufficient for training because the model learns from
        the physics SDD label, not from the ensemble structure itself.
        """
        # Build list of per-member Datasets with independent noise
        ds_base = ds.isel(member=0)
        member_datasets = []
        for i in range(n_members):
            rng = np.random.default_rng(seed + i)
            ds_m = ds_base.copy(deep=True)
            for var, sigma in NOISE_SIGMAS.items():
                if var in ds_m:
                    noise = rng.normal(0.0, sigma, ds_m[var].shape).astype(np.float32)
                    ds_m[var] = ds_m[var] + noise
            member_datasets.append(ds_m)

        ds_perturbed = xr.concat(member_datasets, dim="member")
        ds_perturbed["member"] = np.arange(n_members)
        return ds_perturbed

    def harmonize(
        self,
        wn2_ds: xr.Dataset,
        hycom_ds: xr.Dataset,
    ) -> xr.Dataset:
        """
        Regrid and merge WeatherNext 2 and HYCOM datasets onto TARGET_LON / TARGET_LAT.

        Parameters
        ----------
        wn2_ds : xr.Dataset
            WeatherNext 2 output from WeatherNext2Harvester.fetch_ensemble().
            Expected dimensions: (member, time, latitude, longitude).
        hycom_ds : xr.Dataset
            HYCOM output from HYCOMLoader.fetch_tile(), already on TARGET_DEPTHS_M.
            Expected dimensions: (time, depth, lat, lon).

        Returns
        -------
        merged : xr.Dataset
            Dimensions: (member, time, depth, latitude, longitude)
            - WeatherNext 2 variables are 2D in (latitude, longitude).
            - HYCOM variables include the depth dimension.
            - All variables share (member, time, latitude, longitude).
            - CF conventions: coordinate names standardised to 'latitude' and
              'longitude'; units attributes preserved from source datasets.
            - The 'member' dimension is essential: downstream SVaR is estimated
              from the empirical quantiles of the 64-member SDD distribution.
        """
        logger.info("Regridding WeatherNext 2 to 0.25-degree target grid.")
        wn2_regridded = wn2_ds.interp(
            latitude=TARGET_LAT, longitude=TARGET_LON, method="linear"
        )

        # Auto-expand single-member input (ERA5 proxy path)
        if wn2_regridded.dims.get("member", 1) == 1:
            logger.info("member=1 detected — expanding to 64 synthetic members.")
            wn2_regridded = DataHarmonizer.expand_and_perturb(wn2_regridded)

        logger.info("Regridding HYCOM to 0.25-degree target grid.")
        hycom_regridded = hycom_ds.interp(
            lat=TARGET_LAT, lon=TARGET_LON, method="linear"
        ).rename({"lat": "latitude", "lon": "longitude"})

        # Broadcast HYCOM across the member dimension
        hycom_broadcast = hycom_regridded.expand_dims(
            {"member": wn2_regridded["member"]}
        )

        merged = xr.merge(
            [wn2_regridded, hycom_broadcast],
            join="inner",  # retain only overlapping times
        )

        merged.attrs.update(
            {
                "Conventions": "CF-1.8",
                "title": "MHW Risk Profiler — Harmonized Atmosphere-Ocean Dataset",
                "institution": "mhw-risk-profiler pipeline",
                "source": "WeatherNext 2 (GEE FGN ensemble) + HYCOM GLBv0.08",
                "history": f"Created by harvester.py",
                "comment": (
                    "64-member WeatherNext 2 ensemble paired with deterministic "
                    "HYCOM reanalysis. Use the 'member' dimension for SVaR estimation."
                ),
            }
        )

        logger.info("Harmonized dataset: %s", merged)
        return merged


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_ingestion_pipeline(
    start_date: str,
    end_date: str,
    bbox: tuple[float, float, float, float],
    gcs_bucket: str,
    output_dir: str = "data/processed",
    service_account_key: Optional[str] = None,
    n_members: int = 64,
) -> Path:
    """
    End-to-end ingestion: authenticate, fetch, harmonize, and save.

    This is the function called by the Dockerfile ENTRYPOINT and by the
    FastAPI endpoint. It orchestrates the three classes above in sequence.

    Parameters
    ----------
    start_date : str
        ISO 8601 start date for the analysis window, e.g. "2023-06-01".
    end_date : str
        ISO 8601 end date (inclusive), e.g. "2023-08-31".
    bbox : tuple of float
        (lon_min, lat_min, lon_max, lat_max) — spatial region of interest.
        For salmon farm applications: include a 2-degree buffer beyond the
        lease polygon to capture upstream atmospheric and oceanic forcing.
    gcs_bucket : str
        GCS bucket name for WeatherNext 2 Zarr cache.
    output_dir : str
        Local directory for saving the harmonized output.
    service_account_key : str, optional
        Path to GCP service account JSON. None uses Application Default Credentials.
    n_members : int
        WeatherNext 2 ensemble members to retrieve (1–64).

    Returns
    -------
    output_path : Path
        Absolute path to the saved Zarr store in output_dir.
        The calling process should print this path as verification evidence.
    """
    # -- Fetch WeatherNext 2 ------------------------------------------------
    wn2 = WeatherNext2Harvester(
        gcs_bucket=gcs_bucket,
        service_account_key=service_account_key,
    )
    wn2.authenticate()
    wn2_ds = wn2.fetch_ensemble(start_date, end_date, bbox, n_members=n_members)

    # -- Fetch HYCOM --------------------------------------------------------
    hycom = HYCOMLoader()
    hycom_ds = hycom.fetch_tile(start_date, end_date, bbox)

    # -- Harmonize ----------------------------------------------------------
    harmonizer = DataHarmonizer()
    merged = harmonizer.harmonize(wn2_ds, hycom_ds)

    # -- Save ---------------------------------------------------------------
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"harmonized_{start_date}_{end_date}.zarr"

    logger.info("Saving harmonized dataset to %s", output_path)
    merged.to_zarr(output_path, mode="w", consolidated=True)

    # Verification print — required by CLAUDE.md Verification Gate
    saved = xr.open_zarr(output_path, consolidated=True)
    print("=" * 60)
    print("VERIFICATION — Harmonized dataset written successfully")
    print(f"Path : {output_path.resolve()}")
    print(saved)
    print("=" * 60)

    return output_path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _wait_for_ee_task(task, poll_interval_s: int = 30) -> None:
    """
    Block until a GEE batch Export task completes or raises on failure.

    Parameters
    ----------
    task : ee.batch.Task
        The submitted GEE export task object.
    poll_interval_s : int
        Seconds between status polls. GEE export tasks typically take
        2–20 minutes depending on region size and ensemble members requested.
    """
    import time

    logger.info("Waiting for GEE task: %s", task.id)
    while True:
        status = task.status()
        state = status["state"]
        logger.info("GEE task state: %s", state)
        if state == "COMPLETED":
            return
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"GEE export task failed: {status.get('error_message')}")
        time.sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="MHW Risk Profiler — Data Ingestion")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    parser.add_argument("--gcs-bucket", required=True, help="GCS bucket for WeatherNext 2 cache")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--key", default=None, help="Path to GCP service account JSON")
    parser.add_argument("--n_members", type=int, default=64, dest="n_members")
    args = parser.parse_args()

    out = run_ingestion_pipeline(
        start_date=args.start,
        end_date=args.end,
        bbox=tuple(args.bbox),
        gcs_bucket=args.gcs_bucket,
        output_dir=args.output_dir,
        service_account_key=args.key,
        n_members=args.n_members,
    )
    print(f"Pipeline complete. Output: {out}")
