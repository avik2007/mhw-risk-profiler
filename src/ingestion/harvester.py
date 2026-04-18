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
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Optional

import dask
import dask.array as da
import gcsfs
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

import zarr.errors

try:
    import ee
except ImportError:  # earthengine-api not installed in non-GEE environments
    ee = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _gcs_safe_write(ds: xr.Dataset, gcs_uri: str, consolidated: bool = True) -> None:
    """Write an xr.Dataset to a GCS Zarr store, handling zarr v3 + gcsfs incompatibility.

    zarr v3 calls ``delete_dir()`` unconditionally when opening with ``mode="w"``,
    even if the target path has never been written to.  On GCS, gcsfs raises
    OSError 404 for a delete on a non-existent path, crashing the write before
    any data is transferred.

    This helper works around the issue by:
      1. Manually deleting existing content with gcsfs (guarded by ``fs.exists``
         so a non-existent path is silently ignored).
      2. Writing with ``mode="a"`` (append/create-if-absent), which does NOT call
         ``delete_dir()``, producing an identical result to ``mode="w"`` on a
         freshly cleared path.

    Race-safety: if a concurrent writer finishes between our fs.exists() check and
    our to_zarr() call, zarr v3 raises ContainsArrayError.  We catch it, check if
    the store is now complete, and either return (cache hit) or delete-and-retry once.

    After a successful write, a ``.complete`` sentinel is written so _gcs_complete()
    can recognise the store without relying on zarr v3 consolidated metadata.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to write.
    gcs_uri : str
        GCS destination URI, e.g. "gs://bucket/hycom/tiles/2022/".
    consolidated : bool
        Whether to write consolidated metadata (default True).
    """
    fs = gcsfs.GCSFileSystem()
    path = gcs_uri.removeprefix("gs://").rstrip("/")
    if fs.exists(path):
        fs.rm(path, recursive=True)
    try:
        ds.to_zarr(gcs_uri, mode="a", consolidated=consolidated)
    except zarr.errors.ContainsArrayError:
        if _gcs_complete(fs, gcs_uri):
            return
        if fs.exists(path):
            fs.rm(path, recursive=True)
        ds.to_zarr(gcs_uri, mode="a", consolidated=consolidated)
    fs.touch(f"{path}/.complete")


def _gcs_complete(fs: gcsfs.GCSFileSystem, gcs_uri: str) -> bool:
    """Return True only if gcs_uri contains a complete Zarr store.

    Primary check: ``.complete`` sentinel written by _gcs_safe_write after a
    successful to_zarr() call.

    Fallback: ``water_v/zarr.json`` presence, which covers HYCOM monthly/annual
    tiles written before this sentinel scheme was introduced.  zarr v3 does not
    write ``.zmetadata`` (the original sentinel), so that check always returned
    False and the idempotency guard was silently broken.

    Parameters
    ----------
    fs : gcsfs.GCSFileSystem
        Authenticated GCS filesystem instance.
    gcs_uri : str
        GCS URI of the Zarr store, e.g. "gs://bucket/hycom/tiles/2022/".
        The leading "gs://" prefix and any trailing slash are stripped before
        the existence check, matching gcsfs path conventions.
    """
    base = gcs_uri.removeprefix("gs://").rstrip("/")
    if fs.exists(f"{base}/.complete"):
        return True
    # Fallback for HYCOM tiles written before the .complete sentinel was introduced.
    return fs.exists(f"{base}/water_v/zarr.json")

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
    # sea_surface_temperature excluded: WN2 SST is masked over land, returning defaultValue=0
    # for ~25% of the GoM bbox (coastline pixels). HYCOM is the authoritative SST source.
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

    def authenticate(self) -> None:
        """
        Authenticate with both Google Earth Engine and Google Cloud Storage.

        GEE authentication uses the earthengine-api flow (service account or OAuth).
        GCS authentication uses google-cloud-storage with the same credential source.

        This method is idempotent — safe to call multiple times.
        """
        if not self._ee_initialized:
            if self._key:
                import json
                with open(self._key) as fh:
                    _key_data = json.load(fh)
                credentials = ee.ServiceAccountCredentials(
                    email=_key_data["client_email"], key_file=self._key
                )
                ee.Initialize(credentials=credentials)
            else:
                # Application Default Credentials — standard in Cloud Run / Vertex AI
                ee.Initialize()
            self._ee_initialized = True
            logger.info("GEE authentication successful.")

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
            ISO 8601 start date, e.g. "2022-01-01".
        end_date : str
            ISO 8601 end date, EXCLUSIVE (GEE filterDate convention), e.g. "2023-01-01"
            to include all of 2022. Pass year+1-01-01 to cover a full calendar year.
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
        if not self._ee_initialized:
            raise RuntimeError("Call authenticate() before fetch_ensemble().")

        lon_min, lat_min, lon_max, lat_max = bbox
        region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

        logger.info(
            "Querying WeatherNext 2: %s to %s, bbox=%s, members=%d",
            start_date, end_date, bbox, n_members,
        )

        # WN2 structure: each image = one member × one init_time × one forecast_hour.
        # Filter to 00Z initialization only (one forecast per calendar day) and
        # forecast_hour=24 (next-day forecast), yielding 365 × 64 images per year.
        # ee.Filter.date acts on system:time_start (Unix ms of init window start).
        collection = (
            ee.ImageCollection(WN2_GEE_ASSET)
            .filter(ee.Filter.date(start_date, end_date))
            .filter(ee.Filter.stringEndsWith("start_time", "T00:00:00Z"))
            .filter(ee.Filter.eq("forecast_hour", 24))
        )

        zarr_uri = self._fetch_and_write_zarr(
            collection, region, start_date, end_date, n_members
        )
        ds = xr.open_zarr(zarr_uri, chunks="auto", consolidated=True)

        logger.info("WeatherNext 2 dataset loaded: %s", ds)
        return ds

    def fetch_and_cache(
        self,
        year: int,
        bbox: tuple[float, float, float, float],
        gcs_uri: str,
        n_members: int = 64,
    ) -> None:
        """
        Fetch one full calendar year of WeatherNext 2 data and write to GCS as Zarr.

        Authenticates lazily (idempotent — safe to call even if authenticate() was
        already invoked). Uses .zmetadata as the completeness marker so partial writes
        from a failed run are never treated as a cache hit.

        Parameters
        ----------
        year : int
            Calendar year to fetch (Jan 1 – Dec 31).
            Coverage: 2022-01-01 to present (WN2 starts 2022-01-01).
        bbox : tuple of float
            (lon_min, lat_min, lon_max, lat_max) in decimal degrees WGS84.
            Must be ≤ 5°×5° for the sampleRectangle compute path (GoM standard
            bbox of (-71.0, 41.0, -66.0, 45.0) satisfies this at 5°×4°).
        gcs_uri : str
            GCS destination for the annual Zarr store.
            Convention: "gs://<bucket>/weathernext2/cache/wn2_{year}-01-01_{year}-12-31_m{n}.zarr"
            If the store is already complete (.zmetadata present), returns immediately.
        n_members : int
            Number of ensemble members to retrieve (1–64). Default 64 for production SVaR.

        Side effects
        ------------
        Writes a consolidated Zarr store at gcs_uri with dims (member, time, latitude, longitude)
        and variables: sea_surface_temperature [K], 2m_temperature [K],
        10m_u_component_of_wind [m/s], 10m_v_component_of_wind [m/s],
        mean_sea_level_pressure [Pa].
        """
        if not self._ee_initialized:
            self.authenticate()

        fs = gcsfs.GCSFileSystem()
        if _gcs_complete(fs, gcs_uri):
            logger.info("Cache hit — skipping WN2 fetch for %d: %s", year, gcs_uri)
            return

        lon_min, lat_min, lon_max, lat_max = bbox
        region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

        # end_date is exclusive in GEE filterDate — use Jan 1 of next year to
        # include all of Dec 31. Same fix applied to ERA5Harvester.fetch_and_cache().
        collection = (
            ee.ImageCollection(WN2_GEE_ASSET)
            .filter(ee.Filter.date(f"{year}-01-01", f"{year + 1}-01-01"))
            .filter(ee.Filter.stringEndsWith("start_time", "T00:00:00Z"))
            .filter(ee.Filter.eq("forecast_hour", 24))
        )

        logger.info("Fetching WeatherNext 2 for year %d -> %s", year, gcs_uri)
        ds = self._build_dataset(collection, region, n_members, gcs_uri)

        if self._key:
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", self._key)
        _gcs_safe_write(ds, gcs_uri)
        logger.info("WN2 year %d Zarr written to %s", year, gcs_uri)

    def _build_dataset(
        self,
        collection,
        region,
        n_members: int,
        gcs_uri: str,
    ) -> xr.Dataset:
        """
        Fetch WeatherNext 2 pixel values from a pre-filtered ImageCollection and
        assemble them into an xr.Dataset with dims (member, time, latitude, longitude).

        Uses GEE's synchronous sampleRectangle compute path. One API call is made
        per calendar day: all n_members images for that day are merged with toBands()
        into a single image (n_members × 5 bands), then sampleRectangle extracts the
        full spatial grid in one round trip. For 365 days this yields ~365 GEE calls.

        WN2 native resolution is 0.25° (~27,830 m). For the GoM bbox (5°×4°) the
        pixel grid is 21×17 = 357 pixels per band. The 320-band combined image
        (64 members × 5 variables) returns ~114,240 float values per call — well
        within GEE's sampleRectangle limits.

        Band naming after toBands():  "{system:index}_{band_name}"
        where system:index = "{start_compact}_{end_compact}_{member_int}"
        and member_int is the integer member ID (0–63).
        Members are sorted numerically via a derived 'member_int' property before
        toBands(), so band index 0 = member 0, index 1 = member 1, etc.

        Parameters
        ----------
        collection : ee.ImageCollection
            WN2 collection already filtered to 00Z init, forecast_hour=24, and the
            target date range. Expected size: 365 × n_members images per year.
        region : ee.Geometry
            Spatial extent for pixel extraction.
        n_members : int
            Number of ensemble members per day.
        gcs_uri : str
            GCS URI of the final annual Zarr store (e.g. gs://bucket/wn2_2022.zarr).
            Per-day intermediates are written to {gcs_uri}_daily/d{YYYYMMDD}/.

        Returns
        -------
        ds : xr.Dataset
            Dimensions: (member, time, latitude, longitude)
            Variables: WN2_VARIABLES [K, K, m/s, m/s, Pa]
            member coordinate: integer 0 … n_members-1
            time coordinate: one np.datetime64 per calendar day
        """
        import time as _time

        target_bands = list(WN2_VARIABLES)

        # Get all unique calendar dates covered by the filtered collection.
        # start_time format: "2022-01-01T00:00:00Z" — take the date part only.
        all_start_times = collection.aggregate_array("start_time").getInfo()
        if not all_start_times:
            raise RuntimeError(
                "WeatherNext 2 collection is empty after filtering. "
                "Check date range, 00Z filter, and forecast_hour=24 filter."
            )
        unique_dates = sorted({st[:10] for st in all_start_times})
        logger.info("WN2 collection covers %d days, %d total images",
                    len(unique_dates), len(all_start_times))

        fs = gcsfs.GCSFileSystem()
        daily_base = gcs_uri.rstrip("/") + "_daily"
        datasets: list = []
        lat_coords: Optional[np.ndarray] = None
        lon_coords: Optional[np.ndarray] = None

        for i, date_str in enumerate(unique_dates):
            day_uri = f"{daily_base}/d{date_str.replace('-', '')}/"
            if _gcs_complete(fs, day_uri):
                ds_day = xr.open_zarr(day_uri, chunks={})
                if lat_coords is None:
                    lat_coords = ds_day["latitude"].values
                    lon_coords = ds_day["longitude"].values
                datasets.append(ds_day)
                logger.info("WN2 cache hit: %s (%d/%d)", date_str, i + 1, len(unique_dates))
                continue

            d = _date.fromisoformat(date_str)
            next_date_str = (d + timedelta(days=1)).isoformat()
            # system:index format: "{YYYYMMDDHHMM}_{YYYYMMDDHHMM}_{member_int}"
            # start_compact = init time at 00Z, end_compact = 24h later at 00Z.
            start_compact = d.strftime("%Y%m%d0000")
            end_compact   = (d + timedelta(days=1)).strftime("%Y%m%d0000")

            # Query the asset directly with a 1-day date filter rather than
            # filtering the year-level collection. This gives GEE a computation
            # graph of ~64 images instead of ~23,360, avoiding server-side
            # timeout on the sampleRectangle call.
            day_coll = (
                ee.ImageCollection(WN2_GEE_ASSET)
                .filter(ee.Filter.date(date_str, next_date_str))
                .filter(ee.Filter.stringEndsWith("start_time", "T00:00:00Z"))
                .filter(ee.Filter.eq("forecast_hour", 24))
                .map(lambda img: img.set(
                    "member_int",
                    ee.Number.parse(img.getString("ensemble_member"))
                ))
                .sort("member_int")
                .limit(n_members)
            )

            # Merge all member images into one multi-band image and extract pixels.
            # Band names: "{start_compact}_{end_compact}_{member_int}_{var}"
            combined = day_coll.select(target_bands).toBands()

            # Retry up to 5 times with exponential backoff (60, 120, 240, 480 s).
            for attempt in range(5):
                try:
                    sample = combined.sampleRectangle(region=region, defaultValue=0).getInfo()
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise
                    wait = 60 * (2 ** attempt)
                    logger.warning(
                        "sampleRectangle failed for %s (attempt %d/5): %s — retrying in %ds",
                        date_str, attempt + 1, exc, wait,
                    )
                    _time.sleep(wait)
            props = sample["properties"]

            # Derive lat/lon coordinates from the first day's pixel grid.
            if lat_coords is None:
                first_key = f"{start_compact}_{end_compact}_0_{target_bands[0]}"
                first_arr = np.array(props[first_key])  # (n_lat, n_lon)
                n_lat, n_lon = first_arr.shape
                bounds = region.bounds().getInfo()["coordinates"][0]
                lon_coords = np.linspace(bounds[0][0], bounds[2][0], n_lon)
                lat_coords = np.linspace(bounds[0][1], bounds[2][1], n_lat)

            data_vars_day = {}
            for var in target_bands:
                member_arrays = [
                    np.array(props[f"{start_compact}_{end_compact}_{m}_{var}"])
                    for m in range(n_members)
                ]
                arr = np.array(member_arrays, dtype=np.float32)  # (n_members, n_lat, n_lon)
                data_vars_day[var] = (["member", "latitude", "longitude"], arr)

            ds_day = xr.Dataset(
                data_vars_day,
                coords={
                    "member":    np.arange(n_members),
                    "latitude":  lat_coords,
                    "longitude": lon_coords,
                },
            ).expand_dims({"time": [np.datetime64(date_str)]})

            _gcs_safe_write(ds_day, day_uri)
            datasets.append(ds_day)
            logger.info("WN2 fetched %s (%d/%d)", date_str, i + 1, len(unique_dates))

        return xr.concat(datasets, dim="time")

    def _fetch_and_write_zarr(
        self,
        collection,
        region,
        start_date: str,
        end_date: str,
        n_members: int,
    ) -> str:
        """
        Build an xr.Dataset from a filtered WN2 ImageCollection and write it to GCS.

        This is the internal cache layer for fetch_ensemble(). The GCS URI is derived
        deterministically from start_date, end_date, and n_members so that repeated
        calls with the same arguments are idempotent.

        Uses _gcs_complete() (.zmetadata check) as the cache sentinel — a partial
        write from a failed run is never treated as a cache hit.

        Parameters
        ----------
        collection : ee.ImageCollection
            WN2 collection filtered to the target date range, 00Z init, fh=24.
        region : ee.Geometry
            Spatial extent for sampleRectangle pixel extraction.
        start_date, end_date : str
            ISO 8601 dates used to build the deterministic GCS cache key.
            end_date should be exclusive (e.g. "2023-01-01" for full 2022).
        n_members : int
            Number of ensemble members to extract.

        Returns
        -------
        gcs_uri : str
            GCS URI of the written (or pre-existing) Zarr store.
        """
        store_key = f"{self._gcs_prefix}/wn2_{start_date}_{end_date}_m{n_members}.zarr"
        gcs_uri = f"gs://{self._gcs_bucket}/{store_key}"

        fs = gcsfs.GCSFileSystem()
        if _gcs_complete(fs, gcs_uri):
            logger.info("Cache hit for WeatherNext 2 Zarr: %s", gcs_uri)
            return gcs_uri

        logger.info("Fetching WeatherNext 2 via sampleRectangle compute path.")
        if self._key:
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", self._key)

        ds = self._build_dataset(collection, region, n_members, gcs_uri)
        _gcs_safe_write(ds, gcs_uri)
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

        Data is fetched month-by-month to limit preemption exposure.  Each month
        is written to an intermediate Zarr at ``{gcs_uri}monthly/m{MM}/`` before
        the 12 monthly stores are concatenated into the final annual store at
        ``gcs_uri``.  Both levels use ``_gcs_complete`` (checks for ``.zmetadata``)
        as the idempotency marker, so a partial write from a preempted spot VM is
        never treated as a cache hit.

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
            GCS destination for the annual tile, e.g. "gs://bucket/hycom/tiles/2022/".
            Monthly intermediates are written to "gs://bucket/hycom/tiles/2022/monthly/mMM/".
            If the annual store is already complete (.zmetadata present), returns immediately.
            If some monthly stores are already complete, those months are skipped.

        Side effects
        ------------
        Writes monthly Zarr stores at ``{gcs_uri}monthly/m{MM}/`` (12 stores) and a
        consolidated annual Zarr at ``gcs_uri``, all with dims (time, depth, lat, lon)
        and variables: water_temp [°C], salinity [psu], water_u [m/s], water_v [m/s].
        This annual Zarr store is the canonical HYCOM input tile for the MHW detection and
        SVaR analytics pipeline; it is read by run_data_prep.py, train_era5.py, and
        train_wn2.py via xr.open_zarr().
        Credentials are read from GOOGLE_APPLICATION_CREDENTIALS automatically by gcsfs.
        """
        import calendar

        fs = gcsfs.GCSFileSystem()
        if _gcs_complete(fs, gcs_uri):
            logger.info("Cache hit — skipping HYCOM fetch for %d: %s", year, gcs_uri)
            return

        base = gcs_uri.rstrip("/")
        month_uris: list[str] = []

        for month in range(1, 13):
            _, last_day = calendar.monthrange(year, month)
            start_date = f"{year}-{month:02d}-01"
            end_date   = f"{year}-{month:02d}-{last_day:02d}"
            month_uri  = f"{base}/monthly/m{month:02d}/"

            if not _gcs_complete(fs, month_uri):
                logger.info(
                    "Fetching HYCOM %d-%02d (%s to %s)...", year, month, start_date, end_date
                )
                ds_month = self.fetch_tile(start_date, end_date, bbox)
                _gcs_safe_write(ds_month, month_uri)
                logger.info("HYCOM %d-%02d written to %s", year, month, month_uri)
            else:
                logger.info("Cache hit — HYCOM %d-%02d: %s", year, month, month_uri)

            month_uris.append(month_uri)

        # Concatenate all 12 monthly stores into the annual tile.
        # Monthly data is already on GCS; xr.open_zarr reads lazily so the concat
        # and subsequent to_zarr stream data through the VM in Dask chunks rather
        # than loading the full year into memory at once.
        logger.info("Concatenating 12 monthly HYCOM tiles for year %d...", year)
        monthly_datasets = [xr.open_zarr(uri, chunks="auto") for uri in month_uris]
        ds_annual = xr.concat(monthly_datasets, dim="time")
        _gcs_safe_write(ds_annual, gcs_uri)
        logger.info("HYCOM year %d annual store written to %s", year, gcs_uri)

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
