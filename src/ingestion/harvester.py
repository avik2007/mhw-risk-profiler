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

# WeatherNext 2 GEE asset path (Google public data)
WN2_GEE_ASSET = "projects/gcp-public-data-weathernext/assets/59572747_3_0"

# HYCOM THREDDS base URL for GLBv0.08 reanalysis (1/12-degree, daily)
HYCOM_THREDDS_BASE = (
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z"
)

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
                credentials = ee.ServiceAccountCredentials(
                    email=None, key_file=self._key
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
        all ensemble members as a single xr.Dataset with a 'member' dimension.

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
            Number of ensemble members to retrieve (max 64 for WeatherNext 2).
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

        collection = (
            ee.ImageCollection(WN2_GEE_ASSET)
            .filter(date_filter)
            .filter(ee.Filter.eq("number_of_members", n_members))
            .select(WN2_VARIABLES)
        )

        # Export to GCS as Zarr and open lazily with Dask
        zarr_uri = self._export_to_gcs(collection, region, start_date, end_date)
        ds = xr.open_zarr(zarr_uri, chunks="auto", consolidated=True)

        logger.info("WeatherNext 2 dataset loaded: %s", ds)
        return ds

    def _export_to_gcs(
        self,
        collection,
        region,
        start_date: str,
        end_date: str,
    ) -> str:
        """
        Export a GEE ImageCollection to GCS as a consolidated Zarr store.

        Parameters
        ----------
        collection : ee.ImageCollection
            The filtered WeatherNext 2 collection to export.
        region : ee.Geometry
            Spatial extent for the export clip.
        start_date, end_date : str
            Used to construct a unique, deterministic GCS object key,
            enabling cache hits on repeated queries for the same date range.

        Returns
        -------
        gcs_uri : str
            GCS URI of the exported Zarr store: "gs://<bucket>/<prefix>/<key>.zarr"
            Suitable for direct use with xr.open_zarr().
        """
        import ee

        store_key = f"{self._gcs_prefix}/wn2_{start_date}_{end_date}.zarr"
        gcs_uri = f"gs://{self._gcs_bucket}/{store_key}"

        # Check for cached export to avoid redundant GEE Tasks
        bucket = self._gcs_client.bucket(self._gcs_bucket)
        if any(bucket.list_blobs(prefix=store_key)):
            logger.info("Cache hit for WeatherNext 2 Zarr: %s", gcs_uri)
            return gcs_uri

        logger.info("Submitting GEE export task to GCS: %s", gcs_uri)
        task = ee.batch.Export.image.toCloudStorage(
            image=collection.toBands(),
            description=f"wn2_export_{start_date}_{end_date}",
            bucket=self._gcs_bucket,
            fileNamePrefix=store_key,
            region=region,
            scale=27750,  # ~0.25-degree at equator in meters
            fileFormat="GeoTIFF",  # GEE does not export Zarr natively; converted post-export
            maxPixels=int(1e10),
        )
        task.start()
        _wait_for_ee_task(task)
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

    def __init__(self, thredds_url: str = HYCOM_THREDDS_BASE) -> None:
        """
        Parameters
        ----------
        thredds_url : str
            OPeNDAP URL for the HYCOM THREDDS server.
            Default points to HYCOM GLBv0.08 expt_93.0 (Jan 1994 – Dec 2015).
            For real-time use, substitute the relevant nowcast/forecast experiment URL.
        """
        self._url = thredds_url

    def fetch_tile(
        self,
        start_date: str,
        end_date: str,
        bbox: tuple[float, float, float, float],
    ) -> xr.Dataset:
        """
        Download a spatiotemporal tile of HYCOM 3D temperature and salinity,
        then interpolate from native hybrid levels to TARGET_DEPTHS_M.

        Parameters
        ----------
        start_date : str
            ISO 8601 start date, e.g. "2023-06-01".
        end_date : str
            ISO 8601 end date, inclusive, e.g. "2023-08-31".
        bbox : tuple of float
            (lon_min, lat_min, lon_max, lat_max) in decimal degrees WGS84.

        Returns
        -------
        ds_std : xr.Dataset
            Dimensions: (time, depth, latitude, longitude)
            Variables: water_temp [deg C], salinity [psu], water_u [m/s], water_v [m/s]
            Depth coordinate: TARGET_DEPTHS_M [m], positive downward.

            Physical note: The mixed layer depth (MLD) is implicitly encoded here —
            near-isothermal profiles indicate deep mixing (MHW-suppressing), while
            sharp thermoclines indicate strong stratification (MHW-amplifying).
            The 1D-CNN uses this vertical structure as its primary feature.
        """
        lon_min, lat_min, lon_max, lat_max = bbox

        logger.info("Fetching HYCOM tile: %s to %s, bbox=%s", start_date, end_date, bbox)

        ds_raw = xr.open_dataset(
            self._url,
            engine="netcdf4",
            chunks={"time": 10, "depth": -1, "lat": 100, "lon": 100},
        ).sel(
            time=slice(start_date, end_date),
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        )[HYCOM_VARIABLES]

        ds_std = self._interpolate_to_standard_depths(ds_raw)
        logger.info("HYCOM tile loaded and interpolated: %s", ds_std)
        return ds_std

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
    parser.add_argument("--members", type=int, default=64)
    args = parser.parse_args()

    out = run_ingestion_pipeline(
        start_date=args.start,
        end_date=args.end,
        bbox=tuple(args.bbox),
        gcs_bucket=args.gcs_bucket,
        output_dir=args.output_dir,
        service_account_key=args.key,
        n_members=args.members,
    )
    print(f"Pipeline complete. Output: {out}")
