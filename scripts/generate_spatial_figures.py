"""
generate_spatial_figures.py — Generate spatial SST figures and XAI grid for README.

Reads WN2 + HYCOM zarr from GCS, picks best snapshot (max ensemble spread date),
plots with cartopy, stitches existing XAI PNGs into a 2×2 seasonal grid.

Requires:
  GOOGLE_APPLICATION_CREDENTIALS — path to GCP service account JSON
  MHW_GCS_BUCKET                 — GCS bucket URI, e.g. "gs://mhw-risk-cache"

Outputs (all to docs/assets/figures/):
  wn2_sst_mean_spread.png
  hycom_sst.png
  xai_attribution_grid.png
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import xarray as xr
import gcsfs
from PIL import Image

# ── constants ─────────────────────────────────────────────────────────────────
BUCKET      = os.environ.get("MHW_GCS_BUCKET", "gs://mhw-risk-cache").rstrip("/")
CRED        = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
WN2_YEARS   = [2023, 2022]          # try newest first
HYCOM_YEARS = [2023, 2022]
BBOX        = (-71.0, 41.0, -66.0, 45.0)   # lon_min, lat_min, lon_max, lat_max
OUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "docs", "assets", "figures")
XAI_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "results", "xai")


def _fs():
    """Return authenticated gcsfs filesystem."""
    if CRED:
        return gcsfs.GCSFileSystem(token=CRED)
    return gcsfs.GCSFileSystem()   # falls back to ADC


def _open_wn2(year: int) -> xr.Dataset | None:
    """Open WN2 annual zarr from GCS. Returns None if not found."""
    uri = f"{BUCKET}/weathernext2/cache/wn2_{year}-01-01_{year}-12-31_m64.zarr"
    try:
        fs = _fs()
        store = fs.get_mapper(uri.removeprefix("gs://"))
        ds = xr.open_zarr(store, consolidated=True)
        print(f"  WN2 {year}: opened {uri}")
        return ds
    except Exception as exc:
        print(f"  WN2 {year}: {exc}")
        return None


def _open_hycom(year: int) -> xr.Dataset | None:
    """Open HYCOM annual tile zarr from GCS. Returns None if not found."""
    uri = f"{BUCKET}/hycom/tiles/{year}/"
    try:
        fs = _fs()
        store = fs.get_mapper(uri.removeprefix("gs://"))
        ds = xr.open_zarr(store, consolidated=False)
        print(f"  HYCOM {year}: opened {uri}")
        return ds
    except Exception as exc:
        print(f"  HYCOM {year}: {exc}")
        return None


def _best_snapshot_wn2(ds: xr.Dataset) -> tuple[int, str]:
    """
    Find the time index with highest mean ensemble spread of sea_surface_temperature.

    Returns (time_index, date_string).
    Spatial mean of per-pixel ensemble std gives one spread value per day.
    """
    sst = ds["sea_surface_temperature"]   # (member, time, lat, lon)
    print("  Computing ensemble spread per day (loading ~small slice)...")
    # std over member dim → (time, lat, lon); then mean over lat/lon → (time,)
    spread_daily = sst.std(dim="member").mean(dim=["latitude", "longitude"]).compute()
    best_idx = int(spread_daily.argmax().values)
    best_date = str(ds.time.values[best_idx])[:10]
    print(f"  Best snapshot: idx={best_idx}, date={best_date}, "
          f"spread={float(spread_daily[best_idx]):.4f} K")
    return best_idx, best_date


def plot_wn2(ds: xr.Dataset, t_idx: int, date_str: str, out_path: str):
    """
    Two-panel figure: WN2 ensemble mean SST (left) + ensemble spread (right).

    Parameters
    ----------
    ds       : WN2 xr.Dataset with dim (member, time, latitude, longitude)
    t_idx    : Time index of the selected snapshot
    date_str : Human-readable date for subtitle
    out_path : Output PNG path
    """
    sst = ds["sea_surface_temperature"].isel(time=t_idx).compute()  # (member, lat, lon)

    # Convert K → °C for readability
    sst_c = sst - 273.15

    mean_sst  = sst_c.mean(dim="member")   # (lat, lon)
    spread    = sst_c.std(dim="member")    # (lat, lon)

    lat = ds.latitude.values
    lon = ds.longitude.values

    proj = ccrs.PlateCarree()
    lon_min, lat_min, lon_max, lat_max = BBOX

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    fig.suptitle(
        f"WeatherNext 2 — Sea Surface Temperature  ({date_str})\n"
        "Gulf of Maine  |  64-member FGN ensemble",
        fontsize=12, fontweight="bold",
    )

    # ── left: ensemble mean ──────────────────────────────────────────────────
    ax = axes[0]
    ax.set_extent([lon_min - 0.5, lon_max + 0.5, lat_min - 0.5, lat_max + 0.5], crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="#d4c5a9", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=3)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5,
                 xlocs=range(-72, -65, 1), ylocs=range(40, 47, 1))

    cmap_sst = plt.cm.RdYlBu_r.copy()
    cmap_sst.set_bad(color="#d4c5a9")   # NaN → land colour, eliminates white gaps
    vmin, vmax = float(mean_sst.min()), float(mean_sst.max())
    im0 = ax.pcolormesh(lon, lat, mean_sst.values, transform=proj,
                        cmap=cmap_sst, vmin=vmin, vmax=vmax, zorder=1)
    plt.colorbar(im0, ax=ax, orientation="horizontal", pad=0.04,
                 label="SST (°C)", shrink=0.85)
    ax.set_title("Ensemble Mean", fontsize=10)

    # ── right: ensemble spread ────────────────────────────────────────────────
    ax = axes[1]
    ax.set_extent([lon_min - 0.5, lon_max + 0.5, lat_min - 0.5, lat_max + 0.5], crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="#d4c5a9", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=3)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5,
                 xlocs=range(-72, -65, 1), ylocs=range(40, 47, 1))

    cmap_spread = plt.cm.YlOrRd.copy()
    cmap_spread.set_bad(color="#d4c5a9")
    smax = float(spread.max()) or 1.0
    im1 = ax.pcolormesh(lon, lat, spread.values, transform=proj,
                        cmap=cmap_spread, vmin=0, vmax=smax, zorder=1)
    plt.colorbar(im1, ax=ax, orientation="horizontal", pad=0.04,
                 label="Ensemble Spread — std (°C)", shrink=0.85)
    ax.set_title("Ensemble Spread (std across 64 members)", fontsize=10)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_hycom(ds: xr.Dataset, date_str: str, out_path: str):
    """
    Single-panel HYCOM surface SST for the given date (or nearest available date).

    Parameters
    ----------
    ds       : HYCOM xr.Dataset with dims (time, depth, lat, lon) or (time, lat, lon)
    date_str : Target date string YYYY-MM-DD
    out_path : Output PNG path
    """
    # Select surface (depth=0) and nearest time
    sst_var = "water_temp"
    da = ds[sst_var]

    # Handle depth dimension if present
    if "depth" in da.dims:
        da = da.sel(depth=0, method="nearest")
    elif "Depth" in da.dims:
        da = da.isel(Depth=0)

    # Select nearest time
    try:
        da = da.sel(time=date_str, method="nearest")
    except Exception:
        da = da.isel(time=0)

    da = da.compute()

    # Determine lat/lon coordinate names
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat = ds[lat_name].values
    lon = ds[lon_name].values

    proj = ccrs.PlateCarree()
    lon_min, lat_min, lon_max, lat_max = BBOX

    # Clip to bbox
    lat_mask = (lat >= lat_min - 0.5) & (lat <= lat_max + 0.5)
    lon_mask = (lon >= lon_min - 0.5) & (lon <= lon_max + 0.5)
    lat_cl = lat[lat_mask]
    lon_cl = lon[lon_mask]
    da_cl  = da.values[np.ix_(lat_mask, lon_mask)]

    fig, ax = plt.subplots(
        figsize=(7, 5),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    actual_date = str(da.time.values)[:10] if hasattr(da, "time") else date_str
    fig.suptitle(
        f"HYCOM — Sea Surface Temperature  ({actual_date})\n"
        "Gulf of Maine  |  0.08° resolution",
        fontsize=12, fontweight="bold",
    )

    ax.set_extent([lon_min - 0.5, lon_max + 0.5, lat_min - 0.5, lat_max + 0.5], crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="#d4c5a9", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=3)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5,
                 xlocs=range(-72, -65, 1), ylocs=range(40, 47, 1))

    data_clean = np.where(np.isfinite(da_cl), da_cl, np.nan)
    vmin = np.nanpercentile(data_clean, 2)
    vmax = np.nanpercentile(data_clean, 98)
    im = ax.pcolormesh(lon_cl, lat_cl, data_clean, transform=proj,
                       cmap="RdYlBu_r", vmin=vmin, vmax=vmax, zorder=1)
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04,
                 label="SST (°C)", shrink=0.85)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def stitch_xai_grid(xai_dir: str, out_path: str):
    """
    Stitch 4 seasonal XAI attribution PNGs into a 2×2 matplotlib figure.

    Parameters
    ----------
    xai_dir  : Directory containing ig_attribution_DJF/MAM/JJA/SON.png
    out_path : Output PNG path
    """
    seasons      = ["DJF", "MAM", "JJA", "SON"]
    season_labels = {"DJF": "Winter", "MAM": "Spring", "JJA": "Summer", "SON": "Fall"}
    files   = {s: os.path.join(xai_dir, f"ig_attribution_{s}.png") for s in seasons}

    missing = [s for s, p in files.items() if not os.path.exists(p)]
    if missing:
        print(f"  WARNING: missing XAI files for seasons: {missing}")

    imgs = {}
    for s, p in files.items():
        if os.path.exists(p):
            imgs[s] = Image.open(p)

    if not imgs:
        print("  No XAI images found — skipping xai_attribution_grid.png")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(
        "Integrated Gradients Attribution — Seasonal Comparison (ERA5 vs WeatherNext 2)",
        fontsize=13, fontweight="bold",
    )

    for ax, s in zip(axes.flat, seasons):
        if s in imgs:
            ax.imshow(np.array(imgs[s]))
            ax.set_title(season_labels[s], fontsize=12, fontweight="bold")
        else:
            ax.text(0.5, 0.5, f"Missing: {s}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
        ax.axis("off")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. WN2 spatial figure ─────────────────────────────────────────────────
    print("\n[1/3] WN2 spatial figure")
    wn2_ds = None
    for yr in WN2_YEARS:
        wn2_ds = _open_wn2(yr)
        if wn2_ds is not None:
            break
    if wn2_ds is None:
        print("  ERROR: could not open any WN2 zarr — skipping WN2 figure")
        best_date = "unknown"
    else:
        t_idx, best_date = _best_snapshot_wn2(wn2_ds)
        plot_wn2(wn2_ds, t_idx, best_date,
                 os.path.join(OUT_DIR, "wn2_sst_mean_spread.png"))

    # ── 2. HYCOM spatial figure ───────────────────────────────────────────────
    print("\n[2/3] HYCOM spatial figure")
    hycom_ds = None
    for yr in HYCOM_YEARS:
        hycom_ds = _open_hycom(yr)
        if hycom_ds is not None:
            break
    if hycom_ds is None:
        print("  ERROR: could not open any HYCOM zarr — skipping HYCOM figure")
    else:
        plot_hycom(hycom_ds, best_date,
                   os.path.join(OUT_DIR, "hycom_sst.png"))

    # ── 3. XAI attribution grid ───────────────────────────────────────────────
    print("\n[3/3] XAI attribution grid")
    stitch_xai_grid(XAI_DIR, os.path.join(OUT_DIR, "xai_attribution_grid.png"))

    print("\nDone. Figures in:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
