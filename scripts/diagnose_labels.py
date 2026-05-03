#!/usr/bin/env python3
"""
diagnose_labels.py — Spot-check SDD label magnitude before WN2 training.

Loads 2022 WN2 + HYCOM from GCS, calls build_tensors(), prints intermediate
stats to confirm whether ~250 degC*day labels are a bug or physically real.

Expected GoM values (Gulf of Maine, 2022):
  - MHW mask fraction: 0.05–0.30 (5–30% of days/cells in event)
  - SDD mean: 10–100 degC*day
  - SDD max: 50–300 degC*day (could be high for warm 2022)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import xarray as xr

from src.ingestion.harvester import DataHarmonizer
from src.analytics.mhw_detection import compute_mhw_mask
from src.analytics.sdd import accumulate_sdd

bucket = os.environ.get("MHW_GCS_BUCKET", "").rstrip("/")
if not bucket:
    raise RuntimeError("MHW_GCS_BUCKET not set")

print("Loading threshold from GCS...")
threshold = xr.open_zarr(f"{bucket}/hycom/climatology/")["sst_threshold_90"]
print(f"  threshold dims: {threshold.dims}")
print(f"  threshold range: {float(threshold.min()):.2f} – {float(threshold.max()):.2f} degC")
print(f"  threshold mean:  {float(threshold.mean()):.2f} degC")

print("\nLoading WN2 2022 + HYCOM 2022 from GCS...")
wn2   = xr.open_zarr(f"{bucket}/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr", chunks="auto")
hycom = xr.open_zarr(f"{bucket}/hycom/tiles/2022/", chunks="auto")

harmonizer = DataHarmonizer()
merged = harmonizer.harmonize(wn2, hycom)
print(f"  merged vars: {list(merged.data_vars)}")

# SST for label (HYCOM depth=0, degC)
sst_celsius = merged["water_temp"].isel(depth=0)
print(f"\nSST (HYCOM depth=0) stats:")
sst_vals = sst_celsius.isel(member=0).values
print(f"  mean={np.nanmean(sst_vals):.2f}  min={np.nanmin(sst_vals):.2f}  max={np.nanmax(sst_vals):.2f} degC")

# Regrid threshold to merged grid
rename_map = {}
if "lat" in threshold.dims:
    rename_map["lat"] = "latitude"
if "lon" in threshold.dims:
    rename_map["lon"] = "longitude"
threshold_regrid = threshold.rename(rename_map) if rename_map else threshold
threshold_regrid = threshold_regrid.interp(
    latitude=merged.latitude.values, longitude=merged.longitude.values
)
print(f"\nThreshold (regridded) stats:")
print(f"  mean={float(threshold_regrid.mean()):.2f}  min={float(threshold_regrid.min()):.2f}  max={float(threshold_regrid.max()):.2f} degC")

# Compute MHW mask
print("\nComputing MHW mask...")
mhw_mask = compute_mhw_mask(sst_celsius, threshold_regrid)
mask_frac = float(mhw_mask.values.mean())
print(f"  MHW mask fraction (frac days/cells in event): {mask_frac:.4f}")
print(f"  Expected: 0.05–0.30  {'OK' if 0.0 < mask_frac < 0.50 else 'SUSPICIOUS'}")

# Compute SDD
print("\nComputing SDD...")
sdd = accumulate_sdd(sst_celsius, threshold_regrid, mhw_mask)
sdd_spatial_mean = sdd.mean(dim=["latitude", "longitude"])  # (member,)
print(f"  SDD per-member spatial mean: {sdd_spatial_mean.values}")
label_mean = float(sdd_spatial_mean.mean())
label_max  = float(sdd_spatial_mean.max())
print(f"\n  label mean across members: {label_mean:.2f} degC*day")
print(f"  label max  across members: {label_max:.2f} degC*day")
print(f"  Expected range: 10–300 degC*day for warm GoM year")
print(f"  Loss at epoch 1 (pred~0): approx {label_mean**2:.0f}  (observed ERA5 loss: 63000)")

print("\nDone.")
