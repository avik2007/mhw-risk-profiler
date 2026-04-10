#!/usr/bin/env python3
"""
scope_wn2_asset.py — Inspect WeatherNext 2 GEE asset schema.

Run once to understand the time axis structure before implementing train_wn2.py.
Requires GEE authentication: export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json

Usage:
    conda run -n mhw-risk python scripts/scope_wn2_asset.py
"""
import os
import ee

KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if KEY:
    import json
    with open(KEY) as fh:
        _kd = json.load(fh)
    creds = ee.ServiceAccountCredentials(email=_kd["client_email"], key_file=KEY)
    ee.Initialize(credentials=creds)
else:
    ee.Initialize()

ASSET = "projects/gcp-public-data-weathernext/assets/weathernext_2_0_0"
col = ee.ImageCollection(ASSET)

print("=== WeatherNext 2 Asset Schema ===\n")
print(f"Total images: {col.size().getInfo()}")

# Inspect first image
first = ee.Image(col.first())
print(f"\nFirst image date:       {first.date().format('YYYY-MM-dd').getInfo()}")
print(f"First image band count: {first.bandNames().size().getInfo()}")
print(f"First 20 band names:    {first.bandNames().slice(0, 20).getInfo()}")
print(f"First image properties: {first.propertyNames().getInfo()}")

# Check if 'system:time_start' is present (daily time series) or init+lead structure
props = first.toDictionary(first.propertyNames()).getInfo()
for k in ["system:time_start", "initialization_time", "lead_time", "forecast_hour",
          "number_of_members", "forecast_reference_time"]:
    print(f"  {k}: {props.get(k, '<not present>')}")

# Inspect last image to determine coverage end date
last = ee.Image(col.sort("system:time_start", False).first())
print(f"\nLast image date:  {last.date().format('YYYY-MM-dd').getInfo()}")

# Sample 5 consecutive images to check time spacing
print("\nFirst 5 image dates (time axis structure):")
imgs = col.sort("system:time_start").toList(5)
for i in range(5):
    img = ee.Image(imgs.get(i))
    print(f"  [{i}] {img.date().format('YYYY-MM-dd HH:mm').getInfo()}")

print("\n=== Interpretation guide ===")
print("If dates are daily (YYYY-MM-DD 00:00): daily time series -> treat like ERA5.")
print("If dates are 6-hourly or include HH:MM: sub-daily -> must resample to daily.")
print("If 'initialization_time' exists: forecast run structure -> need lead-time handling.")
