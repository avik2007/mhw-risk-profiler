# mhw-gcp-env-commands.md
# GCP / Conda Environment Commands — mhw-risk-profiler
# -------------------------------------------------------
# Canonical setup and verification steps for local dev and CI.
# Run in order when onboarding a new machine or debugging an auth failure.

---

## Conda Environment

```bash
# Create and activate the project environment (first time)
conda create -n mhw-risk python=3.11 -y
conda activate mhw-risk
pip install -r requirements.txt

# Activate on subsequent sessions
conda activate mhw-risk
```

---

## GEE Authentication

```bash
# Service account path — set before running harvester.py
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json"

# Verify authentication is functional (should print project/email, no error)
python -c "import ee; ee.Initialize(); print(ee.data.getAssetRoots())"

# Application Default Credentials (Cloud Run / GKE — no JSON needed)
gcloud auth application-default login
```

---

## GCS Bucket Permissions Check

```bash
# List top-level objects in the WeatherNext 2 cache bucket
gsutil ls gs://<bucket>/weathernext2/cache/

# Write permission smoke test (creates and deletes a temp object)
echo "ping" | gsutil cp - gs://<bucket>/.write_test && gsutil rm gs://<bucket>/.write_test
```

---

## HYCOM OPeNDAP Connectivity

```bash
# Verify THREDDS is reachable (expect HTTP 200)
curl -s -o /dev/null -w "%{http_code}" \
  "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/2020/t000.nc.das"
```

---

## Ingestion Smoke Test (minimal bbox, 2 ensemble members)

```bash
# Run harvester with a 1°×1° patch for 3 days — should complete in < 5 min
python src/ingestion/harvester.py \
  --start 2020-08-01 --end 2020-08-03 \
  --bbox '[-70.5, 43.0, -69.5, 44.0]' \
  --n_members 2 \
  --output data/processed/smoke_test.zarr
```

---

## Dask / Zarr Lazy-Open Check

```python
import xarray as xr
ds = xr.open_zarr("data/processed/smoke_test.zarr", chunks="auto")
print(ds)          # Should print Dataset repr without loading data into RAM
print(ds.chunks)   # Confirm Dask chunking is active
```

---

## HYCOM Vertical Profile Sanity Check

```python
import xarray as xr
ds = xr.open_zarr("data/processed/smoke_test.zarr")

# Print T/S profile at a single lat/lon point — thermocline should be visible
ts_profile = ds[["water_temp", "salinity"]].isel(lat=0, lon=0, time=0, member=0)
print(ts_profile.to_dataframe())
# Expected: temperature drops sharply between ~50–200 m (thermocline);
#           salinity shows halocline structure if region supports it.
```
