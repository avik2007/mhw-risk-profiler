# Plan: WN2 SST Fix + WN2 Training Run
# Date: 2026-04-20
# Status: APPROVED (session 17) — ready for execution

---

## Context

ERA5 training complete (50 epochs, session 16). ERA5 spread=0.00 accepted limitation.
WN2 training blocked by latent crash: `build_tensors()` calls `merged["sea_surface_temperature"]`
but WN2 GCS tiles have NO SST — was removed (session 7) due to land mask defaultValue=0 bug.

Decision (session 17): add SST back with proper NaN masking (replace 0 K → NaN).
WN2 has 64 genuine ensemble members → real SST spread → real MHW spread → real SVaR quantiles.

---

## Success Criteria

- [ ] `harvester.py`: `WN2_VARIABLES` contains `"sea_surface_temperature"`
- [ ] `harvester.py`: after GEE fetch, pixels where `sst == 0` replaced with `NaN` before GCS write
- [ ] All 68 tests pass (no regressions)
- [ ] WN2 2022 GCS tile has `sea_surface_temperature` variable (verify with `xr.open_zarr`)
- [ ] WN2 2023 GCS tile has `sea_surface_temperature` variable
- [ ] `train_wn2.py --epochs 50` completes without crash
- [ ] `spread > 0` in training log (real MHW spread from 64 WN2 members)
- [ ] Weights saved: `data/models/wn2_weights.pt`, `wn2_best_weights.pt`

---

## Step 1 — Code: Add SST to WN2_VARIABLES + land mask fix

**File:** `src/ingestion/harvester.py`

Sub-tasks:
1a. Find `WN2_VARIABLES` list. Add `"sea_surface_temperature"` to it.
1b. In `WeatherNext2Harvester.fetch_and_cache()` (or `_build_dataset()`), after fetching SST
    from GEE via `sampleRectangle`, replace pixels where value == 0 (land defaultValue) with NaN.
    Implementation note: the fetch returns a numpy array or xarray variable — use
    `np.where(sst_values == 0, np.nan, sst_values)` before assembling the Dataset.
1c. Run `pytest tests/ -v` — must pass all 68 tests.

**Deliverable:** code change + all tests green.

---

## Step 2 — Code: Commit + push

```bash
git add src/ingestion/harvester.py
git commit -m "fix: add SST back to WN2_VARIABLES with land mask (0→NaN) for WN2 training"
git push
```

---

## Step 3 — VM: Re-fetch WN2 2022 tile with SST

On mhw-data-prep (or mhw-wn2-prep if still alive):

```bash
# Pull latest code
cd ~/mhw-risk-profiler && git pull

# Delete stale 2022 WN2 zarr (no SST) + sentinel
gsutil -m rm -r gs://mhw-risk-cache/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr
gsutil rm gs://mhw-risk-cache/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr.complete

# Re-fetch (use nohup + disown — conda run dies on SSH detach)
nohup env \
  GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
  MHW_GCS_BUCKET=gs://mhw-risk-cache \
  /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/run_wn2_prep.py \
  --year 2022 \
  >> ~/nohup_wn2_2022.log 2>&1 </dev/null & disown $!
```

**ETA:** ~55 min/year (9 sec/day × 365 days).
**Verify:** `xr.open_zarr("gs://mhw-risk-cache/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr")` — must show `sea_surface_temperature`.

---

## Step 4 — VM: Re-fetch WN2 2023 tile with SST

Same as Step 3, but 2023:

```bash
gsutil -m rm -r gs://mhw-risk-cache/weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr
gsutil rm gs://mhw-risk-cache/weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr.complete

nohup env \
  GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
  MHW_GCS_BUCKET=gs://mhw-risk-cache \
  /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/run_wn2_prep.py \
  --year 2023 \
  >> ~/nohup_wn2_2023.log 2>&1 </dev/null & disown $!
```

Can run 2022 and 2023 in parallel on separate VMs to cut total time to ~55 min.

---

## Step 5 — VM: Run train_wn2.py --epochs 50

After both WN2 tiles have `.complete` sentinels:

```bash
nohup env \
  GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
  MHW_GCS_BUCKET=gs://mhw-risk-cache \
  /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/train_wn2.py --epochs 50 \
  >> ~/train_wn2.log 2>&1 </dev/null & disown $!
```

**Expected outputs:**
- `data/models/wn2_weights.pt`
- `data/models/wn2_best_weights.pt`
- `data/results/wn2_training_log.csv`
- `data/results/plots/wn2_*.png`
- `spread > 0` (64 genuine WN2 members = real MHW probability mass)

---

## Questions for Gemini Review

1. **Land mask strategy**: replacing SST==0 with NaN assumes `defaultValue=0` is the only land
   artifact. Is there a more robust masking approach (e.g., GEE `mask()` or `updateMask()` before
   `sampleRectangle`)? Should we use the HYCOM land mask as ground truth instead?

2. **SST units**: WN2 SST is in Kelvin (confirmed ~275–291 K over ocean in session 7 smoke test).
   HYCOM SST is in Celsius. `build_tensors()` must convert WN2 SST from K → °C before harmonizing.
   Does `harmonize()` handle this, or does the fix need a unit conversion step?

3. **Noise sigma for WN2 SST**: `NOISE_SIGMAS` has an SST entry (σ=0.5 K) retained from ERA5
   `expand_and_perturb()`. WN2 already has 64 real members — should SST noise be applied or skipped
   for WN2? (Gemini: flag if double-noising would corrupt the real ensemble spread.)

4. **2022 WN2 daily tile structure**: WN2 is written per-day to `{gcs_uri}_daily/d{YYYYMMDD}/`.
   Adding SST to `WN2_VARIABLES` means per-day caches on GCS also need to be deleted before re-fetch.
   Does `run_wn2_prep.py` handle the `_daily/` subdirectory, or must user delete manually?
