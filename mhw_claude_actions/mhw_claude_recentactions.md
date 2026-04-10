# mhw_claude_recentactions.md
# Completed actions log — what was done and when
# -----------------------------------------------

---

## [2026-04-10] ERA5/WN2 Dual Training Plan — Execution Started (Tasks 0 & 1 complete)

Executing plan at `docs/superpowers/plans/2026-04-10-era5-wn2-dual-training.md`
using `superpowers:subagent-driven-development` skill.

### Completed this session:

**Task 1 — matplotlib dependency** ✅
- Added `matplotlib>=3.8.0` to `requirements.txt` under Scientific utilities
- Installed in `mhw-risk` conda env; verified import (`3.10.8`)
- Commit: `88765f1`

**HYCOM URL fix** ✅
- Switched `GLBv0.08` → `GLBy0.08` in both `HYCOM_THREDDS_TS` and `HYCOM_THREDDS_UV` constants in `src/ingestion/harvester.py`
- GLBy0.08/expt_93.0 covers 2018-12-04 to 2024-09-04; needed for WN2 2022/2023 training periods
- Commit: `7012a5f`

**Task 0 — WN2 GEE asset scoping** ✅ (done_with_concerns addressed)
- Created `scripts/scope_wn2_asset.py`
- Ran against live GEE; output captured to `docs/superpowers/specs/wn2_asset_schema.txt`
- Key finding: WN2 is a **forecast run structure** (not daily time series), covering **2022-present only**
  - 4 init times/day (00Z, 06Z, 12Z, 18Z), 15-day horizon, 64 FGN members
  - Recommended harvesting: filter to 00Z init + forecast_hour=24 → one 24h-ahead per member per day
  - ERA5 TRAIN_PERIOD (2018/2019) is valid; WN2 must use 2022/2023
- Updated `docs/superpowers/specs/2026-04-10-era5-wn2-xai-comparison-design.md` with Phase 0 Findings
- Fixed `col.size().getInfo()` hang in scope script
- Commits: `d0bf97d`, `b84a199`

### Plan change (approved by user):
- ERA5: TRAIN_PERIOD=2018, VAL_PERIOD=2019 (unchanged)
- WN2: TRAIN_PERIOD=2022, VAL_PERIOD=2023 (new)
- `_train_utils.py` must export BOTH sets: `ERA5_TRAIN_PERIOD`, `ERA5_VAL_PERIOD`, `WN2_TRAIN_PERIOD`, `WN2_VAL_PERIOD`
- Each training script imports its own set

### Current HEAD: `b84a199`

---

## [2026-03-30] HYCOM EDA Notebook Created

1. Created `notebooks/hycom_eda.ipynb` — 10-section exploratory notebook using existing
   `data/processed/hycom_2019-08-01_2019-08-03.zarr` (no network required).
2. Sections: dataset structure, surface temperature map (with 18°C contour), all 4 variables
   side-by-side, depth profiles (what the CNN sees), Hovmöller diagram, SST time series with
   SDD shading, current vectors, all-profiles overlay, T-S diagram, xarray patterns.
3. Registered `mhw-risk` conda env as a Jupyter kernel (`ipykernel install --user --name mhw-risk`).
4. Launch: `conda run -n mhw-risk jupyter notebook notebooks/hycom_eda.ipynb`

---

## [2026-03-30] Analytics Plan Revised — WN2 Proxy Training Dropped

Plan at `docs/superpowers/plans/2026-03-30-hycom-proxy-training.md` revised after discussion.
Decisions made:
- WN2 proxy training (tasks 2–4) dropped — no fake WN2 data; wait for real GEE whitelist
- MHW threshold changed from constant 18°C → location-varying per grid cell
- Two tasks remain, deferred to next session:
  1. `src/analytics/payout.py` — parametric payout engine (pure math)
  2. `scripts/compute_hycom_climatology.py` — fetch 2yr HYCOM surface SST, compute
     90th-percentile threshold per (dayofyear, lat, lon), save to
     `data/processed/hycom_sst_threshold.zarr`

---

## [2026-03-30] HYCOM-Proxy Training Pipeline Plan Written (Tentative — superseded above)

---

## [2026-03-27] HYCOM Zarr Verification — Steps 3 & 4 DONE

1. Wrote `scripts/verify_hycom_zarr.py` — fetches HYCOM tile, writes local Zarr, verifies steps 3 & 4.
2. Ran the script; both steps passed:
   - **Step 3** (Vertical Coordinate Sanity Check): T/S profile at 43.5°N 70°W printed from
     `data/processed/hycom_2019-08-01_2019-08-03.zarr`. Thermocline confirmed:
     19.8°C (0m) → 17.2°C (5m) → 13.2°C (10m) → 10.5°C (20m) → 9.3°C (30m) → 8.8°C (50m) → 7.9°C (75m).
     NaN at 100–300 m expected (seafloor depth ~100 m in Gulf of Maine).
   - **Step 4** (Dask Scaling Test): `xr.open_zarr` returned (time=24, depth=11, lat=26, lon=13);
     all 4 variables confirmed as `dask.array` (lazy, not eager); no OOM.
     Disk size: 744 KB (well within the MB target).

---

## [2026-03-27] Ensemble Connectivity Smoke Test (Step 2) — Partial

### HYCOM Side: PASSED

1. Created `mhw-risk` conda environment (python=3.11); installed all requirements.
2. Added `google-cloud-storage>=2.14.0` and `gcsfs>=2024.2.0` to `requirements.txt` (were missing).
3. Fixed 5 bugs in `harvester.py`:
   - `ServiceAccountCredentials(email=None)` → extract email from JSON key file.
   - Corrected WeatherNext 2 GEE asset path: `59572747_3_0` → `weathernext_2_0_0`.
   - `_export_to_gcs` (GeoTIFF + `xr.open_zarr`) replaced with `_fetch_and_write_zarr` (sampleRectangle compute path + `gs://` URI Zarr write).
   - `HYCOM_THREDDS_BASE` split into `HYCOM_THREDDS_TS` (ts3z) + `HYCOM_THREDDS_UV` (uv3z) — T/S and currents are separate THREDDS datasets.
   - CLI arg `--members` renamed to `--n_members` to match docs.
4. Rewrote `HYCOMLoader.fetch_tile`:
   - Opens both ts3z and uv3z with `decode_times=False`.
   - Slices time by raw float index (avoids OPeNDAP hang from full-axis sort).
   - Converts bbox longitude -180..180 → 0..360 for HYCOM slicing; converts back after load.
   - Merges T/S and UV datasets before interpolation.
5. Wrote `scripts/smoke_test_gee.py` — 3-stage standalone connectivity test.
6. Verification evidence (HYCOM, 2019-08-01 to 2019-08-03, Gulf of Maine 1°×1°):
   - Dataset: (time=24, depth=11, lat=26, lon=13), all 4 variables loaded.
   - T/S profile at 43.5°N 70°W: 19.8°C at 0m → 7.9°C at 75m; NaN below (seafloor ~100m).
   - Thermocline confirmed visible (August Gulf of Maine summer stratification).

### WeatherNext 2 Side: BLOCKED

- GEE auth works (service account authenticated OK).
- Asset path corrected to `weathernext_2_0_0`.
- Access denied: the WeatherNext Data Request form must be submitted at developers.google.com/weathernext/guides/earth-engine to whitelist the service account.
- **User action required**: submit the form, then re-run `python scripts/smoke_test_gee.py`.

---

## [2026-03-27] Docker Engine Installed and Verified

1. Removed conflicting Ubuntu-repo Docker packages — none were present; system was clean.
2. Added Docker's official apt repo (Noble / amd64) to `/etc/apt/sources.list.d/docker.list`.
3. Installed: docker-ce 29.3.1, docker-ce-cli, containerd.io, docker-buildx-plugin, docker-compose-plugin v5.1.1.
4. Added avik2007 to docker group; enabled and started daemon via systemd (active/running).
5. Verification gate passed: hello-world OK, compose v5.1.1 OK, mhw-risk-profiler:latest image built OK (all 6 layers).

---

## [2026-03-24] Cloud Infrastructure Initialized

### Actions Completed

1. GCP project created; $300 free credits activated.

2. Earth Engine API enabled on the project. Registered account at code.earthengine.google.com/register
   as Contributor (Noncommercial, 1,000 EECU-hours). Billing account linked (required for Contributor
   tier, but EE usage itself does not charge).

3. Service account created with required IAM roles.
   (See `mondal-mhw-gcp-info.md` for account email and role list.)

4. GCS bucket created with Standard storage class, Hierarchical namespace enabled, and public
   access prevention enforced.
   (See `mondal-mhw-gcp-info.md` for bucket name and region.)

5. JSON key secured under `~/.config/gcp-keys/` with `chmod 600`.
   `GOOGLE_APPLICATION_CREDENTIALS` environment variable configured to point to this path.
   (See `mondal-mhw-gcp-info.md` for exact path.)

6. Smoke test passed: Auth OK, Bucket accessible, Contents empty (expected).

7. Monthly budget alert configured at 50%, 90%, and 100% thresholds.

---

## [2026-03-24] Day 1 (Session 3) — Maintenance: Cloud Calibration Task Setup

### Actions Completed

1. Replaced ACTIVE task in `mhw_claude_todo.md` from "Implement GEE Python API Harvester"
   to "GCP Environment Calibration & Ingestion Testing" — four sub-steps with explicit
   verification evidence requirements (IAM/Auth, Ensemble smoke test, HYCOM vertical profile,
   Dask lazy-open).

2. Archived the previous ACTIVE "Implement GEE Python API Harvester" entry to this log
   (see Session 2 entry below).

3. Updated `CLAUDE.md` with a dedicated GCP/Conda Environment Commands section covering
   authentication setup, Conda env activation, and OPeNDAP connectivity checks.

---

## [2026-03-24] Day 1 (Session 2) — Ingestion Engine Implementation

### Actions Completed

1. Corrected task priority: GEE/HYCOM ingestion engine set as ACTIVE before model work.

2. Implemented `src/ingestion/harvester.py` — production-ready, three-class ingestion engine:
   - `WeatherNext2Harvester`: GEE authentication (service account + ADC), queries
     `gcp-public-data-weathernext` FGN ensemble, exports to GCS Zarr with cache-hit logic.
   - `HYCOMLoader`: Fetches HYCOM GLBv0.08 via OPeNDAP/THREDDS; interpolates from native
     hybrid coordinate (Z-level / Sigma / Isopycnal) to TARGET_DEPTHS_M standard levels.
   - `DataHarmonizer`: Regrids both sources to 0.25-degree TARGET_LAT/LON grid; broadcasts
     deterministic HYCOM across 64 WeatherNext 2 ensemble members; writes CF-1.8 metadata.
   - `run_ingestion_pipeline()`: End-to-end orchestration with verification print gate.

3. All functions include verbose physical oceanography header comments per CLAUDE.md style rules.

4. Harvester saves harmonized output to `data/processed/harmonized_<start>_<end>.zarr`
   and prints Dataset repr as verification evidence.

5. Created `requirements.txt` with pinned minor-version dependencies.

6. Updated `mhw_claude_todo.md`: GEE Harvester set ACTIVE, 1D-CNN/Transformer moved to QUEUED.

---

## [2026-03-24] Day 1 — Project Infrastructure Setup

### Actions Completed

1. Acknowledged existing `mhw_ai_research/` folder containing Gemini, Perplexity, and
   NotebookLM deep-dives on MHW risk and marine habitat suitability.

2. Created core source directory structure:
   - `src/ingestion/`  — GEE API + Xarray/Dask harmonization layer
   - `src/models/`     — PyTorch 1D-CNN + Transformer architecture
   - `src/analytics/`  — MHW Stress Degree Day and Financial VaR logic
   - All directories initialized with `__init__.py` stubs.

3. Created `CLAUDE.md` in the project root with ArgoEBUS-inspired principles:
   Plan Mode, Self-Improvement Loop, Science-to-Engineering Boundary, Style, Verification Gate.

4. Created `mhw-repo-architecture.md` — annotated directory tree with pipe notation,
   separated from `CLAUDE.md` per user instruction.

5. Created `data/` directory with subdirectories `raw/`, `processed/`, `cache/`.
   Each initialized with `.gitkeep` to track structure without committing data.

6. Created `.gitignore` — explicitly excludes `data/` and `mhw_ai_research/`;
   notebooks retained (`.ipynb` not ignored) to preserve R&D visibility.

7. Created `Dockerfile` — `python:3.11-slim` base, system spatial libs (`libgdal-dev`,
   `libnetcdf-dev`), all core Python dependencies, WORKDIR `/app`, ingestion entrypoint.

8. Created `requirements.txt` — pinned minor versions for earthengine-api, xarray, dask,
   netCDF4, zarr, torch, captum, fastapi, uvicorn, numpy, pandas, scipy.

9. Created `README.md` — full SETS Framework framing (Ecological, Social/Financial,
   Technological), Science-to-Insight pipeline ASCII diagram, Quickstart, Data Sources.

10. Populated `mhw_claude_todo.md` with Day 1 priority task:
    "Drafting the GEE Python API harvester for WeatherNext 2 Zarr data and HYCOM NetCDF alignment."

---
