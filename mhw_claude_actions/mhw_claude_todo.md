# mhw_claude_todo.md
# Active task queue — current and next actions
# ---------------------------------------------

---

## ACTIVE — RESUME HERE NEXT SESSION

### ERA5/WN2 Dual Training Plan — In Progress (Tasks 2–7 remaining)

**Skill to invoke first:** `superpowers:subagent-driven-development`
**Plan file:** `docs/superpowers/plans/2026-04-10-era5-wn2-dual-training.md`
**Spec file:** `docs/superpowers/specs/2026-04-10-era5-wn2-xai-comparison-design.md`
**Current HEAD:** `b84a199`

#### Status of all 8 tasks:
| # | Task | Status |
|---|------|--------|
| 0 | WN2 GEE asset scoping (`scripts/scope_wn2_asset.py`) | ✅ DONE |
| 1 | Add `matplotlib>=3.8.0` to `requirements.txt` | ✅ DONE |
| 2 | ERA5Harvester + `expand_and_perturb` + 3 unit tests | ⬜ NOT STARTED — start here |
| 3 | Shared `scripts/_train_utils.py` + 3 unit tests | ⬜ blocked by Task 2 |
| 4 | `scripts/train_era5.py` dry-run gate | ⬜ blocked by Task 3 |
| 5 | `scripts/train_wn2.py` dry-run gate | ⬜ blocked by Task 3 |
| 6 | `scripts/compare_xai.py` dry-run gate | ⬜ blocked by Tasks 4+5 |
| 7 | Full test suite + all 3 dry-runs in sequence | ⬜ blocked by all above |

#### Critical plan amendment (approved this session):
- **ERA5** uses `TRAIN_PERIOD = ("2018-01-01", "2018-12-31")`, `VAL_PERIOD = ("2019-01-01", "2019-12-31")`
- **WN2** uses `TRAIN_PERIOD = ("2022-01-01", "2022-12-31")`, `VAL_PERIOD = ("2023-01-01", "2023-12-31")`
  - WN2 only covers 2022-present (GEE forecast run structure, not daily reanalysis)
- **HYCOM URL** already fixed: `GLBv0.08` → `GLBy0.08` in `harvester.py` (commit `7012a5f`)
  - GLBy0.08/expt_93.0 covers 2018-12-04 to 2024-09-04 — covers both ERA5 and WN2 periods
- **`_train_utils.py`** must export FOUR period constants (not two):
  ```python
  ERA5_TRAIN_PERIOD = ("2018-01-01", "2018-12-31")
  ERA5_VAL_PERIOD   = ("2019-01-01", "2019-12-31")
  WN2_TRAIN_PERIOD  = ("2022-01-01", "2022-12-31")
  WN2_VAL_PERIOD    = ("2023-01-01", "2023-12-31")
  ```
  `train_era5.py` imports `ERA5_TRAIN_PERIOD, ERA5_VAL_PERIOD`.
  `train_wn2.py` imports `WN2_TRAIN_PERIOD, WN2_VAL_PERIOD`.

#### How to resume:
1. Read the plan file fully (it has verbatim code for every task — paste into subagents, don't make them read the file)
2. Use `superpowers:subagent-driven-development` — dispatch one implementer per task, then spec reviewer, then code quality reviewer
3. Task 2 is the first to implement. Its files: `src/ingestion/era5_harvester.py`, `tests/test_era5_harvester.py`, modify `src/ingestion/harvester.py` (add `NOISE_SIGMAS` + `expand_and_perturb`), modify `src/ingestion/__init__.py`
4. Apply the WN2 period amendment when implementing Task 3 (`_train_utils.py`)

---

## NEXT (after dual training plan complete)

### Pre-WeatherNext Analytics Completions
**Plan:** `docs/superpowers/plans/2026-03-30-hycom-proxy-training.md`
**Status:** Confirmed — deferred.

Two tasks, no WN2 needed, no proxy training:
1. `src/analytics/payout.py` — parametric insurance payout engine (pure math, no data)
2. `scripts/compute_hycom_climatology.py` — fetch 2 years of HYCOM surface SST (depth=0),
   run `compute_climatology()` → location-varying 90th-percentile threshold per (dayofyear, lat, lon),
   save to `data/processed/hycom_sst_threshold.zarr` (network required, surface-only = fast)

---

## PENDING (external blocker — no code work needed)

### WeatherNext 2 GEE Access — WN2 whitelisting confirmed, but harvesting strategy needs update
**Status:** GEE whitelist approved. However, WN2 is a forecast run structure (not daily time series).
See `docs/superpowers/specs/wn2_asset_schema.txt` for the full schema findings.

**When implementing `train_wn2.py` for real run**, `WeatherNext2Harvester.fetch_ensemble()` must filter:
- `start_time` ending in `T00:00:00Z` (00Z init only)
- `forecast_hour = 24` (24h-ahead forecast → one per member per day)
- This gives 365 × 64 images/year — matching ERA5's daily structure

---

## COMPLETED

- [2026-04-10] matplotlib>=3.8.0 added to requirements.txt and installed
- [2026-04-10] HYCOM URLs updated GLBv0.08 → GLBy0.08 (covers 2022-2024)
- [2026-04-10] WN2 asset scoped: forecast run structure, 2022-present, 64 FGN members
- [2026-03-24] GCP project created, $300 free credits activated
- [2026-03-24] Earth Engine API enabled; account registered as Contributor (Noncommercial)
- [2026-03-24] Service account created with required IAM roles (see mondal-mhw-gcp-info.md)
- [2026-03-24] GCS bucket created (see mondal-mhw-gcp-info.md for name/region/config)
- [2026-03-24] ADC configured via GOOGLE_APPLICATION_CREDENTIALS
- [2026-03-24] Smoke test passed: Auth OK, bucket accessible, contents empty
- [2026-03-27] Docker Engine 29.3.1 installed and verified
- [2026-03-27] mhw-risk conda env created; requirements.txt updated (gcsfs, google-cloud-storage)
- [2026-03-27] harvester.py: 5 bugs fixed (WN2 asset path, email=None, GeoTIFF→Zarr, HYCOM lon/time, CLI arg)
- [2026-03-27] HYCOM OPeNDAP verified: ts3z + uv3z fetched, interpolated to TARGET_DEPTHS_M, thermocline confirmed
- [2026-03-27] Step 3 DONE: T/S profile from data/processed/ — 19.8°C→7.9°C thermocline (0→75m), NaN below seafloor
- [2026-03-27] Step 4 DONE: Dask lazy-open confirmed (time=24, depth=11, lat=26, lon=13), 744 KB on disk, no OOM
- [2026-03-27] 1D-CNN + Transformer architecture implemented: cnn1d.py, transformer.py, ensemble_wrapper.py
- [2026-03-27] Smoke test passed (python -m src.models.ensemble_wrapper): 567,330 params
- [2026-03-30] HYCOM EDA notebook created (notebooks/hycom_eda.ipynb, 10 sections, offline)
- [2026-03-27] MHW Detection & SVaR Analytics implemented and tested (src/analytics/)

---

## QUEUED

### [LONG TERM] Extended SST Climatology — HYCOM Experiments + OISST
**Goal**: Replace the 2-year HYCOM expt_93.0 baseline with a longer historical record
suitable for a statistically robust 90th-percentile MHW threshold (Hobday 2016 recommends
≥30 years).

**Two avenues:**
1. Longer HYCOM runs — GLBv0.08 has expt_91.x, 92.x going back to ~1994. Check THREDDS catalog.
2. NOAA OISST v2.1 — daily 0.25-degree, 1981–present, standard MHW literature baseline.

**When to tackle**: Before production deployment. Not blocking current dev work.

---

### [LOW PRIORITY] XAI Option C — Member-Level Attribution Variance (ERA5 vs WN2)
**Prerequisite**: Phase 3 XAI comparison (compare_xai.py) complete.
**Output**: `data/results/xai_member_variance.json`

---

### [LOW PRIORITY] MTSFT: FFT-enriched Transformer for Periodic SST Features
**Prerequisite**: Baseline MHWRiskModel XAI validated on standard architecture first.
