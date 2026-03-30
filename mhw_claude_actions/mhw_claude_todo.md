# mhw_claude_todo.md
# Active task queue — current and next actions
# ---------------------------------------------

---

## ACTIVE

*None.*

---

## NEXT (approved plan — ready to execute when confirmed)

### Pre-WeatherNext Analytics Completions
**Plan:** `docs/superpowers/plans/2026-03-30-hycom-proxy-training.md`
**Status:** Confirmed — deferred to next session.

Two tasks, no WN2 needed, no proxy training:
1. `src/analytics/payout.py` — parametric insurance payout engine (pure math, no data)
2. `scripts/compute_hycom_climatology.py` — fetch 2 years of HYCOM surface SST (depth=0),
   run `compute_climatology()` → location-varying 90th-percentile threshold per (dayofyear, lat, lon),
   save to `data/processed/hycom_sst_threshold.zarr` (network required, surface-only = fast)

---

## PENDING (external blocker — no code work needed)

### WeatherNext 2 GEE Access — Complete Step 2 of the previous calibration task
**Blocker**: WeatherNext Data Request form submitted at developers.google.com/weathernext/guides/earth-engine.
Waiting for Google to whitelist `mhw-harvester@mhw-risk-profiler.iam.gserviceaccount.com`.

**When approved**, run:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
conda run -n mhw-risk python scripts/smoke_test_gee.py
```
Expected: Stage 1 and Stage 3 pass immediately; Stage 2 (WeatherNext → GCS Zarr) passes once whitelisted.
Evidence required: Zarr written to `gs://mhw-risk-cache/weathernext2/cache/wn2_*.zarr`; Dataset repr printed.

---

## COMPLETED

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
  - CNN1dEncoder: 3 Conv1d layers (4→32→64→128), residual skip, AdaptiveAvgPool1d — vertical translational invariance
  - TransformerEncoder: 4-layer pre-norm, 8 heads, d_model=128, sinusoidal pos-enc, time=90, features=5
  - LeakyGate: α=0.1, gate ∈ [0.1,0.9], both streams always contribute, gate value exposed for regime monitoring
  - MHWRiskModel: output (sdd, latent, gate) — Softplus head, Captum IG hook on latent
  - Design spec: docs/superpowers/specs/2026-03-27-cnn-transformer-mhw-design.md
- [2026-03-27] Smoke test passed (python -m src.models.ensemble_wrapper): 567,330 params
  - Test 1: shapes (2,4), (2,4,128), (2,4); SDD range [0.597, 0.624]; gate range [0.494, 0.511]
  - Test 2: member 0 SDD 0.6620 > others mean 0.5971 (delta 0.065); no cross-member leakage
  - Test 3: Captum IG shapes match inputs; HYCOM attr L2=0.0045, WN2 attr L2=0.0003
- [2026-03-30] HYCOM EDA notebook created (`notebooks/hycom_eda.ipynb`, 10 sections, offline)
- [2026-03-27] MHW Detection & SVaR Analytics implemented and tested (src/analytics/)
  - mhw_detection.py: compute_climatology(), compute_mhw_mask() — Hobday 2016 Category I
    - Consecutive-day filter: forward run-length + backward propagation pass
    - Modernised to grouped.quantile() for xarray 2024.x compatibility
  - sdd.py: accumulate_sdd() — thermal load above threshold, MHW-mask gated, xarray-native
  - svar.py: compute_svar(), compute_ensemble_stats() — ensemble quantile VaR, population std
  - Full test suite: 26 tests (test_mhw_detection.py × 9, test_sdd.py × 6, test_svar.py × 11)
  - Integration smoke test PASSED: HYCOM proxy, 18°C threshold, 338 locations, SVaR_95 mean 40.5 degC.day
  - End-to-end with real WeatherNext 2 blocked until Google whitelist (see PENDING)

---

## QUEUED

### [LONG TERM] Extended SST Climatology — HYCOM Experiments + OISST
**Goal**: Replace the 2-year HYCOM expt_93.0 baseline with a longer historical record
suitable for a statistically robust 90th-percentile MHW threshold (Hobday 2016 recommends
≥30 years). Relevant to this project and at least one other.

**Two avenues to investigate:**
1. **Longer HYCOM runs** — HYCOM GLBv0.08 has multiple experiments covering earlier periods
   (expt_91.0, expt_91.1, expt_91.2, expt_92.8, expt_92.9). Check THREDDS catalog at
   `https://tds.hycom.org/thredds/catalog.html` for coverage dates and whether ts3z/uv3z
   are available for each. GLBv0.08 potentially covers back to 1994.
2. **NOAA OISST** (Optimum Interpolation SST, v2.1) — daily, 0.25-degree global,
   1981–present. NetCDF via OPeNDAP or bulk download from NCEI. No depth structure
   (surface only), but 40+ years is ideal for climatology. Already used as the standard
   baseline in most published MHW literature. Access: `https://www.ncei.noaa.gov/products/optimum-interpolation-sst`

**When to tackle**: Before production deployment of the MHW threshold. Not blocking
current development work (2-year proxy threshold is sufficient for pipeline validation).

---

### [LOW PRIORITY] MTSFT: FFT-enriched Transformer for Periodic SST Features
**Goal**: Upgrade `TransformerEncoder` to Multi-Temporal Scale Fusion Transformer (MTSFT)
architecture per `NotebookLM-MHWRiskprofiler-deepdive.txt`. Enrich Transformer input with
FFT-derived spectral features (periodic components of SST signal) concatenated to the 5
raw WN2 variables before the attention stack.

**Prerequisite**: Baseline MHWRiskModel verified and Captum IG interpretability validated
on the standard architecture first.

**Caution**: Must re-verify Captum IG attribution remains interpretable over mixed
raw+spectral feature space after FFT enrichment.

