# mhw_claude_todo.md
# Active task queue — current and next actions
# ---------------------------------------------

---

## ACTIVE

*None.*

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

---

## QUEUED

### [LOW PRIORITY] MTSFT: FFT-enriched Transformer for Periodic SST Features
**Goal**: Upgrade `TransformerEncoder` to Multi-Temporal Scale Fusion Transformer (MTSFT)
architecture per `NotebookLM-MHWRiskprofiler-deepdive.txt`. Enrich Transformer input with
FFT-derived spectral features (periodic components of SST signal) concatenated to the 5
raw WN2 variables before the attention stack.

**Prerequisite**: Baseline MHWRiskModel verified and Captum IG interpretability validated
on the standard architecture first.

**Caution**: Must re-verify Captum IG attribution remains interpretable over mixed
raw+spectral feature space after FFT enrichment.

---

### Implement MHW Detection & SVaR Analytics in src/analytics/
**Goal**: Compute Stress Degree Days (SDD) from harmonized output and estimate
Stochastic Value-at-Risk (SVaR) from the 64-member ensemble distribution.

Sub-steps:
1. `mhw_detection.py` — detect MHW events using Hobday et al. (2016) Category I threshold.
2. `sdd.py` — accumulate Stress Degree Days above the MHW threshold.
3. `svar.py` — estimate SVaR from the empirical quantiles of the 64-member SDD distribution.
4. Smoke test with synthetic data; verification via printed output.

---
