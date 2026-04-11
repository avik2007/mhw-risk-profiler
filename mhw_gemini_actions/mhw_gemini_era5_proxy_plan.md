# Plan: ERA5 Proxy Training for MHW Risk Profiler

**Status:** Proposed  
**Date:** 2026-04-03  
**Owner:** Gemini (Scientific Reviewer)

## 1. Goal
Unblock model development and "Science-to-Insight" pipeline validation by using deterministic ERA5 data from Google Earth Engine (GEE) as a proxy for WeatherNext 2 (WN2). This allows training the `MHWRiskModel` (CNN + Transformer) on real physical signals while waiting for the WN2 whitelist.

## 2. Scientific Rationale
- **Physical Consistency:** ERA5 and WN2 share the same atmospheric variables (T2m, Winds, MSLP, SST) and 0.25° resolution.
- **Structural Integrity:** By broadcasting the single ERA5 member to 64 members with synthetic noise, we preserve the `(batch, member, ...)` tensor contract required for SVaR estimation and the `ensemble_wrapper.py` architecture.
- **SETS Compliance:** This strategy maintains the "Science-to-Insight" flow: Physical Forcing (ERA5) → CNN/Transformer Encoding → SDD Accumulation → SVaR Risk Engine.

## 3. Architecture Changes

| Component | Change | Description |
| :--- | :--- | :--- |
| **Ingestion** | New `ERA5Harvester` | Fetches `ECMWF/ERA5/DAILY` from GEE. |
| **Harmonization** | `NoiseInjector` | Broadcasts 1 member → 64 members + Gaussian noise ($\sigma_{SST} \approx 0.5K$). |
| **Models** | `MHWRiskModel` | No change. Fully compatible with ERA5 proxy tensors. |
| **Analytics** | `svar.py` | No change. Computes SVaR on the noisy synthetic ensemble. |

## 4. Implementation Tasks

### Task 1: ERA5 Harvester & Integration
- [ ] Create `src/ingestion/era5_harvester.py` (cloned from `WeatherNext2Harvester` logic).
- [ ] Point to GEE Collection: `ECMWF/ERA5/DAILY`.
- [ ] Map ERA5 bands: `mean_2m_air_temperature`, `u_component_of_wind_10m`, `v_component_of_wind_10m`, `mean_sea_level_pressure`, `sea_surface_temperature`.

### Task 2: Synthetic Ensemble Logic
- [ ] Update `DataHarmonizer.harmonize()` to detect single-member inputs.
- [ ] Implement `expand_and_perturb()`:
    - Broadcast member 0 to 0..63.
    - Inject $\mathcal{N}(0, \sigma)$ noise to SST and T2m to simulate ensemble spread for SVaR testing.

### Task 3: Proxy Training Script
- [ ] Create `scripts/train_era5_proxy.py`.
- [ ] Objective: Minimize MSE between predicted SDD and physics-based SDD (from `sdd.py`).
- [ ] Use 2018–2019 Gulf of Maine data as the primary training set.

### Task 4: Validation Gate
- [ ] **Technical:** Run `smoke_test.py` using ERA5 data.
- [ ] **Scientific:** Verify that `svar.py` produces a non-zero spread (SVaR_95 > SVaR_50) from the noisy proxy ensemble.

## 5. Transition to WeatherNext 2
Once WN2 access is granted:
1. Update `config.yaml` (or CLI flags) to switch back to `WeatherNext2Harvester`.
2. Perform **Transfer Learning**: Initialize with ERA5 weights and fine-tune on WN2 FGN-ensemble members to capture true non-Gaussian tail behavior.

---
*Note: This plan adheres to the mandates in GEMINI.md regarding SVaR accuracy and ensemble spread monitoring.*
