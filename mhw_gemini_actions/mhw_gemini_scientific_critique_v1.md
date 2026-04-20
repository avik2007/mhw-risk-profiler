# Scientific & Technical Critique: MHW Risk Profiler (v1)
**Date:** 2026-04-20
**From:** Gemini (Reviewer/Scientist)
**To:** Claude (Architect/Implementer)

## 🚩 CRITICAL: Scientific Integrity (MHW Climatology)
The current training and validation strategy uses a **2-year baseline (2022–2023)** to compute the 90th-percentile MHW threshold. This is a fundamental oceanographic flaw.

- **The Problem:** Per Hobday et al. (2016), a climatology requires a **30-year baseline** (e.g., 1982–2011). With only 2 years of data, the 90th percentile is effectively the **maximum** of those two years.
- **The Consequence:** Detection of MHWs in the same 2-year period becomes mathematically improbable. The Stress Degree Day (SDD) accumulation will be zero for almost all ensemble members, and the resulting SVaR quantiles will provide **zero financial risk insight**.
- **The Fix:** Re-compute the climatology using a longer period (at least 10-20 years if 30 is not possible) and apply an **11-day moving window** to smooth the daily threshold.

## ⚠️ Technical Flaws in WN2 SST Fix Plan
The 5-step plan in `docs/superpowers/plans/2026-04-20-wn2-sst-fix-and-training.md` has three "blocker" issues:

1. **Daily Cache Poisoning:** Deleting the annual Zarr store is insufficient. `WeatherNext2Harvester` checks for per-day caches in `{gcs_uri}_daily/d{YYYYMMDD}/`. If these are not deleted, the script will skip the SST fetch, and the new annual Zarr will still be missing SST.
   - **Action:** Add `gsutil -m rm -r gs://..._daily/` to the plan.

2. **Land Mask Inconsistency:** Masking only SST (`0 K -> NaN`) while leaving atmospheric variables (wind, pressure) active over land creates a physical mismatch. 
   - **Action:** Apply the same land mask to **all** atmospheric variables in `_build_dataset` to ensure the model sees a consistent Ocean-only domain.

3. **Unit Mismatch (K vs. °C):** WN2 SST is in Kelvin; HYCOM is in Celsius. 
   - **Action:** Ensure `DataHarmonizer.harmonize()` standardizes all SST inputs to Celsius BEFORE they reach the SDD accumulation logic.

## 📉 Training Strategy Risks
- **ERA5 Spread:** "Accepting" 0.00 spread for ERA5 is acceptable for debugging, but the **i.i.d. Gaussian noise (σ=0.5 K)** added to ERA5 inputs is a poor proxy for the structured, physically-coherent uncertainty of the WN2 FGN ensemble. 
- **The Risk:** The model may overfit to Gaussian noise and fail to generalize when exposed to real meteorological extremes in WN2.

---
**Recommendation:** DO NOT proceed with 50 epochs of training until the Climatology Baseline is addressed and the Cache Invalidation is added to the plan.
