# MHW Gemini Todo

## [2026-04-20] LinkedIn Post Drafts Ready for Review

Two post drafts written for the ERA5 MHW XAI results, saved at:
- `docs/linkedin/2026-04-20-mhw-era5-draft-a.md` — short (~300 words)
- `docs/linkedin/2026-04-20-mhw-era5-draft-b.md` — medium (~600 words)

Both have `[INSERT]` placeholders for actual B3 results (SVaR map, IG attribution chart, variable percentages).
Hook: "Can atmospheric data predict the financial impact of ocean heatwaves?"
Audience: ML + climate science. WN2 follow-up promised in both.

Gemini: please review scientific accuracy of the framing and suggest any improvements to the physical interpretation sections before B3 results are available.

---

## [2026-04-20] ACTION REQUIRED — Review WN2 SST Fix Plan

Claude has written a 5-step execution plan at:
`docs/superpowers/plans/2026-04-20-wn2-sst-fix-and-training.md`

**What the plan does:**
- Adds `sea_surface_temperature` back to `WN2_VARIABLES` in `harvester.py`
- Masks land pixels (SST==0 K → NaN) before GCS write
- Deletes stale WN2 2022/2023 GCS tiles (no SST), re-fetches with SST
- Runs `train_wn2.py --epochs 50` to get real ensemble spread + SVaR quantiles

**Please evaluate (4 open questions at bottom of plan file):**
1. Is `SST==0 → NaN` the right land mask strategy, or use GEE `updateMask()` / HYCOM land mask?
2. Does `harmonize()` already convert WN2 SST from Kelvin → Celsius, or is a unit conversion step needed in the plan?
3. Should `expand_and_perturb()` SST noise (σ=0.5 K) be skipped for WN2 (already 64 real members)?
4. Must the `_daily/` per-day GCS subdirectories be deleted manually, or does `run_wn2_prep.py` handle cache invalidation?

---

- [ ] Monitor regression loss (MSE) and SVaR stability for the 2022/2023 dual training runs on GCP.
- [x] Ensure all functions in `src/models/` and new `fetch_and_cache` methods have verbose header comments.
- [ ] Review harmonization logic in `src/ingestion/harvester.py` for CF-1.8 compliance.
- [ ] Monitor SVaR accuracy and ensemble spread during execution once WN2 access is granted.
- [ ] Monitor regression loss curves (Train vs. Val) and Mean Squared Error (MSE) to detect over- or under-fitting.
- [x] Incorporate XAI output review into the documentation workflow.
- [ ] Compare 90th-percentile MHW thresholds against fixed biological thresholds (e.g., 18°C for salmon).
