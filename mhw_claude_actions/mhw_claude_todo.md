# mhw_claude_todo.md
# Active task queue — current and next actions
# ---------------------------------------------
# Completed sessions → mhw_claude_recentactions.md
# Sessions 1–2 and all prior work fully documented there.

---

## ACTIVE — Data prep running on GCP VM

**VM:** `mhw-data-prep`, us-central1-a, e2-standard-4, on-demand (no preemption)
**Job:** `run_data_prep.py` — PID 1323, logging to `~/mhw-risk-profiler/data_prep.log`
**Monitor:** Cron job `d489cf43` checks GCS every hour at :07, posts status here
**Launch command (if restart needed):**
```bash
cd ~/mhw-risk-profiler && \
MHW_GCS_BUCKET=gs://mhw-risk-cache \
GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
nohup /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/run_data_prep.py \
>> data_prep.log 2>&1 </dev/null & disown $!
```

**5 paths to complete (checked via .zmetadata):**
- [ ] `hycom/tiles/2022`
- [ ] `hycom/tiles/2023`
- [ ] `hycom/climatology`
- [ ] `era5/2022`
- [ ] `era5/2023`

---

## NEXT — ERA5 real training run

**Trigger:** All 5 GCS paths above show COMPLETE
**Command (on VM):**
```bash
cd ~/mhw-risk-profiler && \
MHW_GCS_BUCKET=gs://mhw-risk-cache \
GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
nohup /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/train_era5.py --epochs 50 \
>> train_era5.log 2>&1 </dev/null & disown $!
```

**Expected outputs (real data, not synthetic):**
- `data/results/era5_real/loss_curve.png` — train vs. val MSE, 50 epochs
- `data/results/era5_real/mhw_threshold_map.png` — 90th-pct SST threshold on GoM grid
- `data/results/era5_real/svar_output.zarr` — SVaR_95/50/05 per grid cell
- `data/results/xai/ig_attribution_real_{DJF,MAM,JJA,SON}.png` — real IG attribution

---

## NEXT — LinkedIn post (ERA5 story)

**Trigger:** ERA5 real training complete + results reviewed with user
**Content plan:**
- Story: XAI attribution (Integrated Gradients) revealing which atmospheric/ocean variables
  drive marine heatwave financial risk — what the NN learned from real ERA5 + HYCOM data
- Visuals: real loss curves, IG attribution bar charts by season, SVaR map on GoM grid
- Promise: same analysis coming with WeatherNext 2 ensemble (64-member probabilistic)

**Pre-post checklist:**
- [ ] Review loss curves with user — confirm convergence, no overfit
- [ ] Review IG attribution plots — fact-check physical interpretation (SST dominance, seasonality)
- [ ] Review SVaR map — confirm spatial pattern makes physical sense
- [ ] Draft copy with user before publishing

**NO SYNTHETIC DATA IN THE POST. Real runs only.**

---

## PARALLEL (build now while data prep runs) — WN2 infrastructure

**Goal:** Have WN2 code ready to deploy immediately after the LinkedIn post decision point.

### Task W1 — `WeatherNext2Harvester.fetch_ensemble()`
File: `src/ingestion/harvester.py`
- Implement 00Z init filter (`start_time` ends in `T00:00:00Z`)
- Implement `forecast_hour=24` filter → one 24h-ahead forecast per member per day
- Output: 365 × 64 images/year matching ERA5 daily structure
- Reference: `docs/superpowers/specs/wn2_asset_schema.txt`
- Tests: cache hit, cache miss, filter correctness, exception propagation

### Task W2 — Extend `run_data_prep.py` with WN2 step
- Add step 6: fetch WN2 2022 → `weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr`
- Add step 7: fetch WN2 2023 → `weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr`
- Both guarded by `.zmetadata` idempotency check
- Tests: step skip on existing .zmetadata, fetch triggered on miss

### Deployment decision point (after LinkedIn post):
- If ERA5 training still running when W1+W2 are done → spin up second VM, run WN2 fetch in parallel
- If ERA5 training done and post is drafted → queue WN2 for next session

---

## QUEUED — WN2 real training + XAI comparison

**Prerequisite:** W1 + W2 complete, WN2 GCS tiles populated
**Steps:**
1. `train_wn2.py --epochs 50` with `MHW_GCS_BUCKET` set
2. `compare_xai.py` on both real ERA5 + WN2 results
3. Update LinkedIn post or write a follow-up

**Expected outputs:**
- `data/results/wn2_real/loss_curve.png`
- `data/results/xai/xai_comparison_real.json` — ERA5 vs WN2 IG, all 4 seasons
- `data/results/xai/ig_attribution_wn2_real_{DJF,MAM,JJA,SON}.png`

---

## QUEUED — Long-term

### Extended SST Climatology
Replace 2-year HYCOM baseline with ≥30-year record (Hobday 2016).
Options: longer HYCOM experiments (expt_91.x/92.x back to ~1994) or NOAA OISST v2.1.
Not blocking current work.

### XAI Option C — Member-Level Attribution Variance
Prerequisite: ERA5 vs WN2 XAI comparison complete.

### MTSFT — FFT-enriched Transformer
Prerequisite: baseline MHWRiskModel XAI validated.

### Vertex AI Migration
Prerequisite: GCP data prep pipeline complete and validated. Migrate training to managed
Vertex AI custom jobs for GPU access, Cloud Console logging, consistent GCS artifact storage.
