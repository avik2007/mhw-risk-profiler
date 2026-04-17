# mhw_claude_todo.md
# Active task queue — current and next actions
# ---------------------------------------------
# Completed sessions → mhw_claude_recentactions.md

---

## ACTIVE — Data prep running on two GCP VMs

### VM 1: `mhw-data-prep` — HYCOM + ERA5
**Job:** `run_data_prep.py` — PID 4883, logging to `~/nohup_data_prep.log`
**Steps 1-5:** HYCOM 2022, HYCOM 2023, climatology, ERA5 2022, ERA5 2023
**Monitor:** Cron job `d489cf43` checks GCS every hour at :07
**Launch command (if restart needed):**
```bash
nohup env \
  GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
  MHW_GCS_BUCKET=gs://mhw-risk-cache \
  /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/run_data_prep.py \
  >> ~/nohup_data_prep.log 2>&1 </dev/null & disown $!
```

**5 paths to complete (checked via .zmetadata):**
- [ ] `gs://mhw-risk-cache/hycom/tiles/2022/.zmetadata`
- [ ] `gs://mhw-risk-cache/hycom/tiles/2023/.zmetadata`
- [ ] `gs://mhw-risk-cache/hycom/climatology/.zmetadata`
- [ ] `gs://mhw-risk-cache/era5/2022/.zmetadata`
- [ ] `gs://mhw-risk-cache/era5/2023/.zmetadata`

### VM 2: `mhw-wn2-prep` — WeatherNext 2
**Job:** `run_wn2_prep.py` — PID 964, logging to `~/nohup_wn2_prep.log`
**Steps 1-2:** WN2 2022, WN2 2023 (~55 min/year via GEE sampleRectangle)
**Launch command (if restart needed):**
```bash
nohup env \
  GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
  MHW_GCS_BUCKET=gs://mhw-risk-cache \
  /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/run_wn2_prep.py \
  >> ~/nohup_wn2_prep.log 2>&1 </dev/null & disown $!
```

**2 paths to complete:**
- [ ] `gs://mhw-risk-cache/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr/.zmetadata`
- [ ] `gs://mhw-risk-cache/weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr/.zmetadata`

---

## NEXT — ERA5 real training run

**Trigger:** All 5 HYCOM/ERA5 GCS paths show COMPLETE
**Command (on mhw-data-prep or fresh VM):**
```bash
nohup env \
  GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
  MHW_GCS_BUCKET=gs://mhw-risk-cache \
  /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/train_era5.py --epochs 50 \
  >> ~/train_era5.log 2>&1 </dev/null & disown $!
```

**Expected outputs (real data, not synthetic):**
- `data/models/era5_weights.pt` — final epoch weights
- `data/models/era5_best_weights.pt` — best val loss weights
- `data/results/era5_training_log.csv` — per-epoch metrics
- `data/results/plots/era5_*.png` — loss curve, IG attribution, SVaR map

**Note:** ERA5 inputs are real reanalysis; 64 ensemble members are synthetic
(Gaussian noise via expand_and_perturb). This is acceptable for training.
Results are real and may be published.

---

## NEXT — LinkedIn post (ERA5 story)

**Trigger:** ERA5 real training complete + results reviewed with user
**Content plan:**
- XAI attribution (Integrated Gradients) revealing which variables drive MHW financial risk
- Visuals: real loss curves, IG attribution bar charts by season, SVaR map on GoM grid
- Promise: same analysis coming with WeatherNext 2 ensemble (64-member probabilistic)

**Pre-post checklist:**
- [ ] Review loss curves — confirm convergence, no overfit
- [ ] Review IG attribution plots — fact-check physical interpretation
- [ ] Review SVaR map — confirm spatial pattern makes physical sense
- [ ] Draft copy with user before publishing

**NO SYNTHETIC DATA IN THE POST. Real runs only.**

---

## QUEUED — WN2 real training + XAI comparison

**Prerequisite:** WN2 GCS tiles complete + ERA5 training reviewed
**Steps:**
1. `train_wn2.py --epochs 50` with `MHW_GCS_BUCKET` set
2. `compare_xai.py` on both real ERA5 + WN2 results
3. LinkedIn follow-up or update

**Expected outputs:**
- `data/results/plots/wn2_*.png`
- `data/results/xai/xai_comparison_real.json` — ERA5 vs WN2 IG, all 4 seasons

---

## QUEUED — Long-term

### Extended SST Climatology
Replace 2-year HYCOM baseline with ≥30-year record (Hobday 2016).
Options: longer HYCOM experiments (expt_91.x/92.x back to ~1994) or NOAA OISST v2.1.

### XAI Option C — Member-Level Attribution Variance
Prerequisite: ERA5 vs WN2 XAI comparison complete.

### MTSFT — FFT-enriched Transformer
Prerequisite: baseline MHWRiskModel XAI validated.

### Vertex AI Migration
Prerequisite: GCP data prep pipeline complete and validated.
