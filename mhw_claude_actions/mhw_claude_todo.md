# mhw_claude_todo.md
# Active task queue — current and next actions
# ---------------------------------------------
# Completed sessions → mhw_claude_recentactions.md

---

## ACTIVE — ERA5 training in progress (~00:45 UTC 2026-04-20)

### Status at session 16 (~00:45 UTC 2026-04-20)

**All GCS sentinels:** COMPLETE (5/5)
**Training:** RUNNING on `mhw-data-prep` (n1-highmem-8, 52 GB), PID 4547
**3 OOM bugs fixed this session (commits 87adafb, 2eb89c9, 814ee9d):**
1. `harmonize()` interpolated ERA5 to global 721×1440 grid → 485 GB → fixed: clip TARGET_LAT/LON to input bbox
2. `threshold` had `lat`/`lon` dims vs `latitude`/`longitude` in merged → 15 GB outer-product → fixed: rename+interp in `build_tensors()`
3. Interp condition used mismatched-size DataArrays → replaced with unconditional `.values` interp
**Memory peak observed:** ~3.4 GB RSS (vs 52 GB before fixes)

### VM status as of ~00:45 UTC 2026-04-20

| VM | Task | Status |
|----|------|--------|
| `mhw-data-prep` (n1-highmem-8) | ERA5 training, 50 epochs | **RUNNING** PID 4547 |
| All other VMs | — | COMPLETED/STOPPED |

**Manual restart (if training dies):**
```bash
gcloud compute ssh mhw-data-prep --zone=us-central1-a -- "
cd ~/mhw-risk-profiler && git pull origin main && \
> ~/nohup_train_era5.log && \
nohup env \
  GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
  MHW_GCS_BUCKET=gs://mhw-risk-cache \
  /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/train_era5.py --epochs 50 \
  >> ~/nohup_train_era5.log 2>&1 </dev/null & disown \$!"
```

**Check log:**
```bash
gcloud compute ssh mhw-data-prep --zone=us-central1-a -- "tail -30 ~/nohup_train_era5.log"
```

---

## BLOCKED — Pre-training gates (resolve before training run)

### [P1] ~~Verify sea_surface_temperature in ECMWF/ERA5/HOURLY~~ — RESOLVED (2026-04-18)
### [P2] ~~Spot-check HYCOM monthly tile GCS paths~~ — RESOLVED (2026-04-18)
### [P3-INFRA] ~~Fix 2023 monthly `.complete` sentinels~~ — RESOLVED (2026-04-19, session 14)
### [P-OOM] ~~Fix 3 OOM bugs in harmonize()+build_tensors()~~ — RESOLVED (2026-04-20, session 16)

---

## NEXT — ERA5 real training run (start manually next session)

**Trigger:** Verify all 5 HYCOM/ERA5 GCS `.complete` sentinels, then launch.
**Checklist before launch:**
- [ ] `gs://mhw-risk-cache/hycom/tiles/2022/.complete`
- [ ] `gs://mhw-risk-cache/hycom/tiles/2023/.complete`
- [ ] `gs://mhw-risk-cache/hycom/climatology/.complete`
- [x] `gs://mhw-risk-cache/era5/2022/.complete`
- [x] `gs://mhw-risk-cache/era5/2023/.complete`

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

**Caveat (Gemini Session 9):** SDD labels computed from 2-year (2022-2023) climatology baseline.
Hobday (2016) requires ≥30 years. Threshold is inflated vs true baseline — SVaR estimates
will be conservative. Must include this disclaimer in config JSON, training log, and any
LinkedIn post. Does not block training but blocks publication claims.

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

## QUEUED — After ERA5 training validated

### [P3] Build `payout.py` — parametric payout engine
Convert model SVaR output → loss exceedance curve → insurance payout trigger.
- Input: SDD per grid cell, configurable fixed biological threshold (18°C salmon, ~15°C kelp)
- Output: payout amount per member per threshold breach
- Pattern: piecewise linear (attachment point → cap → linear proportional payout)
- Source: UNEP Finance Initiative, Willis/Rare, AXA Climate, Bozec et al. (2025)
Prerequisite: ERA5 training complete + SVaR output validated.

### [P4] CF-1.8 compliance audit of `harmonize()` output
Spot-check merged dataset: coordinate names, units, `cell_methods` on aggregated vars,
`Conventions` attr, depth `positive: down`. Low risk but required for publication-grade data.

---

## QUEUED — WN2 real training + XAI comparison

**Prerequisite:** WN2 GCS tiles complete (DONE) + ERA5 training reviewed
**Steps:**
1. `train_wn2.py --epochs 50` with `MHW_GCS_BUCKET` set
2. `compare_xai.py` on both real ERA5 + WN2 results
3. LinkedIn follow-up or update

**Expected outputs:**
- `data/results/plots/wn2_*.png`
- `data/results/xai/xai_comparison_real.json` — ERA5 vs WN2 IG, all 4 seasons

---

## QUEUED — Long-term

### [P5] Extended SST Climatology — NOAA OISST v2.1 (1981–present)
Replace 2-year HYCOM baseline with ≥30-year record (Hobday 2016 requirement).
Preferred path: NOAA OISST v2.1 — daily, 0.25-degree, matches our grid exactly, freely available.
Alternative: HYCOM expt_91.x/92.x back to ~1994 (~30 years).
Recompute climatology GCS tile, retrain both ERA5 and WN2 runs.
**This is the primary scientific validity fix — resolves Gemini Critical Risk flag.**
Prerequisite: ERA5 + WN2 baseline training reviewed.

### [P6] XAI Option C — Member-Level Attribution Variance
Per-member IG across 64 WN2 members to quantify attribution uncertainty.
Prerequisite: ERA5 vs WN2 XAI comparison complete.

### [P7] MTSFT — FFT-enriched Transformer
Replace `TransformerEncoder` with frequency-domain-augmented version for better seasonal cycle capture.
Prerequisite: baseline MHWRiskModel XAI validated.

### [P8] Vertex AI Migration
Move training off spot VMs onto managed infrastructure with preemption handling.
Prerequisite: full pipeline validated end-to-end.

### [P9] Fix OOM in load_real_data() — lazy/chunked data loading
`harmonize()` + `build_tensors()` currently materializes ~13GB on a 16GB VM (OOM).
Workaround: upgraded to `n1-highmem-8` (52GB). Proper fix: refactor `load_real_data()`
in `train_era5.py` (and `train_wn2.py`) to load lazily, process in spatial/temporal chunks,
and avoid full materialization before tensor conversion.
Prerequisite: baseline training validated end-to-end first.
