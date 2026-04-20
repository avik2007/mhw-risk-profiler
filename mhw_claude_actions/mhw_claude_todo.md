# mhw_claude_todo.md
# Active task queue — current and next actions
# ---------------------------------------------
# Completed sessions → mhw_claude_recentactions.md

---

## ACTIVE — Fix WN2 SST missing from tiles + OISST 30yr baseline

### Plan: `docs/superpowers/plans/2026-04-20-oisst-climatology-and-wn2-sst-fix.md`

Two parallel tracks. Track A code DONE. Track B code is next session's first task.

---

### Track A — OISST 30yr climatology (PARALLEL with Track B)

- [x] **A1 DONE** — `compute_climatology()` updated with `window=11` rolling (commit `019bdb3`)
- [x] **A2 DONE** — `fetch_oisst_climatology.py` written + 6 tests (commit `d6b0584`)
- [x] **A3 RUNNING** — OISST fetch launched locally (nohup_oisst.log), ETA 30-90 min
  ```bash
  gsutil -m rm -r gs://mhw-risk-cache/hycom/climatology/
  nohup env GOOGLE_APPLICATION_CREDENTIALS=... MHW_GCS_BUCKET=gs://mhw-risk-cache \
    /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/fetch_oisst_climatology.py \
    >> ~/nohup_oisst.log 2>&1 </dev/null & disown $!
  ```
  ETA: ~30-90 min. Verify THREDDS URL reachable first (curl check in plan).

---

### Track B — WN2 SST fix (PARALLEL with Track A)

- [x] **B1 DONE** — Add SST to `WN2_VARIABLES` + NaN mask in `_build_dataset()` (commit `bf10412`)
  - `sea_surface_temperature` re-added to WN2_VARIABLES; `arr[arr == 0.0] = np.nan` in inner loop
  - `test_wn2_variables_includes_sst` + `test_sst_zero_pixels_masked_to_nan` added; 78/78 pass

- [x] **B2 RUNNING** — WN2 2022+2023 re-fetch launched locally in parallel (nohup_wn2_2022.log, nohup_wn2_2023.log), ETA ~55 min each

- [ ] **B3 VM** — After A3 + B2 both complete:
  - Retrain ERA5 with new 30yr threshold: `train_era5.py --epochs 50`
  - Train WN2: `train_wn2.py --epochs 50`
  - Verify `spread > 0` and `SVaR_95 > SVaR_50 > SVaR_05`

---

### Gemini findings resolved this session:
- ✅ Climatology baseline (critical): → OISST 30yr (Track A)
- ✅ Cache poisoning: → `_daily/` dirs explicitly deleted in B2
- ✅ Land mask inconsistency: → xarray skipna handles naturally; no code change needed
- ✅ Unit mismatch (K→°C): → already handled in `_train_utils.py:117`; no change needed

### ERA5 training artifacts (v1, 2yr threshold — superseded after A3+B3):
| Metric | Epoch 1 | Epoch 50 |
|--------|---------|---------|
| train_loss | 6111 | 6043 |
| val_loss | 2149 | 2100 |
| SVaR_95 | 0.57 °C·day | 1.10 °C·day |
| spread | 0.00 | 0.00 |

**GCS artifacts (v1):** `gs://mhw-risk-cache/era5/training_results/`

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
