# mhw_claude_todo.md
# Active task queue — current and next actions
# ---------------------------------------------
# Completed sessions → mhw_claude_recentactions.md

---

## DONE (sessions 24–26)

- SDD label fix (commit 79853d7), ERA5 retrain done (train=63k, val=38k, SVaR_95=1.34, spread=0)
- WN2 2023 annual zarr consolidated from daily cache (EXIT_CODE:0, 2026-04-21 22:08)
- SDD label magnitude diagnosed: 252 °C·day is REAL (2022 GoM record-warm year vs 30yr OISST baseline; 50% MHW mask fraction is physically valid)
- WN2 dim order bug found + fixed in `harmonize()`: WN2 zarr stores (time, member, lat, lon); added transpose to (member, time, lat, lon) before merge
- WN2 50-epoch training complete (2026-04-22 00:06): train=63308, val=38201, SVaR_95=1.09, spread=0.00, gate=0.769 — results pulled locally
- diagnose_labels.log confirms: 252.77 °C·day labels, 50.8% MHW mask, all 64 member labels IDENTICAL (HYCOM deterministic — spread=0 by construction)
- Root cause of flat loss confirmed: grad_clip=1.0 clips MSE gradient ~500 every step → near-zero learning

---

## ACTIVE — Fix grad_clip + label normalization, then retrain

**Root cause:** grad_clip=1.0 with MSE gradient ≈ 2×(pred−label) ≈ −500 → clipped every epoch → near-zero learning. Both ERA5 and WN2 stuck. WN2 spread=0 because HYCOM labels identical across members + gate depth-dominant (0.77).

**Why label norm fixes it:**
- Model output at init ≈ 0.5–2.0 (Softplus of near-zero linear weights)
- Without norm: loss=(1−250)²=62001, gradient=−498 → clipped every step → near-zero learning
- With norm: label=250/250=1.0, loss=(0.008−1)²≈0.98, gradient≈−2 → grad_clip rarely triggers → full LR updates
- Denorm at inference (×250) restores physical units for SVaR output

**Implementation — 3 files to touch:**

**1. `scripts/_train_utils.py`** — add constant + normalize in `build_tensors()` + denorm in `run_svar_inference()`:
```python
LABEL_NORM = 250.0  # deg C * day — approx label scale for GoM 2022-2023

# in build_tensors(), after label_arr is computed:
label_arr = label_arr / LABEL_NORM

# in run_svar_inference(), after model forward pass:
sdd = sdd * LABEL_NORM  # restore physical units before quantile
```

**2. `scripts/train_era5.py`** — change grad_clip:
```python
grad_clip_max_norm=10.0  # was 1.0
```

**3. `scripts/train_wn2.py`** — same grad_clip change:
```python
grad_clip_max_norm=10.0  # was 1.0
```

**Steps:**
- [ ] Implement above 3 changes locally
- [ ] **COMMIT** `src/ingestion/harvester.py` dim-order fix (transpose in `harmonize()`) — SCP'd to VM, not committed
- [ ] SCP updated scripts to mhw-training VM
- [ ] Retrain ERA5 (50 epochs) — verify normalized loss converges toward 0, spread still 0 (by design — ERA5 labels identical across members)
- [ ] Retrain WN2 (50 epochs) — verify spread > 0 if gate recalibrates toward atm-dominant

**Hypothesis:** `compute_mhw_mask` / `accumulate_sdd` not correctly aligning `dayofyear` threshold to
calendar dates in `merged.time` → every day treated as exceeding threshold → SDD accumulates for full 365 days.

**Diagnostic (run on VM, no training needed):**
```python
# In a quick script: load merged, threshold, call build_tensors(), print label stats
print(label_t.mean(), label_t.max())  # expect ~20-100 °C·day; if ~250 → bug confirmed
# Also spot-check: what fraction of days have mhw_mask=True?
```

**Files to inspect:** `src/analytics/mhw_detection.py` (`compute_mhw_mask`), `src/analytics/sdd.py` (`accumulate_sdd`)

**Fix if confirmed:** Ensure `threshold.sel(dayofyear=merged.time.dt.dayofyear)` alignment is correct before mask computation.

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

## DONE (but results invalid) — ERA5 50-epoch run on mhw-training VM (session 23)
Results invalid due to SDD label bug. See CRITICAL BUG section above.
- train_loss: 37079→36888 | val_loss: 20045→19912 | SVaR_95: 0.65→1.12 | spread=0.00 (all epochs)
- Artifacts on VM: `mhw-training:/home/avik2007/mhw-risk-profiler/data/results/`
- VM still running: `mhw-training` (us-central1-c, e2-highmem-8) — reuse for retrain after bug fix

## INFRA — GPU quota request in-flight
- User submitted GCP quota request for `GPUS_ALL_REGIONS` limit=2 (submitted session 23)
- Once approved: attach T4 to `mhw-training` for faster WN2 training
- Machine image available: `mhw-data-prep-img` (global)

## ACTIVE — ERA5 real training run (mhw-data-prep VM, PID 28544, ~/train_era5.log)

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
