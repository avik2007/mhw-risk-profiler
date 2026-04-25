# mhw_claude_todo.md
# Active task queue — current and next actions
# ---------------------------------------------
# Completed sessions → mhw_claude_recentactions.md

---

## DONE (sessions 24–27)

- SDD label fix (commit 79853d7), ERA5 retrain done (train=63k, val=38k, SVaR_95=1.34, spread=0)
- WN2 2023 annual zarr consolidated from daily cache (EXIT_CODE:0, 2026-04-21 22:08)
- SDD label magnitude diagnosed: 252 °C·day is REAL (2022 GoM record-warm year vs 30yr OISST baseline; 50% MHW mask fraction is physically valid)
- WN2 dim order bug found + fixed in `harmonize()`: WN2 zarr stores (time, member, lat, lon); added transpose to (member, time, lat, lon) before merge
- WN2 50-epoch training complete (2026-04-22 00:06): train=63308, val=38201, SVaR_95=1.09, spread=0.00, gate=0.769 — results pulled locally
- diagnose_labels.log confirms: 252.77 °C·day labels, 50.8% MHW mask, all 64 member labels IDENTICAL (HYCOM deterministic — spread=0 by construction)
- Root cause of flat loss confirmed: grad_clip=1.0 clips MSE gradient ~500 every step → near-zero learning
- **LABEL_NORM=250.0 + grad_clip=10.0 fix implemented** (commit 86a2614), ERA5 retrain complete: train→0.0001, val→0.046, SVaR_95=1.00 (normalized), spread=0.00 (expected). Model converged.

---

## QUEUED — README figure polish (post-commit)

- [ ] WN2 mean SST plot: mask cold-blue artifact pixels (Cape Cod coastal WN2 cells with ~0°C values leaking through; clip values < 5°C or apply stricter NaN mask in `generate_spatial_figures.py`)
- [ ] Network diagram: review sublabel font sizes at final dpi — may need further compression

---

## ACTIVE — WN2 SVaR inference running (PID 26740, mhw-training VM)

- ERA5 training + SVaR complete (EXIT_CODE:0). WN2 50 epochs done, SVaR inference in progress.
- ERA5 results: train→0.0001, val→0.046, SVaR_95=1.00 normalized (250 °C·day), spread=0, gate=0.338 (atm)
- WN2 results: train→0.025, val→0.0001, SVaR_95=0.81 normalized (~203 °C·day), spread=0, gate=0.669 (ocean)
- spread=0 for both: HYCOM labels identical across members — expected, not a bug

**Next:**
- [ ] Wait for WN2 SVaR inference to finish (monitor PID 26740)
- [ ] Pull all results locally: `gcloud compute scp mhw-training:~/mhw-risk-profiler/data/results/ data/results/ --recurse --zone=us-central1-c`
- [ ] Review loss curves + SVaR maps + pred-vs-actual plots
- [ ] Decide: accept spread=0 and proceed to payout.py, OR investigate architectural fix for spread (e.g. per-member HYCOM perturbation so labels differ across members)
- [ ] LinkedIn ERA5 post prep (pending results review)

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
