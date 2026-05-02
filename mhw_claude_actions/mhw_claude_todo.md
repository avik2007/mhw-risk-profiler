# mhw_claude_todo.md
# Active task queue — current and next actions
# ---------------------------------------------
# Completed sessions → mhw_claude_recentactions.md

---

## ACTIVE — Option B parallel retrain on 2 VMs (relaunched 2026-04-30 after NaN fix)

**LinkedIn deadline: 2026-05-01**

| | VM | Zone | PID | Log | Status |
|---|---|---|---|---|---|
| ERA5 | mhw-training | us-central1-c, e2-highmem-8 | 184046 | ~/train_era5.log | ✅ COMPLETE — best ep 5. Results NOT yet pulled. |
| WN2  | mhw-training-wn2 | us-central1-c, e2-highmem-8 | 79192  | ~/train_wn2.log | ✅ COMPLETE — best ep 11, val_loss=0.077224, gate≈0.445, spread=0.217. Results pulled locally. |

Running `python scripts/train_*.py --epochs 50` on **25c9d02** (f522bbd + forward-fill NaN fix).

### Bug found + fixed (session 30, 2026-04-30)
Both runs launched 2026-04-27 crashed silently — all metrics NaN from epoch 1, early-stop at epoch 10.
**Root cause:** `build_tensors` land mask checked only `water_temp[depth=0]`. Shallow coastal cells (GoM continental shelf) pass the surface check but have NaN at depth levels 2–10 (below local seafloor). 194,560/627,968 (31%) of `hycom_t` values were NaN → CNN forward → NaN loss → NaN gradients → corrupt weights.
**Fix (commit 25c9d02):** forward-fill NaN along depth axis after computing `hycom_raw`. Last valid depth value propagated downward; `AdaptiveAvgPool1d` averages smoothly across all depth levels.

### WN2 NaN — RESOLVED (session 31, commit 88a5fe2)
WN2 SST NaN for ~86/357 GoM cells (all 365 days, all 64 members) — WN2 land-sea mask omits SST at coastal cells. Fix: `wn2_sst_valid` mask added to `build_tensors()`. n_cells 223→161. Verified clean: hycom_NaN=0, wn2_NaN=0, label_NaN=0 for both 2022 (train) and 2023 (val).

### Note: stdout block-buffered
Python stdout redirected to log file = block-buffered. Epoch lines don't appear in log until buffer fills or script exits. Use file timestamps (config.json, best_weights.pt) to monitor progress — not `tail`.

### Pull results
- [x] WN2 results pulled → `data/results_wn2/results/`
- [ ] ERA5 results NOT yet pulled — `gcloud compute scp --recurse mhw-training:~/mhw-risk-profiler/data/results/ data/results_era5/ --zone=us-central1-c`

### WN2 results review (session 31) — gate collapse + flat pred-vs-actual
- WN2 trained 21 epochs, best ep 11 (val_loss=0.077224). Early stopped at ep 21.
- Gate stable ~0.44 throughout. No per-cell differentiation.
- Flat pred-vs-actual: model predicts ~1.25–1.40 normalized regardless of actual SDD (0.5–2.1).
- Root cause: deterministic HYCOM labels (all 64 members identical) → no gradient signal for spatial gate differentiation or spread estimation.
- Spread grew 0.009→0.217 but reflects ONLY the WN2 atmospheric input noise, not actual HYCOM label variance.
- **Same diagnosis applies to ERA5 run (best ep 5).**
- Fix options: (A) quantile loss — cheap partial improvement; (B) P5 OISST 30yr retraining — correct fix.

### After ERA5 results pulled
- [ ] Run final XAI: `MHW_GCS_BUCKET=gs://mhw-risk-cache GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/compare_xai.py --use-gcs --n-steps 50`
- [ ] Validate "Zonal Wind Story" (WN2 → U-wind > V-wind, ERA5 → blunted by σ=0.3 noise)
- [ ] Stop mhw-training-wn2 (and possibly mhw-training) to halt VM costs — after LinkedIn validated
- [ ] Draft LinkedIn copy — review with user before publishing

### Cleanup pending
- [ ] Decide whether to keep `mhw-data-prep-img` (T4-baked, blocks e2 reuse) or delete + replace with `mhw-training-img`
- [ ] `~/results_pre_optionB/` and `~/models_pre_optionB/` on mhw-training — keep until LinkedIn validated, then remove
- [ ] `~/results_imaged/` and `~/models_imaged/` on mhw-training-wn2 — same

---

## DONE — Session 29 (2026-04-27)

- ✅ Reviewed Gemini session 29 spatial-batching upgrade via `code-review-graph::detect_changes`
- ✅ Found + fixed 5 bugs (commit f522bbd, pushed to origin/main):
  1. compare_xai.get_season_tensors stale spatial-mean → per-cell with `.contiguous()`
  2. save_plots cell-0-only gate hist + scatter → flatten across all cells × members
  3. save_plots full-tensor forward → mini-batch
  4. train_*.py gate_mean = last batch → accumulate val_gates across all batches
  5. SVaR + plots used last-epoch weights → reload best_weights.pt before inference
- ✅ Smoke-tested all 3 scripts via dry-run (all green)
- ✅ Provisioned 2nd VM via fresh no-GPU machine image (workaround for `mhw-data-prep-img` T4 lock-in)
- ✅ Backed up prior mhw-training results to `~/results_pre_optionB/`
- ✅ Both training runs launched in parallel on f522bbd code

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

## QUEUED — README/GitHub visualization polish (priority 3, after LinkedIn)

- [ ] WN2 mean SST plot: mask cold-blue artifact pixels (Cape Cod coastal WN2 cells with ~0°C values leaking through; clip values < 5°C or apply stricter NaN mask in `generate_spatial_figures.py`)
- [ ] Network diagram: review sublabel font sizes at final dpi — may need further compression
- [ ] Push polished figures + README updates to GitHub

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

## QUEUED — GeoAI Inference Orchestrator Pivot (Thread 4) — PRIORITY 4

**Trigger:** WN2 SVaR review accepted + LinkedIn ERA5 post published + README/GitHub viz polish merged.
**Plan file:** `docs/superpowers/plans/2026-04-25-geoai-orchestrator-pivot.md`
**Source:** `mhw_ai_research/Gemini-Iterative Weather Model Pipeline Thread 4.txt`

**Locked decisions (D1–D6):**
- D1 engine: GraphCast v1; Aurora as future-state generalization
- D2 FGN: Perlin/spatially-coherent noise on initial conditions ("Poor Man's FGN"); latent-space injection deferred
- D3 CRPS: skip v1; revisit if extremes underpredicted
- D4 init conditions: ERA5 via GEE v1; HRES/GFS future-state
- D5 scope: wrap existing `harvester.py` + `era5_harvester.py`, no rip
- D6 GPU: T4 if quota approved, CPU fallback acceptable for v1

**Phases:**
- [ ] Phase 0 — prerequisites green (WN2 review + LinkedIn + GitHub viz done; GPU status known)
- [ ] Phase 1 — `src/providers/` skeleton: `base.py`, `weathernext.py` wrapper, `era5.py` wrapper, schema tests
- [ ] Phase 2 — `realtime.py` (ERA5 GEE near-real-time fetch + GCS cache)
- [ ] Phase 3 — `graphcast_emulator.py` (stock weights, 10-day rollout, day-0 RMSE smoke vs ERA5 < 1°C)
- [ ] Phase 4 — `fgn_wrapper.py` (N=64 Perlin perturbations, ensemble spread Spearman > 0.7 vs WN2 overlap)
- [ ] Phase 5 — CRPS fine-tune (skipped v1, doc only)
- [ ] Phase 6 — `configs/settings.yaml` + `scripts/run_profiler.py --model {wn2|graphcast}`
- [ ] Phase 7 — README reframe ("Foundation Model Orchestrator for Spatial Finance"), `mhw-repo-architecture.md` update, recentactions append
- [ ] Phase 8 — validation gates (schema, smoke, spread, backtest 2023, end-to-end on yesterday)

**Future-state generalizations (post-v1):**
- Aurora provider (multi-modal, thermocline-sensitive)
- Latent-space FGN injection (`h_{t+1}=GNN(h_t,z)`)
- CRPS / Energy Score fine-tune
- HRES / GFS realtime providers
- Vertex AI deployment (ties to existing P8)

**Risks:**
- GraphCast weight license — confirm derivative inference allowed
- `graphcast` package JAX/CUDA pinning
- Perlin σ calibration (HYCOM spread=0 complicates baseline; use ERA5+synthetic as fallback)
- HRES open-data license (future-state)

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
