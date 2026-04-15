# mhw_claude_todo.md
# Active task queue — current and next actions
# ---------------------------------------------

---

## 3-SESSION DELIVERY PLAN (target: presentable graphics + training evidence)

### Session 1 — GCP pipeline code complete [DONE ✅]
**HEAD:** `2d37d7e` | 55/55 tests passing | 10 commits

---

### Session 2 — Proxy training run locally → first presentable artifacts [DONE ✅]
**HEAD:** `89c9ed2` | 55/55 tests passing | 3 commits

**Artifacts produced:**
- `data/results/plots/era5_loss_curve.png` — train 122→66, val 105→57 over 30 epochs
- `data/results/plots/era5_svar_curve.png`, `era5_spread_curve.png`, `era5_gate_hist.png`, `era5_pred_vs_actual.png`
- `data/results/era5_svar.zarr` — SVaR_95/50/05/spread at lat=3, lon=4 synthetic GoM grid
- `data/results/xai/ig_attribution_{DJF,MAM,JJA,SON}.png` — 76–77 KB each, SST top var
- `data/results/xai/xai_comparison.json` — 4 seasons, gate≈0.472
- `data/models/era5_best_weights.pt` — 2.3 MB

---

### Session 3 — Real data run on spot GCE VM
**Goal:** Real loss curves + MHW threshold maps on Gulf of Maine grid using 2022 HYCOM+ERA5.
**Prerequisite:** Session 2 complete. GCS populated via `run_data_prep.py` on `e2-standard-2` (~$0.05/run).
**See:** `docs/gcp-data-prep-runbook.md` for VM setup commands.

**Expected outputs:**
- Real `train_era5.py` run on GCS data
- `data/results/era5_real/loss_curve.png`
- `data/results/era5_real/mhw_threshold_map.png` — 90th-pct threshold on GoM grid

---

## PENDING (external blocker — no code work needed)

### WeatherNext 2 GEE Access — real-run harvesting strategy
**Status:** GEE whitelist approved. WN2 is a forecast run structure (not daily time series).
See `docs/superpowers/specs/wn2_asset_schema.txt` for full schema findings.

**When implementing `train_wn2.py` for real run**, `WeatherNext2Harvester.fetch_ensemble()` must filter:
- `start_time` ending in `T00:00:00Z` (00Z init only)
- `forecast_hour = 24` (24h-ahead forecast → one per member per day)
- This gives 365 × 64 images/year — matching ERA5's daily structure

---

## QUEUED

### [LONG TERM] Extended SST Climatology — HYCOM Experiments + OISST
**Goal**: Replace the 2-year HYCOM expt_93.0 baseline with a longer historical record
suitable for a statistically robust 90th-percentile MHW threshold (Hobday 2016 recommends
≥30 years).

**Two avenues:**
1. Longer HYCOM runs — GLBv0.08 has expt_91.x, 92.x going back to ~1994. Check THREDDS catalog.
2. NOAA OISST v2.1 — daily 0.25-degree, 1981–present, standard MHW literature baseline.

**When to tackle**: Before production deployment. Not blocking current dev work.

---

### [LOW PRIORITY] XAI Option C — Member-Level Attribution Variance (ERA5 vs WN2)
**Prerequisite**: Phase 3 XAI comparison (compare_xai.py) complete.
**Output**: `data/results/xai_member_variance.json`

---

### [LOW PRIORITY] MTSFT: FFT-enriched Transformer for Periodic SST Features
**Prerequisite**: Baseline MHWRiskModel XAI validated on standard architecture first.

---

### [FUTURE] Vertex AI Custom Job for Training Pipeline
**Context**: Real ERA5 and WN2 training runs are planned for GCP (n2-standard-8 or T4 GPU).
Once the spot GCE data prep pipeline is working, migrate training scripts to Vertex AI custom
jobs for managed infrastructure, built-in Cloud Console logging, and consistent artifact
storage in GCS.
**Prerequisite**: GCP data prep pipeline (spot GCE VM + GCS caching) complete and validated.
