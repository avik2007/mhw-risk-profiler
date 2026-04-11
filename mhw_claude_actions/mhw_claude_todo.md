# mhw_claude_todo.md
# Active task queue — current and next actions
# ---------------------------------------------

---

## ACTIVE — RESUME HERE NEXT SESSION

### GCP Data Pipeline — Execute Plan (Subagent-Driven)

**Current HEAD:** `b8bfb26`

**Plan:** `docs/superpowers/plans/2026-04-10-gcp-data-pipeline.md`
**Spec:** `docs/superpowers/specs/2026-04-10-gcp-data-pipeline-design.md`

Use `superpowers:subagent-driven-development` to execute the plan task-by-task.

**7 tasks in order:**
1. `HYCOMLoader.fetch_and_cache()` — TDD, adds to `src/ingestion/harvester.py`
2. `ERA5Harvester.fetch_and_cache()` — TDD, adds to `src/ingestion/era5_harvester.py`
3. `_train_utils.py` period alignment — 2022/2023 shared constants, remove ERA5/WN2 pairs
4. `train_era5.py` GCS-only `load_real_data()` — replaces live OPeNDAP/GEE calls, fixes `ds["threshold"]` bug
5. `train_wn2.py` GCS-only `load_real_data()` — same pattern, removes `GCS_BUCKET` dependency
6. `scripts/run_data_prep.py` — idempotent orchestrator for spot GCE VM
7. `docs/gcp-data-prep-runbook.md` — gcloud VM setup and verification commands

After all tasks pass tests and dry-runs confirm no regressions, the pipeline is ready
to run on a spot GCE VM (`e2-standard-2`, ~$0.05/run). See runbook for VM setup.

---

## NEXT (after pipeline implementation)

### Run data prep job on spot GCE VM
See `docs/gcp-data-prep-runbook.md` for exact commands.
Prerequisite: all 7 pipeline tasks complete and tests passing.

### Real training runs on GCP
ERA5 and WN2 real runs on GCP (n2-standard-8 or T4 GPU).
Prerequisite: GCS data prep complete (`hycom/tiles/2022`, `hycom/tiles/2023`,
`hycom/climatology`, `era5/2022`, `era5/2023` all populated).

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
