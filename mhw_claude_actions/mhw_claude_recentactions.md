# mhw_claude_recentactions.md
# Completed actions log — what was done and when
# -----------------------------------------------

---

## [2026-04-16] Session 6 — On-demand VM, job running, hourly GCS monitor active

### Context
VM `mhw-data-prep` was preempted again (3rd time at same HYCOM 2022-01 fetch point).
Decision: switch from SPOT provisioning to on-demand to eliminate preemption entirely.

### CLAUDE.md updates
- Created global `~/.claude/CLAUDE.md` with four behavioral principles:
  Think Before Coding, Simplicity First, Surgical Changes, Define Success.
  Also includes a "Consulting the User" trigger list and Coding Standards section.
- Added §6 (No Assumptions), §7 (Simplicity First / Surgical Changes), §8 (Tests-First)
  to project-level `CLAUDE.md` with mhw-specific callouts on scientific choices.

### VM migration: SPOT → on-demand
- `gcloud compute instances set-disk-auto-delete mhw-data-prep --no-auto-delete`
  — disabled auto-delete on boot disk before VM deletion.
- `gcloud compute instances delete mhw-data-prep --zone=us-central1-a --quiet`
  — deleted the spot VM; boot disk retained (50 GB, READY).
- `gcloud compute instances create mhw-data-prep --machine-type=e2-standard-4 --disk=name=mhw-data-prep,boot=yes,auto-delete=yes --no-preemptible --scopes=cloud-platform`
  — new on-demand VM created from existing boot disk. No PREEMPTIBLE flag confirmed.
- All prior setup (conda, mhw-risk env, .bashrc env vars, service account key) preserved
  on the boot disk — no re-setup required.

### Job launch fix: conda run → direct Python
- `conda run` does not survive SSH session detachment (process killed on disconnect).
- Fix: use the env's Python binary directly with env vars set inline:
  ```bash
  MHW_GCS_BUCKET=gs://mhw-risk-cache \
  GOOGLE_APPLICATION_CREDENTIALS=/home/avik2007/.config/gcp-keys/mhw-harvester.json \
  nohup /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/run_data_prep.py \
  >> data_prep.log 2>&1 </dev/null &
  disown $!
  ```
- PID 1323 confirmed alive; log shows HYCOM 2022-01 OPeNDAP fetch in progress.

### Hourly GCS monitor
- Cron job `d489cf43` scheduled at `:07` past every hour (session-only, auto-expires 7 days).
- Checks `.zmetadata` for all 5 paths; reports status each hour; stops and notifies user
  when all 5 are complete.
- Cancel with: `CronDelete d489cf43`

### To reconnect manually
```bash
gcloud compute ssh mhw-data-prep --zone=us-central1-a -- "tail -30 ~/mhw-risk-profiler/data_prep.log"
```

### GCS verification (run after job completes)
```bash
for path in hycom/tiles/2022 hycom/tiles/2023 hycom/climatology era5/2022 era5/2023; do
  gcloud storage ls gs://mhw-risk-cache/$path/.zmetadata 2>/dev/null && echo "$path: OK" || echo "$path: MISSING"
done
```

### Next step after completion
`train_era5.py --epochs 50` with `MHW_GCS_BUCKET=gs://mhw-risk-cache`

### Lesson recorded
`conda run` does not survive SSH session detachment — always use the env's Python binary
directly (`/home/avik2007/miniconda3/envs/mhw-risk/bin/python`) with `nohup` + `disown`.

---

## [2026-04-16] Session 5 — VM restart, env vars persisted, run_data_prep.py running

### Context
VM was preempted again. SSH failed with IAP error 4003 (backend not running). Restarted VM.
`data_prep.log` existed but was 0 bytes — script had never successfully run in prior sessions
due to two missing env vars.

### Root causes fixed

**1 — `MHW_GCS_BUCKET` was never set**
Script raised `RuntimeError: MHW_GCS_BUCKET env var not set` immediately. Correct bucket URI
identified (see `mondal-mhw-gcp-info.md`).

**2 — `GOOGLE_APPLICATION_CREDENTIALS` tilde not expanding**
`export GOOGLE_APPLICATION_CREDENTIALS="~/.config/gcp-keys/mhw-harvester.json"` passes a
literal `~` to Python; fixed to `"$HOME/.config/gcp-keys/mhw-harvester.json"`.

**3 — Conda not in PATH on VM**
`conda` was installed under `~/miniconda3/` but not initialized. Fixed with:
`/home/avik2007/miniconda3/bin/conda init bash` → modifies `~/.bashrc`.

### Persisted to `~/.bashrc` on VM
Both env vars and conda init are now in `~/.bashrc` — no manual setup needed after future
VM restarts or preemptions.

### Status
`run_data_prep.py` started and logging. HYCOM 2022-01 fetch confirmed in log. Job running
in tmux session `data_prep`.

---

## [2026-04-15] Session 4 — run_data_prep.py pipeline hardening (3 bugs fixed)

### Context
Spot VM `mhw-data-prep` found TERMINATED (preempted again); GCS `gs://mhw-risk-cache/` empty.
Diagnosed three root causes and fixed all before restarting.

### Bug fixes committed

**Bug 1 — Partial-write false cache hit (`harvester.py`, `era5_harvester.py`, `run_data_prep.py`)**
Both `HYCOMLoader.fetch_and_cache()` and `ERA5Harvester.fetch_and_cache()` checked path existence
with `gcsfs.exists(dir_path)`, which returns True as soon as any chunk file is written. A mid-write
preemption left an incomplete Zarr store that passed the idempotency check on restart, silently
producing corrupt data. Fix: check for `.zmetadata` (written *last* by `to_zarr(consolidated=True)`)
as the completeness marker. Added `_gcs_complete()` module-level helper in `harvester.py`;
`run_data_prep.py` `_gcs_exists()` renamed `_gcs_complete()` with same fix.

**Bug 2 — Annual HYCOM fetch atomicity (`harvester.py`)**
`HYCOMLoader.fetch_and_cache()` fetched a full calendar year in one OPeNDAP call + one to_zarr
write. A preemption mid-write lost all progress. Fix: iterate month-by-month, writing 12
intermediate Zarrs at `{gcs_uri}monthly/mMM/` (each with its own `.zmetadata` idempotency check),
then concatenate lazily and write the annual tile. Max preemption loss: ~20-30 min per month
instead of 3+ hours for the full year. Partial resumption tested (months 1-6 done → only 7-12 fetched).

**Bug 3 — ERA5 `filterDate` off-by-one (`era5_harvester.py`)**
GEE `filterDate(start, end)` treats `end` as exclusive. Passing `end_date = "2022-12-31"` silently
omitted Dec 31 from every ERA5 year. Fix: `end_date = f"{year + 1}-01-01"`.

### Test suite
58/58 tests passing. Added 3 new HYCOM tests (zmetadata check, partial resume, 12-month coverage);
updated ERA5 cache-hit path assertion and end-date expectation.

### Commit
See git log for commit hash.

### To restart the job
```bash
gcloud compute instances start mhw-data-prep --zone=us-central1-a
gcloud compute ssh mhw-data-prep --zone=us-central1-a
# then: tmux new -s dataprep
# conda run -n mhw-risk python scripts/run_data_prep.py 2>&1 | tee data_prep.log
```

---

## [2026-04-15] Session 3 — run_data_prep.py restarted on spot GCE VM

### Context
Spot VM `mhw-data-prep` (us-central1-a, **e2-standard-4**, 16 GB — resized from e2-standard-2 to fix OOM kill)
was preempted overnight. GCS was empty. VM restarted and resized. Bucket name was also corrected
(was `mhw-data-cache` which was wrong — correct name in `mondal-mhw-gcp-info.md`).

### Active job (as of 2026-04-15 evening)
- **tmux session:** `dataprep` on VM `mhw-data-prep`
- **Command running:** `conda run -n mhw-risk python scripts/run_data_prep.py 2>&1 | tee data_prep.log`
- **Log file on VM:** `~/mhw-risk-profiler/data_prep.log`
- **Status:** Running — log not yet producing output (early in HYCOM OPeNDAP fetch)
- **Estimated runtime:** 3–5 hours from job start

### To reconnect
```bash
gcloud compute ssh mhw-data-prep --zone=us-central1-a
tmux attach -t dataprep
```

### To check progress without SSH
```bash
gcloud compute ssh mhw-data-prep --zone=us-central1-a -- "tail -20 ~/mhw-risk-profiler/data_prep.log"
```

### GCS verification (run after job completes)
```bash
for path in hycom/tiles/2022 hycom/tiles/2023 hycom/climatology era5/2022 era5/2023; do
  gcloud storage ls gs://mhw-data-cache/$path/ 2>/dev/null && echo "$path: OK" || echo "$path: MISSING"
done
```

### Next step after completion
`train_era5.py --epochs 50` with `MHW_GCS_BUCKET=gs://mhw-data-cache` (Session 3 goal).

---

## [2026-04-14] GCP Data Pipeline — Session 1 Complete (7 tasks, 10 commits)

### Summary
Executed `docs/superpowers/plans/2026-04-10-gcp-data-pipeline.md` via subagent-driven development. All 7 tasks shipped. HEAD: `2d37d7e`. 55/55 tests passing. Both dry-runs clean.

### Task 1 — `HYCOMLoader.fetch_and_cache()` (`src/ingestion/harvester.py`)
- Added `import gcsfs` at module level.
- Method: idempotent GCS cache check (`gcsfs.GCSFileSystem().exists()`, strips `gs://` prefix via `removeprefix`); on miss calls `self.fetch_tile(f"{year}-01-01", f"{year}-12-31", bbox)` then `ds.to_zarr(gcs_uri, mode="w", consolidated=True)`.
- Docstring: physical meaning of all params, HYCOM 0–360 longitude convention documented, downstream consumers named (`run_data_prep.py`, `train_era5.py`, `train_wn2.py`). Satisfies CLAUDE.md §4.
- Tests (`TestHYCOMLoaderFetchAndCache`, `tests/test_harvester_cache.py`): 4 tests — cache hit, cache miss, `gs://` prefix stripping, exception propagation without GCS write.
- Commits: `1e3bb16`, `90434e0`

### Task 2 — `ERA5Harvester.fetch_and_cache()` (`src/ingestion/era5_harvester.py`)
- Same pattern as Task 1; adds `_initialized` guard: raises `RuntimeError("Call authenticate() before fetch_and_cache().")` if auth not done.
- Tests (`TestERA5HarvesterFetchAndCache`): 4 tests — cache hit, cache miss, unauthenticated guard, exception propagation.
- Total cache tests: 8 across both classes.
- Commits: `3222eb1`, `a09ba76`

### Task 3 — Period constant alignment (`scripts/_train_utils.py`)
- Removed `ERA5_TRAIN_PERIOD`, `ERA5_VAL_PERIOD`, `WN2_TRAIN_PERIOD`, `WN2_VAL_PERIOD` and backward-compat aliases.
- New: `TRAIN_PERIOD = ("2022-01-01", "2022-12-31")`, `VAL_PERIOD = ("2023-01-01", "2023-12-31")`.
- Comment explains rationale: ERA5 covers 1979–present on GEE; WN2 covers 2022–present; HYCOM GLBy0.08/expt_93.0 through 2024-09-04. Unified period enables apples-to-apples XAI comparison.
- Both `train_era5.py` and `train_wn2.py` imports updated atomically in same commit.
- Commit: `aac0f8b`

### Task 4 — GCS-only `load_real_data()` in `train_era5.py`
- Removed live `ERA5Harvester.fetch()` and `HYCOMLoader.fetch_tile()` calls from `load_real_data()`.
- Reads from `MHW_GCS_BUCKET` env var via `xr.open_zarr(..., chunks="auto")`.
- Threshold: `xr.open_zarr(f"{bucket}/hycom/climatology/")["sst_threshold_90"]` — fixes pre-existing bug where key was `"threshold"`.
- Train: `era5/2022/` + `hycom/tiles/2022/`; Val: `era5/2023/` + `hycom/tiles/2023/`.
- Raises `RuntimeError` with actionable message if `MHW_GCS_BUCKET` not set.
- Module docstring updated: "2018/2019" → "2022/2023"; usage block updated to GCS workflow.
- Dry-run (`--dry-run --epochs 2`) passes — synthetic path unaffected.
- Commits: `1c21eb2`, `2d37d7e` (docstring fix)

### Task 5 — GCS-only `load_real_data()` in `train_wn2.py`
- Same pattern as Task 4.
- WN2 train: `weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr`; Val: `wn2_2023-01-01_2023-12-31_m64.zarr`.
- HYCOM tiles shared with ERA5 path.
- Old `GCS_BUCKET` env var replaced with `MHW_GCS_BUCKET`. Threshold key fixed to `"sst_threshold_90"`.
- Dry-run passes.
- Commit: `1a3438c`

### Task 6 — `scripts/run_data_prep.py` (new file, 127 lines)
- Idempotent 5-step orchestrator for spot GCE VM (`e2-standard-2`, ~$0.05–0.09/run).
- Steps: HYCOM tiles 2022 → HYCOM tiles 2023 → HYCOM climatology → ERA5 tiles 2022 → ERA5 tiles 2023 → WN2 verification.
- Climatology step reads from GCS (no second OPeNDAP fetch); calls `compute_climatology(sst_all, percentile=90.0)`; writes `sst_threshold_90` to `{bucket}/hycom/climatology/`.
- Each step guarded by `_gcs_exists()` — skip if already present.
- Raises `RuntimeError` if `MHW_GCS_BUCKET` not set.
- Commit: `1d979ff`

### Task 7 — `docs/gcp-data-prep-runbook.md` (new file)
- 7-section runbook: create spot VM → SSH → env setup → run job → verify outputs → delete VM → run real training.
- Includes full `gcloud` commands, `conda` env setup, `gcloud compute scp` for credentials, Python verification snippet opening all 5 GCS Zarr paths.
- Commit: `f55f393`

### Commit log (oldest → newest)
1. `1e3bb16` feat: add HYCOMLoader.fetch_and_cache() with GCS idempotent caching
2. `90434e0` fix: improve HYCOMLoader.fetch_and_cache() docstring and add exception propagation test
3. `3222eb1` feat: add ERA5Harvester.fetch_and_cache() with GCS idempotent caching
4. `a09ba76` test: add ERA5Harvester.fetch_and_cache() exception propagation test
5. `aac0f8b` refactor: align ERA5 and WN2 training periods to 2022/2023 shared constants
6. `1c21eb2` feat: train_era5.py load_real_data() reads from GCS; align to 2022/2023 periods
7. `1a3438c` feat: train_wn2.py load_real_data() reads from GCS; remove GCS_BUCKET dependency
8. `1d979ff` feat: add run_data_prep.py — idempotent GCS data prep orchestrator for spot GCE
9. `f55f393` docs: add GCP data prep runbook for spot GCE VM setup and job execution
10. `2d37d7e` docs: fix train_era5.py module docstring — update 2018/2019 to 2022/2023

### Next session (Session 2)
Run `train_era5.py --dry-run` with extended epochs to produce presentable artifacts:
- `data/results/era5_proxy/loss_curve.png` — train vs. val MSE
- `data/results/era5_proxy/svar_output.zarr` — SVaR quantiles per grid cell
- `data/results/xai/ig_attribution_<season>.png` — per-season Captum IG heatmaps

---

## [2026-04-14] Proxy Training Run — Session 2 Complete (3 tasks, 3 commits)

### Summary
Produced first presentable artifacts from local proxy training run using `train_era5.py --dry-run --epochs 30`. All artifacts confirmed on disk. HEAD: `89c9ed2`. 55/55 tests passing.

### Task S2-T1 — ERA5 proxy training run, 30 epochs
- Ran `train_era5.py --dry-run --epochs 30` with synthetic 3×4 GoM grid (64 members, seed=42).
- Training converged: train loss 122 → 66, val loss 105 → 57 over 30 epochs.
- Artifacts saved under `data/results/plots/`:
  - `era5_loss_curve.png` — train vs. val MSE per epoch
  - `era5_svar_curve.png` — SVaR_95/50/05 quantile traces
  - `era5_spread_curve.png` — ensemble spread over time
  - `era5_gate_hist.png` — LeakyGate activation histogram
  - `era5_pred_vs_actual.png` — scatter of predicted vs. actual SDD
- `data/models/era5_best_weights.pt` saved (2.3 MB).
- Commit: `110c031`

### Task S2-T2 — Synthetic SVaR in dry-run (`train_era5.py`)
- `merged_val = None` in dry-run caused `run_svar_inference()` to skip. Fixed by building a synthetic xarray Dataset matching inference expectations:
  - `pd.date_range("2022-01-01", periods=120, freq="D")`, 3 lats, 4 lons, 64 members, seed=42
  - HYCOM variables: shape `(M, T, 11, lat, lon)` float32; WN2 variables: `(M, T, lat, lon)` float32 with +280 K offset
- `data/results/era5_svar.zarr` written: `SVaR_95`, `SVaR_50`, `SVaR_05`, `spread` at each grid cell.
- Commit: included in `110c031`

### Task S2-T3 — XAI attribution plots (`scripts/compare_xai.py`)
- Added `save_attribution_plots(result: dict, out_dir: str) -> None` to `compare_xai.py`.
- Produces one 2-panel bar chart per season (DJF/MAM/JJA/SON): atmospheric variables (top panel, ERA5=steelblue) and HYCOM variables (bottom panel).
- Docstring explains IG physical meaning: dimensionless normalized gradient × input magnitude; higher = stronger driver of latent risk signal; justifies use as parametric insurance trigger validity evidence.
- Four PNGs saved to `data/results/xai/`: `ig_attribution_DJF.png`, `ig_attribution_MAM.png`, `ig_attribution_JJA.png`, `ig_attribution_SON.png` — 76–77 KB each. SST confirmed as top variable in all seasons.
- `data/results/xai/xai_comparison.json` updated: 4 seasons, gate≈0.472.
- Code quality reviewer flagged missing physical interpretation in docstring; fixed with expanded Notes section.
- Commits: `22f5732`, `89c9ed2`

### Commit log (oldest → newest)
1. `110c031` feat: add synthetic SVaR to train_era5.py dry-run; run 30-epoch proxy training
2. `22f5732` feat: add save_attribution_plots() to compare_xai.py; produce XAI PNGs
3. `89c9ed2` fix: expand save_attribution_plots() docstring with IG physical interpretation

### Artifacts on disk
- `data/results/plots/era5_loss_curve.png`
- `data/results/plots/era5_svar_curve.png`
- `data/results/plots/era5_spread_curve.png`
- `data/results/plots/era5_gate_hist.png`
- `data/results/plots/era5_pred_vs_actual.png`
- `data/results/era5_svar.zarr`
- `data/results/xai/ig_attribution_{DJF,MAM,JJA,SON}.png`
- `data/results/xai/xai_comparison.json`
- `data/models/era5_best_weights.pt`

### Next session (Session 3)
Real data run on spot GCE VM (`e2-standard-2`, ~$0.05/run):
1. `run_data_prep.py` — populate GCS with 2022/2023 HYCOM tiles, climatology, ERA5 tiles
2. `train_era5.py --epochs 50` with `MHW_GCS_BUCKET` set — real loss curves
3. `data/results/era5_real/loss_curve.png` + `era5_real/mhw_threshold_map.png`

---

## [2026-04-10] GCP Data Pipeline — Design + Plan Complete

### Context
User observed that OPeNDAP HYCOM fetching running locally was slow and resource-intensive.
Decision: move all data loading to GCP (spot GCE VM + GCS caching), training reads GCS only.

### Design decisions (brainstormed and approved)
- **Architecture**: Option 3 — GCS-aware harvesters + thin orchestrator (not monolithic script, not shell script)
- **Scope**: Full — all training data pre-fetched (HYCOM tiles, HYCOM climatology, ERA5 tiles, WN2 verified)
- **Execution model**: Spot GCE VM (`e2-standard-2`, ~$0.05/run) — idempotent, can resume after preemption
- **Training load path**: GCS-only, no OPeNDAP/GEE fallback. `MHW_GCS_BUCKET` env var required for real runs.
- **ERA5 period alignment**: ERA5 moved from 2018/2019 → **2022/2023** to match WN2 (apples-to-apples XAI comparison). The 2018/2019 split was a legacy artifact of the old HYCOM coverage constraint, now resolved.

### Artifacts committed
- `docs/superpowers/specs/2026-04-10-gcp-data-pipeline-design.md` — full design spec (`1397ff6`)
- `docs/superpowers/plans/2026-04-10-gcp-data-pipeline.md` — 7-task TDD implementation plan (`b8bfb26`)
- `mhw_claude_actions/mhw_claude_todo.md` — Vertex AI future task added

### GCS layout approved
```
gs://<bucket>/
  weathernext2/cache/       # existing WN2 cache (no change)
  hycom/climatology/        # 90th-pct threshold (dayofyear, lat, lon)
  hycom/tiles/2022/         # shared ERA5 + WN2 train tile
  hycom/tiles/2023/         # shared ERA5 + WN2 val tile
  era5/2022/                # ERA5 train year (member=1)
  era5/2023/                # ERA5 val year
```

### Pre-existing bug identified (to be fixed in Task 4/5)
Both `train_era5.py` and `train_wn2.py` read `ds["threshold"]` from the threshold Zarr,
but it is saved as `ds["sst_threshold_90"]`. The new `load_real_data()` implementations fix this.

### Plan ready for execution
Use `superpowers:subagent-driven-development` next session.

---

## [2026-04-10] Task 7 — Full test suite + all 3 dry-runs complete

**32/32 tests passed** (`pytest tests/ -v`) — no failures, no warnings after fix below.

**FutureWarning fix:** `test_era5_harvester.py` replaced `ds.dims[...]` with `ds.sizes[...]`
throughout — xarray deprecated `.dims` as a length mapping in favour of `.sizes`.
Commit: `83bced6`

**3 dry-runs in sequence — all clean:**
- `train_era5.py --dry-run --epochs 5`: 5 epochs, artifacts saved under `data/` (CUDA)
- `train_wn2.py  --dry-run --epochs 5`: 5 epochs, artifacts saved under `data/` (CUDA)
- `compare_xai.py --dry-run`: 4 seasons, `xai_comparison.json` saved, no warnings

**ERA5/WN2 dual training plan fully complete.** All 8 tasks done.

---

## [2026-04-10] compare_xai.py — OOM bug fixed, dry-run gate passed

**Context:** Running the Task 6 dry-run was crashing the local machine (swap thrash).

**Root cause:** Captum IG with `n_steps=50` stacks all 50 alpha-scaled inputs along the batch
dimension before the forward pass. Inside `MHWRiskModel.forward()`, the member dimension is
flattened into batch: effective batch = `n_steps × N_MEMBERS = 50 × 64 = 3200`. The Transformer
attention weights `(3200, 8, 90, 90)` consumed ~828 MB per layer × 4 layers = ~3.3 GB simultaneously,
exhausting RAM and hammering swap to disk.

**Fix:** Added `internal_batch_size=5` to `ig.attribute()` in `run_season_ig()`. This caps the
effective Transformer batch at `5 × 64 = 320`, reducing peak attention memory to ~330 MB.
Also added `.detach()` before `abs()` on attribution tensors to suppress a spurious Captum
autograd warning.

**Verification:** `conda run -n mhw-risk python scripts/compare_xai.py --dry-run` — clean output,
all 4 seasons, `xai_comparison.json` saved. No warnings.

**Lesson recorded:** `mhw_claude_lessons.md` + `CLAUDE.md` Lessons Applied section updated.

**Status:** compare_xai.py changes uncommitted (pending Task 7 commit).

---

## [2026-04-10] ERA5/WN2 Dual Training Plan — Execution Started (Tasks 0 & 1 complete)

Executing plan at `docs/superpowers/plans/2026-04-10-era5-wn2-dual-training.md`
using `superpowers:subagent-driven-development` skill.

### Completed this session:

**Task 1 — matplotlib dependency** ✅
- Added `matplotlib>=3.8.0` to `requirements.txt` under Scientific utilities
- Installed in `mhw-risk` conda env; verified import (`3.10.8`)
- Commit: `88765f1`

**HYCOM URL fix** ✅
- Switched `GLBv0.08` → `GLBy0.08` in both `HYCOM_THREDDS_TS` and `HYCOM_THREDDS_UV` constants in `src/ingestion/harvester.py`
- GLBy0.08/expt_93.0 covers 2018-12-04 to 2024-09-04; needed for WN2 2022/2023 training periods
- Commit: `7012a5f`

**Task 0 — WN2 GEE asset scoping** ✅ (done_with_concerns addressed)
- Created `scripts/scope_wn2_asset.py`
- Ran against live GEE; output captured to `docs/superpowers/specs/wn2_asset_schema.txt`
- Key finding: WN2 is a **forecast run structure** (not daily time series), covering **2022-present only**
  - 4 init times/day (00Z, 06Z, 12Z, 18Z), 15-day horizon, 64 FGN members
  - Recommended harvesting: filter to 00Z init + forecast_hour=24 → one 24h-ahead per member per day
  - ERA5 TRAIN_PERIOD (2018/2019) is valid; WN2 must use 2022/2023
- Updated `docs/superpowers/specs/2026-04-10-era5-wn2-xai-comparison-design.md` with Phase 0 Findings
- Fixed `col.size().getInfo()` hang in scope script
- Commits: `d0bf97d`, `b84a199`

### Plan change (approved by user):
- ERA5: TRAIN_PERIOD=2018, VAL_PERIOD=2019 (unchanged)
- WN2: TRAIN_PERIOD=2022, VAL_PERIOD=2023 (new)
- `_train_utils.py` must export BOTH sets: `ERA5_TRAIN_PERIOD`, `ERA5_VAL_PERIOD`, `WN2_TRAIN_PERIOD`, `WN2_VAL_PERIOD`
- Each training script imports its own set

### Current HEAD: `b84a199`

---

## [2026-03-30] HYCOM EDA Notebook Created

1. Created `notebooks/hycom_eda.ipynb` — 10-section exploratory notebook using existing
   `data/processed/hycom_2019-08-01_2019-08-03.zarr` (no network required).
2. Sections: dataset structure, surface temperature map (with 18°C contour), all 4 variables
   side-by-side, depth profiles (what the CNN sees), Hovmöller diagram, SST time series with
   SDD shading, current vectors, all-profiles overlay, T-S diagram, xarray patterns.
3. Registered `mhw-risk` conda env as a Jupyter kernel (`ipykernel install --user --name mhw-risk`).
4. Launch: `conda run -n mhw-risk jupyter notebook notebooks/hycom_eda.ipynb`

---

## [2026-03-30] Analytics Plan Revised — WN2 Proxy Training Dropped

Plan at `docs/superpowers/plans/2026-03-30-hycom-proxy-training.md` revised after discussion.
Decisions made:
- WN2 proxy training (tasks 2–4) dropped — no fake WN2 data; wait for real GEE whitelist
- MHW threshold changed from constant 18°C → location-varying per grid cell
- Two tasks remain, deferred to next session:
  1. `src/analytics/payout.py` — parametric payout engine (pure math)
  2. `scripts/compute_hycom_climatology.py` — fetch 2yr HYCOM surface SST, compute
     90th-percentile threshold per (dayofyear, lat, lon), save to
     `data/processed/hycom_sst_threshold.zarr`

---

## [2026-03-30] HYCOM-Proxy Training Pipeline Plan Written (Tentative — superseded above)

---

## [2026-03-27] HYCOM Zarr Verification — Steps 3 & 4 DONE

1. Wrote `scripts/verify_hycom_zarr.py` — fetches HYCOM tile, writes local Zarr, verifies steps 3 & 4.
2. Ran the script; both steps passed:
   - **Step 3** (Vertical Coordinate Sanity Check): T/S profile at 43.5°N 70°W printed from
     `data/processed/hycom_2019-08-01_2019-08-03.zarr`. Thermocline confirmed:
     19.8°C (0m) → 17.2°C (5m) → 13.2°C (10m) → 10.5°C (20m) → 9.3°C (30m) → 8.8°C (50m) → 7.9°C (75m).
     NaN at 100–300 m expected (seafloor depth ~100 m in Gulf of Maine).
   - **Step 4** (Dask Scaling Test): `xr.open_zarr` returned (time=24, depth=11, lat=26, lon=13);
     all 4 variables confirmed as `dask.array` (lazy, not eager); no OOM.
     Disk size: 744 KB (well within the MB target).

---

## [2026-03-27] Ensemble Connectivity Smoke Test (Step 2) — Partial

### HYCOM Side: PASSED

1. Created `mhw-risk` conda environment (python=3.11); installed all requirements.
2. Added `google-cloud-storage>=2.14.0` and `gcsfs>=2024.2.0` to `requirements.txt` (were missing).
3. Fixed 5 bugs in `harvester.py`:
   - `ServiceAccountCredentials(email=None)` → extract email from JSON key file.
   - Corrected WeatherNext 2 GEE asset path: `59572747_3_0` → `weathernext_2_0_0`.
   - `_export_to_gcs` (GeoTIFF + `xr.open_zarr`) replaced with `_fetch_and_write_zarr` (sampleRectangle compute path + `gs://` URI Zarr write).
   - `HYCOM_THREDDS_BASE` split into `HYCOM_THREDDS_TS` (ts3z) + `HYCOM_THREDDS_UV` (uv3z) — T/S and currents are separate THREDDS datasets.
   - CLI arg `--members` renamed to `--n_members` to match docs.
4. Rewrote `HYCOMLoader.fetch_tile`:
   - Opens both ts3z and uv3z with `decode_times=False`.
   - Slices time by raw float index (avoids OPeNDAP hang from full-axis sort).
   - Converts bbox longitude -180..180 → 0..360 for HYCOM slicing; converts back after load.
   - Merges T/S and UV datasets before interpolation.
5. Wrote `scripts/smoke_test_gee.py` — 3-stage standalone connectivity test.
6. Verification evidence (HYCOM, 2019-08-01 to 2019-08-03, Gulf of Maine 1°×1°):
   - Dataset: (time=24, depth=11, lat=26, lon=13), all 4 variables loaded.
   - T/S profile at 43.5°N 70°W: 19.8°C at 0m → 7.9°C at 75m; NaN below (seafloor ~100m).
   - Thermocline confirmed visible (August Gulf of Maine summer stratification).

### WeatherNext 2 Side: BLOCKED

- GEE auth works (service account authenticated OK).
- Asset path corrected to `weathernext_2_0_0`.
- Access denied: the WeatherNext Data Request form must be submitted at developers.google.com/weathernext/guides/earth-engine to whitelist the service account.
- **User action required**: submit the form, then re-run `python scripts/smoke_test_gee.py`.

---

## [2026-03-27] Docker Engine Installed and Verified

1. Removed conflicting Ubuntu-repo Docker packages — none were present; system was clean.
2. Added Docker's official apt repo (Noble / amd64) to `/etc/apt/sources.list.d/docker.list`.
3. Installed: docker-ce 29.3.1, docker-ce-cli, containerd.io, docker-buildx-plugin, docker-compose-plugin v5.1.1.
4. Added avik2007 to docker group; enabled and started daemon via systemd (active/running).
5. Verification gate passed: hello-world OK, compose v5.1.1 OK, mhw-risk-profiler:latest image built OK (all 6 layers).

---

## [2026-03-24] Cloud Infrastructure Initialized

### Actions Completed

1. GCP project created; $300 free credits activated.

2. Earth Engine API enabled on the project. Registered account at code.earthengine.google.com/register
   as Contributor (Noncommercial, 1,000 EECU-hours). Billing account linked (required for Contributor
   tier, but EE usage itself does not charge).

3. Service account created with required IAM roles.
   (See `mondal-mhw-gcp-info.md` for account email and role list.)

4. GCS bucket created with Standard storage class, Hierarchical namespace enabled, and public
   access prevention enforced.
   (See `mondal-mhw-gcp-info.md` for bucket name and region.)

5. JSON key secured under `~/.config/gcp-keys/` with `chmod 600`.
   `GOOGLE_APPLICATION_CREDENTIALS` environment variable configured to point to this path.
   (See `mondal-mhw-gcp-info.md` for exact path.)

6. Smoke test passed: Auth OK, Bucket accessible, Contents empty (expected).

7. Monthly budget alert configured at 50%, 90%, and 100% thresholds.

---

## [2026-03-24] Day 1 (Session 3) — Maintenance: Cloud Calibration Task Setup

### Actions Completed

1. Replaced ACTIVE task in `mhw_claude_todo.md` from "Implement GEE Python API Harvester"
   to "GCP Environment Calibration & Ingestion Testing" — four sub-steps with explicit
   verification evidence requirements (IAM/Auth, Ensemble smoke test, HYCOM vertical profile,
   Dask lazy-open).

2. Archived the previous ACTIVE "Implement GEE Python API Harvester" entry to this log
   (see Session 2 entry below).

3. Updated `CLAUDE.md` with a dedicated GCP/Conda Environment Commands section covering
   authentication setup, Conda env activation, and OPeNDAP connectivity checks.

---

## [2026-03-24] Day 1 (Session 2) — Ingestion Engine Implementation

### Actions Completed

1. Corrected task priority: GEE/HYCOM ingestion engine set as ACTIVE before model work.

2. Implemented `src/ingestion/harvester.py` — production-ready, three-class ingestion engine:
   - `WeatherNext2Harvester`: GEE authentication (service account + ADC), queries
     `gcp-public-data-weathernext` FGN ensemble, exports to GCS Zarr with cache-hit logic.
   - `HYCOMLoader`: Fetches HYCOM GLBv0.08 via OPeNDAP/THREDDS; interpolates from native
     hybrid coordinate (Z-level / Sigma / Isopycnal) to TARGET_DEPTHS_M standard levels.
   - `DataHarmonizer`: Regrids both sources to 0.25-degree TARGET_LAT/LON grid; broadcasts
     deterministic HYCOM across 64 WeatherNext 2 ensemble members; writes CF-1.8 metadata.
   - `run_ingestion_pipeline()`: End-to-end orchestration with verification print gate.

3. All functions include verbose physical oceanography header comments per CLAUDE.md style rules.

4. Harvester saves harmonized output to `data/processed/harmonized_<start>_<end>.zarr`
   and prints Dataset repr as verification evidence.

5. Created `requirements.txt` with pinned minor-version dependencies.

6. Updated `mhw_claude_todo.md`: GEE Harvester set ACTIVE, 1D-CNN/Transformer moved to QUEUED.

---

## [2026-03-24] Day 1 — Project Infrastructure Setup

### Actions Completed

1. Acknowledged existing `mhw_ai_research/` folder containing Gemini, Perplexity, and
   NotebookLM deep-dives on MHW risk and marine habitat suitability.

2. Created core source directory structure:
   - `src/ingestion/`  — GEE API + Xarray/Dask harmonization layer
   - `src/models/`     — PyTorch 1D-CNN + Transformer architecture
   - `src/analytics/`  — MHW Stress Degree Day and Financial VaR logic
   - All directories initialized with `__init__.py` stubs.

3. Created `CLAUDE.md` in the project root with ArgoEBUS-inspired principles:
   Plan Mode, Self-Improvement Loop, Science-to-Engineering Boundary, Style, Verification Gate.

4. Created `mhw-repo-architecture.md` — annotated directory tree with pipe notation,
   separated from `CLAUDE.md` per user instruction.

5. Created `data/` directory with subdirectories `raw/`, `processed/`, `cache/`.
   Each initialized with `.gitkeep` to track structure without committing data.

6. Created `.gitignore` — explicitly excludes `data/` and `mhw_ai_research/`;
   notebooks retained (`.ipynb` not ignored) to preserve R&D visibility.

7. Created `Dockerfile` — `python:3.11-slim` base, system spatial libs (`libgdal-dev`,
   `libnetcdf-dev`), all core Python dependencies, WORKDIR `/app`, ingestion entrypoint.

8. Created `requirements.txt` — pinned minor versions for earthengine-api, xarray, dask,
   netCDF4, zarr, torch, captum, fastapi, uvicorn, numpy, pandas, scipy.

9. Created `README.md` — full SETS Framework framing (Ecological, Social/Financial,
   Technological), Science-to-Insight pipeline ASCII diagram, Quickstart, Data Sources.

10. Populated `mhw_claude_todo.md` with Day 1 priority task:
    "Drafting the GEE Python API harvester for WeatherNext 2 Zarr data and HYCOM NetCDF alignment."

---
