# mhw_claude_recentactions.md
# Completed actions log — what was done and when
# -----------------------------------------------

---

## [2026-04-22] Session 26 — WN2 50-epoch results reviewed; label diagnostic run; grad_clip root cause confirmed

### What happened
- **WN2 results pulled** from mhw-training VM (`/home/avik2007/data/` not `~/mhw-risk-profiler/data/` — script ran from `~`): train=63308, val=38201, SVaR_95=1.09, spread=0.00, gate=0.769
- **ERA5 results confirmed**: train=63224, val=38103, SVaR_95=1.34, spread=0.00, gate=0.281 (atm-dominant, opposite to WN2)
- **Pred vs actual**: model predicts ~1 °C·day vs labels ~200 °C·day — 200× underprediction. Gate histogram shows all 64 cells identical value.
- **diagnose_labels.log** run on VM: SST mean=11.95°C, threshold mean=10.87°C, MHW mask=50.8%, SDD=252.77 °C·day for ALL 64 members (identical to 8 decimal places — HYCOM deterministic, no member variation)
- **Root cause confirmed**: grad_clip=1.0 clips gradient ~500 every step → near-zero learning. Label scale mismatch (250 vs model output ~1) is primary driver.
- **Spread=0 physics**: HYCOM is a single deterministic run — all 64 member labels are identical → no gradient signal to differentiate member predictions → spread=0 by construction. Fix may yield non-zero spread only if atm stream (WN2 varies per member) dominates gate post-fix.
- **Model uses Softplus** (not ReLU) in regression head — correct for SDD≥0 constraint, not the cause of gradient issues.
- **Monitor** armed (task bucoe4n0g) polling VM every 30s for new training activity.

### Key decisions
- 252 °C·day labels + 50.8% MHW mask = physically valid, not bugs
- Fix is label normalization (÷250) + grad_clip→10.0, not architectural change
- spread>0 not guaranteed post-fix — depends on gate recalibration toward atm-dominant

### Next
- Implement: `LABEL_NORM=250.0` in `_train_utils.py`, `grad_clip_max_norm=10.0` in both training scripts
- Commit harvester.py dim-order fix (still uncommitted)
- SCP to VM, retrain ERA5 then WN2

---

## [2026-04-21] Session 25 — WN2 2023 zarr consolidated; label diagnosis; dim-order fix; WN2 training launched

### What happened
- **WN2 training PID 4296 crashed**: `wn2_2023 zarr` missing from GCS (only daily tiles present, annual not consolidated)
- **ERA5 PID 4050 was still running**: killed after confirming epoch 50/50 done (train=63k, val=38k, SVaR_95=1.34)
- **WN2 2023 consolidated**: ran `run_wn2_prep.py --year 2023` on mhw-training; 365 daily tiles → annual zarr with `.complete` sentinel written (22:08)
- **Reviewed train_wn2.py + _train_utils.py** before training: all params confirmed (lr=1e-4, epochs=50, seq_len=90, N_MEMBERS=64, grad_clip=1.0)
- **Diagnosed SDD label magnitude**: 252 °C·day confirmed real — 2022 GoM was record-warm year; 50% MHW mask fraction against 30yr OISST baseline is physically valid. No code bug.
- **WN2 dim-order bug found**: WN2 zarrs store `(time, member, lat, lon)` due to `expand_dims({"time":...})` in `_build_dataset()` placing time at dim 0; ERA5 goes through `expand_and_perturb` which corrects order. Fix: added `transpose("member","time","latitude","longitude")` in `harmonize()` after expand_and_perturb check. SCP'd to mhw-training (not committed).
- **WN2 training launched**: PID 17145, `~/train_wn2.log`, 22:41. stdout fully buffered — output only visible on process exit. Process healthy at turn 20 (230% CPU, 3.2GB RSS, 98 CPU-min).
- **Grad-clip concern**: 200× mismatch between label (~252) and model output (~1-2) means MSE gradient ≈ -500 → clipped every epoch → near-zero learning. Fix options: increase max_norm or normalize labels. Decision deferred until WN2 results reviewed.

---

## [2026-04-21] Session 24 — SDD label fix applied; ERA5 retrain done; WN2 training launched

### What happened
- **Verified HYCOM water_temp[depth=0]**: Gulf of Maine (lat 41-45°N, lon -71 to -66°W), 0-16°C range across seasons — physically correct. Not Gulf of Mexico.
- **Fixed SDD label bug** in `scripts/_train_utils.py:117`: replaced `merged["sea_surface_temperature"] - 273.15` (ERA5, Kelvin, sparse) with `merged["water_temp"].isel(depth=0)` (HYCOM surface, °C, full ocean grid). Updated docstrings. Added `test_build_tensors_label_uses_hycom_sst` — 4/4 tests pass.
- **Committed + pushed** `79853d7` — "fix: use HYCOM water_temp[depth=0] for SDD label instead of ERA5 SST"
- **ERA5 retrain (50 epochs)** on `mhw-training` (PID 4050, then restarted with PYTHONUNBUFFERED=1): train_loss 63474→63225, val_loss 38191→38104, SVaR_95 1.12→1.34, spread=0.00. Loss 2× higher than before (labels now real, not ~0). Only 0.4% improvement over 50 epochs — model barely moved.
- **Root cause of flat ERA5 loss**: model predicts ~1.3 °C·day; HYCOM SDD labels ~250 °C·day → 200× scale mismatch. MSE of (1.3-250)² ≈ 63k matches observed loss. Grad clip at norm=1.0 caps every update → near-zero learning.
- **Spread=0 accepted for ERA5**: HYCOM is broadcast identically to all 64 Gaussian-noise ERA5 members → label is identical per member → spread=0 by construction. Scientifically acceptable — ERA5 is a proxy; spread is a WN2 property.
- **Correction saved to memory**: Gaussian noise applied to ERA5 (single reanalysis → 64 synthetic members), NOT WN2 (genuine 64-member physical ensemble).
- **WN2 training launched** on `mhw-training` (PID 4296, `~/train_wn2.log`). Both 2022+2023 WN2 zarrs confirmed in GCS.
- **New high-priority bug flagged**: SDD labels ~250 °C·day is suspect (expected 10-100 °C·day for GoM). Hypothesis: `compute_mhw_mask` not aligning `dayofyear` threshold to calendar dates → all days exceed threshold → full-year SDD accumulation.
- **GPU quota**: request in-flight; retry in 2 days. `GPUS_ALL_REGIONS` quota, not regional.

### Key decisions
- ERA5 label scale bug (250 °C·day) is a separate issue from the SST-source fix — diagnose after WN2 training
- Label normalization and LR tuning deferred until SDD magnitude bug is diagnosed
- spread=0 is acceptable for ERA5 proxy; expect spread > 0 from WN2 genuine ensemble

### Next
- Check WN2 training log: `tail -5 ~/train_wn2.log` on `mhw-training`
- Diagnose SDD label magnitude: print `label_t.mean()/max()`, check `compute_mhw_mask` dayofyear alignment in `src/analytics/mhw_detection.py`
- After WN2 done + SDD bug fixed: retrain ERA5 with label normalization (divide by mean SDD, store scale in config)
- GPU quota retry in ~2 days

---

## [2026-04-21] Session 23 — ERA5 50-epoch run complete; SDD label bug discovered

### What happened
- **Verified all 5 GCS sentinels** green: hycom/tiles/2022, /2023, climatology, era5/2022, /2023
- **Cleaned up 6 idle VMs**: mhw-2022-m06/07/09/12, mhw-2023-m12, mhw-hycom2023-prep — all terminated
- **GPU quota = 0**: T4/L4 exhausted across all zones + project has GPUS_ALL_REGIONS=0. User submitted quota request (limit=2). Machine image `mhw-data-prep-img` created for future GPU VM.
- **Created `mhw-training`** (us-central1-c, e2-highmem-8, 64GB, CPU-only) from machine image — `mhw-data-prep` exhausted in us-central1-a
- **ERA5 50-epoch run completed** on `mhw-training` (PID 1232): train_loss 37079→36888, val_loss 20045→19912, SVaR_95 0.65→1.12, spread=0.00 throughout
- **Root cause investigation**: plots pulled locally. Two bugs found in `build_tensors()`:
  - **Bug 1 (critical)**: SDD label uses `merged["sea_surface_temperature"]` (ERA5, Kelvin, only 4×5 coverage). Interpolated to 17×21 grid → 337 cells have fill-value SST (-272°C). Those cells → SDD=0 via clip. ~50 valid cells have SDD=600-1200. Spatial mean ≈ 193. This explains 100× underprediction and is NOT a scale issue — it's the **wrong SST variable**.
  - **Bug 2 (consequence)**: spread=0 because all 64 members see the same garbage-dominated spatial average.
- **Fix identified**: use `merged["water_temp"].isel(depth=0)` (HYCOM surface, Celsius, full GoM coverage) instead of ERA5 SST for label computation. One-line change in `_train_utils.py:117`.
- **50-epoch run results are invalid** — discard, retrain after fix.

### Key decisions
- HYCOM water_temp[depth=0] is the correct SST for SDD label — ERA5 SST lacks ocean grid coverage
- mhw-training VM (us-central1-c) is the active training machine — keep running, reuse for retrain
- Do NOT add label normalization or LR changes until HYCOM SST fix is verified — scale issue was a symptom, not the disease

### Next
- Verify HYCOM water_temp[depth=0] sample values (confirm ~15-25°C on VM)
- Fix `scripts/_train_utils.py` line ~117: swap SST source
- Retrain ERA5 50 epochs, verify spread > 0 and SVaR_95 > SVaR_50 > SVaR_05

---

## [2026-04-20] Session 22 — B2 fixed + done; ERA5 training launched on mhw-data-prep VM

### What happened
- **B2 ROOT CAUSE**: Process died at day 113/365 in `_build_dataset()` — `_gcs_safe_write` for annual store was never called. Not a silent failure; incomplete run.
- **B2 FIX** (`harvester.py:fetch_and_cache()`): added encoding pop + rechunk({"time":30}) after `_build_dataset()` returns, before `_gcs_safe_write` — same fix as HYCOM annual tile. Mirrors lesson `[2026-04-19]`.
- **B2 DONE**: Re-ran `run_wn2_prep.py --year 2022` locally. Cache hits for days 1-113, fetched 114-365, annual Zarr written at 19:20 local: `gs://mhw-risk-cache/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr`
- **All 5 sentinels verified green**: hycom/tiles/2022, hycom/tiles/2023, hycom/climatology, era5/2022, era5/2023 — all `.complete` present
- **ERA5 training launched on mhw-data-prep VM** (PID 28544, `~/train_era5.log`): `train_era5.py --epochs 50`
- **Local training killed** (PID 32789): was running locally on WSL2 → IO-bound over home internet → 0.5% CPU, hung in Dask GCS download. Killed and moved to VM.
- **VM training status**: still in data-load phase as of session end (Dask PerformanceWarning lines only, no epoch output yet). User disconnected with Ctrl+D (not Ctrl+B D) — process should survive as `nohup + disown`.

### Key decisions
- Always run training on mhw-data-prep (co-located with GCS) — local WSL2 is IO-bound
- WN2 annual write needs encoding pop + rechunk same as HYCOM (both concat zarr-backed daily/monthly stores)

### Next
- SSH back to mhw-data-prep, verify PID 28544 alive, check `~/train_era5.log` for epoch output
- After 50 epochs: review loss curves + SVaR metrics, then decide on LinkedIn post

---

## [2026-04-20] Session 21 — A3 fixed (ERDDAP); WN2 zarr silent-write bug discovered

### What happened
- **A3 ROOT CAUSE FOUND**: two bugs in `fetch_oisst_climatology.py`:
  1. 10,800 per-day NCEI global downloads (1.7 MB each) → NCEI rate-limits after N requests → returns HTML with status 200 → HDF error on tempfile
  2. OISST native lon is 0-360; `sel(lon=slice(-71,-66))` returned empty data silently
- **A3 FIX** (commit `ed131ee`): switched to ERDDAP griddap `ncdcOisst21Agg_LonPM180` — 30 annual requests × ~500 KB GoM subset; server-side spatial filter; -180/180 lon convention; Content-Type guard
- **A3 DONE**: climatology written to `gs://mhw-risk-cache/hycom/climatology/` at 23:13
- **B2 INVESTIGATION**: WN2 log shows 113/365 days "fetched" for 2022, but `gsutil ls -r` on zarr path returns NO objects — zarr write is failing silently. `gcsfs.ls()` shows HNS directory nodes (not real data files). 2022 + 2023 zarr stores have zero actual chunks on GCS.
- **B2 STATUS**: NOT DONE — need to diagnose and fix silent zarr write failure, then re-run

### Key decisions
- ERDDAP griddap is the correct pattern for all NCEI OISST access — never use per-day direct HTTPS
- `gsutil ls -r` is authoritative for GCS content; `gcsfs.ls()` can return false positives on HNS buckets

## [2026-04-20] Session 20 — OISST fetch debugging; WN2 resume; 3 URL/engine fixes

### What happened
- **Diagnosed overnight failures**: OISST died (WSL shutdown, ~20% done); WN2 2022+2023 died (gRPC [Errno 11] from parallel GEE sessions, 45/365 days each with SST)
- **HYCOM 2022+2023**: confirmed `.complete` sentinels present — those are DONE
- **WN2 resumed**: launched sequentially in tmux `wn2` (2022→2023); cache hits for days 1-45, fetching new from day 46
- **OISST fix 1** (commit `b19c1bf`): parallelized month fetch with `ThreadPoolExecutor(max_workers=6)` → ~6x speedup (~1.5hr)
- **OISST fix 2** (commit `7217a55`): THREDDS OPeNDAP URL returns 400 (moved); updated to direct HTTPS path
- **OISST fix 3** (commit `9499c64`): `netcdf4` engine treats `https://` as OPeNDAP (parser error); switched to `requests.get` + temp file
- **OISST fix 4** (commit `e7116ed`): `netcdf4` backend rejects BytesIO; switched to `tempfile.NamedTemporaryFile` + `os.unlink`
- Tests updated for new mock pattern (requests.get + open_dataset); 6/6 pass
- OISST now running on VM (PID 21120 or successor after re-launches); all months fetching correctly

### Key decisions
- Run WN2 sequentially (not parallel) — parallel GEE sessions cause gRPC resource exhaustion
- Use tempfile (not BytesIO) for OISST daily NC files — most compatible with netcdf4 engine

## [2026-04-20] Session 19 — B1 code done; A3+B2 launched; LinkedIn drafts written

### What happened
- **B1 DONE** (commit `bf10412`): `sea_surface_temperature` re-added to `WN2_VARIABLES`; `arr[arr==0.0]=np.nan` land mask added in `_build_dataset()` inner loop; 2 new tests; 78/78 pass
- **run_wn2_prep.py** updated with `--year` flag (commit `969bd06`) to enable parallel per-year VM execution
- **A3 LAUNCHED**: deleted old GCS climatology, OISST 1982–2011 fetch running locally (`~/nohup_oisst.log`), ETA 30-90 min
- **B2 LAUNCHED**: deleted all WN2 2022+2023 GCS cache (12.2k objects), both years re-fetching in parallel locally (`~/nohup_wn2_2022.log`, `~/nohup_wn2_2023.log`), ETA ~55 min each; confirmed progress at day 9/365 each
- **LinkedIn drafts written** (commit `84d0ff6`): `docs/linkedin/2026-04-20-mhw-era5-draft-a.md` (short) + `draft-b.md` (medium), both with `[INSERT]` placeholders for B3 results

### Key decisions
- Confirmed Gaussian noise does NOT bias IG attributions — ERA5 XAI results are scientifically valid and postable
- SVaR spread limitation (synthetic ensemble) is the caveat, not attribution validity
- Lead post visual: SVaR map, emphasized visual: IG bar chart; promise WN2 follow-up

### Next
- Wait for A3 + B2 to complete, then B3: `train_era5.py --epochs 50` + `train_wn2.py --epochs 50`
- Fill in LinkedIn draft placeholders with B3 results; review with user before posting

---

## [2026-04-20] Session 18 — OISST 30yr climatology code complete; plan written for parallel WN2+OISST execution

### What happened

**Gemini critique evaluated:**
- Reviewed `mhw_gemini_scientific_critique_v1.md` (session 17 Gemini findings)
- Cache poisoning (#1): REAL blocker — `_daily/` dirs must be deleted before WN2 re-fetch
- Land mask inconsistency (#2): NOT a blocker — xarray `skipna=True` handles NaN land pixels in spatial mean naturally; no code change needed
- Unit mismatch (#3): NOT a blocker — `_train_utils.py:117` already converts K→°C before SDD

**Plan written:**
- `docs/superpowers/plans/2026-04-20-oisst-climatology-and-wn2-sst-fix.md`
- Two parallel tracks: Track A (OISST 30yr climatology) + Track B (WN2 SST fix)
- P5 (OISST) moved up: runs in parallel with WN2 re-fetch, not after

**Track A code — COMPLETE (2 commits):**
- `compute_climatology()` in `src/analytics/mhw_detection.py`: added `window=11` param + scipy `uniform_filter1d` with `mode="wrap"` (circular, no boundary NaN). commit `019bdb3`
- `scripts/fetch_oisst_climatology.py` created: fetches OISST v2.1 1982-2011 for GoM bbox via NCEI THREDDS OPeNDAP month-by-month, computes 90th-pct threshold with 11-day window, writes `sst_threshold_90` to GCS. commit `d6b0584`
- 76/76 tests passing (6 new OISST tests + 2 rolling window tests)

**Track B code — NOT YET STARTED:**
- Next session start here: add `sea_surface_temperature` back to `WN2_VARIABLES` in `harvester.py:177`, add NaN mask (`arr[arr == 0.0] = np.nan`) in `_build_dataset()` inner loop ~line 548

### Key decisions
- `window=1` disables smoothing in `compute_climatology()` (backward compatible)
- THREDDS URL needs curl verification on VM before running A3 (documented in plan)
- ERA5 must be retrained after new 30yr threshold is in GCS (SDD labels change)
- Delete `_daily/` dirs explicitly before WN2 re-fetch (cache poisoning fix)

### State at session 18 end
- Local: 76 tests passing; commits `019bdb3`, `d6b0584` on main; not pushed yet
- GCS: old 2yr HYCOM climatology still in place (not deleted yet — VM step A3)
- Next: push commits, then Track B code (B1), then VM work (A3 + B2 in parallel)

---

## [2026-04-20] Session 17 — ERA5 results reviewed, WN2 SST bug diagnosed

### What happened
- Reviewed ERA5 training log (GCS): train 6111→6043, val 2149→2100, spread=0.00 throughout
- Diagnosed spread=0 root cause fully:
  - `WN2_VARIABLES` excludes SST (harvester.py line 182); only atmospheric vars fetched from WN2
  - ERA5 SST IS fetched (era5_harvester.py) and IS noised (NOISE_SIGMAS σ=0.5K), but SST crosses HYCOM threshold on same days for all members → noise never flips MHW boundary → spread=0
  - This is an accepted limitation: ERA5 = deterministic reanalysis, synthetic noise cannot produce genuine ensemble spread
- Diagnosed WN2 training latent crash: WN2 GCS tiles have NO sea_surface_temperature (confirmed gsutil ls); `build_tensors()` line 117 calls `merged["sea_surface_temperature"]` → KeyError
- Decision: add SST back to WN2_VARIABLES + land mask (defaultValue=0→NaN) + re-fetch WN2 2022/2023 tiles
- ERA5 model valid for XAI (IG attribution on deterministic SST→SDD mapping); spread/SVaR story requires WN2

### State at session 17 end
- No code changes this session (plan defined, execution deferred to next session)
- mhw-data-prep VM: may still be running — check/stop before next data prep task
- Next: implement plan (5 steps) at top of todo.md

---

## [2026-04-20] Session 16 — 3 OOM bugs fixed, ERA5 training complete (50 epochs)

### Context
Resumed from Session 15. Training had OOM-killed twice (total-vm 58 GB, anon-rss 52 GB) before reaching epoch 1. Diagnosed and fixed 3 separate OOM-causing bugs.

### Bug 1 — harmonize() global grid interp (commit 87adafb)
- `harmonize()` interpolated ERA5 (4×5 GoM tile) to global `TARGET_LAT` (721) × `TARGET_LON` (1440) grids
- After `expand_and_perturb(64 members)`: 64 × 365 × 721 × 1440 × 5 vars × 4 bytes ≈ 485 GB → OOM
- Fix: compute union bbox from input data inside `harmonize()`, filter TARGET_LAT/LON to bbox before interp
- After fix: 64 × 365 × 17 × 21 × 5 vars × 4 bytes ≈ 165 MB ✓

### Bug 2 — threshold coord name mismatch → outer-product (commit 2eb89c9)
- HYCOM climatology saved with `lat`/`lon` dims (101×63 native grid)
- `merged` ERA5 uses `latitude`/`longitude` (17×21 target grid)
- `compute_mhw_mask(sst, threshold)` → xarray outer-products (17×21) × (101×63) → ~15 GB array → OOM
- Fix: rename `lat`→`latitude`, `lon`→`longitude` then interp to `merged.latitude.values` in `build_tensors()` before calling `compute_mhw_mask`

### Bug 3 — interp condition with mismatched-size DataArrays (commit 814ee9d)
- The condition `(threshold_regrid.latitude != merged.latitude).any()` compared 101-element vs 17-element arrays
- Could produce incorrect result (remaining latitude=5 mismatch in mask_np shape)
- Fix: unconditionally interp with `.values` (plain numpy) instead of DataArray

### Training completed
- All 68 tests passing throughout
- Training ran 50 epochs: train 6111→6043, val 2149→2100
- `spread=0.00` throughout: Gaussian noise in ERA5 vars doesn't differentiate SDD per member (known limitation)
- SVaR_95=SVaR_50=SVaR_05 (degenerate spread, expected)
- Weights saved: `data/models/era5_weights.pt`, `era5_best_weights.pt`
- Plots saved: `data/results/plots/era5_*.png` (5 files)
- All artifacts backed up to `gs://mhw-risk-cache/era5/training_results/`

### State at session 16 end (~01:30 UTC 2026-04-20)
- mhw-data-prep (n1-highmem-8): training complete, VM still running
- GCS: all training artifacts backed up
- Next: review results with user, decide on spread=0.00 / SDD scale issues before LinkedIn post

---

## [2026-04-19] Session 15 — Data prep complete, ERA5 training launched on upgraded VM

### Context
Resumed from Session 14. All HYCOM data still in flight. Worked through 4 recovery steps and got training running.

### Actions taken

**Step 1: 2023 .complete sentinels** — all 12 monthly tiles had data but no sentinel. Used `gsutil cp /dev/null` (gsutil touch not supported) to write all 12.

**Step 2: 2022 monthly verification** — all 12 .complete present (dedicated VMs completed overnight).

**Step 3: mhw-hycom2023-prep restarted** — found 2023 annual already complete (cache hit). Logged "HYCOM 2023 complete."

**Step 4: mhw-data-prep restarted** — stale code on VM was same commit as local (2305080). Previous crashes were from before fix was deployed. New run: all 12 monthly cache hits → annual assembly succeeded (17:47 UTC) → climatology computed → ERA5/WN2 cache hits → "Data prep complete."

**Training cron installed** — `~/launch_training_when_ready.sh` polls `climatology/.complete` every 5 min, self-removes after trigger. Fired at 18:00 UTC.

**OOM on e2-standard-4 (16GB)** — `load_real_data()` materializes ~13GB; process OOM-killed twice. Upgraded VM to `n1-highmem-8` (52GB). Training restarted: PID 987, 20.7% mem (~11GB), healthy headroom.

**[P9] added to todo** — lazy/chunked data loading fix for `load_real_data()` in train_era5.py + train_wn2.py.

### State at session 15 end (~21:00 UTC 2026-04-19)
- All 5 GCS sentinels: COMPLETE
- mhw-data-prep (n1-highmem-8): ERA5 training running, PID 987
- Training log: `~/nohup_train_era5.log` on mhw-data-prep

---

## [2026-04-19] Session 14 — Concurrent-write catastrophe diagnosed, Option A recovery, rechunk bug fixed

### Context
Resumed from Session 13. Found multi-VM chaos: mhw-data-prep (sequential m01→m12) AND dedicated month VMs were writing to the same zarr paths concurrently. mhw-hycom2023-prep still crashed. Three separate bugs encountered and fixed.

### Catastrophe: concurrent writes across VMs (April 19)
- Session 13 left mhw-data-prep running sequentially AND 9 dedicated VMs running in parallel
- Both mhw-data-prep and mhw-2022-m01 (and others) were writing to same GCS zarr paths simultaneously
- mhw-2022-m06 and mhw-2022-m07 VMs were BOTH fetching months 7-9 concurrently (double concurrent write)
- Neither VM had any Python process alive — both crashed from mutual interference
- Root cause: no coordination between mhw-data-prep sequential logic and dedicated per-month VMs
- **Lesson: never run mhw-data-prep sequential AND dedicated month VMs simultaneously**

### Recovery (Option A):
- Killed mhw-data-prep PID 937 (was re-fetching m01 sequentially)
- Restarted mhw-2022-m06 → m06 only, mhw-2022-m07 → m07 only
- Started mhw-data-prep → m09 then m12 sequentially (only uncovered months)
- Touched 2023 monthly .complete sentinels (all 12) via `gsutil cp /dev/null`
- Restarted mhw-hycom2023-prep for 2023 annual assembly

### Bug fixed: mhw-hycom2023-prep recurring crash (commit `2305080`)
- After .complete sentinel fix, 2023-prep found all 12 cache hits and proceeded to annual concat
- Crashed: `ValueError: Zarr requires uniform chunk sizes except for final chunk` on `water_temp`
- Root cause: `xr.concat` of 12 monthly zarrs produces irregular dask chunks along time axis (each month has per-month zarr chunk spec; 28-31 days/month = non-uniform chunk sizes)
- Previous fix (pop `encoding["chunks"]`) cleared metadata mismatch but dask chunk array itself remained irregular
- Fix: `ds_annual = ds_annual.chunk({"time": 30})` after encoding pop, before `_gcs_safe_write`
- Pushed, pulled on mhw-hycom2023-prep and mhw-data-prep; 2023-prep restarted and running (108% CPU)

### State at session 14 (~04:10 UTC 2026-04-19 = 21:10 PDT 2026-04-18)
- 2022 monthly m01-m11: all dedicated VMs running, started ~00:00 UTC, 4+ hrs in, OPeNDAP still downloading
- 2022 m09: mhw-data-prep fetching (started 00:40 UTC); m12 queued after m09
- 2022 annual assembly: NOT started — need to trigger manually after all 12 .complete present
- 2023 annual assembly: mhw-hycom2023-prep running with rechunk fix (started 04:10 UTC)
- Training: blocked on 2022 annual + climatology + 2023 annual

### Lesson applied
- `[2026-04-19]` xr.concat of N monthly zarrs → irregular dask time chunks even after encoding pop → rechunk({"time": 30}) required before annual write

---

## [2026-04-18] Session 13 — Two harvester bugs fixed, 9 dedicated VMs spun up, 2022 re-fetch in progress

### Context
Resumed from Session 12. mhw-data-prep had crashed on 2022 annual assembly due to two bugs. All 2022 monthly tiles deleted by crash.

### Bugs fixed (commit `633d0c8`, pushed, pulled on all VMs)

**Bug A — `_gcs_safe_write` recursive delete wiped 2022 monthly tiles:**
- HNS bucket: `fs.exists("hycom/tiles/2022/")` = True (parent dir of monthly tiles)
- `_gcs_safe_write` called `fs.rm(path, recursive=True)` before annual write → deleted all m01-m12
- Fix: added `preserve_dirs=("monthly",)` parameter + `_clear_store()` inner helper that skips listed subdirs
- Annual write now passes `preserve_dirs=("monthly",)`

**Bug B — xr.concat encoding misalignment ValueError:**
- Monthly zarrs carry `encoding["chunks"]` that don't align with Dask chunks of concatenated dataset
- Fix: `ds_annual[var].encoding.pop("chunks", None)` for all data vars before `_gcs_safe_write`

### Recovery actions
- Deleted partial 2022 annual zarr from GCS (`zarr.json` + `time/` only)
- Restarted `mhw-data-prep` (PID 937) with `run_data_prep.py` (handles all 2022 months + annual + climatology)
- Restarted `mhw-hycom2023-prep` (2023 m11+m12 + annual assembly)
- Spun up 9 dedicated per-month VMs (`mhw-2022-m01/02/03/04/05/06/08/10/11`) — each self-terminates after job
- m07, m09, m12 never got dedicated VMs (GCP quota/rate limits); covered by `mhw-data-prep` sequential
- Set up CronCreate job `aeb30b31` polling every 30min for completion → PushNotification

### New issue discovered at session end
- 2023 monthly tile directories exist in GCS (m01-m12, all vars + zarr.json) but NO `.complete` sentinels
- Old VMs (pre-fix code) wrote data but didn't touch `.complete`
- `mhw-hycom2023-prep` CRASHED at 22:48 UTC with asyncio loop error
- **Action needed next session:** `gsutil touch` all 12 2023 monthly sentinels, restart `mhw-hycom2023-prep`

### State at session end (~01:30 UTC 2026-04-19)
- 2022 monthly: 9 dedicated VMs fetching (m01-m06/m08/m10/m11); m07/m09/m12 via mhw-data-prep sequential; ~0 .complete yet
- 2023 monthly: data present, .complete MISSING for all 12 months
- mhw-hycom2023-prep: CRASHED, needs restart after 2023 sentinels fixed
- mhw-data-prep: RUNNING (PID 937), fetching 2022 m01
- CronCreate `aeb30b31`: active, will PushNotify on completion

---

## [2026-04-18] Session 12 — P1 resolved, VM monitoring, training deferred to next session

### Actions
- Checked all 8 VM logs and confirmed processes alive via `pgrep` (all mid-fetch, no stalls)
- Resolved P1 blocker: confirmed `sea_surface_temperature` present in `ECMWF/ERA5/HOURLY` via GEE service account auth; removed TODO comment from `era5_harvester.py:46`
- Confirmed WN2 fully complete: both hourly zarrs have `.complete`, both daily zarrs have all 365 day-subdirs (d20220101–d20221231, d20230101–d20231231)
- Revised training ETA to ~05:30 UTC — user asleep by then; training deferred to next session

### State at session end (04:13 UTC)
- 8 VMs still running; most fetching ~1.5–2h in; m12 outlier at 3h (alive, just slow)
- All 3 HYCOM `.complete` sentinels still pending; ERA5 + WN2 sentinels confirmed
- **Next session: verify sentinels, then launch `train_era5.py --epochs 50`**

---

## [2026-04-18] Session 11 — Two crash bugs fixed, 3 new VMs, training ETA cut by 4h

### Context
Resumed after /clear. 6 VMs running HYCOM data prep. Checked all logs and found crashes.

### Crashes found and root-caused
`mhw-data-prep` and `mhw-hycom2022-m7-9` both crashed with `zarr.errors.ContainsArrayError`
while trying to write HYCOM 2022 m07 concurrently.

**Bug 1 — `_gcs_complete` always returned False (`harvester.py` line 103):**
zarr v3 never writes `.zmetadata` (which was the sentinel). Confirmed: no `.zmetadata` exists
on any completed HYCOM month (m01-m10). Idempotency guard was silently dead since zarr v3
was introduced. Every VM restart re-fetched all months from scratch.

**Bug 2 — TOCTOU race in `_gcs_safe_write` (`harvester.py`):**
Two VMs both checked `_gcs_complete` → False (because Bug 1), both fetched m07 for 2h,
both called `_gcs_safe_write`. First writer succeeded. Second writer hit `mode="a"` with
existing arrays → `ContainsArrayError`. Also found same broken check inlined in
`era5_harvester.py::fetch_and_cache` (checking `.zmetadata` directly).

### Fixes applied (commit `633d0c8`)

**`harvester.py`:**
- Added `import zarr.errors`
- `_gcs_complete`: now checks `.complete` sentinel first, then falls back to
  `water_v/zarr.json` (covers all HYCOM tiles written before this fix)
- `_gcs_safe_write`: catches `ContainsArrayError` → if store now complete, return;
  else delete and retry once. Writes `.complete` sentinel after successful write.

**`era5_harvester.py`:**
- Imported `_gcs_complete` from `harvester.py`
- Replaced inline `.zmetadata` check with `_gcs_complete(fs, gcs_uri)`

### GCS remediation
- Deleted partial m07 (`lat/`, `salinity/`, `water_temp/`, `water_u/` only — missing `water_v/`, `time/`, `depth/`, `lon/`)
- Wrote `.complete` sentinels for: WN2 2022+2023, ERA5 2022+2023
- HYCOM monthly tiles (m01-m06, m10-m11 for 2022; m05, m09 for 2023) covered by `water_v/zarr.json` fallback

### VM actions
- Both crashed VMs (`mhw-data-prep`, `mhw-hycom2022-m7-9`) restarted with new code
- `mhw-data-prep` confirmed cache-hitting m01-m06 immediately after restart ✓
- `mhw-hycom2022-m7-9` stopped after confirming it was now redundant (m08+m09 covered by dedicated VMs)
- `mhw-era5-prep` stopped (ERA5 complete, VM idle)
- 3 new VMs created from snapshot `mhw-hycom-worker-snap` (from `mhw-hycom2022-m7-9` disk):
  - `mhw-hycom2022-m8` — HYCOM 2022 m08 only, fetching since 02:37 UTC
  - `mhw-hycom2022-m9` — HYCOM 2022 m09 only, fetching since ~02:45 UTC
  - `mhw-hycom2023-m4` — HYCOM 2023 m04 only, fetching since ~02:45 UTC
- Hit 32/32 CPU quota after first new VM → stopped 2 VMs to free 8 CPUs, then created 2 more

### Training ETA improvement
- Before: ~09:15 UTC (2022 annual blocked on m07→m08→m09 sequential)
- After: ~05:15 UTC (m08+m09 written in parallel, mhw-data-prep cache-hits both)
- Net gain: ~4 hours

### P2 resolved
HYCOM monthly tile paths confirmed correct in production (no double-slash keys).

---

## [2026-04-17] Session 10 — Code Review Fixes, Gemini Evaluation, Pipeline Monitoring

### Context
6 parallel HYCOM VMs running. WN2 + ERA5 tiles complete. Enacting pre-/clear protocol.

### VM Events
- `mhw-hycom2023-prep` PID 983 hung on 2023-m01 for 1h+ (3 log lines, OPeNDAP stall). Killed + restarted as PID 2186 at 20:09 UTC. Fresh logs confirmed.
- `mhw-wn2-prep` stopped — WN2 2022 + 2023 GCS tiles confirmed complete. VM can be deleted.
- All 5 HYCOM VMs alive as of ~20:45 UTC, fetching first months, ETAs ~22:00–22:40 UTC.

### Code Fixes Applied (code-reviewer subagent findings)

**`src/ingestion/harvester.py`**
- Fixed `_fetch_and_write_zarr` (line 571): `_build_dataset` call was missing `gcs_uri` 4th arg added in a prior refactor → would raise `TypeError` on `fetch_ensemble()`. Passed `gcs_uri` to fix.

**`scripts/run_hycom_months_prep.py`**
- Fixed double-slash bug: `annual_base` had trailing `/` → `month_uri` produced `//monthly/` → GCS key mismatch broke idempotency check with `HYCOMLoader.fetch_and_cache`. Fixed by stripping trailing slash from `annual_base` and using explicit `/` in `month_uri`.
- Added `GOOGLE_APPLICATION_CREDENTIALS` startup validation check.
- Added non-overlap comment warning for parallel VM month ranges.

**`src/ingestion/era5_harvester.py`**
- Added ERA5 HOURLY coverage count check: raises `ValueError` if `n_images < days_in_year * 24` (prevents silent partial-year corruption).
- Added `TODO` comment on `sea_surface_temperature` band availability in `ECMWF/ERA5/HOURLY` — must verify via `ee.ImageCollection("ECMWF/ERA5/HOURLY").first().bandNames().getInfo()` before training.
- Fixed docstring: `ECMWF/ERA5/DAILY` → `ECMWF/ERA5/HOURLY`.

**False positive (NOT fixed):** Code reviewer flagged ERA5 off-by-one in date loop (`while d < end_d`). Analysis confirmed correct: `end_d = 2023-01-01` and `d < end_d` correctly includes 2022-12-31. No change.

### Gemini Session 9 Evaluation

| Finding | Action |
|---------|--------|
| 2-yr climatology baseline (Critical scientific risk) | Acknowledged. Caveat added to todo + training section. Long-term fix: NOAA OISST v2.1 ≥30yr (P5). Does not block training. |
| Parallel GCS write race conditions | Addressed via non-overlap comment. Per-VM ranges confirmed non-overlapping. Idempotency is correct via `.zmetadata`. |
| `.zmetadata`-only idempotency check | Intentional design — `to_zarr(consolidated=True)` writes `.zmetadata` last, making it a reliable sentinel. |
| payout.py missing | Added as P3 in todo queue (post ERA5 training). |
| CF-1.8 compliance audit | Added as P4 in todo queue. |

### Config Changes
- Global `~/.claude/CLAUDE.md`: added caveman default + /clear-every-20-turns recommendation.
- Project `CLAUDE.md`: added Pre-/clear Protocol section (update todo + recentactions before /clear).

### Priority Queue Written to todo.md
- P1: Verify SST band in ECMWF/ERA5/HOURLY (blocks training correctness)
- P2: Spot-check GCS monthly tile paths (confirms double-slash fix in production)
- P3: Build `payout.py` parametric payout engine
- P4: CF-1.8 compliance audit of `harmonize()` output
- P5: NOAA OISST v2.1 ≥30yr climatology (primary scientific validity fix)
- P6: XAI Option C — member-level attribution variance
- P7: MTSFT — FFT-enriched Transformer
- P8: Vertex AI migration

---

## [2026-04-17] Session 9 — Gemini Pipeline Audit (FOR CLAUDE EVALUATION)

### Context
Gemini performed a multi-agent audit of the GCP data prep pipeline using specialized `@reviewer` and `@scientist` personas.

### Action Required for Claude
- Evaluate the findings in `mhw_gemini_actions/mhw_gemini_recentactions.md` (Session 9).
- **Critical**: Address the 2-year vs. 30-year climatology baseline risk flagged by `@scientist`.
- **Engineering**: Address the parallel GCS write race conditions flagged by `@reviewer` in `scripts/run_hycom_months_prep.py`.
- **Constraint**: Gemini was restricted to read-only mode; implementation of fixes is deferred to Claude.

---

## [2026-04-17] Session 8 — WN2 complete, ERA5 complete, HYCOM parallelized across 6 VMs

### Context
Continuing from Session 7. WN2 was crashing mid-run; ERA5 collection was stale (ended 2020).
This session: fixed both, launched parallel HYCOM month VMs, confirmed WN2+ERA5 complete.

### WN2 per-day crash-safe fix — COMPLETE
- Root cause: no intermediate saves → any crash forced restart from day 1.
- Fix in `src/ingestion/harvester.py` `WeatherNext2Harvester._build_dataset()`:
  - Per-day GCS Zarr writes to `{gcs_uri}_daily/d{YYYYMMDD}/` before accumulating.
  - `_gcs_complete()` check at loop start → crash-safe resume (cache hits for saved days).
  - Retries: 3 → 5 attempts; fixed 60s → exponential backoff (60, 120, 240, 480, 960s).
- Commit: `e5f6e0a`

### WN2 — BOTH YEARS COMPLETE
- 2022: `gs://mhw-risk-cache/weathernext2/cache/wn2_2022-01-01_2022-12-31_m64.zarr`
- 2023: `gs://mhw-risk-cache/weathernext2/cache/wn2_2023-01-01_2023-12-31_m64.zarr`
- Completed: 2026-04-17 ~19:42 UTC on VM `mhw-wn2-prep`

### ERA5 collection fix (ECMWF/ERA5/DAILY → HOURLY) — COMPLETE
- Root cause: `ECMWF/ERA5/DAILY` ends 2020-07-09 and is no longer updated.
- Fix in `src/ingestion/era5_harvester.py`:
  - `GEE_COLLECTION = "ECMWF/ERA5/HOURLY"` (covers 1940–present)
  - `ERA5_BANDS["temperature_2m"]` key: `"2m_temperature"` (was `"mean_2m_air_temperature"`)
  - Added `from datetime import date as _date, timedelta`
  - Rewrote fetch loop: per-day `.filterDate(date_str, next_date_str).mean()` — 24 hourly
    images averaged server-side → physically equivalent to what ERA5/DAILY computed.
  - Added `logger.info("ERA5 fetched %s (%d/%d)", ...)` per-day progress logging.
- `scripts/run_era5_prep.py`: added explicit `era5.authenticate()` call before year loop.
- Commit: `056bc47`

### ERA5 — BOTH YEARS COMPLETE
- 2022: `gs://mhw-risk-cache/era5/2022/.zmetadata`
- 2023: `gs://mhw-risk-cache/era5/2023/.zmetadata`
- Completed: 2026-04-17 ~19:17 UTC on VM `mhw-era5-prep`

### HYCOM parallelization — 4 new VMs launched
- Added `scripts/run_hycom_months_prep.py`: fetches a configurable month range for any year.
  - Reads `HYCOM_YEAR`, `HYCOM_START_MONTH`, `HYCOM_END_MONTH` from env.
  - Calls `fetch_tile()` + `_gcs_safe_write()` directly → per-month idempotent GCS writes.
  - Does NOT assemble annual tile (left to sequential VM's `fetch_and_cache()`).
- Commit: `4d37b06`
- Four new VMs created from `mhw-wn2-snap`, all RUNNING as of session end:

| VM | HYCOM_YEAR | START_MONTH | END_MONTH | First month started |
|----|-----------|-------------|-----------|---------------------|
| `mhw-hycom2022-m7-9` | 2022 | 7 | 9 | m07 @ 19:36 |
| `mhw-hycom2022-m10-12` | 2022 | 10 | 12 | m10 @ 19:36 |
| `mhw-hycom2023-m5-8` | 2023 | 5 | 8 | m05 @ 19:36 |
| `mhw-hycom2023-m9-12` | 2023 | 9 | 12 | m09 @ 19:36 |

- Sequential VMs `mhw-data-prep` (2022) and `mhw-hycom2023-prep` (2023) continue running.
  They will assemble the annual Zarr tiles once all months are cached.

### HYCOM progress as of session end
- 2022: months 1-6 complete; months 7-9 and 10-12 now fetching in parallel.
  - m07 overlap between `mhw-data-prep` and `mhw-hycom2022-m7-9` is benign (idempotent write).
- 2023: month 1 in progress on `mhw-hycom2023-prep`; months 5-8 and 9-12 on parallel VMs.

### New scripts committed this session
- `scripts/run_hycom_months_prep.py` (commit `4d37b06`)
- `scripts/run_era5_prep.py` (commits `8e38f0d`, `e0bb541`, `6e37e54`)
- `scripts/run_hycom2023_prep.py` (commit `a890d49`)

---

## [2026-04-17] Session 7 — WN2 infrastructure complete, second VM running, both data preps active

### Context
Continuing from Session 6. All code work completed this session; both VMs launched and running.

### WN2 infrastructure (Tasks W1 + W2) — COMPLETE
- `WeatherNext2Harvester.fetch_and_cache()` and `_build_dataset()` implemented in `harvester.py`
- `run_data_prep.py` extended with WN2 steps 6 & 7
- `run_wn2_prep.py` written as standalone script for parallel VM execution
- 68/68 tests passing (21 new: `test_wn2_harvester.py` + updated `test_harvester_cache.py`)

### zarr v3 + gcsfs bug fix — `_gcs_safe_write()`
- Root cause: zarr v3 calls `delete_dir()` unconditionally in `mode="w"` even on
  non-existent GCS paths; gcsfs raises OSError 404.
- Fix: `_gcs_safe_write()` helper in `harvester.py` — clears with `fs.exists()`/`fs.rm()`
  then writes with `mode="a"` (no `delete_dir` call). Used by all GCS write paths.
- ERA5 and WN2 tests updated to mock `_gcs_safe_write` directly.

### WN2 SST bug found and fixed
- WN2 `sea_surface_temperature` band is land-masked — ~25% of GoM bbox pixels return
  `defaultValue=0` (0 K, physically impossible). Diagnosed via `reduceRegion` vs
  `sampleRectangle` comparison: real SST range 275–291 K over ocean only.
- Fix: removed `sea_surface_temperature` from `WN2_VARIABLES`. WN2 now provides 4
  atmospheric variables only (2m_temp, u/v wind, MSLP). HYCOM remains authoritative SST.
- `NOISE_SIGMAS` SST entry retained — still valid for ERA5 `expand_and_perturb()`.

### WN2 smoke test — PASSED
- `scripts/wn2_smoke_test.py`: 3-day fetch (Jan 1-3 2022), 64 members, GCS write + read-back.
- Output confirmed: `(member=64, time=3, latitude=17, longitude=21)`, 4 variables, no SST.
- ~9 sec/day fetch rate → ~55 min/year for full 365-day run.

### Second VM: `mhw-wn2-prep`
- Snapshot `mhw-wn2-snap` taken from `mhw-data-prep` boot disk (conda env + credentials preserved).
- VM created: e2-standard-4, us-central1-a, on-demand, from snapshot.
- `run_wn2_prep.py` running at PID 964, logging to `~/nohup_wn2_prep.log`.
- GEE confirmed: 23,360 images found for 2022 (365 days × 64 members), fetching in progress.

### run_data_prep.py job restart (mhw-data-prep)
- Previous launch used `conda run` which doesn't survive SSH detachment.
- Restarted with `nohup env ... /path/to/python run_data_prep.py >> log 2>&1 </dev/null & disown $!`
- PID 4883, confirmed alive, fetching HYCOM 2022-01 as of session end.

### LinkedIn post decision
- Post will use real ERA5 + HYCOM results only. Synthetic ensemble noise on ERA5 inputs
  is acceptable for training; "no synthetic data" rule applies to published results.

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
