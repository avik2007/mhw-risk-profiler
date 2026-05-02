# mhw_claude_lessons.md
# Root-cause fixes and non-obvious discoveries
# ---------------------------------------------
# Format: [YYYY-MM-DD] <lesson> | Why it matters

---

## [2026-03-27] HYCOM OPeNDAP & WeatherNext 2 GEE Integration

- WeatherNext 2 correct GEE asset path: `projects/gcp-public-data-weathernext/assets/weathernext_2_0_0` (NOT `59572747_3_0`). Source: developers.google.com/weathernext/guides/earth-engine.
- WeatherNext 2 GEE access requires submitting the WeatherNext Data Request form — the service account must be whitelisted before any ImageCollection call will succeed.
- HYCOM GLBv0.08 expt_93.0/ts3z covers 2018-01-01 to 2020-02-19 (3-hourly), NOT 1994-2015 as originally assumed. Plan smoke test dates accordingly.
- HYCOM stores T/S (`ts3z`) and currents (`uv3z`) in separate OPeNDAP datasets — a single URL cannot load all four HYCOM_VARIABLES.
- HYCOM longitude coordinate is 0–360; bbox inputs in -180–180 must be converted with `% 360` before slicing; convert back after loading.
- HYCOM OPeNDAP time axis has unit `hours since 2000-01-01 00:00:00` but xarray reports it as `hours since analysis` — always open with `decode_times=False` and decode manually.
- Never call `sortby("time")` on a full OPeNDAP dataset before slicing — loads the entire time index and hangs. Filter time by raw float index first (`isel(time=np.where(mask)[0])`), then decode.
- `gcsfs.GCSMap` is zarr v2 API; zarr 3.x was installed (`zarr>=2.17` satisfies 3.x). Use direct `gs://` URI in `to_zarr()` / `open_zarr()` instead — gcsfs registers as an fsspec backend and handles credentials via `GOOGLE_APPLICATION_CREDENTIALS`.
- `google-cloud-storage` and `gcsfs` were missing from requirements.txt — both are required at import time in `WeatherNext2Harvester.authenticate()`.

---

## [2026-04-14] GCP Data Pipeline Implementation

- `xarray.Dataset.to_zarr` is read-only on xarray ≥2026.x — `patch.object(instance, "to_zarr")` silently fails. Use `patch("xarray.Dataset.to_zarr")` (class-level patch) for all `to_zarr` mocks. Confirmed working on xarray 2026.2.0.
- `str.removeprefix("gs://")` requires Python 3.9+. Always add a comment: `# Python 3.9+ — enforced by Dockerfile (python:3.11-slim)` so future readers don't have to audit the Dockerfile.
- `gcsfs.GCSFileSystem().exists()` expects the path WITHOUT the `gs://` prefix — always strip it before calling. Pattern: `path = gcs_uri.removeprefix("gs://"); fs.exists(path)`.
- The threshold Zarr variable is named `"sst_threshold_90"` (not `"threshold"`) — `ds["threshold"]` in the old `load_real_data()` was a latent bug that would only surface during a real GCS training run. Fix is in Tasks 4 & 5 (`train_era5.py`, `train_wn2.py`).
- When adding `fetch_and_cache()` to a class that requires authentication (ERA5Harvester), always add an `_initialized` guard that raises `RuntimeError` with the exact method name to call (`"authenticate()"`). Classes that don't require auth (HYCOMLoader) need no such guard — document the asymmetry in a test comment to prevent future confusion.
- Exception propagation test is mandatory for any GCS write method: assert that if the upstream fetch raises, `to_zarr` is never called and no partial Zarr is written. This is the key resilience property for spot VM idempotency.
- The HYCOM climatology step in `run_data_prep.py` must read SST from the already-cached GCS Zarr tiles (not re-fetch OPeNDAP) — otherwise the spot VM pays a second 1–2 hr OPeNDAP fetch. Pattern: write HYCOM tiles first, then open them with `xr.open_zarr()` for the climatology computation.
- Module-level docstrings in training scripts go stale when period constants change — update them in the same commit that changes the constants, or catch it in final review. `train_era5.py` still said "2018/2019" after Task 3 moved to 2022/2023; required a follow-up commit.

---

## [2026-04-27] Spatial-batching refactor downstream effects + GCE machine-image GPU lock-in

- When refactoring training inputs from spatial-mean (1, M, …) to per-cell (N_cells, M, …), audit ALL downstream code that consumed the old shape: `save_plots` gate hist / scatter that index `[0]` will silently slice cell-0 only; XAI scripts using spatial-mean tensors will feed OOD inputs to the trained model and produce unreliable attributions. Lesson: refactors that change tensor shape need a hand-trace of every `[0]` index and every `mean(dim=...)` in dependent code, not just the changed file.
- After spatial-batching, Captum IG with `(N_cells, M, …)` per-cell tensors needs cell-chunking. Default `internal_batch_size=5` was tuned for `(1, M=64)` → 320-profile budget. With `cells_per_chunk × M × internal_batch_size`, keep `cells_per_chunk=1` to preserve the same budget; iterate over cells and accumulate `attr.abs().sum(dim=0)`, divide by `n_processed` at end.
- Model uses `view()` (not `reshape`) for member-flatten — non-contiguous tensors from numpy advanced/scalar indexing fail. Always `.contiguous()` and explicitly `.to(torch.float32)` on tensors built from xarray `.values` before feeding the model.
- Mini-batch any forward pass that feeds the full validation tensor (~N_cells × 64 sequences) to a Transformer-based model; one shot allocates the full attention matrix and OOMs even on 64GB VMs.
- After training with early stopping (`patience=N`), the in-memory `model` holds the last-epoch state, NOT the best-val state. Always `model.load_state_dict(torch.load(f"{prefix}_best_weights.pt"))` before any final inference / plotting / SVaR generation, otherwise reported metrics reflect the patience-degraded final epoch.
- Logging `gate.mean().item()` at the end of a mini-batch validation loop captures only the LAST mini-batch's gate (variable is overwritten each iteration). Accumulate `val_gates.append(gate.cpu())` inside the loop and `torch.cat(val_gates).mean()` outside for honest aggregation.
- **GCE machine-image GPU lock-in**: a machine image created from an instance with an attached GPU (e.g. T4) carries the accelerator config such that creating a new instance from it on a non-GPU machine family (e2/n2) fails with `[machine-type, accelerator-type] features are not compatible` in EVERY zone. `--accelerator=type=...,count=0` is also rejected. Workaround: snapshot a CLEAN no-GPU running instance into a fresh machine image and provision from that. Don't bake GPU into machine images intended for reuse.

---

## [2026-04-10] Captum IG + large member dimension causes disk thrashing

- With `N_MEMBERS=64` and Captum's default full-batch IG (`n_steps=50`), the effective Transformer batch is `n_steps × M = 50 × 64 = 3200`. Attention weights `(3200, 8, 90, 90)` consume ~828 MB per layer × 4 layers = ~3.3 GB, exhausting RAM and hammering swap to disk.
- Fix: pass `internal_batch_size=5` to `ig.attribute()`. This caps the Transformer batch at `5 × 64 = 320`, reducing peak attention memory to ~330 MB.
- Rule: any Captum IG call that passes the member dimension inside the batch tensor MUST set `internal_batch_size ≤ 10` to stay under ~1 GB peak.

---

## [2026-03-24] Cloud Infrastructure & Credential Handling

- `Storage Object Admin` alone causes a 403 `storage.buckets.get` error — bucket-level access
  requires an additional role (use `Storage Bucket Viewer (Beta)`).
- `Storage Legacy Bucket Reader` does not appear in the GCP console — the working substitute
  is `Storage Bucket Viewer (Beta)`.
- Earth Engine registration at code.earthengine.google.com/register is a separate step from
  GCP IAM — easy to miss; the API must be enabled AND the account must be registered.
- Contributor tier (noncommercial) requires an active billing account but does not charge for
  EE usage — necessary to unlock WeatherNext 2 + HYCOM workloads at scale.
- Run `chmod 600` on the JSON key immediately after download.
- Never store the JSON key inside the project directory — use `~/.config/gcp-keys/`.
- Add `**/*.json` and `.env` to `.gitignore` before the first `git add`.
- Use `us-central1` for the GCS bucket — co-located with Earth Engine and Vertex AI,
  minimising egress latency and cross-region charges.
- Enable Hierarchical namespace on the bucket — optimizes Zarr directory operations
  (list and rename are O(1) rather than O(n) under flat namespace).
- Use Standard storage class for Zarr training caches — frequent reads make Nearline/Coldline
  retrieval fees costly; Standard has no minimum storage duration penalty.

---

## [2026-03-24] Do not ignore .ipynb files in .gitignore for this project.
Why: R&D notebooks in this repo are part of the science-to-engineering record.
Suppressing them from version control would hide the reasoning chain behind
model architecture and threshold choices. Only checkpoint directories are ignored.

---
## [2026-04-16] conda run does not survive SSH session detachment

**Lesson:** `conda run -n <env> python script.py` spawns a subprocess tree managed by conda.
When the SSH session closes, the entire process group is killed regardless of `nohup` or `&`.

**Fix:** Use the conda env's Python binary directly:
```bash
nohup /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/run_data_prep.py \
  >> data_prep.log 2>&1 </dev/null &
disown $!
```
Set env vars inline before the command. `disown` ensures the shell releases the process
before the SSH session closes.

---

## [2026-05-01] Session 31 — WN2 SST land-sea mask NaN + test assertion fix

- **WN2 SST NaN root cause:** WeatherNext 2 is an atmospheric model that defines SST only as a lower boundary condition. Its land-sea mask omits SST for ~86/357 GoM grid cells (persistent across all 365 days and all 64 members). The 4 other WN2 vars (U10, V10, 2m_temp, MSLP) are complete everywhere. Of the 86 NaN cells, 62 passed the HYCOM ocean mask into `build_tensors`, giving 1/5 vars = 20% NaN per cell = NaN loss from epoch 1.
- **Fix (commit 88a5fe2):** Added `wn2_sst_valid = ~merged["sea_surface_temperature"].isel(member=0, time=0).isnull()` to the combined mask in `build_tensors`. Reduces n_cells 223→161. No-op for ERA5 (ERA5 SST fully covers the domain).
- **Diagnostic pattern:** `wt.isnan().float().mean(dim=[1,2,3])` per cell showed exactly 0.2 for affected cells — immediately identifiable as exactly 1 of 5 vars being fully NaN. Then confirmed by checking raw WN2 zarr: 86 cells NaN for ALL 365 days, 0 partial-NaN cells.
- **test_build_tensors_shapes pre-existing bug:** Expected n_cells=1 but correct value is n_lat×n_lon=20 (all synthetic cells valid, no NaN in test data). Fixed assertion to `n_cells = 4 * 5`.

---
