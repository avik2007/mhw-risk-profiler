# GeoAI Inference Orchestrator Pivot (Thread 4)

**Date:** 2026-04-25
**Source:** `mhw_ai_research/Gemini-Iterative Weather Model Pipeline Thread 4.txt`
**Status:** Approved (decisions D1–D6 locked to top recommendations; generalizations retained as future-state)
**Sequencing:** Executes AFTER (1) WN2 SVaR review, (2) LinkedIn ERA5 post, (3) README/GitHub visualization polish.

---

## 1. Pivot Summary

Refactor `mhw-risk-profiler` from a WeatherNext 2-specific tool into a model-agnostic
**Weather Inference Orchestrator**. Use GraphCast (and later Aurora) to emulate WN2-style
ensembles for present-day (2026) data — solving WN2's 2-year temporal cap and post-May 2026
deprecation timeline.

Apple meteorology contact's tip: build adoptable, modular pipelines (provider pattern), not
single-source scripts.

**Out of scope:** financial tipping-point search via perturbation (user rejected this framing).
This pivot is about **temporal coverage + adoptability**, not Monte Carlo sensitivity.

---

## 2. Locked Decisions (D1–D6)

| ID | Decision | Primary (v1) | Future generalization |
|----|----------|--------------|----------------------|
| D1 | Emulator engine | **GraphCast only** (GNN-native, GCP-optimized, direct WN2 predecessor) | Add Aurora as parallel provider once GraphCast stable; multi-modal thermocline sensitivity for v2 |
| D2 | FGN emulation method | **"Poor Man's FGN"** — spatially coherent Perlin/low-res Gaussian noise on initial conditions, N parallel rollouts. No retrain. | Latent-space noise injection `h_{t+1}=GNN(h_t,z)` — gated by config flag if v1 ensemble spread too narrow |
| D3 | CRPS fine-tune | **Skip for v1** — stock GraphCast weights, accept MSE blur | Replace MSE in final layer with CRPS / Energy Score if extremes underpredicted in backtest |
| D4 | Initial-conditions source | **ERA5 via GEE** (existing pipeline, 5-day lag) | ECMWF HRES open-data for near-real-time; NOAA GFS as free fallback |
| D5 | Scope vs existing code | **Wrap, don't rip** — keep `harvester.py`, `era5_harvester.py` intact, add providers alongside | Deprecate originals once provider parity proven |
| D6 | GPU access | T4 quota request already in-flight (todo line 79). **CPU fallback** acceptable for v1 smoke (~10–20 min/forecast). | A100 / TPU on Vertex AI once orchestrator validated end-to-end (ties to existing P8 Vertex AI migration) |

---

## 3. Phase 0 — Prerequisites

Pivot does not start until ALL of the following are green:

- [ ] WN2 SVaR inference review complete; results accepted (or spread fix decided)
- [ ] LinkedIn ERA5 post published
- [ ] README/GitHub visualization polish complete (WN2 cold-blue artifact, network diagram fonts)
- [ ] GPU quota status known (approved → T4 path; denied → CPU fallback path)

---

## 4. Phase 1 — Provider Pattern Refactor (no behavior change)

**Goal:** introduce abstraction with zero regression in existing WN2/ERA5 training.

- Create `src/providers/` directory.
- `src/providers/base.py`:
  - Abstract class `BaseWeatherProvider`.
  - Methods: `fetch_initial_conditions(date, hours_back) → xr.Dataset`, `run(initial_state, steps) → xr.Dataset`.
  - Class constant `STANDARD_VARS = {"sst","t2m","u10","v10","msl","slhf","sshf","ssrd","tp"}` (CF-1.8 short names).
  - Class constant `STANDARD_DIMS = ("member","time","lat","lon")`.
- `src/providers/weathernext.py` — thin wrapper over existing `harvester.py` (calls existing `fetch_and_cache`, returns standardized cube).
- `src/providers/era5.py` — thin wrapper over `era5_harvester.py`.
- `tests/test_provider_schema.py`:
  - Assert each provider returns dataset with `STANDARD_VARS` ⊆ `data_vars`.
  - Assert dim order matches `STANDARD_DIMS`.
  - Assert CF-1.8 attrs present (`units`, `standard_name`, `Conventions`).

**Exit criterion:** existing `train_era5.py` + `train_wn2.py` runs through provider wrappers, identical loss curves.

---

## 5. Phase 2 — Real-Time Initial-Conditions Provider

**Goal:** unblock present-day inference (D4=a → ERA5 GEE).

- `src/providers/realtime.py` — `RealTimeERA5Provider(BaseWeatherProvider)`:
  - `fetch_initial_conditions(date=None, hours_back=48)` → most recent ERA5 frame (~5-day lag).
  - Cache to GCS: `gs://mhw-risk-cache/initial_conditions/era5/<YYYYMMDD>.zarr`.
- Smoke test: fetch yesterday-minus-5d frame, assert non-null SST over GoM, shape `(time, lat, lon)`.

**Future-state (D4 generalization):** `RealTimeHRESProvider` (ECMWF open-data), `RealTimeGFSProvider` (NOAA). Same interface.

---

## 6. Phase 3 — GraphCast Emulator (D1=a)

- Add dependency: `graphcast` (DeepMind) or `graph-weather` (community port). Spike both, pick the one with cleaner GCS-weights loading.
- Download stock weights → `gs://mhw-risk-cache/models/graphcast/<version>/`.
- `src/providers/graphcast_emulator.py`:
  - `GraphCastEmulator(BaseWeatherProvider)`.
  - Load weights once at init (cache on local disk).
  - `run(initial_state, steps=40)` → 10-day forecast at 6-hr step.
  - Returns standardized xarray cube `(time, lat, lon)` at 0.25°.
- Smoke test: 1 forecast for fixed historical date, compare day-0 SST to ERA5 ground truth (assert RMSE < 1°C over GoM).

**Future-state (D1 generalization):** `AuroraProvider` mirrors interface; user picks via CLI flag `--model aurora`.

---

## 7. Phase 4 — FGN Emulation Wrapper (D2=a)

**Goal:** generate 64-member ensembles from deterministic GraphCast — drop-in replacement for WN2 zarr in existing `harmonize()`.

- `src/providers/fgn_wrapper.py`:
  - `FGNWrapper(provider, n_members=64, noise_kind="perlin", sigma_per_var={...})`.
  - Generate N perturbed initial states with spatially coherent noise (Perlin or low-pass-filtered Gaussian).
  - σ tuned per variable (e.g. SST σ=0.3°C, T2M σ=0.5°C, wind σ=0.5 m/s — calibrate against WN2 ensemble spread on overlap dates).
  - Parallel rollouts (joblib / asyncio batching) → stack on `member` dim.
  - Output schema matches existing WN2 64-member zarr.
- Validation:
  - Ensemble spread > 0 (unlike HYCOM-deterministic SDD case).
  - Spread distribution roughly matches WN2 spread on overlap year (Spearman > 0.7 across grid cells).

**Future-state (D2 generalization):** latent-space noise injection in GNN processor — requires fine-tune. Add `LatentFGNProvider` if Perlin spread proves too narrow.

---

## 8. Phase 5 — CRPS Fine-Tune (skipped in v1, D3=a)

Documented for completeness; not implemented in v1.

If extremes underpredicted in Phase 8 backtest:
- Replace MSE in `scripts/_train_utils.py:compute_loss` with CRPS / Energy Score.
- Fine-tune GraphCast final layer on ERA5 hindcast (frozen encoder).
- Re-run validation.

---

## 9. Phase 6 — CLI + Config

- `configs/settings.yaml`:
  - Bio thresholds (salmon 18°C, kelp 15°C).
  - Depth levels (`TARGET_DEPTHS_M`).
  - Provider defaults (engine, n_members, noise sigmas, weight paths).
  - GCS bucket / paths.
- `scripts/run_profiler.py`:
  ```
  python run_profiler.py \
    --model {wn2|graphcast} \
    --date 2026-04-25 \
    --hours 240 \
    --members 64 \
    --bbox gom \
    --config configs/settings.yaml
  ```
- Pipeline: `provider → FGNWrapper (if deterministic) → harmonize → MHWRiskModel → SDD → SVaR → plots`.

**Future-state (D1 generalization):** add `aurora` to `--model` choices once `AuroraProvider` lands.

---

## 10. Phase 7 — Docs + State Sync

- `README.md` reframe: title section "Foundation Model Orchestrator for Spatial Finance".
- New `README.md` section: "How to add a provider" — subclass `BaseWeatherProvider`, register in `configs/settings.yaml`.
- `mhw-repo-architecture.md` updated with `src/providers/` tree.
- `mhw_claude_actions/mhw_claude_recentactions.md` — append pivot session entry.
- `mhw_claude_actions/mhw_claude_todo.md` — pivot section moved from QUEUED to ACTIVE on kickoff.

---

## 11. Phase 8 — Validation Gates

| Gate | Test | Pass criterion |
|------|------|----------------|
| Schema | `tests/test_provider_schema.py` | All providers conform to `STANDARD_VARS` / `STANDARD_DIMS` / CF-1.8 |
| Smoke | GraphCast 1-forecast day-0 SST RMSE vs ERA5 (GoM) | < 1°C |
| Spread | FGN wrapper ensemble spread vs WN2 historical (overlap date) | Spearman > 0.7 across grid |
| Backtest | Run profiler for 2023 date, compare SVaR map to WN2 historical SVaR | Qualitative pattern match (visual + cosine sim > 0.6) |
| End-to-end | `run_profiler.py --model graphcast --date <yesterday>` | Produces SVaR plot, no errors, runtime < 30 min on T4 |

---

## 12. Risks + Open Items

- GraphCast weight licensing — confirm permissive use for derivative inference pipeline.
- `graphcast` package GCP/CUDA compatibility — may require pinned JAX version.
- Perlin noise calibration — requires WN2 overlap dates with valid 64-member spread; HYCOM-derived spread=0 issue (todo line 39) may complicate calibration; use ERA5+synthetic-noise overlap as fallback baseline.
- ECMWF HRES open-data licensing for future RealTimeHRESProvider.

---

## 13. Career-Pivot Framing (per Gemini coaching, for LinkedIn / Earth Finance pitch)

> "Recognizing the temporal lag in public datasets, I architected an adoptable inference
> pipeline using GraphCast weights to emulate present-day probabilistic ensembles. This
> ensures Spatial Value-at-Risk calculations are grounded in current dynamics, fulfilling
> the 2026 Benchmark of reasoning across real-time multi-layer geospatial data."
