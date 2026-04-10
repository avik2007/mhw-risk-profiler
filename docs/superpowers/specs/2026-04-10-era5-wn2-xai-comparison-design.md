# ERA5 vs WeatherNext 2 — Dual Training & XAI Comparison Design

**Date:** 2026-04-10
**Status:** Approved
**Approach:** Option A — Sequential (ERA5 first, WN2 second)

---

## Goal

Train two independent `MHWRiskModel` instances — one on ERA5 deterministic reanalysis (with synthetic ensemble expansion), one on WeatherNext 2 real FGN ensemble — and compare:

1. **SVaR output** — per-grid-cell Stress VaR from each model
2. **XAI attribution** — Captum Integrated Gradients per season, revealing how ocean-atmosphere feature correlations shift between deterministic reanalysis and probabilistic ensemble inputs

---

## Architecture Overview

```
Phase 0 — WN2 Data Structure Scoping
──────────────────────────────────────
Inspect WN2 GEE asset schema: is data organized as a daily time series
(like ERA5) or as forecast runs with initialization dates and lead times?
This determines how train_wn2.py batches samples and what date range is usable.
Output: written findings added to this spec before implementation begins.

Phase 1 — ERA5 Pipeline
────────────────────────
ERA5Harvester.fetch()                      (member=1, time, lat, lon)
DataHarmonizer.expand_and_perturb()        (member=64, time, lat, lon)  ← synthetic Gaussian spread
train_era5.py                              trains MHWRiskModel from scratch
  → data/models/era5_weights.pt
  → data/models/era5_best_weights.pt       checkpoint at lowest-loss epoch
  → data/results/era5_training_log.csv     per-epoch metrics
  → data/results/era5_config.json          hyperparameters used
  → data/results/era5_svar.zarr            per-grid-cell SVaR

Phase 2 — WeatherNext 2 Pipeline
──────────────────────────────────
WeatherNext2Harvester.fetch_ensemble()     (member=64, time, lat, lon)  ← real FGN draws
train_wn2.py                               trains MHWRiskModel from scratch (no weight sharing)
  → data/models/wn2_weights.pt
  → data/results/wn2_svar.zarr             per-grid-cell SVaR

Phase 3 — XAI Comparison
──────────────────────────
scripts/compare_xai.py
  loads era5_weights.pt + ERA5 inputs
  loads wn2_weights.pt  + WN2 inputs
  runs Captum IG per season (DJF, MAM, JJA, SON)
  → data/results/xai_comparison.json
```

No weight sharing between models. No transfer learning. The WN2 model is trained independently from scratch so that attributions reflect WN2 data structure, not ERA5-inherited features.

---

## Train / Validation Split

Both training scripts use a **temporal split** to detect overfitting:

| Split | Period | Rationale |
|-------|--------|-----------|
| Train | 2018 (365 days) | Model sees this data during weight updates |
| Validation | 2019 (365 days) | Model never trained on this; used only to compute val loss per epoch |

Training loss falling while validation loss rises = overfitting. Both curves are saved per epoch so the pattern is visible. This split also tests year-to-year generalization — essential for an insurance risk product.

---

## Training Artifacts

Both `train_era5.py` and `train_wn2.py` save the following after each run. The prefix (`era5_` or `wn2_`) distinguishes the two pipelines.

**Data artifacts:**

| File | Content |
|------|---------|
| `data/models/{prefix}_weights.pt` | Final epoch weights |
| `data/models/{prefix}_best_weights.pt` | Weights at lowest validation loss epoch |
| `data/results/{prefix}_training_log.csv` | Per-epoch: epoch, train_loss, val_loss, SVaR_95, SVaR_50, SVaR_05, spread, gate_mean |
| `data/results/{prefix}_config.json` | Hyperparameters: lr, epochs, n_members, noise_sigmas, domain, train/val periods |
| `data/results/{prefix}_svar.zarr` | Per-grid-cell SVaR (lat=17, lon=21), variables: SVaR_95, SVaR_50, SVaR_05, spread |

**Image artifacts** (saved to `data/results/plots/`):

| File | What it shows | How to read it |
|------|--------------|----------------|
| `{prefix}_loss_curve.png` | Train loss + val loss vs epoch | Both lines falling = still learning. Val loss rising while train falls = overfitting. Both flat = converged. |
| `{prefix}_svar_curve.png` | SVaR_95, SVaR_50, SVaR_05 vs epoch | Shows how risk estimates stabilize over training. Erratic curves late in training = not converged. |
| `{prefix}_spread_curve.png` | SVaR_95 − SVaR_05 vs epoch | Ensemble spread evolution. Should stabilize. Collapsing to zero = degenerate ensemble. |
| `{prefix}_gate_hist.png` | Histogram of gate values at final epoch | Gate near 1 = depth (HYCOM) dominant. Gate near 0 = atmospheric (WN2/ERA5) dominant. Bimodal = two distinct MHW regimes. |
| `{prefix}_pred_vs_actual.png` | Predicted SDD vs physics SDD scatter (val set) | Points along the diagonal = good fit. Fan shape = heteroscedastic error. Systematic offset = bias. |

All plots are generated with `matplotlib`. No new dependencies — matplotlib is already in the environment.

---

## Phase 1: ERA5 Pipeline

Builds on `docs/superpowers/plans/2026-04-03-era5-proxy-training.md`. The ERA5Harvester, expand_and_perturb, and test suite are unchanged. The training script gains: train/val split, best-weights checkpoint, training log CSV, config JSON, and all image artifacts above.

Key points:
- `ERA5Harvester` fetches `ECMWF/ERA5/DAILY` from GEE, 5 bands renamed to WN2-compatible variable names
- `DataHarmonizer.expand_and_perturb()` broadcasts `member=1 → member=64` with per-variable Gaussian noise (σ values from WN2 published spread)
- Training objective: MSE vs physics SDD from `accumulate_sdd()` using HYCOM surface SST and `hycom_sst_threshold.zarr`
- Domain: Gulf of Maine `lat=[41,45], lon=[-71,-66]`
- Train: 2018, Val: 2019
- 50 epochs (CLI argument), Adam lr=1e-4, gradient clipping max_norm=1.0 (safeguard against large gradients)

---

## Phase 2: train_wn2.py

Symmetric to `train_era5.py` in every way except the harvester and ensemble expansion. Same train/val split (2018/2019), same artifact set, same hyperparameters.

| | train_era5.py | train_wn2.py |
|---|---|---|
| Harvester | `ERA5Harvester.fetch()` | `WeatherNext2Harvester.fetch_ensemble()` |
| expand_and_perturb | Yes (1→64) | No (already 64 real members) |
| Weight init | From scratch | From scratch |
| Epochs / lr / clip | 50 / 1e-4 / 1.0 | Same |
| Train / Val split | 2018 / 2019 | Same |
| All artifact outputs | `era5_*` prefix | `wn2_*` prefix |

`--dry-run` flag required (same as ERA5 script): skips GEE/HYCOM fetches, uses synthetic random tensors of the correct shape.

---

## Per-Grid-Cell SVaR Output

Both training scripts run a **separate inference pass** after the training loop completes. The training loop uses spatially-averaged SDD for MSE stability. The SVaR inference pass restructures data so `batch = lat * lon`:

```python
# Inference pass — post-training
# hycom_t: (lat*lon, member=64, depth=11, channels=4)
# wn2_t:   (lat*lon, member=64, time=90, features=5)

sdd_pred, _, _ = model(hycom_t, wn2_t)            # (lat*lon, 64)
sdd_grid = sdd_pred.view(n_lat, n_lon, n_members)  # (lat, lon, 64)

svar_ds = xr.Dataset({
    "SVaR_95": xr.DataArray(sdd_grid.quantile(0.95, dim=-1), dims=["lat", "lon"]),
    "SVaR_50": xr.DataArray(sdd_grid.quantile(0.50, dim=-1), dims=["lat", "lon"]),
    "SVaR_05": xr.DataArray(sdd_grid.quantile(0.05, dim=-1), dims=["lat", "lon"]),
    "spread":  xr.DataArray(
        sdd_grid.quantile(0.95, dim=-1) - sdd_grid.quantile(0.05, dim=-1),
        dims=["lat", "lon"]
    ),
})
svar_ds.to_zarr("data/results/era5_svar.zarr")  # or wn2_svar.zarr
```

Spatial averaging is left to downstream analysis — the Zarr preserves full (lat, lon) resolution.

---

## Phase 3: XAI Comparison (scripts/compare_xai.py)

### Attribution method

Captum `IntegratedGradients` targeting the `latent` tensor (existing hook point in `MHWRiskModel.forward()`). Uses the same `latent_forward` wrapper as the smoke test in `ensemble_wrapper.py`.

### Per-season partitioning

The time axis is split into four standard meteorological seasons before running IG. For each season:

```python
season_mask = time_index.month.isin(season_months[season])  # e.g. [6,7,8] for JJA
wn2_season  = wn2_t[:, :, season_mask, :]   # (batch, member, T_season, 5)
hycom_season = hycom_t                       # HYCOM is time-invariant per profile
```

IG is run independently per season. Attribution tensors are collapsed:

```
WN2/ERA5 stream:  mean |IG| over (member, time_steps) → 5 scores per variable
HYCOM stream:     mean |IG| over (member, depth_levels) → 4 scores per channel
```

### Output format

`data/results/xai_comparison.json`:

```json
{
  "era5": {
    "DJF": {
      "atm":  {"sea_surface_temperature": 0.042, "2m_temperature": 0.031, "10m_u_component_of_wind": 0.018, "10m_v_component_of_wind": 0.017, "mean_sea_level_pressure": 0.011},
      "hycom": {"water_temp": 0.018, "salinity": 0.009, "water_u": 0.005, "water_v": 0.004},
      "gate_mean": 0.58
    },
    "MAM": { ... },
    "JJA": { ... },
    "SON": { ... }
  },
  "wn2": {
    "DJF": { ... },
    "MAM": { ... },
    "JJA": { ... },
    "SON": { ... }
  },
  "delta": {
    "DJF": {
      "atm":  {"sea_surface_temperature": "+0.025", ...},
      "hycom": {"water_temp": "+0.007", ...}
    },
    "MAM": { ... },
    "JJA": { ... },
    "SON": { ... }
  }
}
```

`delta` = WN2 attribution − ERA5 attribution per variable per season. Positive delta means WN2-trained model weights that variable more heavily.

---

## File Map

| Action | Path | Phase |
|--------|------|-------|
| Scope | WN2 GEE asset schema — findings written to this spec | 0 |
| Already specced | `src/ingestion/era5_harvester.py` | 1 |
| Already specced | `src/ingestion/harvester.py` (`expand_and_perturb`) | 1 |
| **Create** | `scripts/train_era5.py` (replaces prior spec; gains val split + all artifacts) | 1 |
| **Create** | `scripts/train_wn2.py` | 2 |
| **Create** | `scripts/compare_xai.py` | 3 |

No changes to `MHWRiskModel`, `svar.py`, `mhw_detection.py`, `cnn1d.py`, or `transformer.py`.

---

## Validation Gates

**Phase 0:**
- WN2 GEE asset schema inspected; findings (time axis organization, coverage dates, member structure) written to this spec

**Phase 1:**
- `pytest tests/test_era5_harvester.py -v` — 3 tests pass
- `python scripts/train_era5.py --dry-run` — 50 epoch lines, val loss printed each epoch, `spread > 0`
- `data/models/era5_weights.pt` and `era5_best_weights.pt` exist
- `data/results/era5_training_log.csv` has 50 rows with all columns
- `data/results/plots/era5_loss_curve.png` and 4 other plots exist

**Phase 2:**
- `python scripts/train_wn2.py --dry-run` — same gate as ERA5; all `wn2_*` artifacts produced
- `data/results/wn2_svar.zarr` exists with dims `(lat=17, lon=21)`

**Phase 3:**
- `python scripts/compare_xai.py --dry-run` — `xai_comparison.json` exists, all 4 seasons present, all 9 variables scored for both models

---

## Future Extension (Option C — deferred)

Member-level attribution variance: for the WN2 model, compute IG attribution variance *across members* and compare against ERA5 model. Tests whether real FGN ensemble spread produces member-dependent feature importance (i.e., different atmospheric trajectories shift what the model attends to), vs ERA5 synthetic members where attribution should be near-uniform across members.

---

*Scientific rationale: Gemini. Implementation: Claude.*
