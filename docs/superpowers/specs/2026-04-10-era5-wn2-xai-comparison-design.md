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
Phase 1 — ERA5 Pipeline
────────────────────────
ERA5Harvester.fetch()                      (member=1, time, lat, lon)
DataHarmonizer.expand_and_perturb()        (member=64, time, lat, lon)  ← synthetic Gaussian spread
train_era5.py                              trains MHWRiskModel from scratch
  → data/models/era5_weights.pt
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

## Phase 1: ERA5 Pipeline

Fully specced in `docs/superpowers/plans/2026-04-03-era5-proxy-training.md`. No changes.

Key points:
- `ERA5Harvester` fetches `ECMWF/ERA5/DAILY` from GEE, 5 bands renamed to WN2-compatible variable names
- `DataHarmonizer.expand_and_perturb()` broadcasts `member=1 → member=64` with per-variable Gaussian noise (σ values from WN2 published spread)
- Training objective: MSE vs physics SDD from `accumulate_sdd()` using HYCOM surface SST and `hycom_sst_threshold.zarr`
- Domain: Gulf of Maine `lat=[41,45], lon=[-71,-66]`, period 2018–2019
- 50 epochs, Adam lr=1e-4

---

## Phase 2: train_wn2.py

Symmetric to `train_era5.py`. Differences:

| | train_era5.py | train_wn2.py |
|---|---|---|
| Harvester | `ERA5Harvester.fetch()` | `WeatherNext2Harvester.fetch_ensemble()` |
| expand_and_perturb | Yes (1→64) | No (already 64 real members) |
| Weight init | From scratch | From scratch |
| Weights output | `data/models/era5_weights.pt` | `data/models/wn2_weights.pt` |
| SVaR output | `data/results/era5_svar.zarr` | `data/results/wn2_svar.zarr` |

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
| Already specced | `src/ingestion/era5_harvester.py` | 1 |
| Already specced | `src/ingestion/harvester.py` (`expand_and_perturb`) | 1 |
| Already specced | `scripts/train_era5.py` + SVaR inference pass | 1 |
| **Create** | `scripts/train_wn2.py` | 2 |
| **Create** | `scripts/compare_xai.py` | 3 |

No changes to `MHWRiskModel`, `svar.py`, `mhw_detection.py`, `cnn1d.py`, or `transformer.py`.

---

## Validation Gates

**Phase 1:**
- `pytest tests/test_era5_harvester.py -v` — 3 tests pass
- `python scripts/train_era5.py --dry-run` — 50 epoch lines, `spread > 0`, weights + zarr saved

**Phase 2:**
- `python scripts/train_wn2.py --dry-run` — same gate as ERA5
- `data/results/wn2_svar.zarr` exists with dims `(lat=17, lon=21)`

**Phase 3:**
- `python scripts/compare_xai.py --dry-run` — `xai_comparison.json` exists, all 4 seasons present, all 9 variables scored for both models

---

## Future Extension (Option C — deferred)

Member-level attribution variance: for the WN2 model, compute IG attribution variance *across members* and compare against ERA5 model. Tests whether real FGN ensemble spread produces member-dependent feature importance (i.e., different atmospheric trajectories shift what the model attends to), vs ERA5 synthetic members where attribution should be near-uniform across members.

---

*Scientific rationale: Gemini. Implementation: Claude.*
