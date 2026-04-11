# mhw_claude_todo.md
# Active task queue — current and next actions
# ---------------------------------------------

---

## ACTIVE — RESUME HERE NEXT SESSION

### Pre-WeatherNext Analytics — ONE STEP REMAINING

**Current HEAD:** `5e073b7`

**Already done this session:**
- `src/analytics/payout.py` ✅ committed `5ea4427` — 11/11 tests pass
- `scripts/compute_hycom_climatology.py` ✅ committed `5e073b7` — 4/4 tests pass, --help verified

**Remaining: spot-check the threshold Zarr**

The OPeNDAP fetch (`conda run -n mhw-risk python scripts/compute_hycom_climatology.py`)
was running when the session ended. Check if it finished:

```bash
ls -lh data/processed/hycom_sst_threshold.zarr 2>/dev/null || echo "not yet — re-run the script"
```

If the Zarr exists, run the spot-check:

```bash
conda run -n mhw-risk python -c "
import xarray as xr
ds = xr.open_zarr('data/processed/hycom_sst_threshold.zarr')
t = ds['sst_threshold_90']
print('Shape:', dict(t.sizes))
print('Summer peak (doy=213):', t.sel(dayofyear=213).values.mean().round(2), 'degC')
print('Winter trough (doy=15):', t.sel(dayofyear=15).values.mean().round(2), 'degC')
print('Spatial std at summer peak:', t.sel(dayofyear=213).values.std().round(3))
"
```

Expected: summer mean > winter mean; spatial std > 0 (location-varying confirmed).

Then update recentactions and move to the NEXT section below.

---

## NEXT (after analytics completions)

### Real training runs on GCP
ERA5 and WN2 real runs require GCP (n2-standard-8 or T4 GPU). Prerequisite: `hycom_sst_threshold.zarr` from analytics step 2 above.

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
