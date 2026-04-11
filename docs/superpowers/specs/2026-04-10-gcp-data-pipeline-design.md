# GCP Data Pipeline Design
# GCS-Cached Data Prep on Spot GCE + GCS-Only Training

**Date:** 2026-04-10
**Status:** Approved

---

## Problem

All data loading currently runs locally:

- `compute_hycom_climatology.py` fetches 2 years of HYCOM over OPeNDAP (~30-60 min, ~2.4 GB RAM)
- `train_era5.py` and `train_wn2.py` real runs call `HYCOMLoader.fetch_tile()` and `ERA5Harvester.fetch()` live during training, tying up GPU time waiting on external network I/O

This violates the principle that GCP compute should own data-intensive work, and makes training non-reproducible (OPeNDAP availability is not guaranteed).

---

## Decision

**Option 3: GCS-aware harvesters + thin orchestrator.**

- `HYCOMLoader` and `ERA5Harvester` gain `fetch_and_cache(year, bbox, gcs_uri)` methods with idempotent cache-hit logic
- A new `scripts/run_data_prep.py` orchestrator calls these methods for all required years
- Training scripts read exclusively from GCS — no live OPeNDAP or GEE calls during training
- Data prep runs once on a spot GCE VM (`e2-standard-2`, ~$0.05 per run)

ERA5 training period aligned to **2022 train / 2023 val** (same as WN2) for apples-to-apples XAI comparison. The previous split (ERA5 on 2018/2019, WN2 on 2022/2023) was a legacy artifact of the old HYCOM coverage constraint, which no longer applies after switching to `GLBy0.08`.

---

## GCS Layout

```
gs://<bucket>/
  weathernext2/cache/            # existing — WN2 ensemble tiles (no change)
  hycom/
    climatology/                 # hycom_sst_threshold.zarr — 90th-pct per (dayofyear, lat, lon)
    tiles/
      2022/                      # full-year HYCOM tile, GoM bbox — shared ERA5 + WN2 train
      2023/                      # full-year HYCOM tile, GoM bbox — shared ERA5 + WN2 val
  era5/
    2022/                        # full-year ERA5 Zarr, GoM bbox — train
    2023/                        # full-year ERA5 Zarr, GoM bbox — val
```

All leaf nodes are Zarr stores. HYCOM tiles are shared between ERA5 and WN2 (same years, same bbox). WN2 tiles remain under the existing `weathernext2/cache/` path — no changes.

---

## Component Changes

### 1. `HYCOMLoader` (`src/ingestion/harvester.py`)

New method added; existing `fetch_tile()` untouched:

```python
def fetch_and_cache(self, year: int, bbox: tuple, gcs_uri: str) -> None:
    """
    Fetch one full calendar year of HYCOM data and write to GCS as Zarr.

    Parameters
    ----------
    year : int
        Calendar year to fetch (Jan 1 – Dec 31).
    bbox : tuple
        (lon_min, lat_min, lon_max, lat_max) in -180..180 degrees.
    gcs_uri : str
        GCS destination, e.g. "gs://bucket/hycom/tiles/2022/".
        If the URI already exists, returns immediately (idempotent).
    """
```

Cache-hit check: `gcsfs.GCSFileSystem().exists(gcs_uri)`.

### 2. `ERA5Harvester` (`src/ingestion/era5_harvester.py`)

Same pattern:

```python
def fetch_and_cache(self, year: int, bbox: tuple, gcs_uri: str) -> None:
    """
    Fetch one full calendar year of ERA5 data from GEE and write to GCS as Zarr.
    Cache-hit check identical to HYCOMLoader.fetch_and_cache().
    """
```

### 3. `WeatherNext2Harvester` — no changes

Already caches to GCS. The orchestrator verifies expected paths exist and warns if not.

### 4. `scripts/run_data_prep.py` (new file)

Thin orchestrator. Single CLI arg: `--bucket gs://your-bucket`.

Execution order:

1. **HYCOM tiles** — `HYCOMLoader.fetch_and_cache()` for 2022 and 2023 (largest fetches, run first)
2. **HYCOM climatology** — reads the tiles just written from GCS, runs `compute_climatology()`, writes to `gs://bucket/hycom/climatology/`
3. **ERA5 tiles** — `ERA5Harvester.fetch_and_cache()` for 2022 and 2023
4. **WN2 verification** — checks expected GCS paths exist, prints warning if not

Fully idempotent. Re-running after a spot preemption skips completed steps.

### 5. `scripts/_train_utils.py`

- Remove `ERA5_TRAIN_PERIOD`, `ERA5_VAL_PERIOD` (were 2018, 2019)
- Rename `WN2_TRAIN_PERIOD` / `WN2_VAL_PERIOD` to `TRAIN_PERIOD = 2022`, `VAL_PERIOD = 2023` — shared by both training scripts
- `build_tensors()` real-run path replaces all live fetches with `xr.open_zarr(gcs_uri)`
- New required env var: `MHW_GCS_BUCKET` — raises `RuntimeError` if absent during a real run

`train_era5.py` and `train_wn2.py` require no changes.

---

## GCE VM Spec

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Machine type | `e2-standard-2` | 2 vCPU, 8 GB RAM — sufficient for I/O-bound OPeNDAP fetch; GoM tile ~200-400 MB/year |
| Provisioning | Spot (preemptible) | ~60-80% cost reduction; idempotent prep job handles restarts |
| Region | `us-central1` | Matches GCS bucket region — zero intra-region egress cost |
| Boot disk | 50 GB standard PD | Dask spill scratch space |
| Estimated cost | ~$0.05 per full run | ~3 hr × $0.017/hr |

Workflow documented in `docs/gcp-data-prep-runbook.md` (git-tracked, no secrets).

---

## Environment Variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `GOOGLE_APPLICATION_CREDENTIALS` | harvesters, GCS client | GCP service account key path |
| `MHW_GCS_BUCKET` | `_train_utils.py`, `run_data_prep.py` | GCS bucket URI, e.g. `gs://my-bucket` |

---

## Testing

- Dry-run path in `train_era5.py` / `train_wn2.py` unchanged — local development unaffected
- `fetch_and_cache()` methods: unit tests mock `gcsfs.exists()` to verify cache-hit skip logic
- Integration test: run `run_data_prep.py` against a `gs://bucket/test/` prefix, verify Zarr stores are readable

---

## File Disposition

- `scripts/compute_hycom_climatology.py` — superseded by `run_data_prep.py`. Delete after the new pipeline is verified.

---

## Out of Scope

- Vertex AI custom jobs for training (noted in `mhw_claude_todo.md` as future work)
- Extended SST climatology beyond 2022/2023 (noted in todo as long-term)
