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
