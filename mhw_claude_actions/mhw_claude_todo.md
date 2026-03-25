# mhw_claude_todo.md
# Active task queue — current and next actions
# ---------------------------------------------

---

## ACTIVE

### GCP Environment Calibration & Ingestion Testing
**Goal**: Verify that the ingestion engine (`harvester.py`) is physically and technically sound
end-to-end before proceeding to model training. All sub-steps require printed/logged evidence to pass.

Sub-steps:
1. ~~**IAM & Auth Verification**: Confirm GEE Service Account and GCS Bucket permissions are active.~~
   ~~- Validate `GOOGLE_APPLICATION_CREDENTIALS` env var points to a valid service account JSON.~~
   ~~- Run `earthengine authenticate --quiet` or equivalent SDK check; confirm GCS bucket read/write.~~
   **DONE** — Smoke test passed 2026-03-24. Auth OK, bucket accessible, contents empty (expected).
2. **Ensemble Connectivity Test**: Run a smoke test with `harvester.py` using a very small spatial
   bounding box (e.g., 1°×1° patch) and only 2 ensemble members to verify the GEE-to-GCS Zarr path.
   - Expected evidence: Zarr store written to `gs://<bucket>/weathernext2/cache/`; Dataset repr printed.
3. **Vertical Coordinate Sanity Check**: Print a vertical profile of HYCOM Temperature and Salinity
   from `data/processed/` to verify that interpolation to `TARGET_DEPTHS_M` preserves thermocline structure.
   - Expected evidence: T/S values at each TARGET_DEPTHS_M level printed; thermocline gradient visible
     (temperature drops sharply between ~50–200 m; salinity shows halocline if present).
4. **Dask Scaling Test**: Verify that xarray can lazily open the generated Zarr store without
   crashing local memory.
   - Expected evidence: `xr.open_zarr(path)` returns a Dataset with correct dims; `dask.array` chunks
     printed; no OOM error.

---

## COMPLETED

- [2026-03-24] GCP project created, $300 free credits activated
- [2026-03-24] Earth Engine API enabled; account registered as Contributor (Noncommercial)
- [2026-03-24] Service account created with required IAM roles (see mondal-mhw-gcp-info.md)
- [2026-03-24] GCS bucket created (see mondal-mhw-gcp-info.md for name/region/config)
- [2026-03-24] ADC configured via GOOGLE_APPLICATION_CREDENTIALS
- [2026-03-24] Smoke test passed: Auth OK, bucket accessible, contents empty

---

## QUEUED

### Implement the 1D-CNN + Transformer architecture in src/models/
**Goal**: Neural architecture for encoding depth-resolved HYCOM vertical profiles and
long-range WeatherNext 2 SST sequences into MHW severity predictions.

Sub-steps:
1. `cnn1d.py` — 1D-CNN feature extractor for depth-resolved T/S profiles.
2. `transformer.py` — Transformer encoder for temporal SST dependencies.
3. `ensemble_wrapper.py` — wraps both to process all 64 ensemble members independently.
4. Smoke test confirming output tensor shapes; verification via printed output.

---
