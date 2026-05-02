# MHW Gemini Todo

- [x] Analyze IG attribution discrepancy between ERA5 and WN2 (Zonal vs. Meridional wind).
- [x] Implement regularization improvements (AdamW, L2, CosineAnnealingLR) and spatial batching to prevent overfitting.
- [x] Update `compare_xai.py` to support `--use-gcs` loading.
- [ ] Monitor regression loss (MSE) and SVaR stability for the 2022/2023 dual training runs on GCP using the new batching logic.
- [ ] Review the final XAI comparison plots (`ig_attribution_*.png`) once Claude completes the real data run.

---

- [x] Ensure all functions in `src/models/` and new `fetch_and_cache` methods have verbose header comments.
- [ ] Review harmonization logic in `src/ingestion/harvester.py` for CF-1.8 compliance.
- [ ] Monitor SVaR accuracy and ensemble spread during execution once WN2 access is granted.
- [ ] Monitor regression loss curves (Train vs. Val) and Mean Squared Error (MSE) to detect over- or under-fitting.
...
