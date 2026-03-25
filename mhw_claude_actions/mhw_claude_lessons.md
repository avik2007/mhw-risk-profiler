# mhw_claude_lessons.md
# Root-cause fixes and non-obvious discoveries
# ---------------------------------------------
# Format: [YYYY-MM-DD] <lesson> | Why it matters

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
