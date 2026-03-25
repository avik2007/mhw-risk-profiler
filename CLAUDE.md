# CLAUDE.md — mhw-risk-profiler
# ArgoEBUS General Principles (adapted for MHW Financial Risk)
# ---------------------------------------------------------------
# This file governs all AI-assisted development in this repository.
# It is self-updating: any fix recorded in mhw_claude_actions/mhw_claude_lessons.md
# MUST be immediately mirrored here under the "Lessons Applied" section.

---

## Project Identity

**mhw-risk-profiler** harmonizes Google WeatherNext 2 (FGN-based ensemble) and HYCOM
(Hybrid Coordinate Ocean Model) data to calculate Financial Value-at-Risk (VaR) for
aquaculture assets exposed to Marine Heatwave (MHW) events.

---

## General Principles

### 1. Plan Mode
- Required for ANY task with 3 or more steps.
- Before touching code, write a numbered plan. Get acknowledgment or self-review before executing.
- No silent pivots: if the plan changes mid-execution, update the plan first, then continue.

### 2. Self-Improvement Loop
- Every root-cause fix or non-obvious discovery goes into `mhw_claude_actions/mhw_claude_lessons.md`.
- Within the same session, that lesson MUST be reflected here under **Lessons Applied**.
- Format: `[YYYY-MM-DD] <lesson summary>` — one line, no prose.

### 3. Science-to-Engineering Boundary
- Gemini (and Perplexity research in `mhw_ai_research/`) owns hypothesis generation and literature synthesis.
- Claude owns Dockerized, production-ready, reproducible implementation.
- Do not re-litigate scientific choices already settled in `mhw_ai_research/`. Implement them.

### 4. Style
- Responses: concise. No emojis. No filler phrases ("Certainly!", "Great question!").
- Every function must have a verbose header comment explaining:
  - Physical meaning of each input parameter (units, coordinate system, data source).
  - Physical meaning of the output (what quantity, what units, what it represents financially).
  - Any non-obvious numerical choices (e.g., why 90th percentile for MHW threshold).
- Example header format:
  ```python
  def compute_stress_degree_days(sst_anomaly, threshold=0.0):
      """
      Accumulate thermal stress above the MHW baseline threshold.

      Parameters
      ----------
      sst_anomaly : xr.DataArray
          Sea surface temperature anomaly [deg C] relative to the 1982-2011 climatology.
          Positive values indicate warmer-than-baseline conditions.
          Source: WeatherNext 2 ensemble member or HYCOM daily output.
      threshold : float
          Minimum anomaly [deg C] below which no stress is accumulated.
          Default 0.0 follows Hobday et al. (2016) Category I definition.

      Returns
      -------
      sdd : xr.DataArray
          Stress Degree Days [deg C * day] — cumulative thermal load above threshold.
          Used as the trigger variable for parametric insurance payouts and VaR estimation.
      """
  ```

### 5. Verification Gate
- No task is marked "Done" until code execution confirms the output with printed or logged evidence.
- Acceptable evidence: test output, assertion pass, printed shape/values, or saved artefact path.
- "It should work" is not verification. Run it.

---

## Repository Layout

See `mhw-repo-architecture.md` for the full annotated directory tree.

---

## Data Sources

| Source | Format | Access | Role |
|---|---|---|---|
| Google WeatherNext 2 | Zarr (GEE / GCS) | GEE Python API + GCS cache | Atmospheric ensemble, tail-risk |
| HYCOM GLBv0.08 | NetCDF (OPeNDAP) | THREDDS Data Server | 3D thermohaline, subsurface profiles |

Harmonization target: daily, 0.25-degree global grid, aligned time axis, CF-1.8 compliant.

### GCP / GEE Environment Notes
- GEE authentication: service account JSON (local dev) or Application Default Credentials (Cloud Run).
- WeatherNext 2 Zarr cache stored in GCS under `gs://<bucket>/weathernext2/cache/`.
- HYCOM uses OPeNDAP remote access — no local download required; Dask chunks handle memory.
- HYCOM vertical coordinate is hybrid (Z / Sigma / Isopycnal); harvester.py interpolates to
  standard depth levels (TARGET_DEPTHS_M) before harmonization. This is a hard requirement —
  do not skip interpolation or the 1D-CNN will receive physically inconsistent inputs.

See **`mhw-gcp-env-commands.md`** for the full canonical set of Conda, GEE auth, GCS bucket,
OPeNDAP connectivity, smoke test, and Zarr lazy-open verification commands.

---

## Cloud Infrastructure

See **`mondal-mhw-gcp-info.md`** (git-ignored) for bucket name, service account email, IAM roles, and credential paths.

---

## Lessons Applied

<!-- Auto-populated from mhw_claude_actions/mhw_claude_lessons.md -->
<!-- Format: [YYYY-MM-DD] <lesson> -->
[2026-03-24] Storage Object Admin alone causes 403 on bucket ops — add Storage Bucket Viewer (Beta)
[2026-03-24] EE registration at earthengine.google.com/register is separate from GCP IAM
[2026-03-24] chmod 600 JSON key immediately; store under ~/.config/gcp-keys/, never in project dir
[2026-03-24] Use us-central1 + Standard class + Hierarchical namespace for Zarr GCS buckets
