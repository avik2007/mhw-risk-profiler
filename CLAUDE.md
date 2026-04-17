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
- Before touching code, write a detailed numbered plan. Present each step to the user individually
  and wait for explicit approval of that step before presenting the next.
- Do not begin execution until every step has been explained and approved.
- No silent pivots: if the plan changes mid-execution, stop, explain the change, get approval, then continue.

### 1a. Autonomy After Plan Approval (Accept Edits On mode)
Once a detailed plan has been presented step-by-step and the user has approved every step,
Claude may execute the approved plan without requesting permission for each individual action.

**Hard limits — ALWAYS require explicit user permission, no exceptions:**
- Deleting any file or directory
- Creating a new git branch
- Pushing to any remote (git push)

**Everything else** (file edits, running tests, bash commands, installing packages, writing new files,
committing to the current branch) may proceed automatically under an approved plan.

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

### 6. No Assumptions — Consult Before Deciding
- Before making any assumption about requirements, data schema, model behavior, or system state, stop and ask.
- When a decision involves a tradeoff, surface the options explicitly and wait for a decision. Do not pick silently.
- This applies especially to scientific/physical choices (thresholds, aggregation windows, interpolation methods) — do not default to a plausible value without flagging it.

### 7. Simplicity First / Surgical Changes
- Always propose the simplest solution that correctly satisfies the requirement.
- When modifying code, touch only the lines required. No reformatting, no refactoring adjacent code, no unsolicited improvements.
- If the simplest and "architecturally proper" solutions diverge, flag it and ask.

### 8. Tests-First / Verifiable Success Criteria
- Before writing implementation, define explicit success criteria and write or identify tests for them.
- No feature is complete until the defined tests pass and the criteria are verified with evidence.

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
[2026-03-27] WeatherNext 2 correct GEE asset: weathernext_2_0_0 (not 59572747_3_0); requires Data Request form
[2026-03-27] HYCOM expt_93.0/ts3z covers 2018-01-01 to 2020-02-19 (3-hourly); ts3z + uv3z are separate URLs
[2026-03-27] HYCOM longitude is 0-360; decode_times=False required; filter time by raw float index before isel
[2026-03-27] zarr v3 installed (satisfies >=2.17); use gs:// URI in to_zarr/open_zarr, not gcsfs.GCSMap
[2026-03-27] google-cloud-storage and gcsfs missing from requirements.txt — add both
[2026-04-10] Captum IG with N_MEMBERS=64 and n_steps=50 → effective batch 3200 → ~3.3 GB attention → swap thrash; fix: internal_batch_size=5 in ig.attribute()
[2026-04-14] xarray ≥2026.x: to_zarr is read-only on instances — use patch("xarray.Dataset.to_zarr") not patch.object(instance, "to_zarr")
[2026-04-14] gcsfs.exists() requires path WITHOUT gs:// prefix — always strip with removeprefix("gs://") before calling
[2026-04-14] Threshold Zarr key is "sst_threshold_90" not "threshold" — ds["threshold"] was a latent bug fixed in load_real_data()
[2026-04-14] Exception propagation test mandatory for GCS write methods: assert fetch exception prevents to_zarr call
[2026-04-14] Climatology step must read from GCS tiles (not re-fetch OPeNDAP) — saves 1-2 hr on spot VM
[2026-04-14] Module docstrings go stale when periods change — update in same commit or catch in final review
[2026-04-16] conda run does not survive SSH session detachment — use env Python directly: /home/avik2007/miniconda3/envs/mhw-risk/bin/python + nohup + disown $!
