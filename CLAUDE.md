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
[2026-04-18] _gcs_safe_write + HNS bucket: fs.exists("parent/") returns True (HNS dir) → fs.rm recursive wipes monthly tiles. Fix: preserve_dirs=("monthly",) in annual write.
[2026-04-18] xr.concat of monthly zarrs carries encoding["chunks"] misaligned with Dask concat chunks → ValueError in to_zarr. Fix: pop "chunks" from each data_var encoding before annual write.
[2026-04-19] xr.concat of N monthly zarrs produces irregular dask time chunks (each month has its own stored chunk spec, 28-31 days/month). Even after encoding pop, _determine_zarr_chunks rejects non-uniform chunks. Fix: ds_annual.chunk({"time": 30}) after encoding pop, before _gcs_safe_write.
[2026-04-19] NEVER run mhw-data-prep sequential pipeline AND dedicated per-month VMs simultaneously — both write to same zarr paths, causing concurrent-write corruption and VM crashes.
[2026-04-20] harmonize() uses global TARGET_LAT/LON (721×1440); for a GoM bbox tile this produces ~485 GB after expand_and_perturb. Fix: clip TARGET_LAT/LON to input data bbox inside harmonize().
[2026-04-20] HYCOM climatology saved with 'lat'/'lon' dims (0.08° native); merged ERA5 uses 'latitude'/'longitude' (0.25°). Without rename+interp in build_tensors(), xarray outer-products them to ~15 GB → OOM.
[2026-04-20] Pass merged.latitude.values (numpy) not merged.latitude (DataArray) to interp() — DataArray dim metadata can cause misalignment in downstream xarray comparisons.
[2026-04-20] NCEI OISST THREDDS OPeNDAP URL returns 400 — use direct HTTPS: /data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/
[2026-04-20] xarray netcdf4 backend treats https:// as OPeNDAP and rejects BytesIO — use tempfile.NamedTemporaryFile + os.unlink for plain-HTTP NetCDF4 downloads
[2026-04-20] OISST per-day NCEI direct downloads (10800 requests × 1.7 MB global files) triggers rate limiting returning HTML with status 200 — use ERDDAP griddap server-side subset (30 annual requests × ~500 KB GoM slice)
[2026-04-20] OISST native lon is 0-360; GoM bbox sel(lon=slice(-71,-66)) returns empty on direct files — use ncdcOisst21Agg_LonPM180 ERDDAP dataset which uses -180/180 convention
[2026-04-20] Parallel GEE sessions (2 WN2 years simultaneously) cause gRPC [Errno 11] resource exhaustion — run WN2 years sequentially on same machine

## Pre-/clear Protocol

When recommending or executing `/clear` (every ~20 turns per global CLAUDE.md), FIRST:

1. Update `mhw_claude_actions/mhw_claude_todo.md`:
   - Mark any completed items done
   - Add any new tasks or blockers discovered this session
   - Update VM statuses if data prep is still running

2. Update `mhw_claude_actions/mhw_claude_recentactions.md`:
   - Append a dated session summary (what was done, decisions made, fixes applied)
   - Format: `[YYYY-MM-DD] **Session N — Title:** bullet list of actions`

Do NOT /clear until both files are updated. These files are the sole context bridge across sessions.

---

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.

<!-- dgc-policy-v11 -->
# Dual-Graph Context Policy

This project uses a local dual-graph MCP server for efficient context retrieval.

## MANDATORY: Always follow this order

1. **Call `graph_continue` first** — before any file exploration, grep, or code reading.

2. **If `graph_continue` returns `needs_project=true`**: call `graph_scan` with the
   current project directory (`pwd`). Do NOT ask the user.

3. **If `graph_continue` returns `skip=true`**: project has fewer than 5 files.
   Do NOT do broad or recursive exploration. Read only specific files if their names
   are mentioned, or ask the user what to work on.

4. **Read `recommended_files`** using `graph_read` — **one call per file**.
   - `graph_read` accepts a single `file` parameter (string). Call it separately for each
     recommended file. Do NOT pass an array or batch multiple files into one call.
   - `recommended_files` may contain `file::symbol` entries (e.g. `src/auth.ts::handleLogin`).
     Pass them verbatim to `graph_read(file: "src/auth.ts::handleLogin")` — it reads only
     that symbol's lines, not the full file.
   - Example: if `recommended_files` is `["src/auth.ts::handleLogin", "src/db.ts"]`,
     call `graph_read(file: "src/auth.ts::handleLogin")` and `graph_read(file: "src/db.ts")`
     as two separate calls (they can be parallel).

5. **Check `confidence` and obey the caps strictly:**
   - `confidence=high` -> Stop. Do NOT grep or explore further.
   - `confidence=medium` -> If recommended files are insufficient, call `fallback_rg`
     at most `max_supplementary_greps` time(s) with specific terms, then `graph_read`
     at most `max_supplementary_files` additional file(s). Then stop.
   - `confidence=low` -> Call `fallback_rg` at most `max_supplementary_greps` time(s),
     then `graph_read` at most `max_supplementary_files` file(s). Then stop.

## Token Usage

A `token-counter` MCP is available for tracking live token usage.

- To check how many tokens a large file or text will cost **before** reading it:
  `count_tokens({text: "<content>"})`
- To log actual usage after a task completes (if the user asks):
  `log_usage({input_tokens: <est>, output_tokens: <est>, description: "<task>"})`
- To show the user their running session cost:
  `get_session_stats()`

Live dashboard URL is printed at startup next to "Token usage".

## Rules

- Do NOT use `rg`, `grep`, or bash file exploration before calling `graph_continue`.
- Do NOT do broad/recursive exploration at any confidence level.
- `max_supplementary_greps` and `max_supplementary_files` are hard caps - never exceed them.
- Do NOT dump full chat history.
- Do NOT call `graph_retrieve` more than once per turn.
- After edits, call `graph_register_edit` with the changed files. Use `file::symbol` notation (e.g. `src/auth.ts::handleLogin`) when the edit targets a specific function, class, or hook.

## Context Store

Whenever you make a decision, identify a task, note a next step, fact, or blocker during a conversation, call `graph_add_memory`.

**To add an entry:**
```
graph_add_memory(type="decision|task|next|fact|blocker", content="one sentence max 15 words", tags=["topic"], files=["relevant/file.ts"])
```

**Do NOT write context-store.json directly** — always use `graph_add_memory`. It applies pruning and keeps the store healthy.

**Rules:**
- Only log things worth remembering across sessions (not every minor detail)
- `content` must be under 15 words
- `files` lists the files this decision/task relates to (can be empty)
- Log immediately when the item arises — not at session end

## Session End

When the user signals they are done (e.g. "bye", "done", "wrap up", "end session"), proactively update `CONTEXT.md` in the project root with:
- **Current Task**: one sentence on what was being worked on
- **Key Decisions**: bullet list, max 3 items
- **Next Steps**: bullet list, max 3 items

Keep `CONTEXT.md` under 20 lines total. Do NOT summarize the full conversation — only what's needed to resume next session.
