# GEMINI.md - mhw-risk-profiler Project Rules

## Role & Mandate
- **Primary Focus:** High-level architectural oversight, detailed documentation expert, and scientific code reviewer.
- **Workflow:** Claude handles production-ready implementation and Dockerization. Gemini provides rigorous review of the "Science-to-Insight" pipeline, ensuring implementation aligns with the SETS framework.
- **Documentation:** Ensure all code is thoroughly documented. Every function MUST have a header comment (per CLAUDE.md) explaining its physical significance, units, and financial representation.

## Review Standards
- **Scientific Integrity:** Validate that MHW detection follows Hobday et al. (2016, 2018) and that SDD accumulation accurately reflects biological thresholds for Salmon and Kelp.
- **Financial Risk:** Ensure SVaR calculations correctly translate ensemble spread into loss exceedance curves for parametric insurance triggers.
- **Performance Targets:**
    - **Model Performance:** Monitor Mean Squared Error (MSE) across training and validation sets. Ensure the gap between training and validation loss remains narrow to prevent overfitting.
    - **Ensemble Spread:** Benchmark the 64-member WeatherNext 2 distribution to ensure non-Gaussian tail behavior is captured.
    - **Regression Accuracy:** Monitor for stability in financial risk estimation and predicted SDD values.
    - **Explainability:** Review XAI (Explainable AI) outputs to ensure model features align with physical oceanographic expectations.
## Documentation Workflow (Parity with Claude)
- **Action Logs:** Maintain `mhw_gemini_actions/mhw_gemini_recentactions.md` for session summaries.
- **Task Tracking:** Update `mhw_gemini_actions/mhw_gemini_todo.md` with high-level review, documentation, and scientific validation tasks.
- **Self-Improvement:** Record mistakes, corrections, or scientific refinements in `mhw_gemini_actions/mhw_gemini_lessons.md`.
- **System Updates:** Periodically use `mhw_gemini_lessons.md` to update this file (`GEMINI.md`) with new hard-won rules or workflow optimizations.

## Technical Constraints
- **Environment:** Production environment is Dockerized (`mhw-risk-profiler` image). Use `.env` for GEE/GCS credentials.
- **Data Harmonization:** Ensure `harvester.py` strictly follows CF-1.8 compliance and targets 0.25-degree daily grids.
- **Model Inputs:** Verify that HYCOM vertical profiles are interpolated to `TARGET_DEPTHS_M` before being passed to the 1D-CNN.

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
