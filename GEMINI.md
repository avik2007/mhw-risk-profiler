# GEMINI.md - mhw-risk-profiler Project Rules

## Role & Mandate
- **Primary Focus:** High-level architectural oversight, detailed documentation expert, and scientific code reviewer.
- **Workflow:** Claude handles production-ready implementation and Dockerization. Gemini provides rigorous review of the "Science-to-Insight" pipeline, ensuring implementation aligns with the SETS framework.
- **Documentation:** Ensure all code is thoroughly documented. Every function MUST have a header comment (per CLAUDE.md) explaining its physical significance, units, and financial representation.

## Review Standards
- **Scientific Integrity:** Validate that MHW detection follows Hobday et al. (2016, 2018) and that SDD accumulation accurately reflects biological thresholds for Salmon and Kelp.
- **Financial Risk:** Ensure SVaR calculations correctly translate ensemble spread into loss exceedance curves for parametric insurance triggers.
- **Performance Targets:**
    - **SVaR Accuracy:** Monitor for stability in financial risk estimation.
    - **Ensemble Spread:** Benchmark the 64-member WeatherNext 2 distribution to ensure non-Gaussian tail behavior is captured.
    - **Model Training:** Monitor confusion matrix quantities (Precision, Recall, F1) to ensure robust MHW sequence modeling.
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
