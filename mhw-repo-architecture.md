# mhw-repo-architecture.md
# Annotated directory tree for mhw-risk-profiler
# -----------------------------------------------

```
mhw-risk-profiler/
|
|-- mhw_ai_research/                        # Research inputs — do not modify programmatically
|   |-- Gemini-AI-Driven Marine Habitat Suitability.txt
|   |-- NotebookLM-MHWRiskprofiler-deepdive.txt
|   `-- Perplexity-MHWRiskprofiler-deepdive.txt
|
|-- data/                                   # Local data store — git-ignored, never committed
|   |-- raw/                                # Original WeatherNext 2 Zarr tiles and HYCOM NetCDF
|   |-- processed/                          # Harmonized 0.25-deg CF-compliant Zarr outputs
|   `-- cache/                              # Intermediate Dask and GEE export scratch space
|
|-- src/                                    # All production source code lives here
|   |-- __init__.py
|   |
|   |-- ingestion/                          # Data acquisition and harmonization layer
|   |   |-- __init__.py
|   |   `-- harvester.py                    # WeatherNext2Harvester, HYCOMLoader, DataHarmonizer
|   |                                       # + run_ingestion_pipeline() CLI entry point
|   |
|   |-- models/                             # ML architecture for MHW sequence modeling
|   |   |-- __init__.py
|   |   |-- cnn1d.py                        # PyTorch 1D-CNN — local thermal feature extraction
|   |   |-- transformer.py                  # Transformer encoder — long-range SST dependencies
|   |   `-- ensemble_wrapper.py             # Wraps FGN ensemble members for uncertainty propagation
|   |
|   `-- analytics/                          # Risk quantification layer
|       |-- __init__.py
|       |-- stress_degree_days.py           # MHW Stress Degree Day accumulation (Hobday et al.)
|       `-- var_engine.py                   # Financial Value-at-Risk from ensemble SDD distributions
|
|-- mhw_claude_actions/                     # AI-assisted development audit trail
|   |-- mhw_claude_todo.md                  # Active task queue — what is being worked on now
|   |-- mhw_claude_recentactions.md         # Completed actions log — what was done and when
|   `-- mhw_claude_lessons.md               # Root-cause fixes and non-obvious discoveries
|
|-- .gitignore                              # Excludes data/, mhw_ai_research/, secrets, __pycache__
|-- CLAUDE.md                               # AI development principles and verification rules
|-- Dockerfile                              # python:3.11-slim + libgdal-dev + libnetcdf-dev
|-- mhw-repo-architecture.md               # This file
|-- mondal-mhw-gcp-info.md                 # Personal GCP infra details — git-ignored, never commit
|-- README.md                              # Project mission (SETS Framework)
`-- requirements.txt                        # Pinned Python dependencies
```
