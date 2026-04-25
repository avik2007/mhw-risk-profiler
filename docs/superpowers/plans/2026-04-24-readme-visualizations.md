# README Visualizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate four figures for the GitHub README showcasing spatial WN2/HYCOM inputs, network architecture, and XAI attribution.

**Architecture:** Two standalone scripts — `generate_spatial_figures.py` (GCS-dependent) and `generate_network_diagram.py` (standalone). All output to `docs/assets/figures/`.

**Tech Stack:** Python, matplotlib, cartopy, gcsfs, xarray, PIL

---

### Task 1: Network architecture diagram

**Files:**
- Create: `scripts/generate_network_diagram.py`
- Output: `docs/assets/figures/network_architecture.png`

- [ ] Write `scripts/generate_network_diagram.py` — pure matplotlib vertical flow diagram
- [ ] Run: `/home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/generate_network_diagram.py`
- [ ] Verify `docs/assets/figures/network_architecture.png` exists and is non-zero

---

### Task 2: Spatial figures (WN2 + HYCOM) and XAI grid

**Files:**
- Create: `scripts/generate_spatial_figures.py`
- Output: `docs/assets/figures/wn2_sst_mean_spread.png`, `docs/assets/figures/hycom_sst.png`, `docs/assets/figures/xai_attribution_grid.png`

- [ ] Set env: `GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json`
- [ ] Write `scripts/generate_spatial_figures.py`
  - Load WN2 zarr from GCS, pick best snapshot (max ensemble spread date)
  - Plot WN2 mean + spread side-by-side with cartopy
  - Load HYCOM zarr, plot surface SST same date
  - Stitch existing XAI PNGs into 2×2 grid
- [ ] Run: `MHW_GCS_BUCKET=gs://mhw-risk-cache GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json /home/avik2007/miniconda3/envs/mhw-risk/bin/python scripts/generate_spatial_figures.py`
- [ ] Verify all 3 PNGs exist

---

### Task 3: Update README

**Files:**
- Modify: `README.md`

- [ ] Add `## Visualizations` section with embedded figure references
- [ ] Commit all figures + README + scripts
