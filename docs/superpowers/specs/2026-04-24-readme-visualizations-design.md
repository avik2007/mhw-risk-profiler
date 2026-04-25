# README Visualizations Design
**Date:** 2026-04-24  
**Status:** Approved

---

## Goal

Add four figures to the GitHub README to showcase: spatial model inputs (WN2 + HYCOM SST), the dual-stream model architecture, and seasonal XAI attribution comparing ERA5 vs WN2.

---

## Figure Inventory

All figures saved to `docs/assets/figures/` and committed to git.

| File | Content | Source |
|---|---|---|
| `wn2_sst_mean_spread.png` | Two-panel: WN2 ensemble mean SST (left) + ensemble spread (right) | GCS WN2 zarr |
| `hycom_sst.png` | Single-panel: HYCOM 0.08° surface SST, same date as WN2 | GCS HYCOM zarr |
| `xai_attribution_grid.png` | 2×2 grid of existing seasonal IG attribution PNGs (DJF/MAM/JJA/SON) | `data/results/xai/` |
| `network_architecture.png` | Vertical flow diagram of MHWRiskModel architecture | Pure matplotlib, no GCS |

---

## Script A: `scripts/generate_spatial_figures.py`

**Snapshot selection:** Scan WN2 zarr on GCS, compute per-date ensemble std across 64 members, pick date with highest spread. Load HYCOM SST for same date.

**WN2 figure:** Two-panel matplotlib + cartopy PlateCarree. Left: ensemble mean SST with `RdYlBu_r`. Right: ensemble spread (std) with `YlOrRd`. GoM bbox. Coastlines + land mask. Shared colorbar per panel.

**HYCOM figure:** Single panel, same region + colormap as WN2 mean panel, same date.

**XAI grid:** Load 4 existing PNGs from `data/results/xai/`, stitch 2×2 with matplotlib, season labels (DJF/MAM/JJA/SON).

**GCS access:** `gcsfs` + `xr.open_zarr`. Bucket/path from env vars (`GCS_BUCKET`), same pattern as existing ingestion code.

---

## Script B: `scripts/generate_network_diagram.py`

Pure matplotlib, no GCS, no model weights needed.

**Architecture flow (top to bottom):**
```
HYCOM: (batch, 64, 11, 4)          WN2: (batch, 64, 90, 5)
          ↓                                    ↓
  CNN1dEncoder [blue]              TransformerEncoder [green]
  Conv1d(4→32) + BN + ReLU        Linear(5→128) + SinPE
  Conv1d(32→64) + BN + ReLU       4× PreNorm layers
  Conv1d(64→128) + BN + ReLU      (8-head attn, d_ff=256)
  + residual skip (32→128)        Mean pool over 90 days
  AdaptiveAvgPool1d → (B×64,128)  → (B×64,128)
                    ↓                    ↓
              LeakyGate [orange]
              gate ∈ [0.1, 0.9]
              fused: (B×64, 128)
                         ↓
              Linear(128→1) + Softplus
                         ↓
              SDD: (batch, 64)  →  SVaR_95
```

Color-coded boxes: CNN=blue, Transformer=green, Gate=orange, Head=grey. Tensor shapes annotated on arrows.

---

## README Embedding

New `## Visualizations` section after the existing pipeline ASCII diagram:

```markdown
### Spatial Inputs: WeatherNext 2 SST
![WN2 SST mean and ensemble spread](docs/assets/figures/wn2_sst_mean_spread.png)

### Spatial Inputs: HYCOM SST
![HYCOM SST](docs/assets/figures/hycom_sst.png)

### Model Architecture
![Network Architecture](docs/assets/figures/network_architecture.png)

### XAI: Integrated Gradients Attribution (ERA5 vs WN2)
![XAI Attribution](docs/assets/figures/xai_attribution_grid.png)
```

---

## Constraints

- No git push until user reviews generated figures.
- GCS auth via existing env vars / service account — same as ingestion pipeline.
- Use `mhw-risk` conda env (Python at `/home/avik2007/miniconda3/envs/mhw-risk/bin/python`).
- No new dependencies beyond what's already in requirements.txt (`gcsfs`, `matplotlib`, `cartopy`, `PIL`).
