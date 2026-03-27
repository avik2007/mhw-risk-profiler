# Design: 1D-CNN + Transformer Architecture for MHW Risk Profiler
**Date:** 2026-03-27
**Status:** Approved — implementation in progress

---

## Problem

The harmonized atmosphere-ocean dataset (64-member WeatherNext 2 ensemble + HYCOM reanalysis)
needs to be encoded into per-member Stress Degree Day (SDD) predictions for financial
Value-at-Risk estimation. The architecture must be explainable via Captum Integrated
Gradients to satisfy the "Science-to-Insight" requirement for insurance defensibility.

---

## Architecture Overview

Two independent encoders process depth-profile and SST-sequence streams. Their outputs
are fused with a leaky gate and passed through a regression head.

```
HYCOM:  (batch, member, depth=11, channels=4) ─► CNN1dEncoder  ─► (B*M, 128) depth_feat
WN2:    (batch, member, time=90, features=5)  ─► TransformerEncoder ─► (B*M, 128) time_feat
                                                        │
                                                  LeakyGate (α=0.1)
                                                        │
                                                   fused (B*M, 128)  ◄─ Captum hook
                                                        │
                                              Softplus(Linear(128→1))
                                                        │
                                           reshape → sdd (batch, member)
```

**Output tuple:** `(sdd, latent, gate)`
- `sdd`: `(batch, member)` — non-negative SDD prediction per ensemble member
- `latent`: `(batch, member, 128)` — fused embedding for Captum IG attribution
- `gate`: `(batch, member)` — gate value ∈ [0.1, 0.9] for regime monitoring

---

## Module Boundaries

| File | Class | Input | Output |
|---|---|---|---|
| `src/models/cnn1d.py` | `CNN1dEncoder` | `(B, 11, 4)` | `(B, 128)` |
| `src/models/transformer.py` | `TransformerEncoder` | `(B, 90, 5)` | `(B, 128)` |
| `src/models/ensemble_wrapper.py` | `LeakyGate` | `(B, 128)` × 2 | `(B, 128), (B, 1)` |
| `src/models/ensemble_wrapper.py` | `MHWRiskModel` | `(batch, M, 11, 4), (batch, M, 90, 5)` | `(batch, M), (batch, M, 128), (batch, M)` |

`MHWRiskModel` is the sole public API. Encoders are internal components.

---

## CNN1dEncoder Detail

**Purpose:** Extract depth-resolved thermal features from HYCOM vertical profiles.
Detects mixed-layer depth, thermocline strength, and salinity stratification.

- Input permuted to `(B, channels=4, depth=11)` — Conv1d slides along vertical axis
- Layer 1: Conv1d(4→32, k=3, p=1) → BatchNorm → ReLU
- Layer 2: Conv1d(32→64, k=3, p=1) → BatchNorm → ReLU
- Layer 3: Conv1d(64→128, k=3, p=1) → BatchNorm → ReLU
- Residual: Conv1d(32→128, k=1) applied to Layer 1 output, added to Layer 3 output
- AdaptiveAvgPool1d(1) → squeeze → `(B, 128)`

**Residual rationale:** Preserves high-frequency shallow gradient signal (0-10m mixed layer)
through 3 nonlinear transforms. Critical for thermocline detection at variable MLD.

**Pooling rationale:** Vertical translational invariance — thermocline at depth level 3
or 7 produces the same feature magnitude.

---

## TransformerEncoder Detail

**Purpose:** Capture long-range atmospheric forcing patterns over the 90-day season.
Encodes how multi-day wind anomalies and pressure blocking events accumulate MHW risk.

- Linear(5→128) projects WN2 variables to d_model
- Fixed sinusoidal positional encoding (90 steps, 128 dims)
- 4× pre-norm TransformerEncoderLayer: LayerNorm → MHA(8 heads) → residual → LayerNorm → FF(128→256→128) → residual
- mean(dim=1) over time axis → `(B, 128)`

**Pre-norm rationale:** Stable gradient magnitudes through all 4 layers — essential for
Captum IG which backpropagates through the full stack. Post-norm produces exploding
gradients in deeper attention stacks.

**Fixed sinusoidal encoding:** Preserves temporal position semantics under year-to-year
distribution shift (day-45 = mid-season regardless of training year).

---

## LeakyGate Detail

**Purpose:** Fuse depth and time features while guaranteeing both streams contribute.
Gate value exposed for monitoring depth-vs-time dominance (regime switching).

```
gate_logit = Linear(256→1)(cat([depth_feat, time_feat]))
gate = 0.1 + 0.8 × sigmoid(gate_logit)    # ∈ [0.1, 0.9]
fused = gate × depth_feat + (1 - gate) × time_feat
```

Gate → 1.0: depth (HYCOM subsurface) dominates — stratification-driven MHW
Gate → 0.0: time (WN2 atmospheric) dominates — surface forcing-driven MHW

---

## Regression Head

```
sdd = Softplus(Linear(128→1)(latent))
```

Softplus ensures SDD ≥ 0 with smooth gradients — supports insurance tail-risk trend
analysis where gradient of payout with respect to SDD must be continuous.

---

## Verification Requirements

Three smoke tests must pass before task is marked complete:

1. **Shape + non-negativity:** `sdd ∈ (2,4)`, `latent ∈ (2,4,128)`, `gate ∈ (2,4)`, all SDD ≥ 0
2. **Member sensitivity:** +3σ temperature spike on member 0 → `sdd[0,0] > mean(sdd[0,1:])`
3. **Captum IG:** `IntegratedGradients` initialized and attributed without shape mismatch

---

## Deferred

**[LOW PRIORITY] MTSFT — FFT-enriched Transformer**
Upgrade `TransformerEncoder` to Multi-Temporal Scale Fusion Transformer per
`NotebookLM-MHWRiskprofiler-deepdive.txt`. Enrich input with FFT-derived spectral
features alongside raw WN2 variables. Deferred until baseline architecture is
verified and Captum IG interpretability is validated.
