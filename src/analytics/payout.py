"""
payout.py — Parametric insurance payout engine for MHW aquaculture risk
========================================================================
Translates ensemble SVaR estimates into insurance payout amounts for
parametric aquaculture insurance contracts triggered by Stress Degree Days.

Parametric insurance structure
-------------------------------
A parametric contract pays out automatically when a measurable index (here,
SVaR — the 95th-percentile SDD across ensemble members) crosses a pre-agreed
attachment point. There is no claims adjustment; payment is triggered purely
by the index value.

  Attachment point a [°C·day]:  SVaR must exceed this before any payout.
  Cap c [°C·day]:               SVaR at or above this triggers full payout.
  Coverage V [USD]:             Maximum insurance payout (sum insured).

  payout = V × clip((SVaR − a) / (c − a), 0, 1)

Financial motivation
---------------------
SVaR_95 = 40 °C·day with attachment = 20, cap = 60, coverage = $1M →
payout = $1M × (40−20)/(60−20) = $500,000.

The spread between SVaR_50 and SVaR_95 guides reinsurance pricing:
  tight spread  → low model uncertainty → lower uncertainty loading
  wide spread   → high model uncertainty → higher risk premium required

Dependencies: torch>=2.2.0
"""
from __future__ import annotations

import torch


def compute_payout(
    svar: torch.Tensor,
    attachment: float,
    cap: float,
    coverage: float,
) -> torch.Tensor:
    """
    Compute parametric insurance payout from SVaR estimates.

    Parameters
    ----------
    svar : torch.Tensor
        Stochastic Value-at-Risk [°C·day].  Any shape — the function is
        element-wise.  Typically shape (batch,) from compute_svar(), or
        shape (batch, member) for per-member analysis.
    attachment : float
        Attachment point [°C·day].  SVaR below this triggers zero payout.
        Typical range: 15–25 °C·day for Gulf of Maine salmon aquaculture.
    cap : float
        Cap [°C·day].  SVaR at or above this triggers full coverage payout.
        Must be strictly greater than attachment.
        Typical range: 40–80 °C·day depending on species and season length.
    coverage : float
        Sum insured [USD].  Maximum payout per contract.
        Set to the replacement cost of at-risk biomass plus operating losses.

    Returns
    -------
    payout : torch.Tensor
        Insurance payout [USD], same shape as svar.
        All values in [0, coverage].
    """
    if cap <= attachment:
        raise ValueError(
            f"cap ({cap}) must be greater than attachment ({attachment}). "
            "A contract where cap ≤ attachment is undefined (denominator ≤ 0)."
        )
    denom = cap - attachment
    rate = torch.clamp((svar - attachment) / denom, min=0.0, max=1.0)
    return coverage * rate


def compute_expected_loss_ratio(
    sdd: torch.Tensor,
    attachment: float,
    cap: float,
    coverage: float,
) -> dict[str, torch.Tensor]:
    """
    Compute SVaR-based payout and loss ratio statistics from the full ensemble.

    Parameters
    ----------
    sdd : torch.Tensor, shape (batch, member)
        Per-member Stress Degree Days [°C·day] from MHWRiskModel or
        from accumulate_sdd() converted to a tensor.
    attachment : float
        Attachment point [°C·day].  See compute_payout() for details.
    cap : float
        Cap [°C·day].  See compute_payout() for details.
    coverage : float
        Sum insured [USD].  See compute_payout() for details.

    Returns
    -------
    result : dict[str, torch.Tensor]
        All tensors have shape (batch,):
          "svar_95"      — SVaR at 95th percentile [°C·day]
          "payout_95"    — insurance payout triggered by SVaR_95 [USD]
          "loss_ratio_95"— payout_95 / coverage ∈ [0, 1]
          "payout_50"    — payout triggered by SVaR_50 (median scenario) [USD]
    """
    from .svar import compute_ensemble_stats

    stats = compute_ensemble_stats(sdd)
    svar_95   = stats["svar_95"]
    svar_50   = stats["svar_50"]
    payout_95 = compute_payout(svar_95, attachment, cap, coverage)
    payout_50 = compute_payout(svar_50, attachment, cap, coverage)
    return {
        "svar_95":        svar_95,
        "payout_95":      payout_95,
        "loss_ratio_95":  payout_95 / coverage,
        "payout_50":      payout_50,
    }
