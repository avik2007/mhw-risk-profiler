"""
svar.py — Stochastic Value-at-Risk estimation from ensemble SDD distributions
=============================================================================
Computes SVaR from the 64-member WeatherNext 2 ensemble SDD distribution
produced by MHWRiskModel or by the physics-based SDD accumulation pipeline.

Financial motivation
--------------------
SVaR (Stochastic Value-at-Risk) quantifies the tail risk of the ensemble SDD
distribution for parametric insurance pricing:

    SVaR_p = quantile(SDD_members, p)

Interpretation:
  SVaR_95 = 15 °C·day → in the worst 5% of atmospheric scenarios (as represented
  by the 64-member FGN ensemble), the aquaculture site accumulates 15 °C·day of
  thermal stress — sufficient to trigger the insurance payout threshold.

The spread between SVaR_50 and SVaR_95 reflects ensemble forecast uncertainty:
  Tight spread → low model uncertainty → high confidence in risk estimate
  Wide spread  → high model uncertainty → larger insurance risk premium required

This module is the final step of the 'Science-to-Insight' pipeline:
  WeatherNext 2 ensemble → MHWRiskModel → SDD per member → SVaR → payout probability

Dependencies: torch>=2.2.0
"""
from __future__ import annotations

import torch


def compute_svar(
    sdd: torch.Tensor,
    quantile: float = 0.95,
) -> torch.Tensor:
    """
    Estimate Stochastic Value-at-Risk from the ensemble SDD distribution.

    Parameters
    ----------
    sdd : torch.Tensor, shape (batch, member)
        Per-member Stress Degree Days [°C·day].
        Produced by MHWRiskModel.forward() or by accumulate_sdd() converted to tensor.
        All values should be ≥ 0 (guaranteed by MHWRiskModel Softplus head).
    quantile : float
        Probability level for VaR. Must be in (0, 1].
        Common choices:
          0.50 — median (expected SDD under the ensemble)
          0.90 — moderate tail risk (1-in-10 scenario)
          0.95 — standard VaR level used in insurance pricing
          0.99 — extreme tail risk (1-in-100 scenario)

    Returns
    -------
    svar : torch.Tensor, shape (batch,)
        SVaR at the requested quantile level [°C·day].
        One value per batch item (location × season combination).
    """
    if not (0.0 < quantile <= 1.0):
        raise ValueError(
            f"quantile must be in (0, 1], got {quantile}. "
            "Common values: 0.50 (median), 0.95 (standard VaR), 0.99 (extreme tail)."
        )
    return torch.quantile(sdd.float(), q=quantile, dim=1)   # (batch,)


def compute_ensemble_stats(
    sdd: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute a full set of ensemble statistics for reporting and model monitoring.

    Parameters
    ----------
    sdd : torch.Tensor, shape (batch, member)
        Per-member SDD [°C·day] from MHWRiskModel or physics-based accumulation.

    Returns
    -------
    stats : dict[str, torch.Tensor]
        Keys and shapes (all shape (batch,)):
          "mean"    — ensemble mean SDD [°C·day]
          "std"     — ensemble standard deviation (uncertainty spread)
          "svar_50" — median SVaR (expected scenario)
          "svar_90" — 90th-percentile SVaR (moderate tail)
          "svar_95" — 95th-percentile SVaR (standard insurance VaR level)
          "svar_99" — 99th-percentile SVaR (extreme tail, stress-test scenario)

        Financial interpretation of std:
          High std → large spread between worst and median member → higher risk premium
          Low std  → ensemble agrees → lower uncertainty loading on the insurance price
    """
    sdd_f = sdd.float()
    return {
        "mean":    sdd_f.mean(dim=1),
        "std":     sdd_f.std(dim=1),
        "svar_50": torch.quantile(sdd_f, 0.50, dim=1),
        "svar_90": torch.quantile(sdd_f, 0.90, dim=1),
        "svar_95": torch.quantile(sdd_f, 0.95, dim=1),
        "svar_99": torch.quantile(sdd_f, 0.99, dim=1),
    }
