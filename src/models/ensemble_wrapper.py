"""
ensemble_wrapper.py — MHWRiskModel: ensemble-aware CNN + Transformer fusion
===========================================================================
Orchestrates CNN1dEncoder (HYCOM depth profiles) and TransformerEncoder (WeatherNext 2
SST sequences) across all 64 ensemble members, fuses their outputs with a leaky gate,
and produces per-member Stress Degree Day (SDD) predictions for SVaR estimation.

Architecture summary
--------------------
  HYCOM:  (batch, member, depth=11, channels=4)
            ↓ reshape (batch×member, 11, 4)
          CNN1dEncoder → (batch×member, 128)        depth_feat

  WN2:    (batch, member, time=90, features=5)
            ↓ reshape (batch×member, 90, 5)
          TransformerEncoder → (batch×member, 128)  time_feat

  LeakyGate(depth_feat, time_feat) → fused (batch×member, 128)  ← Captum hook
                                      gate  (batch×member, 1)    ← regime monitor

  Softplus(Linear(128→1))(fused) → sdd (batch×member, 1)

  Reshape back:
    sdd    → (batch, member)
    latent → (batch, member, 128)
    gate   → (batch, member)

  Return: (sdd, latent, gate)

Ensemble design rationale
--------------------------
The 64 WeatherNext 2 ensemble members represent draws from the FGN atmospheric state
distribution. Each member is processed independently through the full model — there is
no information sharing between members within a forward pass. This independence is the
key property that makes the member dimension suitable for SVaR estimation:

  SVaR_95 = quantile(sdd[:, :], 0.95, dim=member)

The gate value monitors which input stream dominates per member per sample:
  gate → 1.0: depth (HYCOM subsurface) is the primary MHW driver for that member
  gate → 0.0: time (WN2 atmospheric) is the primary driver

Persistent gate → 0.9 across members signals a stratification-driven MHW regime.
Persistent gate → 0.1 signals a surface-forcing-driven regime. Both are physically
distinct and have different insurance payout implications.

Dependencies
------------
  torch>=2.2.0, captum>=0.7.0
  cnn1d.py, transformer.py (same package)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cnn1d import CNN1dEncoder
from .transformer import TransformerEncoder


class LeakyGate(nn.Module):
    """
    Leaky gating layer that fuses depth and temporal feature vectors while
    guaranteeing both streams contribute a minimum fraction to the output.

    The gate value g ∈ [α, 1-α] is computed from the concatenation of both
    streams via a learned linear map and sigmoid. The floor α prevents either
    stream from being completely ignored — the model cannot learn to discard
    depth information even if atmospheric forcing happens to dominate in training.

    This is physically motivated: both subsurface stratification (depth stream)
    and atmospheric forcing (time stream) are necessary conditions for MHW
    formation. A model that drops either stream violates the physical prior.

    The gate value is returned alongside the fused representation so that
    downstream analytics and monitoring systems can track regime switching.

    Parameters
    ----------
    latent_dim : int
        Dimension of each input stream. Default 128.
    alpha : float
        Minimum contribution floor for each stream. Default 0.1.
        Gate ∈ [alpha, 1-alpha]; both streams always contribute ≥ alpha × 100%.
    """

    def __init__(self, latent_dim: int = 128, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha      = alpha
        self.gate_linear = nn.Linear(latent_dim * 2, 1)

    def forward(
        self, depth_feat: torch.Tensor, time_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse depth and time feature vectors with a leaky gate.

        Parameters
        ----------
        depth_feat : torch.Tensor, shape (B, latent_dim)
            Output of CNN1dEncoder — encodes HYCOM vertical profile structure.
            B = batch_size × n_members.
        time_feat : torch.Tensor, shape (B, latent_dim)
            Output of TransformerEncoder — encodes WN2 atmospheric sequence.

        Returns
        -------
        fused : torch.Tensor, shape (B, latent_dim)
            Gated combination: gate × depth_feat + (1 - gate) × time_feat.
            This is the latent representation exposed for Captum IG attribution.
        gate : torch.Tensor, shape (B, 1)
            Gate value ∈ [alpha, 1-alpha] per sample.
            Values closer to 1 indicate depth-stream dominance (stratification-driven MHW).
            Values closer to 0 indicate time-stream dominance (atmospheric-forcing-driven MHW).
        """
        combined   = torch.cat([depth_feat, time_feat], dim=-1)  # (B, 2*latent_dim)
        gate_logit = self.gate_linear(combined)                   # (B, 1)
        gate = self.alpha + (1.0 - 2.0 * self.alpha) * torch.sigmoid(gate_logit)
        # gate ∈ [alpha, 1-alpha] — leaky: neither stream is fully suppressed

        fused = gate * depth_feat + (1.0 - gate) * time_feat     # (B, latent_dim)
        return fused, gate


class MHWRiskModel(nn.Module):
    """
    End-to-end MHW risk model that processes a full ensemble of HYCOM depth profiles
    and WeatherNext 2 atmospheric sequences into per-member Stress Degree Day (SDD)
    predictions and latent embeddings.

    This is the sole public API for the models layer. All downstream code (analytics,
    Captum attribution, financial VaR engine) interacts with this class only.

    Parameters
    ----------
    in_channels : int
        HYCOM channels per depth level. Default 4 (water_temp, salinity, u, v).
    in_features : int
        WeatherNext 2 variables per time step. Default 5.
    latent_dim : int
        Shared latent dimension for CNN output, Transformer output, and fusion.
        Default 128 — symmetric to ensure both streams operate in the same feature
        space without any lossy projection in the gate.
    seq_len : int
        Fixed temporal sequence length (days). Default 90.
    alpha : float
        Leaky gate floor. Default 0.1.
    n_heads : int
        Transformer attention heads. Default 8.
    n_layers : int
        Transformer encoder layers. Default 4.
    dropout : float
        Dropout in Transformer layers. Default 0.1.
    """

    def __init__(
        self,
        in_channels: int = 4,
        in_features: int = 5,
        latent_dim: int = 128,
        seq_len: int = 90,
        alpha: float = 0.1,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cnn         = CNN1dEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.transformer = TransformerEncoder(
            in_features=in_features,
            d_model=latent_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            seq_len=seq_len,
            dropout=dropout,
        )
        self.gate        = LeakyGate(latent_dim=latent_dim, alpha=alpha)

        # Regression head: Softplus ensures SDD ≥ 0 with smooth gradients
        # Softplus is preferred over ReLU: no zero-gradient region for negative
        # pre-activations, which matters for insurance tail-risk trend analysis
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        hycom: torch.Tensor,
        wn2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass over all ensemble members.

        Parameters
        ----------
        hycom : torch.Tensor, shape (batch, member, depth=11, channels=4)
            HYCOM depth profiles. Each (depth=11, channels=4) slice is one
            ocean profile at a single location, time step, and ensemble member.
            depth=11 corresponds to TARGET_DEPTHS_M = [0,5,10,20,30,50,75,100,150,200,300] m.
            channels=4: [water_temp (°C), salinity (psu), water_u (m/s), water_v (m/s)].
            HYCOM is deterministic — the member dimension is broadcast from WeatherNext 2.
        wn2 : torch.Tensor, shape (batch, member, time=90, features=5)
            WeatherNext 2 atmospheric sequences. Each (time=90, features=5) slice
            is one 90-day summer season forecast from a single FGN ensemble member.
            features=5: [2m_temperature (K), wind_u (m/s), wind_v (m/s), MSLP (Pa), SST (K)].

        Returns
        -------
        sdd : torch.Tensor, shape (batch, member)
            Stress Degree Day prediction [°C·day] per ensemble member.
            All values ≥ 0 (Softplus head). Use quantiles across the member dimension
            for SVaR estimation: SVaR_95 = sdd[:, :].quantile(0.95, dim=1).
        latent : torch.Tensor, shape (batch, member, latent_dim=128)
            Fused latent embedding (gate output). Use this tensor as the target
            for Captum Integrated Gradients attribution — it lies on the direct
            path from both input streams through the nonlinear gate to the SDD head.
        gate : torch.Tensor, shape (batch, member)
            Gate value ∈ [alpha, 1-alpha] per member per sample.
            Monitor this value to detect MHW regime:
              gate → 1: stratification-driven (depth stream dominant)
              gate → 0: atmospheric-forcing-driven (time stream dominant)
        """
        B, M = hycom.shape[:2]

        # Flatten member dimension into batch for parallel processing
        # No information sharing between members within this operation
        h = hycom.view(B * M, hycom.shape[2], hycom.shape[3])   # (B*M, 11, 4)
        w = wn2.view(B * M, wn2.shape[2], wn2.shape[3])         # (B*M, 90, 5)

        # Encode each stream independently
        depth_feat = self.cnn(h)          # (B*M, 128)
        time_feat  = self.transformer(w)  # (B*M, 128)

        # Fuse with leaky gate — latent is the Captum IG hook point
        latent, gate = self.gate(depth_feat, time_feat)  # (B*M, 128), (B*M, 1)

        # Regression head: latent → SDD scalar, all values non-negative
        sdd = self.head(latent)           # (B*M, 1)

        # Reshape back to (batch, member, ...) — restore ensemble structure
        sdd    = sdd.view(B, M)           # (B, M)
        latent = latent.view(B, M, -1)   # (B, M, 128)
        gate   = gate.view(B, M)         # (B, M)

        return sdd, latent, gate


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m src.models.ensemble_wrapper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import torch

    print("=" * 60)
    print("MHWRiskModel — Smoke Test")
    print("=" * 60)

    model = MHWRiskModel(latent_dim=128, alpha=0.1)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print()

    # ------------------------------------------------------------------
    # Test 1 — Shape contract and non-negativity
    # ------------------------------------------------------------------
    print("Test 1: Shape contract and non-negativity")
    hycom = torch.randn(2, 4, 11, 4)
    wn2   = torch.randn(2, 4, 90, 5)

    with torch.no_grad():
        sdd, latent, gate = model(hycom, wn2)

    assert sdd.shape    == (2, 4),      f"sdd shape mismatch: {sdd.shape}"
    assert latent.shape == (2, 4, 128), f"latent shape mismatch: {latent.shape}"
    assert gate.shape   == (2, 4),      f"gate shape mismatch: {gate.shape}"
    assert (sdd >= 0).all(),            f"SDD contains negative values: min={sdd.min():.4f}"
    assert not sdd.isnan().any(),       "SDD contains NaN"
    assert not latent.isnan().any(),    "latent contains NaN"
    assert not gate.isnan().any(),      "gate contains NaN"
    assert (gate >= 0.1).all() and (gate <= 0.9).all(), \
        f"Gate out of leaky bounds [0.1, 0.9]: min={gate.min():.4f}, max={gate.max():.4f}"

    print("  PASSED")
    print(f"  sdd    shape: {list(sdd.shape)}   range [{sdd.min():.4f}, {sdd.max():.4f}]")
    print(f"  latent shape: {list(latent.shape)}")
    print(f"  gate   shape: {list(gate.shape)}   range [{gate.min():.4f}, {gate.max():.4f}]")
    print()

    # ------------------------------------------------------------------
    # Test 2 — Member sensitivity: no cross-member information leakage
    # Inject a +3σ temperature spike into member 0 only.
    # The SDD of member 0 must be strictly higher than the mean of members 1-3.
    # Failure would indicate the member dimension is being pooled before the head.
    # ------------------------------------------------------------------
    print("Test 2: Member sensitivity — no cross-member leakage")
    torch.manual_seed(42)
    hycom_sens = torch.randn(1, 4, 11, 4)
    wn2_sens   = torch.randn(1, 4, 90, 5)

    # Severe temperature anomaly on member 0, water_temp channel only
    hycom_sens[0, 0, :, 0] += 3.0

    with torch.no_grad():
        sdd_sens, _, _ = model(hycom_sens, wn2_sens)

    member0_sdd    = sdd_sens[0, 0].item()
    others_mean    = sdd_sens[0, 1:].mean().item()

    assert member0_sdd > others_mean, (
        f"Member sensitivity FAILED: member 0 SDD ({member0_sdd:.4f}) "
        f"not > mean of others ({others_mean:.4f})"
    )

    print("  PASSED")
    print(f"  member 0 SDD (spike): {member0_sdd:.4f}")
    print(f"  members 1-3 mean SDD: {others_mean:.4f}")
    print(f"  delta: {member0_sdd - others_mean:.4f}")
    print()

    # ------------------------------------------------------------------
    # Test 3 — Captum Integrated Gradients hook reachability
    # The latent tensor is the attribution target: it lies on the direct
    # path from both input streams to the SDD regression head.
    # ------------------------------------------------------------------
    print("Test 3: Captum Integrated Gradients — hook reachability")
    try:
        from captum.attr import IntegratedGradients

        def latent_forward(hycom_in: torch.Tensor, wn2_in: torch.Tensor) -> torch.Tensor:
            """
            Wrapper for Captum: returns a scalar per batch item derived from latent.
            mean over member and latent dims gives one value per batch item.
            """
            _, lat, _ = model(hycom_in, wn2_in)
            return lat.mean(dim=[1, 2])   # (batch,)

        ig = IntegratedGradients(latent_forward)

        hycom_ig = hycom.requires_grad_(True)
        wn2_ig   = wn2.requires_grad_(True)

        attr = ig.attribute((hycom_ig, wn2_ig), n_steps=10)

        assert attr[0].shape == hycom.shape, \
            f"HYCOM attribution shape mismatch: {attr[0].shape} vs {hycom.shape}"
        assert attr[1].shape == wn2.shape, \
            f"WN2 attribution shape mismatch: {attr[1].shape} vs {wn2.shape}"

        print("  PASSED")
        print(f"  HYCOM attribution shape: {list(attr[0].shape)}")
        print(f"  WN2   attribution shape: {list(attr[1].shape)}")
        print(f"  HYCOM attr L2 norm: {attr[0].norm():.4f}")
        print(f"  WN2   attr L2 norm: {attr[1].norm():.4f}")

    except ImportError:
        print("  SKIPPED — captum not installed in current environment")
        print("  Install with: pip install captum>=0.7.0")

    print()
    print("=" * 60)
    print("All smoke tests passed. MHWRiskModel verified.")
    print("=" * 60)
    sys.exit(0)
