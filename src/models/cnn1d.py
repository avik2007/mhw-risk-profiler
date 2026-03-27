"""
cnn1d.py — 1D-CNN depth-profile encoder for MHW Risk Profiler
==============================================================
Encodes HYCOM vertical ocean profiles (temperature, salinity, currents) into a
compact 128-dimensional latent feature vector for downstream fusion with the
WeatherNext 2 temporal context from TransformerEncoder.

Physical motivation
-------------------
The vertical structure of the water column is the primary ocean-side predictor of
Marine Heatwave severity:

  - Mixed Layer Depth (MLD): A shallow MLD means surface heat accumulates rapidly —
    the ocean has little thermal inertia. Conv1d layers with kernel=3 detect the
    depth range over which temperature is nearly isothermal (the mixed layer).
  - Thermocline strength: A sharp temperature gradient between the mixed layer and
    deep water suppresses vertical mixing. The residual skip connection from Layer 1
    ensures this high-frequency signal (large dT/dz at 10-30 m) is preserved through
    three nonlinear transforms.
  - Halocline / density stratification: Salinity controls buoyancy independently of
    temperature. Including salinity as a channel allows the model to distinguish
    temperature-driven from salinity-driven stratification.
  - Horizontal currents (u, v): Lateral heat advection can pre-condition a location
    for MHW before the atmospheric forcing arrives.

Design choices
--------------
  - Input permuted to (B, channels, depth) so Conv1d slides along the vertical axis.
  - AdaptiveAvgPool1d(1) provides vertical translational invariance: the model
    correctly identifies a thermocline at depth level 3 or depth level 7 equally.
    This is critical because Mixed Layer Depth varies spatially and seasonally.
  - Residual skip from Layer 1 (32 channels) to Layer 3 output (128 channels) via
    a 1×1 Conv projection. This follows the ResNet convention: project to match
    channel count, then add elementwise. No activation on the skip path.
  - BatchNorm applied after each Conv and before ReLU for stable training.

Dependencies
------------
  torch>=2.2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNN1dEncoder(nn.Module):
    """
    Hierarchical 1D-CNN that encodes HYCOM depth-resolved profiles into a
    fixed-size latent vector for fusion with the atmospheric Transformer stream.

    Parameters
    ----------
    in_channels : int
        Number of ocean variables per depth level.
        Default 4: [water_temp, salinity, water_u, water_v].
    latent_dim : int
        Output feature dimension.
        Default 128 — matches TransformerEncoder output for symmetric fusion.
    """

    def __init__(self, in_channels: int = 4, latent_dim: int = 128) -> None:
        super().__init__()

        # Layer 1: detect local depth-scale patterns (mixed layer, halocline)
        # kernel=3 spans ~3 depth levels; padding=1 preserves depth sequence length
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)

        # Layer 2: intermediate feature composition across adjacent depth groups
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)

        # Layer 3: high-level depth structure encoding (full thermocline profile)
        self.conv3 = nn.Conv1d(64, latent_dim, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(latent_dim)

        # Residual skip: project Layer 1 output (32-ch) to latent_dim (128-ch)
        # No BatchNorm or activation on the skip path — raw signal passthrough
        self.skip_proj = nn.Conv1d(32, latent_dim, kernel_size=1)

        # Global average pooling over the depth axis → vertical translational invariance
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of HYCOM depth profiles into latent feature vectors.

        Parameters
        ----------
        x : torch.Tensor, shape (B, depth=11, channels=4)
            HYCOM ocean profiles on TARGET_DEPTHS_M standard levels.
            B = batch_size × n_members (reshaped by MHWRiskModel).
            depth=11 → [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300 m].
            channels=4 → [water_temp (°C), salinity (psu), water_u (m/s), water_v (m/s)].

        Returns
        -------
        out : torch.Tensor, shape (B, latent_dim=128)
            Depth-feature vector encoding vertical thermal structure.
            Passed to LeakyGate for fusion with the temporal context vector.
        """
        # Permute to (B, channels, depth) — Conv1d expects (N, C_in, L)
        x = x.permute(0, 2, 1)

        # Layer 1 forward — save output for residual skip
        layer1 = self.relu(self.bn1(self.conv1(x)))      # (B, 32, 11)

        # Layer 2 forward
        layer2 = self.relu(self.bn2(self.conv2(layer1))) # (B, 64, 11)

        # Layer 3 forward
        layer3 = self.relu(self.bn3(self.conv3(layer2))) # (B, 128, 11)

        # Residual: project Layer 1 output → latent_dim, add to Layer 3 output
        # This preserves the high-frequency shallow-gradient signal (0-10 m MLD cue)
        # that would otherwise be attenuated through three nonlinear transforms.
        skip = self.skip_proj(layer1)                    # (B, 128, 11), no activation
        out  = layer3 + skip                             # (B, 128, 11)

        # Global average pool → (B, 128, 1) then squeeze → (B, 128)
        out = self.pool(out).squeeze(-1)

        return out
