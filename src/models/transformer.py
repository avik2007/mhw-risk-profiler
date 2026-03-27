"""
transformer.py — Transformer encoder for WeatherNext 2 SST sequences
=====================================================================
Encodes 90-day WeatherNext 2 atmospheric forcing sequences into a 128-dimensional
temporal context vector for fusion with the HYCOM depth-feature from CNN1dEncoder.

Physical motivation
-------------------
Marine Heatwaves are driven by persistent atmospheric anomalies that evolve over
days-to-weeks timescales. A Transformer encoder is well-suited to capture these
long-range temporal dependencies because:

  - Self-attention has a direct path between any two days in the 90-day window,
    allowing the model to link a week-2 anticyclonic pressure anomaly to a week-7
    SST peak without information degrading through recurrent steps.
  - The 5 WeatherNext 2 features (2m_temperature, wind_u, wind_v, MSLP, SST)
    provide both the forcing signal (winds, pressure) and the integrator (SST).
    Multi-head attention can learn to associate reduced wind stress days with
    subsequent SST anomalies across the attention heads.
  - The 90-day window aligns with the boreal summer season (Jun-Aug) and matches
    the financial cycle for parametric insurance products that pay out on seasonal
    MHW events.

Design choices
--------------
  - Pre-norm convention (LayerNorm before attention sub-layer): produces more stable
    gradient magnitudes than post-norm, which is critical for Captum Integrated
    Gradients that backpropagates through all 4 encoder layers simultaneously.
  - Fixed sinusoidal positional encoding (not learned): preserves temporal position
    semantics under year-to-year distribution shift — day-45 always encodes the
    same phase relationship regardless of the training year.
  - Linear(5→128) input projection matches d_model to the CNN latent_dim, enabling
    direct elementwise fusion in LeakyGate without any dimension mismatch.
  - mean(dim=1) over the 90-day time axis aggregates the seasonal context into a
    single vector. Mean-pooling is preferred over CLS-token for this regression task
    because every time step carries physically relevant information.

Dependencies
------------
  torch>=2.2.0
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Compute fixed sinusoidal positional encodings.

    Encodes the day-of-sequence position as a vector of alternating sine and cosine
    values at geometrically spaced frequencies. This is the original encoding from
    Vaswani et al. (2017) "Attention Is All You Need".

    Parameters
    ----------
    seq_len : int
        Number of time steps (days). For this model, always 90.
    d_model : int
        Model embedding dimension. Must match the linear projection output (128).

    Returns
    -------
    pe : torch.Tensor, shape (1, seq_len, d_model)
        Positional encoding tensor. The leading dimension of 1 allows broadcasting
        over the batch dimension when added to the projected input sequence.

    Physical note
    -------------
    The sinusoidal basis functions encode day-of-season relationships. The model
    can use these positions to learn that "day 45 of the season" (roughly mid-July)
    has different climatological MHW risk than "day 10" (early June), even if the
    raw atmospheric values are similar.
    """
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )  # (d_model/2,)

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0)  # (1, seq_len, d_model)


class _PreNormTransformerLayer(nn.Module):
    """
    Single Transformer encoder layer using pre-norm convention.

    Pre-norm: LayerNorm is applied to the input of each sub-layer (before attention
    and before the feedforward network), rather than to the output. This produces
    more stable gradient flow during backpropagation, which is essential for
    Captum Integrated Gradients attribution through deep attention stacks.

    Architecture per layer:
        x = x + Dropout(Attention(LayerNorm(x)))   # self-attention with residual
        x = x + Dropout(FF(LayerNorm(x)))           # feedforward with residual

    Parameters
    ----------
    d_model : int
        Model dimension. 128 for this architecture.
    n_heads : int
        Number of attention heads. 8 for this architecture (128 / 8 = 16 per head).
    d_ff : int
        Feedforward hidden dimension. 256 for this architecture (2× d_model).
    dropout : float
        Dropout probability applied after attention and after feedforward.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one pre-norm Transformer encoder layer.

        Parameters
        ----------
        x : torch.Tensor, shape (B, seq_len, d_model)

        Returns
        -------
        x : torch.Tensor, shape (B, seq_len, d_model)
        """
        # Self-attention block (pre-norm)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop1(attn_out)

        # Feedforward block (pre-norm)
        x_norm = self.norm2(x)
        x = x + self.drop2(self.ff(x_norm))

        return x


class TransformerEncoder(nn.Module):
    """
    4-layer pre-norm Transformer encoder that summarises a 90-day WeatherNext 2
    atmospheric forcing sequence into a 128-dimensional temporal context vector.

    Parameters
    ----------
    in_features : int
        Number of input variables per time step.
        Default 5: [2m_temperature, wind_u, wind_v, MSLP, SST].
    d_model : int
        Internal model dimension and output latent size.
        Default 128 — matches CNN1dEncoder output for symmetric LeakyGate fusion.
    n_heads : int
        Number of attention heads. Default 8 (16 dims per head at d_model=128).
    n_layers : int
        Number of stacked encoder layers. Default 4.
    d_ff : int
        Feedforward hidden dimension. Default 256 (2× d_model).
    seq_len : int
        Fixed sequence length (days). Default 90.
    dropout : float
        Dropout probability. Default 0.1.
    """

    def __init__(
        self,
        in_features: int = 5,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 256,
        seq_len: int = 90,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model  = d_model
        self.seq_len  = seq_len

        # Project WN2 variables to d_model before adding positional encoding
        self.input_proj = nn.Linear(in_features, d_model)

        # Fixed sinusoidal positional encoding — registered as buffer (not parameter)
        # Buffer moves with .to(device) but is not updated by the optimizer
        pe = _sinusoidal_encoding(seq_len, d_model)
        self.register_buffer("pos_enc", pe)

        self.input_drop = nn.Dropout(dropout)

        # Stack of pre-norm Transformer layers
        self.layers = nn.ModuleList([
            _PreNormTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final LayerNorm before mean-pooling (pre-norm convention applied to output)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of 90-day WeatherNext 2 atmospheric sequences.

        Parameters
        ----------
        x : torch.Tensor, shape (B, time=90, features=5)
            WeatherNext 2 daily atmospheric fields.
            B = batch_size × n_members (reshaped by MHWRiskModel).
            time=90 → one boreal summer season (fixed window, no padding).
            features=5 → [2m_temperature (K), wind_u (m/s), wind_v (m/s),
                           MSLP (Pa), SST (K)].

        Returns
        -------
        out : torch.Tensor, shape (B, d_model=128)
            Temporal context vector summarising 90-day atmospheric forcing.
            Passed to LeakyGate for fusion with the HYCOM depth feature vector.
        """
        # Project to d_model and add positional encoding
        x = self.input_proj(x)                            # (B, 90, 128)
        x = self.input_drop(x + self.pos_enc)             # (B, 90, 128)

        # Pass through 4 pre-norm Transformer encoder layers
        for layer in self.layers:
            x = layer(x)                                  # (B, 90, 128)

        x = self.final_norm(x)                            # (B, 90, 128)

        # Mean-pool over the time axis to produce a single seasonal context vector
        # Mean is preferred over CLS token: every day carries MHW-relevant information
        out = x.mean(dim=1)                               # (B, 128)

        return out
