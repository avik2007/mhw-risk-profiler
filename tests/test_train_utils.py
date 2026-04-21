# tests/test_train_utils.py
"""Unit tests for scripts/_train_utils.py — no GEE/HYCOM calls."""
import sys
sys.path.insert(0, "scripts")  # allow import of scripts/_train_utils

import numpy as np
import pytest
import torch
import xarray as xr

from _train_utils import build_tensors, HYCOM_VARS, WN2_VARS, SEQ_LEN, N_MEMBERS


def _make_merged(n_members=4, n_time=100, n_depth=11, n_lat=4, n_lon=5):
    """Synthetic merged Dataset matching DataHarmonizer.harmonize() output."""
    times = np.array([np.datetime64("2018-01-01") + np.timedelta64(i, "D")
                      for i in range(n_time)])
    wn2_data = {
        v: xr.DataArray(
            np.random.rand(n_members, n_time, n_lat, n_lon).astype(np.float32) + 274.0,
            dims=["member", "time", "latitude", "longitude"],
            coords={"time": times},
        )
        for v in WN2_VARS
    }
    hycom_data = {
        v: xr.DataArray(
            np.random.rand(n_members, n_time, n_depth, n_lat, n_lon).astype(np.float32),
            dims=["member", "time", "depth", "latitude", "longitude"],
            coords={"time": times},
        )
        for v in HYCOM_VARS
    }
    return xr.Dataset({**wn2_data, **hycom_data})


def _make_threshold(n_lat=4, n_lon=5):
    """Synthetic climatological threshold with dims (dayofyear, latitude, longitude)."""
    return xr.DataArray(
        np.full((366, n_lat, n_lon), 273.5, dtype=np.float32),
        dims=["dayofyear", "latitude", "longitude"],
        coords={"dayofyear": np.arange(1, 367)},
    )


def test_build_tensors_shapes():
    """build_tensors() returns tensors with correct shapes."""
    merged = _make_merged(n_members=4, n_time=100)
    threshold = _make_threshold()
    hycom_t, wn2_t, label_t = build_tensors(merged, threshold, seq_len=SEQ_LEN)

    assert hycom_t.shape == (1, 4, 11, 4), f"HYCOM shape mismatch: {hycom_t.shape}"
    assert wn2_t.shape   == (1, 4, SEQ_LEN, 5), f"WN2 shape mismatch: {wn2_t.shape}"
    assert label_t.shape == (1, 4), f"label shape mismatch: {label_t.shape}"


def test_build_tensors_dtype():
    """All output tensors are float32."""
    merged = _make_merged(n_members=4, n_time=100)
    threshold = _make_threshold()
    hycom_t, wn2_t, label_t = build_tensors(merged, threshold, seq_len=SEQ_LEN)

    assert hycom_t.dtype  == torch.float32
    assert wn2_t.dtype    == torch.float32
    assert label_t.dtype  == torch.float32


def test_build_tensors_label_nonneg():
    """SDD label is always >= 0 (Stress Degree Days cannot be negative)."""
    merged = _make_merged(n_members=4, n_time=100)
    threshold = _make_threshold()
    _, _, label_t = build_tensors(merged, threshold, seq_len=SEQ_LEN)
    assert (label_t >= 0).all(), f"Negative SDD values: min={label_t.min():.4f}"


def test_build_tensors_label_uses_hycom_sst():
    """SDD label is non-zero when HYCOM water_temp[depth=0] exceeds threshold.

    Verifies the label path uses HYCOM surface temp (°C) not ERA5 SST (Kelvin).
    If ERA5 SST were used, sst_celsius ≈ 0.85°C vs threshold 5.0°C → SDD = 0 always.
    With HYCOM water_temp[depth=0] = 20.0°C > threshold 5.0°C → SDD > 0.
    """
    n_members, n_time, n_lat, n_lon = 4, 100, 4, 5
    n_depth = 11
    times = np.array([np.datetime64("2018-01-01") + np.timedelta64(i, "D")
                      for i in range(n_time)])

    # ERA5 SST in Kelvin (~274K = ~0.85°C) — below threshold, would yield SDD=0
    wn2_data = {
        v: xr.DataArray(
            np.full((n_members, n_time, n_lat, n_lon), 274.0, dtype=np.float32),
            dims=["member", "time", "latitude", "longitude"],
            coords={"time": times},
        )
        for v in WN2_VARS
    }
    # HYCOM water_temp = 20°C at all depths — well above threshold
    hycom_data = {
        v: xr.DataArray(
            np.full((n_members, n_time, n_depth, n_lat, n_lon), 20.0, dtype=np.float32),
            dims=["member", "time", "depth", "latitude", "longitude"],
            coords={"time": times},
        )
        for v in HYCOM_VARS
    }
    merged = xr.Dataset({**wn2_data, **hycom_data})

    # Threshold = 5°C — well below HYCOM SST, so MHW condition is always active
    threshold = xr.DataArray(
        np.full((366, n_lat, n_lon), 5.0, dtype=np.float32),
        dims=["dayofyear", "latitude", "longitude"],
        coords={"dayofyear": np.arange(1, 367)},
    )

    _, _, label_t = build_tensors(merged, threshold, seq_len=SEQ_LEN)
    assert (label_t > 0).all(), f"Expected SDD > 0 with HYCOM SST=20°C, threshold=5°C; got {label_t}"
