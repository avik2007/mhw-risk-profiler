"""
conftest.py — shared pytest fixtures for analytics unit tests.

All fixtures use synthetic data so tests run without HYCOM/WeatherNext 2 access.
Sizes are kept tiny (member=4, time=10, lat=3, lon=3) for fast execution.
"""
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr


@pytest.fixture
def time_coord():
    """10-day time axis starting 2019-07-01 using pandas DatetimeIndex."""
    return pd.date_range("2019-07-01", periods=10, freq="D")


@pytest.fixture
def sst_above(time_coord):
    """
    SST DataArray where every value is 2°C above a 20°C baseline.
    shape: (member=4, time=10, lat=3, lon=3)
    Every day of every member should trigger MHW detection with threshold=20°C.
    """
    data = np.full((4, 10, 3, 3), 22.0, dtype=np.float32)  # 2°C above threshold
    return xr.DataArray(
        data,
        dims=["member", "time", "lat", "lon"],
        coords={
            "member": np.arange(4),
            "time": time_coord,
            "lat": [42.0, 42.25, 42.5],
            "lon": [-70.0, -69.75, -69.5],
        },
    )


@pytest.fixture
def sst_below(time_coord):
    """
    SST DataArray where every value is 1°C below threshold.
    shape: (member=4, time=10, lat=3, lon=3)
    No MHW should be detected with threshold=20°C.
    """
    data = np.full((4, 10, 3, 3), 19.0, dtype=np.float32)
    return xr.DataArray(
        data,
        dims=["member", "time", "lat", "lon"],
        coords={
            "member": np.arange(4),
            "time": time_coord,
            "lat": [42.0, 42.25, 42.5],
            "lon": [-70.0, -69.75, -69.5],
        },
    )


@pytest.fixture
def sst_mixed(time_coord):
    """
    SST DataArray: first 4 days below threshold, last 6 days above.
    With min_duration=5, only the last 6 days should register as MHW.
    shape: (member=1, time=10, lat=1, lon=1)
    """
    data = np.array([19.0, 19.0, 19.0, 19.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0],
                    dtype=np.float32).reshape(1, 10, 1, 1)
    return xr.DataArray(
        data,
        dims=["member", "time", "lat", "lon"],
        coords={
            "member": [0],
            "time": time_coord,
            "lat": [42.0],
            "lon": [-70.0],
        },
    )


@pytest.fixture
def threshold_20(time_coord):
    """
    Constant 20°C threshold broadcast across dayofyear=365, lat=3, lon=3.
    shape: (dayofyear=365, lat=3, lon=3)
    """
    data = np.full((365, 3, 3), 20.0, dtype=np.float32)
    return xr.DataArray(
        data,
        dims=["dayofyear", "lat", "lon"],
        coords={
            "dayofyear": np.arange(1, 366),
            "lat": [42.0, 42.25, 42.5],
            "lon": [-70.0, -69.75, -69.5],
        },
    )


@pytest.fixture
def threshold_20_small(time_coord):
    """Threshold for sst_mixed fixture (lat=1, lon=1)."""
    data = np.full((365, 1, 1), 20.0, dtype=np.float32)
    return xr.DataArray(
        data,
        dims=["dayofyear", "lat", "lon"],
        coords={
            "dayofyear": np.arange(1, 366),
            "lat": [42.0],
            "lon": [-70.0],
        },
    )


@pytest.fixture
def sdd_tensor():
    """
    Synthetic (batch=2, member=8) SDD tensor.
    Member 7 of batch 0 has a very high SDD (severe MHW member).
    """
    t = torch.zeros(2, 8)
    t[0, :] = torch.tensor([1.0, 1.5, 2.0, 1.2, 1.8, 1.3, 1.6, 10.0])
    t[1, :] = torch.tensor([0.5, 0.8, 0.6, 0.9, 0.7, 0.8, 0.6, 0.7])
    return t
