"""
mhw_detection.py — Marine Heatwave event detection (Hobday et al. 2016, Category I)
=====================================================================================
Detects MHW events from SST DataArrays by comparing against a climatological
90th-percentile threshold and requiring ≥ 5 consecutive days of exceedance.

Physical note
-------------
A Marine Heatwave (Hobday et al. 2016) is defined as a discrete, prolonged, anomalously
warm water event where SST exceeds the local 90th percentile climatological threshold
for five or more consecutive days. This threshold is computed from a historical baseline
period (typically 1982–2011) to avoid conflating warming trends with events.

The 90th percentile is used (not the mean or 95th) because:
  - It is sensitive enough to detect events that cause biological stress before mortality.
  - Salmon begin showing physiological stress at SSTs 2-4°C above their local seasonal
    average, which typically aligns with the 90th percentile in temperate aquaculture zones.

Development proxy
-----------------
When WeatherNext 2 access is unavailable, use HYCOM surface temperature:
    sst_proxy = hycom_ds['water_temp'].isel(depth=0)  # °C, depth index 0 = 0 m
Broadcast to a member dimension before calling compute_mhw_mask().

Dependencies: xarray>=2024.2.0, numpy>=1.26.0
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def compute_climatology(
    sst_historical: xr.DataArray,
    percentile: float = 90.0,
) -> xr.DataArray:
    """
    Compute the climatological SST percentile threshold for each calendar day.

    Groups historical SST by day-of-year and computes the requested percentile
    at each grid point, producing the Hobday et al. baseline threshold array.

    Parameters
    ----------
    sst_historical : xr.DataArray
        Historical daily SST [°C], dims (time, lat, lon).
        Should cover ≥ 3 years for stable percentile estimates.
        Recommended: 1982–2011 (30-year WMO standard period).
    percentile : float
        Percentile to compute. Default 90 per Hobday et al. (2016) Category I.

    Returns
    -------
    threshold : xr.DataArray
        90th-percentile SST [°C] for each calendar day.
        Dims: (dayofyear=365, lat, lon).
        Index dayofyear runs 1–365; day 366 (leap) is omitted by groupby convention.
    """
    grouped = sst_historical.groupby("time.dayofyear")
    threshold = grouped.quantile(percentile / 100.0, dim="time")
    # groupby().quantile() adds a 'quantile' coordinate — drop it
    threshold = threshold.drop_vars("quantile")
    return threshold  # dims: (dayofyear, lat, lon)


def compute_mhw_mask(
    sst: xr.DataArray,
    threshold: xr.DataArray,
    min_duration: int = 5,
) -> xr.DataArray:
    """
    Detect Marine Heatwave events per Hobday et al. (2016) Category I.

    A grid point and time step is flagged as an MHW event if:
      1. SST > threshold at that day-of-year and location.
      2. The exceedance is part of a run of ≥ min_duration consecutive days.

    Parameters
    ----------
    sst : xr.DataArray
        Daily SST [°C], dims (member, time, lat, lon).
        Each member is processed independently — no information sharing.
        When WeatherNext 2 is unavailable, use HYCOM water_temp[depth=0]
        broadcast across a synthetic member dimension.
    threshold : xr.DataArray
        Climatological SST threshold [°C], dims (dayofyear, lat, lon).
        Produced by compute_climatology() or loaded from a precomputed file.
        Must cover all day-of-year values present in sst.time.
    min_duration : int
        Minimum consecutive days above threshold to qualify as an MHW event.
        Default 5 per Hobday et al. (2016). Reduce to 1 for pointwise exceedance.

    Returns
    -------
    mhw_mask : xr.DataArray
        Boolean, same shape as sst: (member, time, lat, lon).
        True where an MHW event is active. Use this mask to gate SDD accumulation.
    """
    # Align threshold to the SST time axis by day-of-year
    doy = sst.time.dt.dayofyear
    thresh_aligned = threshold.sel(dayofyear=doy)      # (time, lat, lon)

    # Pointwise exceedance: SST strictly above threshold
    above = sst > thresh_aligned                        # (member, time, lat, lon), bool

    # Apply consecutive-day filter along time axis per member per grid point
    above_np = above.values.astype(np.int8)             # (member, time, lat, lon)
    mask_np  = _apply_min_duration(above_np, min_duration)

    return xr.DataArray(
        mask_np.astype(bool),
        dims=sst.dims,
        coords=sst.coords,
        attrs={"min_duration_days": min_duration, "method": "Hobday_2016_Cat1"},
    )


def _apply_min_duration(
    above: np.ndarray,
    min_duration: int,
) -> np.ndarray:
    """
    Zero out runs shorter than min_duration along the time axis.

    Uses a forward cumsum pass to compute run lengths ending at each time step,
    then a backward pass to propagate the valid-run flag back to the run start.
    Both passes are fully vectorized over (member, lat, lon) simultaneously.
    Time axis is always axis=1: shape (member, time, lat, lon).

    Parameters
    ----------
    above : np.ndarray
        Integer array (0/1) indicating pointwise threshold exceedance.
        Expected shape: (member, time, lat, lon).
    min_duration : int
        Minimum run length to retain. Runs shorter than this are zeroed out.

    Returns
    -------
    filtered : np.ndarray
        Same shape as above; short runs zeroed out, long runs preserved.

    Example
    -------
    above = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  min_duration=5
    filtered → [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  (run of 6 ≥ 5: kept)

    above = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  min_duration=5
    filtered → [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  (run of 4 < 5: zeroed)
    """
    if min_duration <= 1:
        return above

    above_int = above.astype(np.int32)
    n_time    = above_int.shape[1]

    # -- Forward pass: compute run length ending at each time step --
    # run_len[t] = consecutive 1s ending at t (resets to 0 on a 0)
    run_len = np.zeros_like(above_int)
    for t in range(n_time):
        if t == 0:
            run_len[:, t] = above_int[:, t]
        else:
            run_len[:, t] = np.where(above_int[:, t] == 1,
                                     run_len[:, t - 1] + 1,
                                     0)

    # -- Backward pass: propagate "part of a long run" flag back to run start --
    # A time step t is in a valid run if:
    #   (a) the run ending at t has length >= min_duration, OR
    #   (b) t is True and t+1 is already marked as part of a valid run
    filtered = np.zeros_like(above_int)
    for t in range(n_time - 1, -1, -1):
        long_run_ends_here = (run_len[:, t] >= min_duration)
        if t < n_time - 1:
            carries = (above_int[:, t] == 1) & (filtered[:, t + 1] == 1)
            in_valid_run = long_run_ends_here | carries
        else:
            in_valid_run = long_run_ends_here
        filtered[:, t] = np.where(in_valid_run & (above_int[:, t] == 1), 1, 0)

    return filtered
