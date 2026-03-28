"""
sdd.py — Stress Degree Day accumulation for MHW risk quantification
=====================================================================
Computes Stress Degree Days (SDD) from SST anomalies during MHW events.

Physical note
-------------
Stress Degree Days are the thermal integral of SST above the MHW threshold,
accumulated only during active MHW event days (where mhw_mask is True):

    SDD = Σ_t max(SST_t − threshold_t, 0)   for all t where mhw_mask_t = True

Units: °C·day (thermal load, analogous to heating degree days in building energy).

SDD is the primary trigger variable for parametric insurance payout:
  - Low SDD  (<5 °C·day): no significant biological stress, no payout
  - Medium SDD (5-15 °C·day): sublethal stress, partial payout tier
  - High SDD  (>15 °C·day): mortality risk, full payout triggered

The 64-member ensemble produces 64 SDD realisations, from which SVaR is estimated.

Dependencies: xarray>=2024.2.0, numpy>=1.26.0
"""
from __future__ import annotations

import xarray as xr


def accumulate_sdd(
    sst: xr.DataArray,
    threshold: xr.DataArray,
    mhw_mask: xr.DataArray,
) -> xr.DataArray:
    """
    Accumulate Stress Degree Days above the MHW threshold during flagged event days.

    Parameters
    ----------
    sst : xr.DataArray
        Daily SST [°C], dims (member, time, lat, lon).
        HYCOM surface proxy: hycom_ds['water_temp'].isel(depth=0) broadcast to member dim.
        WeatherNext 2 production: sea_surface_temperature [converted from K to °C].
    threshold : xr.DataArray
        Climatological SST threshold [°C], dims (dayofyear, lat, lon).
        Produced by mhw_detection.compute_climatology().
        Must cover all day-of-year values present in sst.time.
    mhw_mask : xr.DataArray
        Boolean MHW event mask, same dims as sst: (member, time, lat, lon).
        Produced by mhw_detection.compute_mhw_mask().
        Only time steps where mask is True contribute to SDD.

    Returns
    -------
    sdd : xr.DataArray
        Stress Degree Days [°C·day], dims (member, lat, lon).
        All values ≥ 0. Time dimension is summed out.
        Pass sdd.values to torch.tensor() to feed into compute_svar() in svar.py.
    """
    # Align threshold to the SST time axis by day-of-year
    doy = sst.time.dt.dayofyear
    thresh_aligned = threshold.sel(dayofyear=doy)      # (time, lat, lon)

    # SST anomaly above threshold — clipped to 0 (no negative contributions)
    anomaly = (sst - thresh_aligned).clip(min=0.0)     # (member, time, lat, lon)

    # Mask out non-MHW days before summing
    masked_anomaly = anomaly.where(mhw_mask, other=0.0)

    # Sum over time axis to produce cumulative thermal load
    sdd = masked_anomaly.sum(dim="time")               # (member, lat, lon)

    sdd.attrs.update({
        "units": "degC day",
        "long_name": "Stress Degree Days above MHW threshold",
        "description": (
            "Cumulative thermal load above the 90th-percentile climatological SST "
            "threshold, accumulated only during active MHW event days. "
            "Primary trigger variable for parametric insurance payout."
        ),
    })
    return sdd
