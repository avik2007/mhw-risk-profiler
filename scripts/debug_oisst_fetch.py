"""One-shot debug: test ERDDAP fetch of one OISST annual GoM slice."""
import requests
import tempfile
import os
import xarray as xr

URL = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
    "/ncdcOisst21Agg_LonPM180.nc"
    "?sst[(1982-01-01T12:00:00Z):(1982-12-31T12:00:00Z)]"
    "[(0.0)][(41.0):(45.0)][(-71.0):(-66.0)]"
)

print("URL:", URL)
resp = requests.get(URL, timeout=300)
print("status:", resp.status_code)
print("size:", len(resp.content))
print("Content-Type:", resp.headers.get("Content-Type"))
print("first 8 bytes:", resp.content[:8])

with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
    tmp.write(resp.content)
    tmppath = tmp.name

try:
    ds = xr.open_dataset(tmppath, engine="netcdf4")
    print(ds)
    sst = ds["sst"].squeeze("zlev", drop=True)
    print("sst shape:", sst.shape)
    print("sst sample:", sst.isel(time=0).values)
finally:
    os.unlink(tmppath)
