"""One-shot debug: test requests fetch of one OISST daily file."""
import requests
import tempfile
import os
import xarray as xr

URL = (
    "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation"
    "/v2.1/access/avhrr/201001/oisst-avhrr-v02r01.20100101.nc"
)

resp = requests.get(URL, timeout=120)
print("status:", resp.status_code)
print("size:", len(resp.content))
print("Content-Type:", resp.headers.get("Content-Type"))
print("first 8 bytes:", resp.content[:8])

with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
    tmp.write(resp.content)
    tmppath = tmp.name

try:
    ds = xr.open_dataset(tmppath, engine="netcdf4", drop_variables=["ice", "anom", "err"])
    print(ds)
finally:
    os.unlink(tmppath)
