# ---------------------------------------------------------------
# mhw-risk-profiler Dockerfile
# Base: python:3.11-slim
#
# Purpose: Production-ready environment for harmonizing WeatherNext 2
# (Zarr via GEE) and HYCOM (NetCDF) data, running the 1D-CNN +
# Transformer MHW model, and serving VaR results via FastAPI.
#
# System spatial libraries are installed first to satisfy GDAL and
# NetCDF bindings required by earthengine-api, netCDF4, and xarray.
# ---------------------------------------------------------------

FROM python:3.11-slim

# -- System-level spatial dependencies ---------------------------
# libgdal-dev : GDAL C library required by earthengine-api raster I/O
# libnetcdf-dev: NetCDF-C library required by netCDF4 and xarray backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libnetcdf-dev \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -- Working directory -------------------------------------------
WORKDIR /app

# -- Python dependencies -----------------------------------------
# Pinned to minor versions; patch versions resolved at build time.
# Install order: heavy C-extension packages first to leverage layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -- Copy source -------------------------------------------------
COPY src/ ./src/

# -- Entrypoint --------------------------------------------------
# Default entry point targets the ingestion pipeline.
# Override at runtime: docker run <image> python -m src.models.cnn1d
ENTRYPOINT ["python", "-m", "src.ingestion.gee_harvester"]
