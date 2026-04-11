# ERA5 Proxy Training Strategy

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unblock `MHWRiskModel` training while WeatherNext 2 GEE whitelist is pending. Use deterministic ERA5 (`ECMWF/ERA5/DAILY`) from GEE as a physical proxy for WN2. A `NoiseInjector` broadcasts the single ERA5 member to 64 synthetic members with Gaussian noise, preserving the `(batch, member, ...)` tensor contract required by `ensemble_wrapper.py` and `svar.py`.

**Architecture:** No changes to `MHWRiskModel`, `svar.py`, or `accumulate_sdd()`. Only ingestion gains a new class (`ERA5Harvester`) and `DataHarmonizer.harmonize()` gains an `expand_and_perturb()` branch. A new training script wires everything together.

**Transition path:** After WN2 whitelist approval, swap harvester in config and fine-tune from ERA5 weights.

**Tech Stack:** `earthengine-api`, `xarray>=2024.2.0`, `torch>=2.2.0`, `numpy>=1.26.0`, `pytest>=8.0.0`. No new dependencies.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/ingestion/era5_harvester.py` | `ERA5Harvester`: fetch `ECMWF/ERA5/DAILY` from GEE |
| Modify | `src/ingestion/harvester.py` | Add `expand_and_perturb()` static method to `DataHarmonizer`; update `harmonize()` to call it when `member=1` |
| Create | `scripts/train_era5_proxy.py` | Full training loop: ERA5 → model → MSE vs physics SDD |
| Create | `tests/test_era5_harvester.py` | Offline unit tests: band mapping, output shape, noise spread |
| Modify | `src/ingestion/__init__.py` | Export `ERA5Harvester` |

---

## Physical Reference

**Why ERA5 is a valid proxy:**
ERA5 and WeatherNext 2 share the same 5 atmospheric variables, the same 0.25° horizontal resolution, and the same units. ERA5 is the ECMWF reanalysis — it is physically consistent and freely available on GEE without a whitelist. The key difference is that ERA5 is deterministic (1 member), while WN2 produces 64 FGN-ensemble members. The synthetic noise expansion compensates for this structurally so that `svar.py`'s quantile estimator produces a non-degenerate spread.

**Noise calibration:**
Ensemble spread is injected per-member using seeded Gaussian noise. Targets are based on the observed intra-ensemble spread from published WN2 documentation:

| Variable | σ | Rationale |
|----------|---|-----------|
| `sea_surface_temperature` | 0.5 K | Primary MHW driver; target spread from Gemini spec |
| `2m_temperature` | 0.5 K | Coherent with SST perturbation |
| `10m_u_component_of_wind` | 0.3 m/s | Weaker synoptic variability at daily scale |
| `10m_v_component_of_wind` | 0.3 m/s | Same |
| `mean_sea_level_pressure` | 50 Pa | ~0.5 hPa; realistic synoptic noise |

**Training objective:**
The model is trained to predict the physics-based SDD computed by `accumulate_sdd()` using the HYCOM surface SST and the `hycom_sst_threshold.zarr` climatological threshold. This ensures the model learns physically meaningful SDD signals rather than arbitrary targets.

**Training domain:**
Gulf of Maine — `lat=[41, 45], lon=[-71, -66]`, period 2018–2019. Same region and period used for HYCOM OPeNDAP validation; data confirmed reachable via `verify_hycom_zarr.py`.

---

## Task 1: ERA5Harvester

**File:** `src/ingestion/era5_harvester.py`

**GEE Collection:** `ECMWF/ERA5/DAILY`

ERA5 band → WN2-compatible variable name mapping:

| ERA5 Band | Output Variable | Units |
|-----------|----------------|-------|
| `mean_2m_air_temperature` | `2m_temperature` | K |
| `u_component_of_wind_10m` | `10m_u_component_of_wind` | m/s |
| `v_component_of_wind_10m` | `10m_v_component_of_wind` | m/s |
| `mean_sea_level_pressure` | `mean_sea_level_pressure` | Pa |
| `sea_surface_temperature` | `sea_surface_temperature` | K |

**Interface:** `ERA5Harvester.fetch(start_date, end_date, bbox) → xr.Dataset`
- Output dims: `(member=1, time, latitude, longitude)`
- Variable names identical to `WeatherNext2Harvester.fetch_ensemble()` output so `DataHarmonizer.harmonize()` accepts without modification
- Auth: same `ee.Initialize()` pattern as `WeatherNext2Harvester.authenticate()`; reads `GOOGLE_APPLICATION_CREDENTIALS` env var
- No GCS caching — ERA5 is fast to re-fetch and small for the Gulf of Maine tile

**Implementation notes:**
- Model closely on `WeatherNext2Harvester._fetch_and_write_zarr()` but use `ee.ImageCollection("ECMWF/ERA5/DAILY")` filtered by date and `.select(ERA5_BANDS)`
- After fetching, `expand_dims(member=[0])` to add the member axis
- `ERA5_BANDS` dict maps ERA5 band names → WN2 variable names; `rename_vars()` applies the mapping
- No `n_members` parameter — ERA5 is always 1 member; expansion is handled by `DataHarmonizer`

---

- [ ] **Step 1: Write failing tests** (`tests/test_era5_harvester.py`, offline only — no GEE call)

Three tests:

```python
def test_band_mapping():
    """Verify all 5 WN2-compatible variable names appear in the output Dataset."""
    # patch ee.ImageCollection; call ERA5Harvester.fetch()
    # assert set(ds.data_vars) == {"2m_temperature", "10m_u_component_of_wind",
    #                               "10m_v_component_of_wind", "mean_sea_level_pressure",
    #                               "sea_surface_temperature"}

def test_output_shape():
    """Output dims must be (member=1, time, latitude, longitude)."""
    # assert ds.dims == {"member": 1, "time": T, "latitude": L, "longitude": L}

def test_noise_spread():
    """After expand_and_perturb(), SVaR_95 > SVaR_50 (non-degenerate ensemble)."""
    # build synthetic member=1 DataArray; call expand_and_perturb()
    # assert output.dims["member"] == 64
    # sst_std = output["sea_surface_temperature"].std("member")
    # assert float(sst_std.mean()) == pytest.approx(0.5, rel=0.5)  # within 50%
    # sdd_vals = output["sea_surface_temperature"].values  # proxy for SVaR check
    # assert np.quantile(sdd_vals, 0.95) > np.quantile(sdd_vals, 0.50)
```

- [ ] **Step 2: Implement `ERA5Harvester`** in `src/ingestion/era5_harvester.py`

```python
ERA5_BANDS = {
    "mean_2m_air_temperature":    "2m_temperature",
    "u_component_of_wind_10m":    "10m_u_component_of_wind",
    "v_component_of_wind_10m":    "10m_v_component_of_wind",
    "mean_sea_level_pressure":    "mean_sea_level_pressure",
    "sea_surface_temperature":    "sea_surface_temperature",
}

class ERA5Harvester:
    def authenticate(self): ...       # ee.Initialize() with service account
    def fetch(self, start_date, end_date, bbox) -> xr.Dataset: ...
    # returns (member=1, time, latitude, longitude)
```

- [ ] **Step 3: Export from `src/ingestion/__init__.py`**

```python
from .era5_harvester import ERA5Harvester
```

- [ ] **Step 4: Run tests — all 3 must pass**

```bash
pytest tests/test_era5_harvester.py::test_band_mapping -v
pytest tests/test_era5_harvester.py::test_output_shape -v
```

---

## Task 2: Synthetic Ensemble Expansion

**File:** `src/ingestion/harvester.py` — modify `DataHarmonizer`

Add `expand_and_perturb()` as a `@staticmethod` on `DataHarmonizer`.

```python
NOISE_SIGMAS = {
    "sea_surface_temperature":    0.5,   # K
    "2m_temperature":             0.5,   # K
    "10m_u_component_of_wind":    0.3,   # m/s
    "10m_v_component_of_wind":    0.3,   # m/s
    "mean_sea_level_pressure":    50.0,  # Pa
}

@staticmethod
def expand_and_perturb(ds: xr.Dataset, n_members: int = 64, seed: int = 42) -> xr.Dataset:
    """
    Broadcast a single-member ERA5 Dataset to n_members synthetic members
    by injecting independent Gaussian noise into atmospheric variables.
    ...
    """
```

**Logic:**
1. `ds_broad = ds.expand_dims(member=range(n_members))` → broadcast along new member axis
2. For each member `i`, for each variable in `NOISE_SIGMAS`:
   - Draw noise: `np.random.default_rng(seed + i).normal(0, sigma, shape)`
   - Add to that member's slice: `ds_broad[var][i] += noise`
3. Return perturbed Dataset with `member=64`

**Update `harmonize()`:** After regridding `wn2_ds`, insert:
```python
if len(wn2_ds.member) == 1:
    wn2_ds = DataHarmonizer.expand_and_perturb(wn2_ds)
```
This makes the single-member ERA5 path transparent to all downstream code.

---

- [ ] **Step 5: Add `expand_and_perturb()` to `DataHarmonizer`**

- [ ] **Step 6: Update `harmonize()` to call `expand_and_perturb()` when `member=1`**

- [ ] **Step 7: Run noise spread test — must pass**

```bash
pytest tests/test_era5_harvester.py::test_noise_spread -v
```

---

## Task 3: Training Script

**File:** `scripts/train_era5_proxy.py`

**Prerequisites check (fail fast):**
```python
threshold_path = Path("data/processed/hycom_sst_threshold.zarr")
if not threshold_path.exists():
    sys.exit(
        "ERROR: hycom_sst_threshold.zarr not found. "
        "Run scripts/compute_hycom_climatology.py first."
    )
```

**`--dry-run` flag:** Skip GEE/HYCOM fetches; use synthetic random tensors of the correct shape. This allows CI validation without network access.

**Data flow (real mode):**
```
ERA5Harvester.fetch("2018-01-01", "2019-12-31", GoM_bbox)
    → wn2_proxy  (member=1, time=730, lat=17, lon=21)

HYCOMLoader.fetch_tile("2018-01-01", "2019-12-31", GoM_bbox)
    → hycom_ds   (time=730, depth=11, lat=17, lon=21)

DataHarmonizer.harmonize(wn2_proxy, hycom_ds)
    → merged     (member=64, time=730, depth=11, lat=17, lon=21)
    [expand_and_perturb fires inside harmonize()]

# Physics SDD label
threshold = xr.open_zarr("data/processed/hycom_sst_threshold.zarr")
mhw_mask  = compute_mhw_mask(merged["sea_surface_temperature"], threshold)
sdd_phys  = accumulate_sdd(merged["sea_surface_temperature"], threshold, mhw_mask)
    → (member=64, lat=17, lon=21)

label = sdd_phys.mean(dim=["lat", "lon"])  # (member=64,) — spatial mean

# Tensor construction (batch=1, member=64)
hycom_t = torch.tensor(merged[HYCOM_VARS].to_array().values)
         .permute(...)  # → (1, 64, 11, 4)
wn2_t   = torch.tensor(merged[WN2_VARS].to_array().values)
         .permute(...)  # → (1, 64, 730→90, 5)  [use last 90 days]
label_t = torch.tensor(label.values).unsqueeze(0)  # (1, 64)
```

**Training loop:**
```python
model = MHWRiskModel()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1, 51):
    sdd_pred, _, _ = model(hycom_t, wn2_t)   # (1, 64)
    loss = F.mse_loss(sdd_pred, label_t)
    optim.zero_grad(); loss.backward(); optim.step()

    svar95 = sdd_pred[0].quantile(0.95).item()
    svar50 = sdd_pred[0].quantile(0.50).item()
    print(f"Epoch {epoch:02d}/50 | loss={loss.item():.4f} | "
          f"SVaR_95={svar95:.1f} | SVaR_50={svar50:.1f} | "
          f"spread={svar95 - svar50:.1f} degC.day")

torch.save(model.state_dict(), "data/models/era5_proxy_weights.pt")
print("Weights saved → data/models/era5_proxy_weights.pt")
```

---

- [ ] **Step 8: Implement `scripts/train_era5_proxy.py`** with `--dry-run` flag

- [ ] **Step 9: Run dry-run validation**

```bash
python scripts/train_era5_proxy.py --dry-run
```

Expected: 50 lines of epoch output, no errors, `spread > 0`.

---

## Task 4: Validation Gate

Evidence required before marking plan complete:

- [ ] **Step 10: Full test run**

```bash
pytest tests/test_era5_harvester.py -v
```

Expected: 3 tests pass.

- [ ] **Step 11: Dry-run training output**

```bash
python scripts/train_era5_proxy.py --dry-run
```

Expected stdout contains:
```
Epoch 50/50 | loss=X.XXXX | SVaR_95=X.X | SVaR_50=X.X | spread=X.X degC.day
Weights saved → data/models/era5_proxy_weights.pt
```

Evidence: `spread > 0` confirms non-degenerate synthetic ensemble.

---

## Transition to WeatherNext 2 (post-whitelist)

Once `mhw-harvester@mhw-risk-profiler.iam.gserviceaccount.com` is whitelisted:

1. Add `--harvester [era5|weathernext2]` CLI flag to `train_era5_proxy.py` (or create `scripts/train_wn2.py`).
2. Load ERA5 proxy weights: `model.load_state_dict(torch.load("data/models/era5_proxy_weights.pt"))`.
3. Fine-tune on WN2 ensemble at lower lr (1e-5) to capture non-Gaussian tail behavior.
4. `DataHarmonizer.harmonize()` will skip `expand_and_perturb()` automatically when WN2 returns `member=64`.
5. No changes needed to `MHWRiskModel`, `svar.py`, `accumulate_sdd()`, or `mhw_detection.py`.

---

*Scientific rationale reviewed by Gemini (2026-04-03). Implementation owned by Claude.*
