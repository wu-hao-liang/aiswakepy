# Implementation Spec — ShipwakeAIS Python Rewrite

**Based on**: PRD.md
**Status**: ✅ All 12 steps complete (with notes — see Step 12)
**Last reviewed**: 2026-05-09
**Tests**: 145 passing on master

---

## Status Summary

| Step | Module(s) (current) | Status | Notes |
|------|---------------------|--------|-------|
| 0 | (project root) | ✅ done | `pyproject.toml`, `aiswakepy/`, all sub-packages exist |
| 1 | `aiswakepy/config.py` | ✅ done | Pydantic v2; loads from JSON file/string/dict |
| 2 | `aiswakepy/geo/geodesy.py` | ✅ done | `geodetic_distance`, `geodetic_bearing`, `forward_point` (scalar + array) |
| 3 | `aiswakepy/vessel/block_coeff.py` | ✅ done | All three methods (`L_Le`, `B_Le`, `table`) + `ShipDataEDnew.csv` |
| 4 | `aiswakepy/stages/filter.py` | ✅ done **(expanded)** | Now **12 steps** (originally planned 6): added `deduplicate`, `uniformize_vessel_info`, `remove_zero_dimensions`, `remove_invalid_draught`, `clean_error_coords` (kinematic check), `clean_error_speed` (acceleration check), `filter_study_area` |
| 5 | `aiswakepy/stages/depth.py` + `aiswakepy/geo/bathymetry.py` | ✅ done | mikeio mesh + KDTree + tide snap + under-keel filter |
| 6 | `aiswakepy/stages/vessel.py` | ✅ done **(renamed)** | Originally planned as `wave_params.py`; now `vessel.py`. Wave physics (Theta, T, WakeDir) computed here |
| 7 | `aiswakepy/geo/coastline.py` | ✅ done | Includes STRtree builder (per Step 12 fix #4) |
| 8 | `aiswakepy/stages/wave_impact.py` | ✅ done **(renamed)** | Originally planned as `shore_impact.py`; now `wave_impact.py`. Computes per-point impact + ray-coastline intersection |
| 9 | `aiswakepy/viz/wave_map.py` + `aiswakepy/viz/vessel_diagram.py` | ✅ done | Both modules present |
| 10 | `aiswakepy/pipeline.py` + `main.py` + `run_shipwake.ipynb` + `run_shipwake_record.ipynb` | ✅ done | `run_pipeline()` orchestrator, CLI, two notebooks |
| 11 | `validate_pipeline.py` | ✅ done | End-to-end validation script vs MATLAB outputs (see `tests/validation_report.md` if generated) |
| 12 | (multiple) | ⚠️ 5/6 done | See `docs/PERFORMANCE.md` for full status. Outstanding: replace `pipeline.py` `print()` with `rich.console.Console` |

**Architecture differences from original plan** (intentional, post-implementation evolution):
- Stage 6 file renamed `wave_params.py` → `vessel.py` (better reflects content: vessel-derived parameters).
- Stage 8 file renamed `shore_impact.py` → `wave_impact.py` (impact is along the propagation path, not only at shore).
- Stage 4 expanded from 6 to 12 sub-steps to handle real AIS data quality issues (GPS spikes, acceleration anomalies, duplicate fixes, vessel-info inconsistency, draught misreports).
- New `aiswakepy/comparison/ossi.py` module added (not in original plan) for OSSI gauge data matching.

---

## Step 0: Project Scaffolding & Dependencies ✅

**Goal**: Set up the Python project structure, install dependencies, verify imports.

**Tasks**:
1. Create the directory layout under `aiswakepy/` as defined in PRD §5.
2. Create `pyproject.toml` with `uv`; add core dependencies:
   - `numpy`, `pandas`, `pyproj`, `shapely`, `geopandas`, `fiona`
   - `mikeio`, `scipy`, `pydantic>=2`
   - `matplotlib`, `contextily`, `rich`
   - dev: `pytest`, `jupyterlab`
3. Create `aiswakepy/__init__.py` with version string.
4. Create empty `__init__.py` in each sub-package (`stages/`, `geo/`, `vessel/`, `viz/`, `comparison/`).

**Tests**: `test_imports.py` passes.

---

## Step 1: Configuration (`aiswakepy/config.py`) ✅

**Goal**: Load and validate configuration from JSON file, JSON string, or Python dict.

**Tasks**:
1. Pydantic v2 models for each config section.
2. `load_config(source)` accepts file path / JSON string / dict.
3. Default `config.json` shipped at project root.

---

## Step 2: Geodesy Utilities (`aiswakepy/geo/geodesy.py`) ✅

`geodetic_distance`, `geodetic_bearing`, `forward_point` — scalar and vectorised (numpy-array) variants. Wraps `pyproj.Geod`.

---

## Step 3: Block Coefficient & Bow Entry (`aiswakepy/vessel/block_coeff.py`) ✅

Three methods (`L_Le` default, `B_Le`, `table`) with unified interface. `ShipDataEDnew.csv` shipped in `aiswakepy/vessel/`.

---

## Step 4: AIS Filtering & Interpolation (`aiswakepy/stages/filter.py`) ✅ (expanded)

**Originally planned** 6 sub-steps; **actual** 12-step pipeline (driven by real AIS data quality):

| # | Function | Purpose |
|---|----------|---------|
| 1 | `load_ais` | CSV read; parse obstime; drop NaN |
| 2 | `deduplicate` | Drop duplicate (mmsi, obstime) — prevents dt=0 |
| 3 | `uniformize_vessel_info` | Mode-fill width/length/typecargo per MMSI |
| 4 | `remove_zero_dimensions` | Drop width/length/draught ≤ 0 |
| 5 | `remove_invalid_draught` | Drop draught > beam (NEW, post-PRD-v1) |
| 6 | `segment_trajectories` | Split on time gaps > `traj_gap_s` |
| 7 | `clean_error_coords` | Kinematic Consistency Check (GPS spike removal) |
| 8 | `clean_error_speed` | Acceleration Check (replace bad SOG/COG) |
| 9 | `validate_speed` | `SOG = min(reported, geodetic-derived)` (vectorised, Step 12 fix #1) |
| 10 | `interpolate_trajectories` | Cubic Hermite Spline (vectorised, Step 12 fix #3) |
| 11 | `filter_study_area` | Optional polygon clip |
| 12 | `mask_land` | Remove land points (vectorised, Step 12 fix #2) |

---

## Step 5: Bathymetry & Tidal Depth ✅

- `aiswakepy/geo/bathymetry.py`: `load_bathymetry`, `get_depth` (KDTree, parallel queries).
- `aiswakepy/stages/depth.py`: `assign_depth` — bathy + tide + under-keel filter.

---

## Step 6: Wave Parameter Calculation (`aiswakepy/stages/vessel.py`) ✅ (renamed from wave_params.py)

`compute_vessel_params(df, cb_method, g, max_sog_knots, max_bl_ratio)` — fully vectorised. Computes V_ms, Be, L_WL, Cb, α, β, F_m, F_d, H_Kriebel, T, θ, Cel, Tc, WakeDirPort/Starboard. Applies row filters.

---

## Step 7: Coastline Operations (`aiswakepy/geo/coastline.py`) ✅

- `load_coastline`, `build_ray`, `find_shore_intersection`.
- STRtree spatial index (`build_coastline_index`) added in Step 12 fix #4.

---

## Step 8: Shore Impact Calculation (`aiswakepy/stages/wave_impact.py`) ✅ (renamed from shore_impact.py)

`compute_wave_impact(df_vessel, coastline_shp, formula, max_propagation_m, wake_cutoff_m, ...)` — STRtree-accelerated, custom Spinner progress.

---

## Step 9: Visualisation ✅

- `aiswakepy/viz/wave_map.py`: `plot_wave_height_map`, `plot_wave_period_map`, with coastline-binned top-N (Step 12 fix #5).
- `aiswakepy/viz/vessel_diagram.py`: per-vessel wake diagrams.

---

## Step 10: Pipeline Orchestration & Notebook ✅

- `aiswakepy/pipeline.py`: `run_pipeline(config, stages=None)` orchestrator.
- `main.py`: CLI entry — `uv run python main.py --config config.json`.
- `run_shipwake.ipynb`: primary analysis notebook (extensively expanded for empirical formula comparison and regression analysis).
- `run_shipwake_record.ipynb`: clean pipeline record notebook (added 2026-05).

---

## Step 11: End-to-End Validation with Real Data ✅

`validate_pipeline.py` runs the pipeline on `data/AIS_2563.csv` and compares against MATLAB reference (`table_ShoreImpact_*.csv`). Documents expected differences:
- Gravity: 9.78 (Python) vs 9.81 (MATLAB)
- Cb method: type-based `L_Le` (Python) vs old global table (MATLAB)
- Wake directions: COG ± θ (Python) vs COG ± 90° (MATLAB)
- Distance: geodetic WGS84 (Python) vs planar (MATLAB)

---

## Step 12: Performance Optimization ⚠️ 5/6 done

See `docs/PERFORMANCE.md` for detailed status. Summary:

| Fix | Status |
|-----|--------|
| 1. Vectorize `validate_speed` | ✅ |
| 2. Vectorize `mask_land` | ✅ |
| 3. Reduce allocations in `interpolate_trajectories` | ✅ |
| 4. STRtree spatial index for shore intersection | ✅ |
| 5. Coastline-binned top-N visualisation | ✅ (config field renamed `plot_max_points`) |
| 6. Rich console + per-stage timing | ⚠️ partial — timing done, `print()` not migrated to `rich.console.Console` |

---

## Summary: Dependency Graph

```
Step 0: Scaffolding
  │
  ├→ Step 1: Config
  │
  ├→ Step 2: Geodesy
  │    │
  │    ├→ Step 4: AIS Filter (uses geodesy for distance)
  │    │    │
  │    │    └→ Step 5: Depth
  │    │         │
  │    │         └→ Step 6: Wave Params (vessel.py)
  │    │              │
  │    ├→ Step 7: Coastline (uses geodesy for rays)
  │    │    │
  │    │    └→ Step 8: Wave Impact (wave_impact.py)
  │    │
  │    └→ Step 3: Block Coeff (used by Step 6)
  │
  ├→ Step 9: Visualisation (uses wave impact output)
  │
  └→ Step 10: Pipeline + Notebook
       │
       └→ Step 11: Validation
       │
       └→ Step 12: Performance (cross-cutting)
```
