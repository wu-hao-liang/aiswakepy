# Implementation Spec — ShipwakeAIS Python Rewrite

**Based on**: PRD.md
**Status**: ✅ All 12 steps complete; post-v1 evolution documented below
**Last reviewed**: 2026-05-28
**Tests**: 142 passed, 2 skipped (on `feature/dash-interactive-app`)

---

## Status Summary

| Step | Module(s) (current) | Status | Notes |
|------|---------------------|--------|-------|
| 0 | (project root) | ✅ done | `pyproject.toml`, `aiswakepy/`, all sub-packages exist |
| 1 | `aiswakepy/config.py` | ✅ done | Pydantic v2; loads from JSON file/string/dict |
| 2 | `aiswakepy/geo/geodesy.py` | ✅ done | `geodetic_distance`, `geodetic_bearing`, `forward_point` (scalar + array) |
| 3 | `aiswakepy/vessel/block_coeff.py` | ✅ done | All three methods (`L_Le`, `B_Le`, `table`) + `ShipDataEDnew.csv` |
| 4 | `aiswakepy/stages/filter.py` | ✅ done **(expanded + depth folded in)** | Now **13 sub-steps**: original 12-step AIS filter plus under-keel clearance check at the tail (see Step 4 detail) |
| 5 | `aiswakepy/stages/depth.py` + `aiswakepy/geo/bathymetry.py` | ✅ done **(merged into Step 4)** | `depth.py` still exists as a utility module (`assign_depth`); no longer a standalone pipeline stage — called from `filter_ais()` tail |
| 6 | `aiswakepy/stages/vessel.py` | ✅ done **(renamed)** | Originally planned as `wave_params.py`; now `vessel.py`. Wave physics (Theta, T, WakeDir) computed here |
| 7 | `aiswakepy/geo/coastline.py` | ✅ done | Includes STRtree builder (per Step 12 fix #4) |
| 8 | `aiswakepy/stages/wave_impact.py` | ✅ done **(renamed)** | Originally planned as `shore_impact.py`; now `wave_impact.py`. Computes per-point impact + ray-coastline intersection |
| 9 | `aiswakepy/viz/wave_map.py` + `aiswakepy/viz/vessel_diagram.py` + `aiswakepy/viz/report.py` | ✅ done **(extended)** | `report.py` added post-v1: report-quality maps, `top_vessels_table`, `plot_vessel_track_scatter` |
| 10 | `aiswakepy/pipeline.py` + `main.py` + `run_shipwake.ipynb` + `run_shipwake_record.ipynb` + `dash_app.py` | ✅ done **(extended)** | Pipeline now 3 stages (filter+depth, vessel, wave_impact); Dash interactive app added post-v1 |
| 11 | `validate_pipeline.py` | ✅ done | End-to-end validation script vs MATLAB outputs (see `tests/validation_report.md` if generated) |
| 12 | (multiple) | ✅ done | All 6 fixes complete. Fix 6 uses plain `print()` for status — vectorised stages have no per-row progress. See `docs/PERFORMANCE.md` |

**Architecture differences from original plan** (intentional, post-implementation evolution):
- Stage 5 (`depth.py`) merged into the tail of `filter_ais()` — the under-keel check runs on the final interpolated frame; `depth.py` retained as a callable utility.
- Pipeline reduced from 4 runtime stages to **3**: `filter` (includes depth), `vessel`, `wave_impact`. Stage CSV names renumbered: `01_filtered`, `02_vessel`, `03_wave_impact`.
- Stage 6 file renamed `wave_params.py` → `vessel.py` (better reflects content: vessel-derived parameters).
- Stage 8 file renamed `shore_impact.py` → `wave_impact.py` (impact is along the propagation path, not only at shore).
- Stage 4 expanded from 6 to 12 AIS sub-steps + 1 depth sub-step (13 total) to handle real data quality issues.
- New `aiswakepy/comparison/ossi.py` module added (not in original plan) for OSSI gauge data matching.
- New `aiswakepy/viz/report.py` module added (not in original plan) for report-quality plots and top-vessel tables.
- New `dash_app.py` interactive application added (not in original plan) — see Step 10 for details.

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

## Step 4: AIS Filtering + Depth Check (`aiswakepy/stages/filter.py`) ✅ (expanded)

**Originally planned** 6 sub-steps; **actual** 13 sub-steps (12 AIS cleaning steps + 1 depth check at the tail):

| # | Function | Purpose |
|---|----------|---------|
| 1 | `load_ais` | CSV read; parse obstime; drop NaN |
| 2 | `deduplicate` | Drop duplicate (mmsi, obstime) — prevents dt=0 |
| 3 | `uniformize_vessel_info` | Mode-fill width/length/typecargo per MMSI |
| 4 | `remove_zero_dimensions` | Drop width/length/draught ≤ 0 |
| 5 | `remove_invalid_draught` | Drop draught > beam |
| 6 | `segment_trajectories` | Split on time gaps > `traj_gap_s` |
| 7 | `clean_error_coords` | Kinematic Consistency Check (GPS spike removal) |
| 8 | `clean_error_speed` | Acceleration Check (replace bad SOG/COG) |
| 9 | `validate_speed` | `SOG = min(reported, geodetic-derived)` (vectorised) |
| 10 | `interpolate_trajectories` | Cubic Hermite Spline (vectorised) |
| 11 | `filter_study_area` | Optional polygon clip |
| 12 | `mask_land` | Remove land points (vectorised) |
| 13 | `assign_depth` (from `depth.py`) | Bathy + tide → WaterDepth; under-keel filter on final interpolated frame |

The depth check is intentionally at the tail so pre-interpolation points with uncertain draught still constrain the spline; only the final dense frame is checked against bathymetry.

---

## Step 5: Bathymetry & Tidal Depth — merged into Step 4 ✅

`aiswakepy/geo/bathymetry.py` provides `load_bathymetry`, `load_tide`, `snap_to_tide`, `get_depth` (KDTree, parallel queries). `aiswakepy/stages/depth.py` retains `assign_depth()` as a callable utility; it is no longer invoked as a standalone pipeline stage — `filter_ais()` calls it directly at sub-step 13 when `bathy_path` is provided.

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

## Step 9: Visualisation ✅ (extended)

- `aiswakepy/viz/wave_map.py`: `plot_wave_height_map`, `plot_wave_period_map`. Turbo colormap, `vmin=0`, `vmax=ceil(max)`, colorbar matched to axes height via `make_axes_locatable`, coastline-binned top-N downsampling.
- `aiswakepy/viz/vessel_diagram.py`: per-vessel wake diagrams.
- `aiswakepy/viz/report.py` *(added post-v1)*: report-quality wrappers `plot_wave_height_report` / `plot_wave_period_report` (auto-fitted extent), `top_vessels_table` (top-N vessels by peak shore height), `plot_vessel_track_scatter` (speed vs length coloured by vessel type, full 19-category fixed-colour legend).

---

## Step 10: Pipeline Orchestration, Notebook & Dash App ✅ (extended)

- `aiswakepy/pipeline.py`: `run_pipeline(config, stages=None)` orchestrator. **Three runtime stages** (filter+depth, vessel, wave_impact). Stage CSV names: `01_filtered.csv`, `02_vessel.csv`, `03_wave_impact.csv`.
- `main.py`: CLI entry — `uv run python main.py --config config.json`.
- `run_shipwake.ipynb`: primary analysis notebook (empirical formula comparison, regression analysis).
- `run_shipwake_record.ipynb`: clean pipeline record notebook.
- `dash_app.py` *(added post-v1)*: Dash + deck.gl interactive web application. Features:
  - Pipeline runner (filter → vessel → wave_impact) with per-stage progress log
  - Satellite basemap (Esri WorldImagery) with vessel tracks, wave scatter, and coastline overlay
  - Spatial filters: freehand polygon, drag-box, wave-arrival box
  - Attribute filters: vessel type multi-select, MMSI, track similarity
  - Export filtered: saves rerun-ready dataset (AIS subset, tracks, waves, copied input files) to a new `data/` subfolder; default name `<workdir>_filtered`; guard against overwriting existing folders
  - Report plots generated on export: wave height map, wave period map, vessel track scatter, top-10 vessels CSV

---

## Step 11: End-to-End Validation with Real Data ✅

`validate_pipeline.py` runs the pipeline on `data/AIS_2563.csv` and compares against MATLAB reference (`table_ShoreImpact_*.csv`). Documents expected differences:
- Gravity: 9.78 (Python) vs 9.81 (MATLAB)
- Cb method: type-based `L_Le` (Python) vs old global table (MATLAB)
- Wake directions: COG ± θ (Python) vs COG ± 90° (MATLAB)
- Distance: geodetic WGS84 (Python) vs planar (MATLAB)

---

## Step 12: Performance Optimization ✅

See `docs/PERFORMANCE.md` for detailed status. Summary:

| Fix | Status |
|-----|--------|
| 1. Vectorize `validate_speed` | ✅ |
| 2. Vectorize `mask_land` | ✅ |
| 3. Reduce allocations in `interpolate_trajectories` | ✅ |
| 4. STRtree spatial index for shore intersection | ✅ |
| 5. Coastline-binned top-N visualisation | ✅ (config field renamed `plot_max_points`) |
| 6. Per-stage timing + status logging | ✅ (plain `print()` chosen over Rich Console — vectorised stages have no per-row progress to render) |

---

## Summary: Dependency Graph

```
Step 0: Scaffolding
  │
  ├→ Step 1: Config
  │
  ├→ Step 2: Geodesy
  │    │
  │    ├→ Step 4: AIS Filter + Depth check (13 sub-steps)
  │    │    │      [Step 5 depth utility called at sub-step 13]
  │    │    │
  │    │    └→ Step 6: Vessel Params (vessel.py)
  │    │         │
  │    ├→ Step 7: Coastline (uses geodesy for rays)
  │    │    │
  │    │    └→ Step 8: Wave Impact (wave_impact.py)
  │    │
  │    └→ Step 3: Block Coeff (used by Step 6)
  │
  ├→ Step 9: Visualisation (uses wave impact output)
  │
  └→ Step 10: Pipeline + Notebook + Dash App
       │
       └→ Step 11: Validation
       │
       └→ Step 12: Performance (cross-cutting)
```
