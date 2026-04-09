# Implementation Plan — ShipwakeAIS Python Rewrite

**Based on**: PRD.md  
**Approach**: Incremental steps. Each step produces working, tested code before the next begins.

---

## Step 0: Project Scaffolding & Dependencies

**Goal**: Set up the Python project structure, install dependencies, verify imports.

**Tasks**:
1. Create the directory layout under `shipwake/` as defined in PRD §5.
2. Create `pyproject.toml` with `uv`; add core dependencies:
   - `numpy`, `pandas`, `pyproj`, `shapely`, `geopandas`, `fiona`
   - `mikeio`, `scipy`, `pydantic>=2`
   - `matplotlib`, `contextily`
   - dev: `pytest`, `jupyterlab`
3. Create `shipwake/__init__.py` with version string.
4. Create empty `__init__.py` in each sub-package (`stages/`, `geo/`, `vessel/`, `viz/`).

**Tests**:
- `test_imports.py`: verify all top-level packages import without error.
- `uv run pytest` passes.

**Deliverables**: bare project skeleton, `pyproject.toml`, all imports verified.

---

## Step 1: Configuration (`shipwake/config.py`)

**Goal**: Load and validate configuration from JSON file, JSON string, or Python dict.

**Tasks**:
1. Define Pydantic v2 models for each config section (`AisConfig`, `VesselConfig`, `BathymetryConfig`, `CoastlineConfig`, `WaveConfig`, `ImpactConfig`, `OutputConfig`, `ShipwakeConfig`).
2. Implement `load_config(source)` that accepts:
   - `str` ending in `.json` → read file
   - `str` not ending in `.json` → parse as JSON string
   - `dict` → validate directly
3. Ship a default `config.json` with the values from PRD §7.

**Tests** (`tests/test_config.py`):
- Load from a JSON file and verify all fields populated.
- Load from an inline JSON string.
- Load from a Python dict.
- Invalid config (missing required field) raises `ValidationError`.
- Unknown fields are rejected.

---

## Step 2: Geodesy Utilities (`shipwake/geo/geodesy.py`)

**Goal**: Wrap `pyproj.Geod` to provide the distance/bearing/forward-point functions used throughout.

**Tasks**:
1. `geodetic_distance(lon1, lat1, lon2, lat2) -> float` — metres between two points.
2. `geodetic_bearing(lon1, lat1, lon2, lat2) -> float` — forward azimuth in degrees.
3. `forward_point(lon, lat, bearing_deg, distance_m) -> (lon2, lat2)` — Vincenty forward.
4. Vectorised variants accepting numpy arrays for all three functions.

**Tests** (`tests/test_geodesy.py`):
- Known distance: Singapore (103.85, 1.29) to a point 1 km due east — verify within 0.5 m.
- Round-trip: forward_point then distance back should return the original distance.
- Bearing: due north = 0°, due east = 90°.
- Vectorised: 100 random point-pairs match scalar results.

---

## Step 3: Block Coefficient & Bow Entry (`shipwake/vessel/block_coeff.py`)

**Goal**: Implement all three Cb/Le lookup methods with a unified interface.

**Tasks**:
1. `get_vessel_params(length_m, beam_m, ship_type, method="L_Le") -> dict` returning `{'block_coeff', 'bow_entry_m'}`.
2. Method `"L_Le"`: type-based L/Le lookup (PRD §2.5 Method A).
3. Method `"B_Le"`: type-based B/Le lookup (PRD §2.5 Method B).
4. Method `"table"`: load `ShipDataEDnew.csv`, filter by type category, KDTree nearest-neighbour in (LOA, Beam) space (PRD §2.5 Method C).
5. Copy `ShipDataEDnew.csv` into `shipwake/vessel/`.
6. Vectorised wrapper: `get_vessel_params_df(df, method)` accepting a DataFrame with `length`, `width`, `typecargo` columns, returning two new columns.

**Tests** (`tests/test_block_coeff.py`):
- L_Le method: tanker type 80 with L=200 → Cb=0.86, Le=200/7.
- L_Le method: fishing type 30 with L=20 → Cb=0.67, Le=20/3.
- B_Le method: cargo type 70 with B=20 → Cb=0.70, Le=20/0.7.
- Table method: tanker type 80 with L=350, B=63 → match MATLAB `func_cb_tablelooking` output for same input.
- Unknown type code defaults to "all others" category.
- Vectorised: DataFrame with mixed types returns correct per-row results.

---

## Step 4: AIS Filtering & Interpolation (`shipwake/stages/filter.py`)

**Goal**: Load raw AIS CSV, clean, segment, interpolate, and mask land points.

**Tasks**:
1. `load_ais(csv_path) -> DataFrame` — read CSV, parse `obstime` as datetime, retain required columns.
2. `segment_trajectories(df, gap_s=600) -> DataFrame` — sort by mmsi+obstime, assign `segment_id` based on time gaps.
3. `validate_speed(df) -> DataFrame` — compute inter-fix geodetic distance and time delta, derive `v_calc`, set `sog = min(sog_ais, v_calc)`.
4. `interpolate_trajectories(df, spacing_m=20, trigger_m=100) -> DataFrame` — for gaps > trigger_m, insert linearly interpolated points at spacing_m intervals.
5. `mask_land(df, coastline_shp) -> DataFrame` — load coastline polygon via geopandas, remove points inside polygon.
6. `filter_ais(csv_path, coastline_shp, **params) -> DataFrame` — orchestrator calling steps 1–5 in sequence.

**Tests** (`tests/test_filter.py`):
- `load_ais`: synthetic 5-row CSV with known columns → correct dtypes and shape.
- `segment_trajectories`: two fixes 15 min apart for same MMSI → two segments. Two fixes 5 min apart → same segment.
- `validate_speed`: a vessel reporting 10 kts SOG but only moving 50 m in 60 s → sog clamped to ~1.6 kts.
- `interpolate_trajectories`: two points 200 m apart → ~10 interpolated points at 20 m spacing. Points 50 m apart (< trigger) → no interpolation.
- `mask_land`: one point inside a simple polygon, one outside → only outside survives.
- Integration: run `filter_ais` on a small synthetic CSV + a simple rectangular coastline polygon.

---

## Step 5: Bathymetry & Tidal Depth (`shipwake/geo/bathymetry.py` + `shipwake/stages/depth.py`)

**Goal**: Look up static bathymetric depth from mesh, add tidal level, apply under-keel filter.

**Tasks**:
1. `bathymetry.py`:
   - `load_bathymetry(path) -> object` — load `.mesh` via `mikeio.Mesh()` or `.dfsu` via `mikeio.read()`. Cache the geometry.
   - `get_depth(bathy, lons, lats) -> np.ndarray` — look up depth at nearest node/element using mikeio spatial methods (discover exact API during implementation).
2. `load_tide(dfs0_path) -> pd.Series` — read `.dfs0` via `mikeio.read()`, return time-indexed Series.
3. `snap_to_tide(obstimes, tide_series) -> np.ndarray` — snap each AIS timestamp to nearest tide interval, return tidal levels.
4. `stages/depth.py`:
   - `assign_depth(df, bathy_path, tide_dfs0_path=None, underkeel_margin=1.0) -> DataFrame`:
     - Add `WaterDepth` = bathy_depth (+ tide_level if tide provided).
     - Drop rows where depth is NaN (outside mesh).
     - Drop rows where `WaterDepth < draught + underkeel_margin`.
     - Drop rows whose obstime is outside tide series range (if tide provided).

**Tests** (`tests/test_depth.py`):
- `load_bathymetry`: load the actual mesh file from `improved_version/bathy/`, verify it returns a geometry object.
- `get_depth`: query a known point inside the mesh extent → returns a finite positive depth.
- `get_depth`: query a point far outside the mesh → returns NaN or raises.
- `snap_to_tide`: a timestamp at 12:07 with 6-min tide series → snaps to 12:06. Timestamp outside series range → flagged.
- `assign_depth` integration: small synthetic DataFrame with 3 rows (one shallow, one deep, one outside mesh) → correct filtering.

**Note**: this step depends on the actual data files in `improved_version/bathy/` and `improved_version/add_tide/`. Tests that load real files are integration tests; pure logic tests use synthetic data.

---

## Step 6: Wave Parameter Calculation (`shipwake/stages/wave_params.py`)

**Goal**: Vectorised computation of all wave parameters from PRD §FR-4.

**Tasks**:
1. `compute_wave_params(df, cb_method="L_Le", g=9.78, rho=1026, filters=None) -> DataFrame`:
   - Call `get_vessel_params_df` for Cb and Le.
   - Compute all columns from the formulas table (V_ms through WakeDirStarboard).
   - Apply row filters (Froude range, BF, SOG, B/L, depth).
   - Return DataFrame with all computed columns.
2. `export_gis(df) -> DataFrame` — extract the 15-column GIS subset.

**Tests** (`tests/test_wave_params.py`):
- **Single-row hand-calc**: tanker type 80, L=200, B=30, draught=10, SOG=8 kts, depth=15 m. Manually compute expected Cb, Le, Beta, Alpha, FroudeM, FroudeD, H_Kreibel, Theta, WakeDirPort, WakeDirStarboard. Assert each column within 1% tolerance.
- **Froude filter**: a row with SOG=0.5 kts (FroudeM < 0.1) → filtered out.
- **SOG filter**: a row with SOG=15 kts → filtered out.
- **B/L filter**: a row with B=30, L=50 (ratio 0.6) → filtered out.
- **Depth filter**: a row with depth=0 → filtered out.
- **WakeDir uses ±θ, not ±90°**: verify WakeDirPort = COG − θ (not COG − 90).
- **GIS export**: verify 15 columns returned, correct column names.
- **Cross-check against MATLAB output**: load a small slice (first 100 rows) of the existing `*ALL-WaveParameters.csv` from `data/WaveCalc/`. Run the same input through `compute_wave_params` and compare. Allow tolerance for:
  - Cb differences (new type-based method vs old table lookup).
  - Angle differences (±θ vs ±90°).
  - Gravity difference (9.78 vs 9.81).

---

## Step 7: Coastline Operations (`shipwake/geo/coastline.py`)

**Goal**: Load coastline shapefile and provide ray-intersection methods.

**Tasks**:
1. `load_coastline(shp_path) -> MultiPolygon` — read shapefile via geopandas, union all features into a single MultiPolygon.
2. `build_ray(lon, lat, bearing_deg, distance_m) -> LineString` — construct a ray using `forward_point` from geodesy module.
3. `find_shore_intersection(ray, coastline) -> (lon, lat, distance_m) | None` — intersect ray with coastline, return closest intersection point and geodetic distance from ray origin.

**Tests** (`tests/test_coastline.py`):
- Load actual `shpfile/Coast_P1.shp` → returns a valid MultiPolygon with area > 0.
- `build_ray`: from a known point, bearing 90°, 1000 m → endpoint roughly 1 km east.
- `find_shore_intersection`: a ray from sea toward land → returns intersection. A ray pointing out to open sea → returns None.
- Synthetic test: simple rectangular polygon, ray from inside pointing outward → intersects boundary at expected distance.

---

## Step 8: Shore Impact Calculation (`shipwake/stages/shore_impact.py`)

**Goal**: For each wake event, cast rays in port/starboard wake directions, find shore intersection, compute decayed wave height.

**Tasks**:
1. `compute_shore_impact(df_wave, coastline_shp, max_propagation_m=2000, wake_cutoff_m=0.01, g=9.78) -> DataFrame`:
   - Load coastline once.
   - For each row, build port and starboard rays using `WakeDirPort` and `WakeDirStarboard`.
   - Find intersection with coastline.
   - Compute perpendicular distance (ray origin to intersection).
   - Apply wave decay formula: `H_shore = β × (F_m − 0.1)² × (dist / L_WL)^(−1/3) × V_ms² / g`.
   - Filter by `wake_cutoff_m`.
   - Return output DataFrame with columns: MMSI, shore lon/lat, WaveHeight, WavePeriod, DistLoc, DateTime, FroudeM, VesselWidth, VesselLength, SOG, Side.

**Tests** (`tests/test_shore_impact.py`):
- **Hand-calc**: one vessel at known position, known wave params, synthetic rectangular coastline 500 m away → verify H_shore matches manual calculation.
- **No intersection**: vessel with ray pointing to open sea → no output row.
- **Below cutoff**: vessel far from shore producing H_shore < 0.01 m → filtered out.
- **Port and starboard**: one vessel → produces up to 2 output rows (one per side).
- **Distance**: verify DistLoc matches geodetic distance from vessel to shore intersection.

---

## Step 9: Visualisation (`shipwake/viz/`)

**Goal**: Generate wave height/period maps and optional per-vessel diagrams.

**Tasks**:
1. `wave_map.py`:
   - `plot_wave_height_map(df_impact, coastline_shp, output_path)` — scatter plot of shore impact points, colour-coded by WaveHeight, with coastline overlay.
   - `plot_wave_period_map(df_impact, coastline_shp, output_path)` — same, colour-coded by WavePeriod.
2. `vessel_diagram.py`:
   - `plot_vessel_wake(vessel_mmsi, df_wave, df_impact, coastline_shp, output_path)` — single vessel track + wake rays + shore intersection points.

**Tests** (`tests/test_viz.py`):
- `plot_wave_height_map`: generates a PNG file at the specified path, file size > 0.
- `plot_wave_period_map`: same.
- `plot_vessel_wake`: generates a PNG for a single MMSI.
- Plots do not raise exceptions on empty DataFrames (produce blank figure with warning).

---

## Step 10: Pipeline Orchestration & Notebook

**Goal**: Wire all stages together; create the Jupyter notebook interface.

**Tasks**:
1. `shipwake/pipeline.py`:
   - `run_pipeline(config, stages=None) -> dict` — runs selected stages in sequence, passing DataFrames between them. Returns dict of stage outputs.
2. `main.py`:
   - CLI entry: `python -m shipwake --config config.json [--stage filter|depth|wave|impact|viz|all]`.
3. `run_shipwake.ipynb`:
   - Cell 1: path variables.
   - Cell 2: load config.
   - Cells 3–8: one per stage, as specified in PRD §5 notebook layout.

**Tests** (`tests/test_pipeline.py`):
- Run full pipeline on a small synthetic dataset (10 vessels, 50 points each, simple rectangular coastline, synthetic flat bathymetry) → produces shore_impact DataFrame and PNG files.
- Run individual stages in isolation with pre-computed inputs → same results as full pipeline.

---

## Step 11: End-to-End Validation with Real Data

**Goal**: Run the pipeline on the actual example dataset from the project and compare with MATLAB outputs.

**Tasks**:
1. Run pipeline on `data/AIS_2563.csv` (or a representative subset) with actual `Coast_P1.shp` and bathymetry.
2. Compare shore impact results against existing `data/WaveCalc/ShoreImpact/table_ShoreImpact_*.csv`.
3. Document expected differences (gravity 9.78 vs 9.81, Cb method, ±θ vs ±90° angles) and verify remaining values match within tolerance.
4. Generate output plots and visually compare with existing PNGs.

**Tests**:
- This is a manual validation step with documented acceptance criteria, not an automated test suite.
- Record comparison results in `tests/validation_report.md`.

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
  │    │    └→ Step 5: Depth (receives filtered df)
  │    │         │
  │    │         └→ Step 6: Wave Params (receives depth df)
  │    │              │
  │    ├→ Step 7: Coastline (uses geodesy for rays)
  │    │    │
  │    │    └→ Step 8: Shore Impact (uses wave params + coastline)
  │    │
  │    └→ Step 3: Block Coeff (independent, used by Step 6)
  │
  ├→ Step 9: Visualisation (uses shore impact output)
  │
  └→ Step 10: Pipeline + Notebook (wires everything)
       │
       └→ Step 11: Validation
```

Steps 1, 2, 3 can be built in parallel. Steps 4–8 are sequential (each depends on the previous stage's output). Step 9 and 10 follow after the core stages.
