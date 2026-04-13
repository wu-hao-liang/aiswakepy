# Product Requirements Document  
# aiswakepy — Python Rewrite

**Date**: 2026-04-09 (updated)  
**Status**: Current  
**Based on**: REVIEW.md + owner comments + implementation

---

## 1. Purpose & Scope

Rewrite the existing MATLAB shipwake calculation pipeline in Python. The system:

1. Ingests AIS vessel tracking data.
2. Filters, cleans, and interpolates vessel trajectories.
3. Computes ship-wake wave parameters at each vessel position using an empirical formula (Kriebel & Seelig 2005).
4. Determines whether each wake event will intersect a target shoreline/polygon, accounting for wake propagation physics.
5. Applies a distance-decay formula to estimate wave height at the shoreline.
6. Outputs tabular results and geographical visualisations.

---

## 2. Resolved Design Decisions (Owner Comments)

### 2.1 Configuration & Paths
All file paths (AIS input, shapefile, bathymetry, output directory) must be externally configurable — no hardcoded paths.  
**Approach**: JSON config file (`config.json`) loaded at startup. JSON is chosen over YAML because:
- It is natively parseable by AI agents and LLM tool-use frameworks (no extra dependencies).
- It can be passed as an inline string (useful in notebooks or API calls).
- Python's `json` module is in the standard library — no `pyyaml` dependency.

Configuration can be loaded from:
1. A JSON file path: `config = load_config("config.json")`
2. A JSON string: `config = load_config('{"ais": {"raw_csv": "data/AIS_2563.csv"}, ...}')`
3. A Python `dict` directly (for notebook use): `config = load_config({"ais": {...}})`

### 2.1.1 Notebook-First Workflow
The primary user interface is **Jupyter Notebook**, not a CLI. Engineers frequently change input paths during testing and exploration. The design must support this workflow:

- A master notebook (`run_shipwake.ipynb`) with clearly separated cells per stage.
- **Path variables defined in a dedicated cell at the top** of the notebook, easy to edit between runs:
  ```python
  # === EDIT THESE PATHS FOR YOUR RUN ===
  ais_csv       = r"examples/ais/AIS_2563.csv"
  coastline_shp = r"examples/coastline/Coast_P1.shp"
  bathy_file    = r"examples/bathymetry/bathy.mesh"
  output_dir    = r"output/"
  ```
- These variables are passed to each stage function directly — no global state, no hidden config.
- For batch/production runs, the same functions accept a `config.json` file instead.
- Each stage returns a `pandas.DataFrame` that stays in notebook memory, so the user can inspect, filter, or plot interactively between stages.

### 2.2 Wake Propagation Angle — Physics Clarification

The perpendicular beam direction (COG ± 90°) does **not** represent the physical propagation direction of ship wake waves.

In a Kelvin wake pattern, the cusp line (outer envelope of the wake wedge) makes a half-angle of **arcsin(1/3) ≈ 19.47°** relative to the vessel's heading. However, the **wave crests at the cusp propagate outward** at a different angle: **arcsin(1/√3) ≈ 35.26°** relative to the vessel's heading. This is the direction of energy propagation that matters for shoreline impact assessment — it is the angle at which the highest-energy wave crests travel away from the sailing line toward the shore.

In **shallow water**, this propagation angle depends on the **depth Froude number** (F_d = V/√(gh)). The empirical formula:
```
θ = 35.27 × (1 − exp(12 × (F_d − 1)))   [degrees]
```
captures this relationship. Note that the deep-water asymptote (≈ 35.27°) closely matches the theoretical value arcsin(1/√3) ≈ 35.26°.

**Therefore**: when casting a ray to find a shoreline intersection, the correct directions are **COG − θ** (port) and **COG + θ** (starboard), where θ is the computed wake crest propagation angle.  
The old ±90° convention was physically incorrect for determining propagation direction.

> **Note**: the perpendicular distance from the sailing track to the intersection point is still the correct input to the wave height decay formula. This is `dist_perp = dist_ray × sin(θ)`.

### 2.3 Datetime Handling
Raw AIS data timestamps are ISO 8601 UTC strings. All internal processing uses `pandas.Timestamp` (timezone-naive, UTC assumed). No conversion to MATLAB serial dates or Excel serial dates.

### 2.4 Distance Calculation
Use **geodetic (ellipsoidal) distance** via `pyproj.Geod` (WGS84) for all distance and bearing calculations. Do not use great-circle spherical approximation.  

Additionally: vessel movement is uncertain over long distances — a vessel position fix only tells us where the vessel was at that instant, not whether it travelled in a straight line. Therefore, impose a **maximum meaningful wake propagation distance** (default: 2000 m, configurable). Beyond this limit, wake impact is not computed regardless of whether a geometric intersection exists.

### 2.5 Block Coefficient & Bow Entry Length
Three methods are available as selectable backends:

**Method A — L/Le ratio** (`L_Le`, default):  
- Tankers (type 80–89): Cb = 0.86, Le = L/7
- Cargo / Dredger (type 33, 70–79): Cb = 0.80, Le = L/5
- All others: Cb = 0.67, Le = L/3

**Method B — B/Le ratio** (`B_Le`):  
- Tankers: Cb = 0.80, Le = B/1.0
- Cargo / Dredger: Cb = 0.70, Le = B/0.7
- All others: Cb = 0.60, Le = B/0.4

**Method C — Type-filtered table lookup** (`table`):  
Filters `ShipDataEDnew.csv` to the relevant vessel category, then nearest-neighbour lookup in (LOA, Beam) space. Returns Cb and B/Le.

The interface contract:
```python
def get_vessel_params(
    length_m: float, beam_m: float, ship_type: int,
    method: str = "L_Le"
) -> dict:
    """Returns {'block_coeff': float, 'bow_entry_m': float}"""
```

**Requires**: the AIS input must include a `typecargo` (AIS ship type code) column.

### 2.6 Bathymetry & Tidal Water Level
Water depth at each vessel position is the sum of two components:

1. **Static bathymetric depth** — from `.mesh` or `.dfsu` file via `mikeio`. Read the mesh once, cache it, expose a `get_depth(lon, lat)` function backed by `scipy.spatial.KDTree` for fast nearest-node lookup.
2. **Tidal water level** — from predicted tide time series in `.dfs0` file via `mikeio`. Each AIS timestamp is rounded to the nearest interval in the tide series, and the tidal level is added to the static depth.

After combining: **filter out records where `WaterDepth < draught + 1 m`** (insufficient under-keel clearance).

### 2.7 Timezone
Timestamps are treated as UTC throughout. No timezone localisation is required at this stage.

### 2.8 AIS Data Cleaning & Interpolation

The AIS filter stage applies the following steps in order:

| Step | Function | Description |
|------|----------|-------------|
| 1 | `load_ais` | Read CSV; parse obstime; drop NaN obstime/lat/lon |
| 2 | `deduplicate` | Remove duplicate (mmsi, obstime) rows — keeps first. Prevents dt=0 in speed calculations |
| 3 | `uniformize_vessel_info` | Set width/length/typecargo to mode per MMSI — ensures consistent vessel dimensions |
| 4 | `remove_zero_dimensions` | Drop rows where width/length/draught ≤ 0 — required by Kriebel formula |
| 5 | `segment_trajectories` | Sort by mmsi+obstime; split on gaps > `traj_gap_s` (default **180 s**) |
| 6 | `clean_error_coords` | **Kinematic Consistency Check**: flag segments where avg speed > `max_velocity_knots` (default 12); remove GPS spikes by flag count (see §2.8.1) |
| 7 | `clean_error_speed` | **Acceleration Check**: replace SOG/COG where implied acceleration > `max_acceleration_ms2` (default 0.2 m/s²); replacement is distance-weighted average of adjacent finite-difference velocities (see §2.8.2) |
| 8 | `validate_speed` | Secondary cap: `SOG = min(reported, geodetic-computed)` |
| 9 | `interpolate_trajectories` | **Cubic Hermite Spline** at `interp_interval_s` (default 30 s); produces smooth SOG/COG from spline derivatives; draught by nearest-neighbour in time |
| 10 | `filter_study_area` | Optional: keep only points inside a study-area polygon (`study_area_shp`; null = skip) |
| 11 | `mask_land` | Remove points inside coastline polygon |

#### 2.8.1 Kinematic Consistency Check (`clean_error_coords`)

For each consecutive pair `(i, i+1)` in a segment: compute `avg_speed = distance / dt`. If `avg_speed > max_velocity_knots`, flag both endpoints +1.

Resolve by flag count:
- **Flag = 2**: point is a GPS spike (both adjacent segments too fast) → remove.
- **Flag = 1, neighbour = 2**: this point is clean (neighbour was the spike) → keep.
- **Flag = 1, neighbour = 1**: ambiguous drift. Search outward for the nearest other flag-1 point from a *different* suspicious segment; remove all points between. If not found, remove the shorter-in-time half of the trajectory.
- **Flag = 0**: clean → keep.

#### 2.8.2 Acceleration Check (`clean_error_speed`)

For each point, compute the acceleration required for the AIS-reported velocity (from SOG/COG) to match the segment finite-difference velocity within half the time interval:
```
accel_fwd = (v_ais − v_fwd) / (dt_fwd / 2)
accel_bwd = (v_ais − v_bwd) / (dt_bwd / 2)
```
If `|accel|` in either x or y direction exceeds `max_acceleration_ms2`, replace SOG/COG with distance-weighted average of adjacent segment velocities.

---

## 3. Functional Requirements

### FR-1: Configuration Management
- `config.json` defines all inputs, outputs, and tunable parameters.
- Primary interface: **Jupyter notebook** with path variables in a top cell and one cell per stage.
- Programmatic interface: each stage is a pure function accepting a config dict or individual arguments.
- Optional CLI for batch runs: `uv run python main.py --config config.json`.
- Each stage can be run independently if its input DataFrame already exists.

### FR-2: AIS Filtering (`stage: filter`)

Full pipeline as described in §2.8. Output columns:
`mmsi, width, length, draught, obstime, longitude, latitude, sog, cog, typecargo, segment_id`

### FR-3: Water Depth & Tidal Level Assignment (`stage: depth`)
- Look up static bathymetric depth from `.dfsu` or `.mesh` via `mikeio` + `scipy.spatial.KDTree`.
- Look up predicted tidal water level from `.dfs0` via `mikeio`; snap AIS timestamp to nearest interval.
- Total: `WaterDepth = bathy_depth + tide_level`.
- **Under-keel clearance filter**: remove `WaterDepth < draught + underkeel_margin` (default 1 m).
- Points outside mesh extent (NaN depth) are excluded.
- Records whose timestamps fall outside the tidal prediction range are excluded.

### FR-4: Wave Parameter Calculation (`stage: wave`)

Empirical model: **Kriebel & Seelig (2005)** — "An Empirical Model for Ship-Generated Waves".

For each valid AIS point compute:

| Parameter | Formula |
|-----------|---------|
| V_ms | `SOG × 0.5144444` |
| Be (bow entry) | from `get_vessel_params(length, beam, typecargo, method)` |
| L_WL (waterline length) | `length × 0.8` |
| Cb (block coefficient) | from `get_vessel_params(length, beam, typecargo, method)` |
| α | `2.35 × (1 − Cb)` |
| β | `1 + 8 × tanh(0.45 × (L_WL / Be − 2))³` |
| F_m (modified Froude) | `(V_ms / √(g × L_WL)) × exp(α × draught / depth)` |
| F_d (depth Froude) | `V_ms / √(g × depth)` |
| **H_Kriebel** | `β × (F_m − 0.1)² × (Be / (2 × L_WL))^(−1/3) × V_ms² / g` |
| T (wave period) | `0.27 × SOG` |
| E_max | `(ρ × g² × H² × T²) / (16π)`,  ρ = 1026 kg/m³ |
| E_tot | `10.8 × E_max^0.82` |
| **θ (wake half-angle)** | `35.27 × (1 − exp(12 × (F_d − 1)))` degrees |
| Wave celerity component | `V_ms × cos(θ)` |
| T_c | `(2π × Cel) / g` |
| **WakeDirPort** | `COG − θ` (bearing of port cusp line) |
| **WakeDirStarboard** | `COG + θ` (bearing of starboard cusp line) |

**Row filter — discard if**:
- `F_m < 0.1` or `F_m > 0.5`
- `β × (F_m − 0.1)² > 0.4`
- `SOG > 12 knots`
- `beam / length > 0.3`
- `depth ≤ 0`

**Output**: Parquet or CSV with all computed columns; a GIS-ready 15-column export for QGIS/ArcGIS.

### FR-5: Shore Impact Calculation (`stage: impact`)

For each wake event (one port + one starboard ray per point):

1. **Ray construction**: from vessel (lon, lat), bearing = `WakeDirPort` or `WakeDirStarboard`, length = `max_propagation_m` (default 2000 m), via `pyproj.Geod.fwd()`.
2. **Intersection test**: `shapely` ray `LineString` vs coastline `MultiPolygon` boundary. Uses `STRtree` spatial index on decomposed boundary segments for performance.
3. **Select closest intersection** to vessel if multiple exist.
4. **Perpendicular distance**: `dist_perp = dist_ray × sin(θ)` — lateral distance from sailing track, the correct input to the Kriebel decay formula.
5. **Wave height at shore**:
   ```
   H_shore = β × (F_m − 0.1)² × (dist_perp / L_WL)^(−1/3) × V_ms² / g
   ```
6. **Threshold filter**: discard if `H_shore < wake_cutoff_m` (default 0.01 m).
7. **Output row**: MMSI, shore lon/lat, WaveHeight, WavePeriod, DistLoc_km (perpendicular), DateTime, F_m, VesselWidth, VesselLength, SOG, Side.

### FR-6: Output & Visualisation
- Shore impact CSV: one row per intersection event.
- Parquet of all wave parameters (optional).
- **Wave height map**: scatter plot over coastline, colour-coded by H_shore.
  - Points binned at 1 m resolution along coastline (`LineString.project()`).
  - Top `plot_top_n_per_bin` (default 10) wave heights per bin are plotted.
  - Points sorted ascending before scatter so highest waves render on top.
- **Wave period map**: same layout, colour-coded by T.
- **Per-vessel diagram** (optional, configurable): vessel track + wake rays + intersection point.
- All figures saved as PNG to a configurable output directory.
- Console output via `rich.console.Console` with per-stage elapsed timing.

---

## 4. Non-Functional Requirements

| Requirement | Target |
|------------|--------|
| Performance (small) | Process 50 MB AIS CSV (≈100k rows) end-to-end in < 2 minutes |
| Performance (large) | Process 2M AIS records end-to-end in ~3–12 minutes (all optimisations applied) |
| Vectorisation | Stages 2–4 use pandas/numpy vectorised operations; no Python-level row loops |
| Reproducibility | All parameters in config; outputs versioned with input hash |
| Testability | Each stage exposed as a pure function; 118 unit tests, all passing |
| Portability | No OS-specific paths; runs on Windows, Linux, macOS |

---

## 5. Architecture

```
aiswakepy/
├── config.json                   ← Default configuration (JSON)
├── run_shipwake.ipynb            ← Primary notebook interface
├── main.py                       ← CLI entry point for batch runs
├── validate_pipeline.py          ← End-to-end validation vs MATLAB reference
├── PERFORMANCE_PLAN.md           ← Performance optimisation details
├── aiswakepy/
│   ├── __init__.py
│   ├── config.py                 ← Config schema (Pydantic v2) + loader
│   ├── pipeline.py               ← Stage orchestration (Rich console + timing)
│   ├── models/
│   │   └── kriebel.py            ← Kriebel & Seelig (2005) empirical model
│   ├── stages/
│   │   ├── filter.py             ← FR-2: AIS cleaning & interpolation (11 steps)
│   │   ├── depth.py              ← FR-3: bathymetry depth + tidal adjustment
│   │   ├── wave_params.py        ← FR-4: wave parameter calculation
│   │   └── shore_impact.py       ← FR-5: ray-casting, STRtree, decay formula
│   ├── geo/
│   │   ├── geodesy.py            ← pyproj wrappers: distance, bearing, fwd point
│   │   ├── coastline.py          ← Shapefile load, STRtree builder, ray intersection
│   │   └── bathymetry.py         ← mikeio mesh reader + KDTree depth lookup
│   ├── vessel/
│   │   ├── block_coeff.py        ← Cb + bow-entry lookup (L_Le / B_Le / table)
│   │   └── ShipDataEDnew.csv     ← Table-lookup vessel data
│   └── viz/
│       └── wave_map.py           ← Wave height / period maps (coastline-binned)
├── tests/                        ← 118 unit tests
└── examples/                     ← Example data (gitignored — large files)
    ├── ais/
    ├── bathymetry/
    ├── coastline/
    ├── tide/
    └── matlab_reference/
```

### Notebook Layout (`run_shipwake.ipynb`)

```
Cell 1 — Path Configuration       (edit per run)
Cell 2 — Build Config
Cell 3 — Stage 1: AIS Filter      → df_filtered
Cell 4 — Stage 2: Depth + Tide    → df_depth
Cell 5 — Stage 3: Wave Parameters → df_wave
Cell 6 — Stage 4: Shore Impact    → df_impact
Cell 7 — Visualisation
Cell 8 — Export
```

---

## 6. Key Python Package Choices

| Purpose | Package | Rationale |
|---------|---------|-----------|
| Geodetic distance & bearing | `pyproj` (`pyproj.Geod`) | WGS84 ellipsoidal, authoritative, fast |
| Geometry (intersection, polygon) | `shapely` | Robust; `STRtree` for spatial indexing |
| Shapefile I/O | `geopandas` + `fiona` | Industry standard |
| Data tables | `pandas` | AIS and results tables |
| Numerical arrays | `numpy` | Formula evaluation |
| Spline interpolation | `scipy.interpolate.CubicHermiteSpline` | Smooth trajectory interpolation with velocity constraints |
| Nearest-neighbour (depth, Cb) | `scipy.spatial.KDTree` | Fast spatial lookups |
| Bathymetry mesh | `mikeio` | Native DHI `.dfsu`/`.mesh` reader |
| Config schema & validation | `pydantic` v2 | Catches misconfiguration early |
| Config format | `json` (stdlib) | No extra dependency; AI-agent friendly |
| Notebook interface | `jupyterlab` | Primary user interface |
| Plotting | `matplotlib` | Maps and scatter plots |
| Progress bars & logging | `rich` | Stage progress bars and console output |
| Output (optional) | `pyarrow` (Parquet) | Faster I/O than CSV for large intermediates |
| Package manager | `uv` | Fast dependency resolution; always use `uv add`, never pip |

---

## 7. Configuration Schema (`config.json`)

```json
{
  "ais": {
    "raw_csv": "examples/ais/AIS_2563.csv",
    "min_speed_knots": 0.5,
    "traj_gap_s": 180,
    "max_velocity_knots": 12.0,
    "max_acceleration_ms2": 0.2,
    "interp_interval_s": 30.0,
    "study_area_shp": null
  },
  "vessel": {
    "cb_method": "L_Le",
    "block_coeff_csv": "aiswakepy/vessel/ShipDataEDnew.csv",
    "waterline_factor": 0.8
  },
  "bathymetry": {
    "source": "examples/bathymetry/bathy.mesh",
    "tide_dfs0": "examples/tide/tide.dfs0",
    "underkeel_margin_m": 1.0
  },
  "coastline": {
    "shapefile": "examples/coastline/Coast_P1.shp"
  },
  "wave": {
    "max_froude_m": 0.5,
    "min_froude_m": 0.1,
    "max_bf": 0.4,
    "max_sog_knots": 12.0,
    "max_bl_ratio": 0.3,
    "rho_water": 1026,
    "gravity": 9.78
  },
  "impact": {
    "max_propagation_m": 2000,
    "wake_cutoff_m": 0.01
  },
  "output": {
    "directory": "output/",
    "save_parquet": true,
    "plot_wave_height_map": true,
    "plot_period_map": true,
    "plot_vessel_diagrams": false,
    "plot_top_n_per_bin": 10
  }
}
```

---

## 8. Data Flow & File Contracts

```
config.json  (or inline dict in notebook)
     │
     ▼
[Raw AIS CSV — with typecargo column]
     │
     ▼
filter.py  →  df_filtered
     │        cols: mmsi, width, length, draught, obstime,
     │              longitude, latitude, sog, cog, typecargo, segment_id
     │        (11-step pipeline: dedup → clean → Hermite spline → land mask)
     ▼
depth.py   →  df_depth
     │        + WaterDepth column (bathy + tide)
     │        filters: WaterDepth >= draught + underkeel_margin
     ▼
wave_params.py  →  wave_params.parquet  (all cols)
     │             wave_params_gis.csv  (15-col GIS export)
     ▼
shore_impact.py  →  shore_impact.csv
                    output/WaveHeightMap.png   (coastline-binned top-N)
                    output/WavePeriodMap.png
                    output/vessels/[MMSI].png  (if enabled)
```

---

## 9. Open Items / Future Scope

| Item | Priority | Notes |
|------|----------|-------|
| 2D spatial wave height grid | Medium | Bathymetric refraction and shoaling across the domain |
| Wave height validation | Medium | Compare model output against measured data at `NW_SSES_wave_measure_*.shp` stations |
| Multiple target polygons | Low | Support list of target zones (not just one coastline) |
| Parallel shore impact | Low | `concurrent.futures` per-vessel parallelism — STRtree already reduces bottleneck significantly |
| Hermite spline for COG wrap-around | Low | Spline interpolation across 0°/360° discontinuity not yet handled |
