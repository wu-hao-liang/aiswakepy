# Product Requirements Document  
# ShipwakeAIS — Python Rewrite

**Date**: 2026-04-08  
**Status**: Draft  
**Based on**: REVIEW.md + owner comments

---

## 1. Purpose & Scope

Rewrite the existing MATLAB shipwake calculation pipeline in Python. The system:

1. Ingests AIS vessel tracking data.
2. Filters and interpolates vessel trajectories.
3. Computes ship-wake wave parameters at each vessel position using an empirical formula (Kreibel).
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
  ais_csv       = r"data/AIS_2563.csv"
  coastline_shp = r"shpfile/Coast_P1.shp"
  bathy_file    = r"bathy/bathy.dfsu"
  output_dir    = r"output/"
  ```
- These variables are passed to each stage function directly — no global state, no hidden config.
- For batch/production runs, the same functions accept a `config.json` file instead.
- Each stage returns a `pandas.DataFrame` that stays in notebook memory, so the user can inspect, filter, or plot interactively between stages.

### 2.2 Wake Propagation Angle — Physics Clarification

The perpendicular beam direction (COG ± 90°) does **not** represent the physical propagation direction of ship wake waves.

In a Kelvin wake pattern, the cusp line (outer envelope of the wake wedge) makes a half-angle of **arcsin(1/3) ≈ 19.47°** relative to the vessel's heading. However, the **wave crests at the cusp propagate outward** at a different angle: **arcsin(1/√3) ≈ 35.26°** relative to the vessel's heading. This is the direction of energy propagation that matters for shoreline impact assessment — it is the angle at which the highest-energy wave crests travel away from the sailing line toward the shore.

In **shallow water**, this propagation angle depends on the **depth Froude number** (F_d = V/√(gh)). The empirical formula already in the code:
```
θ = 35.27 × (1 − exp(12 × (F_d − 1)))   [degrees]
```
captures this relationship. Note that the deep-water asymptote of this formula (≈ 35.27°) closely matches the theoretical value of arcsin(1/√3) ≈ 35.26°.

**Therefore**: when casting a ray to find a shoreline intersection, the correct directions are **COG − θ** (port) and **COG + θ** (starboard), where θ is the computed wake crest propagation angle.  
The old ±90° convention in `func_calcVesselWave.m` (`WavePort = COG − 90`, `WaveStarboard = COG + 90`) was physically incorrect for determining propagation direction. The Python implementation must use ±θ.

> **Note**: the perpendicular distance from the sailing track to the measurement point is still the correct input to the wave height decay formula (that is a geometric property of the formula, not a propagation direction).

### 2.3 Datetime Handling
Raw AIS data timestamps are ISO 8601 UTC strings. All internal processing uses `pandas.Timestamp` (timezone-naive, UTC assumed). No conversion to MATLAB serial dates or Excel serial dates.

### 2.4 Distance Calculation
Use **geodetic (ellipsoidal) distance** via `pyproj.Geod` (WGS84) for all distance and bearing calculations. Do not use great-circle spherical approximation.  

Additionally: vessel movement is uncertain over long distances — a vessel position fix only tells us where the vessel was at that instant, not whether it travelled in a straight line. Therefore, impose a **maximum meaningful wake propagation distance** (default: 2000 m, configurable). Beyond this limit, wake impact is not computed regardless of whether a geometric intersection exists.

### 2.5 Block Coefficient & Bow Entry Length
Three methods are now available (implemented in `improved_version/`). The Python rewrite must support all three as selectable backends:

**Method A — L/Le ratio** (`func_cb_L_Le`, default):  
Determines Cb and bow entry length from a fixed `length / Le` ratio, selected by AIS ship type code:
- Tankers (type 80–89): Cb = 0.86, Le = L/7
- Cargo / Dredger (type 33, 70–79): Cb = 0.80, Le = L/5
- All others: Cb = 0.67, Le = L/3

**Method B — B/Le ratio** (`func_cb_B_Le`, alternative):  
Same structure but uses beam instead of length:
- Tankers: Cb = 0.80, Le = B/1.0
- Cargo / Dredger: Cb = 0.70, Le = B/0.7
- All others: Cb = 0.60, Le = B/0.4

**Method C — Type-filtered table lookup** (`func_cb_tablelooking`):  
Filters `ShipDataEDnew.csv` (89 rows, categorised by vessel type) to the relevant category, then does nearest-neighbour lookup in (LOA, Beam) space. Returns Cb and B/Le.

The interface contract:

```python
def get_vessel_params(
    length_m: float, beam_m: float, ship_type: int,
    method: str = "L_Le"
) -> dict:
    """Returns {'block_coeff': float, 'bow_entry_m': float}"""
```

**Requires**: the AIS input must include a `typecargo` (AIS ship type code) column. This is a new input requirement compared to the old version.

### 2.6 Bathymetry & Tidal Water Level
Water depth at each vessel position is the sum of two components:

1. **Static bathymetric depth** — from `.mesh` or `.dfsu` file via `mikeio`. For `.mesh` files, load with `msh = mikeio.Mesh("path/to/file.mesh")`; the resulting `msh.geometry` object is the same type as the geometry from a `.dfsu` dataset read via `mikeio.read()`. Use `mikeio`'s built-in spatial methods to look up depth at the nearest node/element (exact API to be confirmed during implementation). Read the mesh once, cache it, expose a `get_depth(lon, lat)` function.
2. **Tidal water level** — from predicted tide time series in `.dfs0` file via `mikeio`. Each AIS timestamp is rounded to the nearest interval in the tide series, and the tidal level is added to the static depth.

After combining: **filter out records where `WaterDepth < draught + 1 m`** (insufficient under-keel clearance — wake formula is not physically valid when the vessel nearly touches the seabed).

Both lookups are combined in a single depth-assignment stage.

### 2.7 Timezone
Timestamps are treated as UTC throughout. No timezone localisation is required at this stage.

### 2.8 AIS Speed Filtering & Interpolation
The current approach (filter zero/very-low-speed records, then interpolate) is correct.  
Improvements in the Python rewrite:
- Use vectorised pandas/numpy operations instead of row-by-row loops.
- Allow the minimum valid speed threshold to be configurable (default: remove records where computed speed < 0.5 knots between fixes, indicative of stationary vessel or GPS drift).
- The interpolation spacing (default 20 m) and trajectory-break gap (default 600 s) remain configurable.

---

## 3. Functional Requirements

### FR-1: Configuration Management
- `config.json` defines all inputs, outputs, and tunable parameters.
- Primary interface: **Jupyter notebook** with path variables in a top cell and one cell per stage.
- Programmatic interface: each stage is a function that accepts a config dict or individual arguments.
- Optional CLI for batch runs: `python -m shipwake --config config.json [--stage all|filter|wave|impact]`.
- Each stage can be run independently if its input CSV/DataFrame already exists.

### FR-2: AIS Filtering (`stage: filter`)
| Step | Requirement |
|------|------------|
| Load | Read raw AIS CSV; retain columns: mmsi, width, length, draught, obstime, longitude, latitude, sog, cog, typecargo |
| Sort | Sort by MMSI, then obstime |
| Speed validation | Compute inter-fix distance and time; derive `v_calc`; use `min(sog_ais, v_calc)` |
| Trajectory segmentation | Gaps > 600 s treated as new segment |
| Interpolation | For inter-fix distance > interpolation threshold (default 100 m), insert intermediate points at 20 m spacing; linear interpolation of all numeric columns |
| Land masking | Remove points inside coastline polygon (loaded from shapefile) |
| Output | Save cleaned CSV (10 standard columns + typecargo + segment_id) |

### FR-3: Water Depth & Tidal Level Assignment (`stage: depth`)
- For each point in the cleaned AIS CSV, look up **static bathymetric depth** from `.dfsu` or `.mesh` via `mikeio`. Mesh loaded once and cached.
- Look up **predicted tidal water level** from `.dfs0` file via `mikeio`. Each AIS timestamp is snapped to the nearest available interval in the tide series.
- Compute total water depth: `WaterDepth = bathy_depth + tide_level`.
- **Under-keel clearance filter**: remove records where `WaterDepth < draught + underkeel_margin` (default 1 m).
- Points outside the mesh extent are flagged (depth = NaN) and excluded from downstream stages.
- Exclude AIS records whose timestamps fall outside the available tidal prediction time range.

### FR-4: Wave Parameter Calculation (`stage: wave`)

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
| H_Kreibel | `β × (F_m − 0.1)² × (Be / (2 × L_WL))^(−1/3) × V_ms² / g` |
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

**Output**: Parquet or CSV with all computed columns; a GIS-ready 15-column export for QGIS/ArcGIS (includes WaterDepth).

### FR-5: Shore Impact Calculation (`stage: impact`)

For each wake event (one port + one starboard ray per point):

1. **Ray construction**: from vessel (lon, lat), bearing = `WakeDirPort` or `WakeDirStarboard`, length = `max_propagation_distance` (default 2000 m), computed via `pyproj.Geod.fwd()`.
2. **Intersection test**: find intersection of ray `LineString` with coastline `MultiPolygon` using `shapely`.
3. **Select closest intersection** to vessel if multiple exist.
4. **Perpendicular distance** from vessel to intersection point — this is the distance used in the decay formula (see note in §2.2), computed geodetically.
5. **Wave height at shore**:
   ```
   H_shore = β × (F_m − 0.1)² × (dist_perp / L_WL)^(−1/3) × V_ms² / g
   ```
6. **Threshold filter**: discard if `H_shore < wake_cutoff` (default 0.01 m).
7. **Output row**: MMSI, shore lon/lat, WaveHeight, WavePeriod, DistLoc (km), DateTime, F_m, VesselWidth, VesselLength, SOG, Side (port/starboard).

### FR-6: Output & Visualisation
- Shore impact CSV: one row per intersection event.
- GIS-ready CSV: lon/lat columns suitable for direct import into QGIS.
- **Wave height map**: scatter plot over coastline, colour-coded by H_shore.
- **Wave period map**: same layout, colour-coded by T.
- **Per-vessel diagram** (optional, configurable): vessel track + wake rays + intersection point.
- All figures saved as PNG to a configurable output directory.

---

## 4. Non-Functional Requirements

| Requirement | Target |
|------------|--------|
| Performance | Process 50 MB AIS CSV (≈100k rows) end-to-end in < 5 minutes on a single core |
| Vectorisation | Stages 2–4 must use pandas/numpy vectorised operations; no Python-level row loops |
| Reproducibility | All random seeds and parameters logged; outputs versioned with input hash |
| Testability | Each stage exposed as a pure function that can be unit-tested independently |
| Portability | No OS-specific paths; runs on Windows, Linux, macOS |

---

## 5. Recommended Architecture

```
shipwake/
├── config.json                   ← Default configuration (JSON)
├── run_shipwake.ipynb            ← Primary notebook interface
├── main.py                       ← Optional CLI entry point for batch runs
├── shipwake/
│   ├── __init__.py
│   ├── config.py                 ← Config schema (Pydantic) + loader (JSON file / string / dict)
│   ├── pipeline.py               ← Stage orchestration (called from notebook or CLI)
│   ├── stages/
│   │   ├── filter.py             ← FR-2: AIS filtering & interpolation
│   │   ├── depth.py              ← FR-3: bathymetry depth lookup
│   │   ├── wave_params.py        ← FR-4: wave parameter calculation
│   │   └── shore_impact.py       ← FR-5: shoreline ray-casting & decay
│   ├── geo/
│   │   ├── geodesy.py            ← pyproj wrappers: distance, bearing, fwd point
│   │   ├── coastline.py          ← Shapefile load, polygon operations (geopandas/shapely)
│   │   └── bathymetry.py         ← mikeio mesh reader + depth interpolation
│   ├── vessel/
│   │   ├── block_coeff.py        ← Cb + bow-entry lookup (pluggable interface)
│   │   └── ship_data.csv         ← Fallback lookup table
│   └── viz/
│       ├── wave_map.py           ← Wave height / period map plots
│       └── vessel_diagram.py     ← Per-vessel wake diagrams
└── tests/
    ├── test_filter.py
    ├── test_wave_params.py
    ├── test_shore_impact.py
    └── test_geodesy.py
```

### Notebook Layout (`run_shipwake.ipynb`)

```
Cell 1 — Path Configuration
    ais_csv, coastline_shp, bathy_file, tide_dfs0, output_dir  (edit per run)

Cell 2 — Load Config & Initialise
    config = load_config(...)  or  build config dict from path variables

Cell 3 — Stage 1: AIS Filter
    df_filtered = filter_ais(ais_csv, coastline_shp, ...)
    df_filtered.head()  # inspect

Cell 4 — Stage 2: Depth + Tide Assignment
    df_depth = assign_depth(df_filtered, bathy_file, tide_dfs0, ...)

Cell 5 — Stage 3: Wave Parameters
    df_wave = compute_wave_params(df_depth, cb_method="L_Le", ...)

Cell 6 — Stage 4: Shore Impact
    df_impact = compute_shore_impact(df_wave, coastline_shp, ...)

Cell 7 — Visualisation
    plot_wave_height_map(df_impact, coastline_shp, output_dir)

Cell 8 — Export
    df_impact.to_csv(output_dir + "shore_impact.csv")
```

Each cell is self-contained: re-run any stage after tweaking parameters without restarting the pipeline.

---

## 6. Key Python Package Choices

| Purpose | Package | Rationale |
|---------|---------|-----------|
| Geodetic distance & bearing | `pyproj` (via `pyproj.Geod`) | WGS84 ellipsoidal, authoritative, fast |
| Ray endpoint computation | `pyproj.Geod.fwd()` | Replaces `m_fdist` Vincenty implementation |
| Geometry (intersection, polygon) | `shapely` | Robust, vectorisable via `geopandas` |
| Shapefile I/O | `geopandas` + `fiona` | Industry standard; wraps OGR |
| Data tables | `pandas` | AIS and results tables |
| Numerical arrays | `numpy` | All formula evaluation |
| Bathymetry mesh | `mikeio` | Native DHI `.dfsu`/`.mesh` reader |
| Config schema & validation | `pydantic` v2 | Catches misconfiguration early |
| Config format | `json` (stdlib) | No extra dependency; AI-agent friendly; inline-string capable |
| Notebook interface | `jupyter` / `jupyterlab` | Primary user interface |
| Plotting | `matplotlib` + `contextily` | Maps with basemap tiles |
| Nearest-neighbour (Cb lookup) | `scipy.spatial.KDTree` | Replaces manual nearest-neighbour |
| CLI (optional) | `argparse` (stdlib) | For batch runs only |
| Output (optional) | `pyarrow` (Parquet) | Faster I/O than CSV for large intermediates |

---

## 7. Configuration Schema (`config.json`)

```json
{
  "ais": {
    "raw_csv": "data/AIS_2563.csv",
    "min_speed_knots": 0.5,
    "interp_spacing_m": 20,
    "traj_gap_s": 600,
    "interp_trigger_m": 100
  },
  "vessel": {
    "cb_method": "L_Le",
    "block_coeff_csv": "aiswakepy/vessel/ship_data.csv",
    "waterline_factor": 0.8
  },
  "bathymetry": {
    "source": "bathy/bathy.dfsu",
    "tide_dfs0": "add_tide/Predicted Water Level (CD)_2024_WestCoast_6min.dfs0",
    "underkeel_margin_m": 1.0
  },
  "coastline": {
    "shapefile": "shpfile/Coast_P1.shp"
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
    "plot_vessel_diagrams": false
  }
}
```

The same structure can be passed as a Python dict in a notebook cell or as an inline JSON string.

---

## 8. Data Flow & File Contracts

```
config.json  (or inline dict in notebook cell)
     │
     ▼
[Raw AIS CSV — with typecargo column]
     │
     ▼
filter.py  →  ais_filtered.csv                                       
     │        cols: mmsi, width, length, draught, obstime,           
     │              lon, lat, sog, cog, typecargo, segment_id        
     ▼                                                               
depth.py   →  ais_with_depth.csv                                     
     │        + WaterDepth column (bathy + tide)                     
     │        reads bathy.dfsu + tide.dfs0 via mikeio (cached)      
     │        filters: WaterDepth >= draught + underkeel_margin      
     ▼                                                               
wave_params.py  →  wave_params_all.csv / .parquet  (all cols)        
                   wave_params_gis.csv             (15-col GIS)      
     │                                                               
     ▼                                                               
shore_impact.py  →  shore_impact_table.csv                           
                    output/WaveHeightMap.png                         
                    output/WavePeriodMap.png                         
                    output/vessels/[MMSI].png  (if enabled)          
```

---

## 9. Open Items / Future Scope

| Item | Priority | Notes |
|------|----------|-------|
| 2D spatial wave height grid | Medium | Bathymetric refraction and shoaling across the domain (currently inactive MATLAB code `func_ShipWakeImpactArea.m`) |
| Wave height validation | Medium | Compare model output against measured data at `NW_SSES_wave_measure_*.shp` stations |
| Multiple target polygons | Low | Support list of target zones (not just one coastline) in a single run |
| Parallel processing | Low | `stage: impact` is the bottleneck — parallelise per-vessel with `concurrent.futures` |
