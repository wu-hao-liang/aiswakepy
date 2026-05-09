# ShipwakeCalculation_WUHL — Project Review

**Reviewed**: 2026-04-08 (updated with `improved_version/` review)  
**Purpose**: Understand the MATLAB codebase in preparation for a Python rewrite.

---

## 1. Directory Structure

```
ShipwakeCalculation_WUHL/
├── calc_WaveHeight_from_AISdata_SSrc.m       ← Main entry point (Jan 27 2023)
├── X1AISfilterSpeed_v3.m                     ← AIS filtering, latest (Jan 26 2023)
├── X1AISfilterSpeed_v2.m                     ← Older filtering (Jan 10 2023) DEPRECATED
├── X2AISfilterSpeed_Fill_v2.m                ← AIS interpolation v2 (Jan 11 2023)
├── X2AISfilterSpeed_Fill.m                   ← Original interpolation (Feb 8 2022) DEPRECATED
├── plotmap.m                                 ← Stub, unused
├── functions/
│   ├── func_calcVesselWave.m                 ← Wave parameter calc, ACTIVE (Jan 27 2023)
│   ├── func_calcVesselWave_csv.m             ← CSV variant, ALTERNATIVE (Jan 13 2023)
│   ├── func_calcVesselWaveUpdate.m           ← Old, different angle calc DEPRECATED
│   ├── func_ShipWakeShoreImpact_61802600.m   ← Shore impact, ACTIVE (Jan 27 2023)
│   ├── func_ShipWakeShoreImpact_61803081.m   ← Project variant, ALTERNATIVE (Jan 15 2023)
│   ├── func_ShipWakeShoreImpact.m            ← Iterative version DEPRECATED (Aug 22 2021)
│   ├── func_ShipWakeShoreImpactOld.m         ← Very old DEPRECATED (Mar 26 2021)
│   ├── func_ShipWakeImpactArea.m             ← 2D bathymetry impact, INACTIVE
│   ├── func_cb.m                             ← Block coefficient lookup, ACTIVE
│   ├── func_concatAIS.m                      ← Data concatenation utility
│   ├── func_textread.m                       ← Text reading utility
│   ├── greatcirc_dist.m                      ← Point-to-N great-circle distance
│   ├── greatcirc_dist_NtoN.m                 ← N-to-N distance array
│   ├── greatcirc_dist_1toN.m                 ← 1-to-N distance
│   ├── greatcirc_dist_ncr02.m                ← Distance variant, ALTERNATIVE
│   ├── interpLine.m                          ← Line interpolation
│   ├── intersections.m                       ← Robust line-line intersection
│   ├── m_fdist.m                             ← Vincenty forward distance/azimuth
│   ├── m_shaperead.m                         ← Shapefile reader
│   ├── ShipDataED.csv / .xlsx                ← Block coefficient lookup table
│   └── ShipData.csv                          ← Older ship data (superseded)
├── improved_version/                          ← **LATEST VERSION (v20240304)**
│   ├── calc_WaveHeight_from_AISdata_SSrc.m   ← Updated main script
│   ├── X1AISfilterSpeed_v3.m                 ← Same as root (carried over)
│   ├── X1AISfilterSpeed_v2.m                 ← DEPRECATED
│   ├── X2AISfilterSpeed_Fill.m               ← DEPRECATED
│   ├── X2AISfilterSpeed_Fill_v2.m            ← DEPRECATED
│   ├── plotmap.m                             ← Stub
│   ├── Shipwake Matlab Script_v20240304.pptx ← Documentation slides
│   ├── add_tide/
│   │   ├── Add_Tide.py                       ← **NEW** Python script: add tidal water level
│   │   ├── interp_6min.mzt                   ← MIKE interpolation setup
│   │   ├── Predicted Water Level (CD)_2024_WestCoast.dfs0      ← Tidal prediction source
│   │   └── Predicted Water Level (CD)_2024_WestCoast_6min.dfs0 ← 6-min interpolated tide
│   ├── bathy/
│   │   ├── 61803960_WestCoast_HD_25m_mCD_Prod_v20260220.mesh   ← **NEW** mesh bathymetry
│   │   ├── depth.xyz                         ← Point cloud
│   │   └── read_bathy.m                      ← Bathymetry reader (uses DHI MATLAB + .mesh)
│   └── functions/
│       ├── func_calcVesselWave.m             ← **UPDATED**: uses type-based Cb & Le lookup
│       ├── func_cb_L_Le.m                    ← **NEW**: Cb + bow entry from L/Le ratio by ship type
│       ├── func_cb_B_Le.m                    ← **NEW**: Cb + bow entry from B/Le ratio by ship type
│       ├── func_cb_tablelooking.m            ← **NEW**: table lookup with type-filtered nearest-neighbour
│       ├── ShipDataEDnew.csv                 ← **NEW**: extended ship data with Le/B column
│       ├── (all other functions carried over from root)
│       └── ...
├── data/
│   ├── AIS_2563.csv                          ← Raw AIS input (~416 KB)
│   ├── AIS_2563_filter.csv                   ← Filtered AIS
│   ├── AIS_2563_interp.csv                   ← Interpolated AIS (~4.2 MB)
│   ├── AIS_2563_WaterDepth.csv               ← AIS + water depth (~4.9 MB)
│   ├── AIS_compare*.csv                      ← Comparison dataset variants
│   ├── depth.xyz                             ← Bathymetry point cloud (~383 KB)
│   ├── SW_mCD_01_20m.mesh                    ← Mesh file (~8.7 MB)
│   └── WaveCalc/
│       ├── *ALL-WaveParameters.csv           ← Full 33-column output (~18-19 MB)
│       ├── *GIS-WaveParameters.csv           ← 14-column GIS-ready output (~8 MB)
│       └── ShoreImpact/
│           ├── table_ShoreImpact_*.csv       ← Shore impact table (~91-98 KB)
│           └── *.png                         ← Visualization outputs
├── bathy/
│   ├── bathySVY21.dfs2 / bathy.dfs2 / bathy.dfsu  ← Bathymetry model files
│   └── old/
├── shpfile/
│   ├── Coast_P1.*                            ← Coastline polygon (WGS84)
│   ├── NW_SSES_wave_measure_1m.*             ← 1m measurement station points
│   ├── NW_SSES_wave_measure_10m.*            ← 10m measurement station points
│   └── NW_SSES_shipwake_measure_triangle.*   ← Triangle measurement zone
└── AIS/
    └── Sebarok/2019/                         ← Archive directory (mostly empty)
```

---

## 2. Active vs Deprecated Files

The `improved_version/` directory (dated v20240304) is the **latest codebase**. Files in the root directory are the older 2023 version.

| File | Location | Status | Notes |
|------|----------|--------|-------|
| `calc_WaveHeight_from_AISdata_SSrc.m` | `improved_version/` | **LATEST** | Updated main script — calls type-based Cb/Le |
| `X1AISfilterSpeed_v3.m` | both | **ACTIVE** | Same in both locations |
| `func_calcVesselWave.m` | `improved_version/functions/` | **LATEST** | Uses `func_cb_L_Le` or `func_cb_B_Le`; accepts `typecargo` column; g=9.78 |
| `func_cb_L_Le.m` | `improved_version/functions/` | **NEW** | Ship-type-based Cb + L/Le ratio lookup |
| `func_cb_B_Le.m` | `improved_version/functions/` | **NEW** | Ship-type-based Cb + B/Le ratio lookup |
| `func_cb_tablelooking.m` | `improved_version/functions/` | **NEW** | Type-filtered nearest-neighbour from `ShipDataEDnew.csv` |
| `Add_Tide.py` | `improved_version/add_tide/` | **NEW** | Python: adds tidal water level, filters draught |
| `read_bathy.m` | `improved_version/bathy/` | **LATEST** | Reads `.mesh` via DHI MATLAB toolbox |
| `func_ShipWakeShoreImpact_61802600.m` | both | **ACTIVE** | Shore impact (unchanged in improved_version) |
| `func_cb.m` | root `functions/` | SUPERSEDED | Replaced by type-based methods in improved_version |
| `func_calcVesselWave.m` | root `functions/` | SUPERSEDED | Old version without type-based Cb/Le |
| `X1AISfilterSpeed_v2.m` | both | DEPRECATED | Replaced by v3 |
| `X2AISfilterSpeed_Fill*.m` | both | DEPRECATED | Replaced by v3 integrated interpolation |
| `func_calcVesselWaveUpdate.m` | both | DEPRECATED | Old angle formulation |
| `func_ShipWakeShoreImpact.m` | both | DEPRECATED | Iterative version |
| `func_ShipWakeShoreImpactOld.m` | both | DEPRECATED | Very old |
| `func_ShipWakeImpactArea.m` | both | INACTIVE | Commented out |
| `plotmap.m` | both | UNUSED | Empty stub |

---

## 3. Overall Workflow (6 Stages — improved_version)

```
[Raw AIS CSV — with typecargo column]
      │
      ▼
 STAGE 1: AIS Filtering & Interpolation   (X1AISfilterSpeed_v3.m)
      │  - Speed validation
      │  - Trajectory interpolation to 20m intervals
      │  - Land-point removal
      ▼
[Cleaned AIS CSV — 10 columns + typecargo]
      │
      ▼
 STAGE 2: Bathymetry Depth Assignment     (read_bathy.m)
      │  - Read .mesh via DHI MATLAB toolbox
      │  - Nearest-element-centre depth lookup
      │  - Append WaterDepth column
      ▼
[AIS + WaterDepth CSV]
      │
      ▼
 STAGE 3: Tidal Water Level Adjustment    (Add_Tide.py — Python)
      │  - Read predicted tide from .dfs0 via mikeio
      │  - Round AIS timestamps to 30-min intervals
      │  - Add tidal level to static bathymetric depth
      │  - Filter out records where WaterDepth < draught + 1 m
      ▼
[AIS + WaterDepth (tide-adjusted) CSV]
      │
      ▼
 STAGE 4: Wave Parameter Calculation      (func_calcVesselWave.m — improved)
      │  - Ship-type-based Cb and bow entry lookup (func_cb_L_Le or func_cb_B_Le)
      │  - Froude number computation (g=9.78)
      │  - Kreibel wave height formula
      │  - Wake spreading angle, wave energy and period
      │  - Row filtering (invalid Froude, speed, beam/length)
      ▼
[Wave Parameters CSV — 33 cols (ALL) / 15 cols (GIS, now includes WaterDepth)]
      │
      ▼
 STAGE 5: Shore Impact Calculation        (func_ShipWakeShoreImpact_61802600.m)
      │  - Cast ray from vessel in port/starboard wake direction
      │  - Find intersection with coastline shapefile
      │  - Apply distance-decay formula to get shore wave height
      │  - Tabulate and visualize results
      ▼
[Shore Impact Table CSV + PNG visualizations]
```

---

## 4. Stage Details

### Stage 1 — AIS Filtering (`X1AISfilterSpeed_v3.m`)

**Input**: Raw AIS CSV (29 columns — many unused)  
**Output**: Cleaned AIS CSV (10 columns: `mmsi, width, length, draught, obstime, longitude, latitude, sog, cog, WaterDepth`)

**Steps**:
1. Load and sort by MMSI + observation time.
2. For each vessel, compute time gaps and great-circle distances between consecutive fixes.
3. Flag gaps > 600 s (10 min) as new trajectory segment.
4. Compute calculated speed `v_calc = distance / time`; take conservative `sog = min(AIS_sog, v_calc)`.
5. For gaps > 100 m, generate intermediate interpolated points at 20 m spacing (linear interpolation of all 9 data columns).
6. Remove all points inside the coastline polygon (`inpolygon()`).

**Key Constants**:
- Trajectory gap threshold: 600 s
- Interpolation spacing: 20 m
- Speed unit conversion: 1 knot = 0.5144444 m/s

---

### Stage 2 — Bathymetry Depth Assignment (`improved_version/bathy/read_bathy.m`)

**NEW in improved_version** — previously depth was assigned externally or inline.

**Input**: Cleaned AIS CSV + `.mesh` bathymetry file (`61803960_WestCoast_HD_25m_mCD_Prod_v20260220.mesh`)  
**Output**: AIS CSV with appended `WaterDepth` column

**Algorithm**:
1. Load mesh using `mzReadMesh()` (DHI MATLAB toolbox).
2. Compute element centre coordinates via `mzCalcElmtCenterCoords()`.
3. For each AIS point, find the nearest mesh element centre by Euclidean distance.
4. Assign the depth (negated, since mesh stores bed level as negative from CD) to `WaterDepth`.

**Note**: This is a brute-force nearest-element lookup (O(N×M)), slow for large datasets. The Python rewrite should use `mikeio` with spatial indexing.

---

### Stage 3 — Tidal Water Level Adjustment (`improved_version/add_tide/Add_Tide.py`)

**NEW in improved_version** — a Python script using `mikeio`.

**Input**: AIS+WaterDepth CSV + predicted tidal water level `.dfs0` file (6-min interval)  
**Output**: AIS CSV with tide-adjusted water depth, filtered for under-keel clearance

**Algorithm**:
1. Read predicted tide time series from `.dfs0` via `mikeio.read()` → pandas Series.
2. Round each AIS observation time to the nearest 30-minute interval (snap to tidal series index).
3. Look up the tidal level at each rounded timestamp.
4. Add tidal level to static bathymetric depth: `WaterDepth = WaterDepth_bathy + tide_level`.
5. **Filter**: remove records where `WaterDepth < draught + 1 m` (insufficient under-keel clearance — vessel would be aground/too shallow for wake formula validity).
6. Exclude AIS records beyond the available tidal prediction time range.

**Key detail**: Tide data is relative to Chart Datum (CD), same as the bathymetry, so the addition is valid without datum correction.

---

### Stage 4 — Wave Parameter Calculation (`improved_version/functions/func_calcVesselWave.m`)

**Input**: AIS CSV with tide-adjusted depth (11 columns: 10 standard + `typecargo`)  
**Output**: Wave parameters CSV (33 columns ALL / 15 columns GIS — now includes WaterDepth)

**Changes from root version**:
- **Gravity**: uses `g = 9.78` (Singapore local gravity) instead of `9.81`.
- **Block coefficient**: determined by **ship type** via `func_cb_L_Le()` or `func_cb_B_Le()` (see §4a below), not by nearest-neighbour table lookup.
- **Bow entry length**: also determined by ship type ratio, not the fixed `width / 1.1` formula.
- **Input column mapping**: skips column 10 (`typecargo`) — `WaterDepth` is now column 11 in the input, mapped to `Processed_Data(:,10)`.
- **GIS output**: now 15 columns (added `WaterDepth` as 15th column).
- **WavePort / WaveStarboard**: still uses `COG ± 90°` (see §8 issue #2 — to be corrected in Python rewrite to use `COG ± θ`).

**Computed columns** (key ones):

| Col | Name | Formula |
|-----|------|---------|
| 11 | SOGms | `SOG × 0.5144444` |
| 12 | BowEntry | `length / L_Le` (L_Le from type lookup) **or** `width / B_Le` (B_Le from type lookup) |
| 13 | LengthWL | `length × 0.8` |
| 17 | BlockCoef (Cb) | From `func_cb_L_Le(typecargo)` or `func_cb_B_Le(typecargo)` |
| 18 | Beta | `1 + 8 × tanh(0.45 × (L/BowEntry − 2))³` |
| 19 | Alpha | `2.35 × (1 − Cb)` |
| 20 | **FroudeM** | `(V/√(g×L)) × exp(α × draught / depth)` |
| 21 | FroudeD | `V / √(g × depth)` |
| 24 | **H_Kreibel** | `β × (FroudeM − 0.1)² × (B/2L)^(−1/3) / g × V²` |
| 25 | T | `0.27 × SOG` (empirical, seconds) |
| 27 | Etot | `10.8 × Emax^0.82` |
| 28 | **Theta** | `35.27 × (1 − exp(12 × (FroudeD − 1)))` degrees |
| 31 | WavePort | `COG − 90°` |
| 32 | WaveStarboard | `COG + 90°` |

**Row filter (unchanged)**:
- `FroudeM < 0.1` or `FroudeM > 0.5`
- `BF > 0.4`
- `SOG > 12 knots`
- `B/L > 0.3`

**GIS output** (15 cols): `mmsi, longitude, latitude, Etot, WavePort, WaveStarboard, H_Kreibel, obstime, Beta, FroudeM, SOGms, width, length, Tc, WaterDepth`

---

### Stage 3 — Shore Impact (`func_ShipWakeShoreImpact_61802600.m`)

**Input**: GIS wave parameters CSV (14 cols) + coastline shapefile (`Coast_P1`)  
**Output**: Shore impact table CSV (11 cols) + PNG plots

**Algorithm** (for each wake event, both port and starboard):
1. Cast a ray from the vessel position in the wake propagation direction (`COG ± 90°`) out to `max_distance = 2000 m`, using Vincenty's formula (`m_fdist`) to get the endpoint.
2. Find all intersections of the ray with the coastline polygon using `intersections()`.
3. Select the closest intersection point (Manhattan distance proxy).
4. If intersection found and distance is within threshold, apply decay formula:

```
H_shore = β × (FroudeM − 0.1)² × (distance_m / L)^(−1/3) / g × V²
```

5. Filter: discard if `H_shore < 0.01 m` (1 cm cutoff).

**Output CSV columns**: `MMSI, Longitude, Latitude, WaveHeight, WavePeriod, DistLoc, WaveTime, FroudeM, VesselWidth, VesselLength, SOG`

**Visualization**: Wave height and wave period 2D maps (color-coded scatter over coastline), plus per-vessel wake diagrams.

---

### 4a. Block Coefficient & Bow Entry Methods (improved_version)

Three methods now exist for determining Cb and bow entry length. The improved_version uses the type-based methods:

#### Method 1: `func_cb_L_Le(typecargo)` — L/Le ratio (currently active in improved_version)

Determines Cb and the ratio `L/Le` (ship length to bow entry length) based on AIS ship type code:

| Ship Type (AIS code) | Cb | L/Le | Le = L / (L/Le) |
|----------------------|-----|------|------------------|
| Tankers (80–89) | 0.86 | 7 | L/7 (blunt bow) |
| Cargo / Dredger (33, 70–79) | 0.80 | 5 | L/5 |
| All others (ferries, tugs, fishing, navy, etc.) | 0.67 | 3 | L/3 (sharp bow) |

#### Method 2: `func_cb_B_Le(typecargo)` — B/Le ratio (alternative, commented out)

Determines Cb and the ratio `B/Le` (beam to bow entry length):

| Ship Type (AIS code) | Cb | B/Le | Le = B / (B/Le) |
|----------------------|-----|------|------------------|
| Tankers (80–89) | 0.80 | 1.0 | B/1.0 |
| Cargo / Dredger (33, 70–79) | 0.70 | 0.7 | B/0.7 |
| All others | 0.60 | 0.4 | B/0.4 |

#### Method 3: `func_cb_tablelooking(LOA, Beam, shipdatafile, typecargo)` — Type-filtered table lookup

More refined version of the old `func_cb.m`: filters `ShipDataEDnew.csv` by ship type category first, then does nearest-neighbour lookup in (LOA, Beam) space within that category. Returns Cb and B/Le ratio.

`ShipDataEDnew.csv` columns: `DWT, Displacement, LOA, LPP, Beam, Depth, Draught, CB, LeB`  
(89 rows, categorised by vessel type: rows 1–22 tankers, 23–59 cargo/carriers, 60–72 ferries, 73–75 fast ferries, 76–84 fishing, 85 tug, 86 heavy lifter, 87 navy frigate, 88 sailing, 89 dredger)

**Key difference from old `func_cb.m`**: the old method did a global nearest-neighbour across all ship types. The new methods either use fixed type-based values (Methods 1&2) or restrict the lookup to the correct vessel category (Method 3).

---

## 5. Key Formulas Summary

| Formula | Expression | Notes |
|---------|-----------|-------|
| Modified Froude (depth-adjusted) | `F_m = (V/√(gL)) × exp(α×d/h)` | α = 2.35(1−Cb) |
| Depth Froude | `F_d = V / √(gh)` | For spreading angle |
| Shape factor | `β = 1 + 8 × tanh(0.45(L/Be − 2))³` | Be = L/(L/Le) or B/(B/Le) by ship type (old: B/1.1) |
| Wave height at origin | `H = β(F_m−0.1)²(B/2L)^(−1/3) V²/g` | Kreibel formula |
| Wake spreading angle | `θ = 35.27(1−exp(12(F_d−1)))` | Degrees |
| Wave period | `T = 0.27 × SOG` | SOG in knots, T in seconds |
| Max wave energy | `E_max = ρg²H²T² / (16π)` | ρ = 1026 kg/m³ |
| Total wave energy | `E_tot = 10.8 × E_max^0.82` | Empirical scaling |
| Wave decay at shore | `H_s = H_0 × (x/L)^(−1/3)` | x = distance to shore |

---

## 6. Coordinate System & Time

- **Spatial**: WGS84 decimal degrees (longitude/latitude); geodetic distances via Vincenty's formula
- **Projection note**: Shapefiles reference SVY21 (Singapore local grid) but coordinates appear stored as WGS84 — verify when porting
- **Time**: MATLAB serial datenum; raw AIS uses ISO 8601 strings; intermediate CSVs use Excel serial date adjusted by `−693960` offset

---

## 7. Input / Output File Formats

| File | Format | Description |
|------|--------|-------------|
| Raw AIS | CSV (29 cols) | MMSI, lat/lon, SOG, COG, obstime, etc. |
| Filtered AIS | CSV (10 cols) | mmsi, width, length, draught, obstime, lon, lat, sog, cog, WaterDepth |
| Wave params ALL | CSV (33 cols) | All intermediate computed values |
| Wave params GIS | CSV (14 cols) | GIS-ready subset |
| Shore impact table | CSV (11 cols) | One row per shoreline intersection |
| Coastline | ESRI Shapefile | Polygon; used for ray-intersection test |
| Bathymetry | `.xyz`, `.dfs2`, `.dfsu` | Point cloud and DHI mesh formats |
| Ship data | CSV/XLSX | Block coefficient lookup (L × B → Cb) |
| Outputs | PNG | Wave height / period maps |

---

## 8. Notable Issues / Quirks

1. **Hardcoded absolute paths** in both root and improved_version (`C:\Projects\...`, `\\sg-nc14\Projects\...`) — must be parameterized.
2. **Wake propagation direction**: both versions still use `COG ± 90°` for WavePort/WaveStarboard. Physically, the correct propagation angle of the wave crest at the cusp is `COG ± θ`, where θ is the computed wake spreading angle (≈ 35.26° in deep water, = `arcsin(1/√3)`). The Python rewrite must correct this — see PRD.md §2.2.
3. **Excel serial date offset**: MATLAB datenum uses a different epoch than Excel. Code applies `−693960` correction. The improved_version passes a `pivot` year of 2000 to `datenum()`. Not an issue in Python (use `pd.to_datetime`).
4. **Mixed distance approaches**: some functions use spherical approximation, others use Vincenty ellipsoidal — standardize with `pyproj` in Python rewrite.
5. **Block coefficient**: improved_version offers three methods (type-based L/Le, type-based B/Le, type-filtered table lookup). The active code uses `func_cb_L_Le`; `func_cb_B_Le` is present but commented out. The old global nearest-neighbour (`func_cb.m`) is superseded.
6. **Gravity constant**: improved_version uses `g = 9.78` (Singapore local gravity). Root version uses `g = 9.81`. The Python rewrite should use a configurable value, defaulting to 9.78 for Singapore.
7. **Tidal adjustment**: the new `Add_Tide.py` rounds timestamps to 30-min intervals for tide lookup, which introduces up to ±15 min of tidal phase error. Acceptable for operational purposes but worth documenting.
8. **Under-keel filter**: `Add_Tide.py` removes records where `WaterDepth < draught + 1 m`. This is a physical validity check — the wake formula is not meaningful when the vessel nearly touches the seabed.
9. **2D bathymetry impact** (`func_ShipWakeImpactArea.m`) is developed but currently commented out in both versions.
10. **No timezone management**: UTC assumed throughout.
11. **AIS interpolation artifacts**: zero speeds and very short segments appear after interpolation — the Froude filters in Stage 4 catch most of these.

---

## 9. Python Rewrite — Key Dependencies

| Capability | Python Library |
|-----------|---------------|
| Geodetic distance / bearing | `pyproj` (via `pyproj.Geod`) |
| Shapefile I/O | `geopandas` + `fiona` |
| Point-in-polygon | `shapely.geometry.Polygon.contains()` |
| Line-line intersection | `shapely.geometry.LineString.intersection()` |
| Data processing | `pandas`, `numpy` |
| Nearest-neighbour lookup | `scipy.spatial.KDTree` |
| Plotting / mapping | `matplotlib` + `contextily` |
| DHI bathymetry (.mesh, .dfsu) | `mikeio` |
| DHI tidal data (.dfs0) | `mikeio` |
| Config validation | `pydantic` v2 |

---

## 10. Recommended File Map for Python Rewrite

| MATLAB (improved_version) | Python module |
|--------------------------|--------------|
| `X1AISfilterSpeed_v3.m` | `stages/filter.py` |
| `read_bathy.m` | `geo/bathymetry.py` (use `mikeio`) |
| `Add_Tide.py` | `stages/depth.py` (merge bathy + tide into one stage) |
| `func_calcVesselWave.m` | `stages/wave_params.py` |
| `func_cb_L_Le.m`, `func_cb_B_Le.m`, `func_cb_tablelooking.m` | `vessel/block_coeff.py` |
| `func_ShipWakeShoreImpact_61802600.m` | `stages/shore_impact.py` |
| `calc_WaveHeight_from_AISdata_SSrc.m` | `pipeline.py` + `run_shipwake.ipynb` |
| `greatcirc_dist*.m`, `m_fdist.m` | `geo/geodesy.py` (replace with `pyproj`) |
| `intersections.m`, `interpLine.m` | (replace with `shapely`) |
| `m_shaperead.m` | (replace with `geopandas`) |
