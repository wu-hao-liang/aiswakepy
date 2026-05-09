# Product Requirements Document
# aiswakepy — Python Rewrite (v1 Baseline)

**Date**: 2026-04-09
**Status**: ✅ Completed — v1 pipeline fully implemented and tested (145 tests pass)
**Implemented by**: Steps 0–11 of `spec/SPEC.md` (Step 12 perf optimisation 5/6 done)
**New requirements**: to be added in a separate PRD v2 document

---

## 1. Purpose & Scope

Rewrite the existing MATLAB shipwake calculation pipeline (`ShipwakeCalculation_WUHL`) in Python. The system:

1. Ingests AIS vessel tracking data.
2. Filters, cleans, and interpolates vessel trajectories.
3. Computes ship-wake wave parameters at each vessel position using Kriebel & Seelig (2005).
4. Determines whether each wake event will intersect a target shoreline, accounting for wake propagation physics.
5. Applies a distance-decay formula to estimate wave height at the shoreline.
6. Outputs tabular results and geographical visualisations.

---

## 2. Key Design Decisions

### 2.1 Configuration & Interface
- All file paths and parameters in a JSON config file (`config.json`), loadable as file path, JSON string, or Python dict.
- Primary interface: **Jupyter Notebook** (`run_shipwake.ipynb`) with path variables in a top cell.
- Programmatic interface: each stage is a pure function.
- CLI for batch runs: `uv run python main.py --config config.json`.

### 2.2 Wake Propagation Angle — Physics

The perpendicular beam direction (COG ± 90°) does **not** represent the physical propagation direction of ship wake waves.

In a Kelvin wake pattern, wave crests at the cusp propagate outward at **arcsin(1/√3) ≈ 35.26°** relative to the vessel's heading. In shallow water this angle depends on depth Froude number:

```
θ = 35.27 × (1 − exp(12 × (F_d − 1)))   [degrees]
```

**Wake ray directions**: `COG − θ` (port) and `COG + θ` (starboard).
**Lateral distance input to decay formula**: `dist_perp = dist_ray × sin(θ)` (not raw ray length).

### 2.3 Distances
Geodetic (ellipsoidal) WGS84 via `pyproj.Geod` for all distances and bearings. Maximum meaningful propagation distance: 2000 m (configurable).

### 2.4 Block Coefficient & Bow Entry Length
Three selectable backends (`cb_method`):
- `L_Le` (default): type-based ratios (tanker Cb=0.86 Le=L/7; cargo Cb=0.80 Le=L/5; other Cb=0.67 Le=L/3)
- `B_Le`: beam-based ratios
- `table`: nearest-neighbour lookup in `ShipDataEDnew.csv` by (LOA, Beam)

### 2.5 Bathymetry & Tidal Water Level
Static depth from `.mesh`/`.dfsu` (mikeio + KDTree); tidal level from `.dfs0` (nearest interval snap). Under-keel clearance filter: drop `WaterDepth < draught + 1 m`.

### 2.6 Datetime
All timestamps UTC, `pandas.Timestamp` timezone-naive internally.

### 2.7 Gravity
g = 9.78 m/s² (Singapore local value).

---

## 3. Functional Requirements (summary)

| ID | Stage | Description |
|----|-------|-------------|
| FR-1 | config | JSON config; notebook-first; each stage a pure function |
| FR-2 | filter | 12-step AIS pipeline: dedup → uniformize → remove-zero-dims → remove-invalid-draught → segment → clean-coords → clean-speed → validate-speed → interpolate (Cubic Hermite Spline) → study-area → mask-land |
| FR-3 | depth | Bathy depth + tide → WaterDepth; under-keel filter |
| FR-4 | vessel | Kriebel & Seelig (2005): F_m, F_d, θ, H, T, wake directions; `SOGms, bow_entry_m, WaterDepth, draught, length, width` required |
| FR-5 | wave_impact | Ray-cast to coastline (STRtree); `dist_perp = dist_ray × sin(θ)`; H_shore decay; arrival time via deep-water Cg = gT/4π |
| FR-6 | viz/output | Shore-impact CSV; wave height + period maps (coastline-binned); optional vessel diagrams |

---

## 4. Empirical Model — Kriebel & Seelig (2005)

Key parameters:
- α = 2.35 × (1 − Cb)
- β = 1 + 8 × tanh(0.45 × (L_WL/Le − 2))³
- F_m = (V/√(g·L_WL)) × exp(α·d/h)
- **H = β × (F_m − 0.1)² × (y/L_WL)^(−1/3) × V²/g**  where y = lateral distance
- T = 0.27 × SOG_knots

Validity limits: 0.1 ≤ F_m ≤ 0.5; β·(F_m−0.1)² ≤ 0.4.

---

## 5. Key Package Choices

| Purpose | Package |
|---------|---------|
| Geodetic distance/bearing | `pyproj.Geod` (WGS84) |
| Geometry / spatial index | `shapely` + STRtree |
| Shapefile I/O | `geopandas` + `fiona` |
| Bathymetry mesh | `mikeio` |
| Spline interpolation | `scipy.interpolate.CubicHermiteSpline` |
| Spatial lookup | `scipy.spatial.KDTree` |
| Config validation | `pydantic` v2 |
| Progress / logging | `rich` |
| Package manager | `uv` (never pip) |

---

## 6. Open Items / Future Scope (v2)

| Item | Priority |
|------|----------|
| AIS-constrained wake detection from gauge time series | High |
| Performance optimisation for 2M AIS records | High — see `docs/PERFORMANCE.md` (5/6 done) |
| Multiple empirical formula comparison & calibration | Medium |
| 2D spatial wave height grid with refraction/shoaling | Medium |
| Multiple target polygons | Low |
