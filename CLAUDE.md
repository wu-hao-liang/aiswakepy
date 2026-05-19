# aiswakepy — Claude Code Context

## Environment
Python virtual environment is built by uv. Always use `uv run python` for any Python execution.

## Project
Python pipeline for AIS-based ship-wake wave height calculation (Kriebel & Seelig 2005 empirical model).

## Setup
- Package manager: `uv` — always use `uv add` or `uv add --dev`, **NEVER pip**
- Run tests: `uv run pytest tests/ -q`  (~141 passed, 2 skipped)
- Run pipeline: `uv run python main.py --config config.json`
- Validate against MATLAB: `uv run python validate_pipeline.py`
- Run Dash app: `uv run python dash_app.py`

## Before starting work
1. Read `PRD.md` — product requirements and physics specification (v1 complete; §6 lists open items)
2. Read `spec/SPEC.md` — 12-step build spec (all steps complete)
3. Read `docs/PERFORMANCE.md` — performance optimisation plan (all 6 fixes done)

## Package structure
```
dash_app.py                Dash + deck.gl interactive app (pipeline runner + map)
scripts/
└── capture_map.py         Headless map screenshot utility
aiswakepy/
├── __init__.py
├── config.py              Pydantic config schema
├── pipeline.py            run_pipeline() orchestrator
├── _progress.py           Spinner helper for per-item progress
├── models/
│   ├── kriebel.py         Kriebel & Seelig (2005) empirical model
│   ├── pianc.py           PIANC Modified empirical model
│   ├── sorensen.py        Sørensen empirical model
│   ├── bhowmik.py         Bhowmik empirical model
│   ├── blaauw.py          Blaauw empirical model
│   ├── gates.py           Gates empirical model
│   └── maynord.py         Maynord empirical model
├── stages/
│   ├── filter.py          AIS load → 12-step cleaning pipeline → mask_land
│   ├── depth.py           assign bathymetry depth + tidal adjustment
│   ├── vessel.py          Kriebel wave params + propagation (Theta, T, WakeDir)
│   └── wave_impact.py     Ray-coastline intersection, wave decay, shore output
├── comparison/
│   ├── ossi.py            Load OSSI gauge data; match AIS events to measurements
│   └── plots.py           Comparison plots
├── geo/
│   ├── bathymetry.py      KDTree mesh lookup
│   ├── coastline.py       Shapefile load, STRtree
│   └── geodesy.py         Geodetic utilities
├── vessel/
│   ├── block_coeff.py     Block coefficient lookup
│   └── ShipDataEDnew.csv  Ship type reference data
└── viz/
    ├── wave_map.py        Wave height / period map plots
    └── vessel_diagram.py  Vessel schematic diagram
```

## Dash server
The Dash app runs in a `screen` session managed by the developer. **Never kill or restart the server** — always leave server lifecycle (start/stop/restart) to the developer.

## Key conventions
- **Progress bars**: use `rich.progress`, NOT tqdm
- **Console logging**: use plain `print()` for stage status (vectorised stages have nothing for Rich to animate). The custom `aiswakepy/_progress.Spinner` covers the few stages with per-item progress.
- **Distances**: geodetic WGS84 via `pyproj.Geod(ellps="WGS84")`
- **Gravity**: g = 9.78 m/s² (Singapore local)
- **Wake directions**: COG ± Theta (NOT COG ± 90°)
- **Shore distance**: perpendicular = dist_ray * sin(Theta) — this is the Kriebel lateral distance `y`
- **Config**: JSON file or inline dict (not YAML)
- **Example data**: `examples/` directory (gitignored — large binary files)

## Docs & specs
- `spec/SPEC.md` — 12-step build spec (all steps complete; Step 12 fix #6 partial)
- `docs/PERFORMANCE.md` — performance optimisation plan (all 6 fixes done)
- `docs/FROUDE_NUMBERS.md` — Froude number reference table for all empirical models
- `docs/MATLAB_REVIEW.md` — original MATLAB codebase review
- `docs/VIZ_DOWNSAMPLING.md` — coastline-binned top-N downsampling algorithm (used in shore-impact maps; reference for upcoming DASH app)
