# aiswakepy — Claude Code Context

## Environment
Python virtual environment is built by uv. Always use `uv run python` for any Python execution.

## Project
Python pipeline for AIS-based ship-wake wave height calculation (Kriebel & Seelig 2005 empirical model).

## Setup
- Package manager: `uv` — always use `uv add` or `uv add --dev`, **NEVER pip**
- Run tests: `uv run pytest tests/ -q`  (145 tests, all pass)
- Run pipeline: `uv run python main.py --config config.json`
- Validate against MATLAB: `uv run python validate_pipeline.py`

## Before starting work
1. Read `PRD.md` — product requirements and physics specification (v1 complete; §6 lists open items)
2. Read `spec/SPEC.md` — 12-step build spec (all steps complete; Step 12 fix #6 partial)
3. Read `docs/PERFORMANCE.md` — performance optimisation plan (5/6 fixes done)

## Package structure
```
aiswakepy/
├── __init__.py
├── config.py              Pydantic config schema
├── pipeline.py            run_pipeline() orchestrator
├── models/
│   ├── kriebel.py         Kriebel & Seelig (2005) empirical model
│   └── formula.py         PIANC Modified empirical model (feature branch)
├── stages/
│   ├── filter.py          AIS load → 12-step cleaning pipeline → mask_land
│   ├── depth.py           assign bathymetry depth + tidal adjustment
│   ├── vessel.py          Kriebel wave params + propagation (Theta, T, WakeDir)
│   └── wave_impact.py     Ray-coastline intersection, wave decay, shore output
├── comparison/
│   └── ossi.py            Load OSSI gauge data; match AIS events to measurements
├── geo/
│   ├── bathymetry.py      KDTree mesh lookup
│   └── coastline.py       Shapefile load, STRtree
└── viz/
    └── wave_map.py        Wave height / period map plots
```

## Key conventions
- **Progress bars**: use `rich.progress`, NOT tqdm
- **Console logging**: use `rich.console.Console`, NOT plain print
- **Distances**: geodetic WGS84 via `pyproj.Geod(ellps="WGS84")`
- **Gravity**: g = 9.78 m/s² (Singapore local)
- **Wake directions**: COG ± Theta (NOT COG ± 90°)
- **Shore distance**: perpendicular = dist_ray * sin(Theta) — this is the Kriebel lateral distance `y`
- **Config**: JSON file or inline dict (not YAML)
- **Example data**: `examples/` directory (gitignored — large binary files)

## Docs & specs
- `spec/SPEC.md` — 12-step build spec (all steps complete; Step 12 fix #6 partial)
- `docs/PERFORMANCE.md` — performance optimisation plan (5/6 done; Rich console migration pending)
- `docs/FROUDE_NUMBERS.md` — Froude number reference table for all empirical models
- `docs/MATLAB_REVIEW.md` — original MATLAB codebase review
