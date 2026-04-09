# aiswakepy — Claude Code Context

## Project
Python pipeline for AIS-based ship-wake wave height calculation (Kriebel & Seelig 2005 empirical model).

## Setup
- Package manager: `uv` — always use `uv add` or `uv add --dev`, **NEVER pip**
- Run tests: `uv run pytest tests/ -q`  (102 tests, all pass)
- Run pipeline: `uv run python main.py --config config.json`
- Validate against MATLAB: `uv run python validate_pipeline.py`

## Before starting work
1. Read `PRD.md` — product requirements and physics specification
2. Read `PLAN.md` — 12-step incremental build plan (Steps 0–11 all complete)
3. Read `PERFORMANCE_PLAN.md` — approved performance optimization plan (pending implementation)

## Package structure
```
aiswakepy/
├── __init__.py
├── config.py              Pydantic config schema
├── pipeline.py            run_pipeline() orchestrator
├── models/
│   └── kriebel.py         Kriebel & Seelig (2005) empirical model
├── stages/
│   ├── filter.py          AIS load → segment → validate_speed → interpolate → mask_land
│   ├── depth.py           assign bathymetry depth + tidal adjustment
│   ├── wave_params.py     Kriebel wave params + propagation (Theta, T, WakeDir)
│   └── shore_impact.py    Ray-coastline intersection, wave decay, shore output
├── geo/
│   ├── bathymetry.py      KDTree mesh lookup
│   └── coastline.py       Shapefile load, STRtree (planned)
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

## Immediate next task
Implement the performance optimizations described in `PERFORMANCE_PLAN.md`.
All 102 existing tests must still pass after each fix.
Run `uv run pytest tests/ -q` after each fix before moving on.
