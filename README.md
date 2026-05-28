# aiswakepy

Python pipeline for estimating ship-wake wave heights at the shoreline from AIS vessel tracking data, based on the Kriebel & Seelig (2005) empirical model.

---

## Overview

Ships generate diverging wake waves that can erode shorelines, disturb moored vessels, and affect recreational waterfronts. **aiswakepy** processes raw AIS records to answer: *which vessels produce the largest waves at a given stretch of coast, and how large are those waves?*

The pipeline covers the full chain from raw AIS CSV to georeferenced shore-impact results:

1. **Filter** — 12-step AIS cleaning: deduplication, segmentation, speed/coordinate validation, cubic Hermite spline interpolation, land-masking, and bathymetric under-keel clearance check.
2. **Vessel** — compute Kriebel & Seelig wave parameters (amplitude, period, wake half-angle θ) at every interpolated vessel position.
3. **Wave impact** — ray-cast each wake to the coastline (Shapely STRtree), apply lateral-distance decay, and record wave height and arrival time at each shore intersection.
4. **Visualisation** — scatter maps (wave height, wave period) and vessel-statistics plots exported as publication-quality PNGs.

An interactive **Dash + deck.gl** web app lets you run the pipeline, explore results on a satellite basemap, apply spatial and attribute filters, and export filtered subsets as self-contained rerun-ready datasets.

---

## Features

- Kriebel & Seelig (2005) primary model plus six additional empirical formulae (PIANC, Sørensen, Maynord, Bhowmik, Blaauw, Gates) for comparison
- Physics-correct wake propagation: rays at `COG ± θ` (not ± 90°), with depth-Froude-dependent angle in shallow water
- Bathymetry from MIKE mesh/dfsu files with tidal water-level correction (dfs0)
- Geodetic distances and bearings via `pyproj.Geod` (WGS84)
- Satellite basemap overlay (Esri WorldImagery via contextily)
- Interactive Dash app with freehand / box / polygon / track filters, vessel-type breakdown, and one-click filtered export
- Validated against MATLAB reference implementation

---

## Requirements

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) package manager

---

## Installation

```bash
git clone https://github.com/wu-hao-liang/aiswakepy.git
cd aiswakepy
uv sync
```

> **Never use `pip install`** — the project is managed exclusively with `uv`.

---

## Quick Start

### Run the pipeline

```bash
uv run python main.py --config config.json
```

### Launch the interactive Dash app

```bash
uv run python dash_app.py
```

Then open `http://localhost:8050` in your browser.

### Run in a Jupyter notebook

Open `run_shipwake.ipynb` and set the path variables in the top cell.

### Run tests

```bash
uv run pytest tests/ -q
```

---

## Configuration

All parameters are specified in a JSON config file. Key sections:

| Section | Purpose |
|---------|---------|
| `ais` | Raw CSV path, interpolation interval, speed/acceleration limits, study-area shapefile |
| `bathymetry` | Mesh/dfsu source, tidal dfs0, under-keel margin |
| `coastline` | Target shoreline shapefile |
| `wave` | Empirical formula selection, Froude limits, gravity (9.78 m/s² for Singapore) |
| `impact` | Maximum propagation distance, minimum wave height cutoff |
| `output` | Output directory, plot flags, stage CSV export |

See `config.json` for a fully annotated example.

---

## Pipeline Stages

```
AIS CSV
  └─▶ filter_ais()         → df_filtered   (01_filtered.csv)
        └─▶ compute_vessel_params() → df_vessel    (02_vessel.csv)
              └─▶ compute_wave_impact()  → df_wave_impact (03_wave_impact.csv)
                    └─▶ plot_wave_height_map()
                         plot_wave_period_map()
                         shore_impact.csv
```

Each stage is a pure function — import and call individually for notebook workflows or chained via `run_pipeline()`.

---

## Package Structure

```
aiswakepy/
├── pipeline.py          run_pipeline() orchestrator
├── config.py            Pydantic v2 config schema
├── stages/
│   ├── filter.py        AIS cleaning + depth clearance check
│   ├── vessel.py        Wake parameters + propagation geometry
│   └── wave_impact.py   Ray–coastline intersection + decay
├── models/              Empirical formula implementations
│   ├── kriebel.py       Kriebel & Seelig (2005)  ← primary
│   ├── pianc.py         PIANC Modified
│   ├── sorensen.py      Sørensen
│   ├── bhowmik.py       Bhowmik
│   ├── blaauw.py        Blaauw
│   ├── gates.py         Gates
│   └── maynord.py       Maynord
├── geo/                 Bathymetry, coastline, geodesy utilities
├── vessel/              Block coefficient lookup + ship type data
├── viz/                 Wave maps, vessel diagrams, report plots
└── comparison/          OSSI gauge matching + formula comparison plots
dash_app.py              Interactive web application
```

---

## Empirical Model

The primary model (Kriebel & Seelig 2005) estimates wave height at lateral distance *y* from the vessel track:

```
H = β × (F_m − 0.1)² × (y / L_WL)^(−1/3) × V² / g
T = 0.27 × SOG_knots
```

where F_m is the depth-modified midship Froude number and β captures hull form via the block coefficient and bow entry length. Validity range: 0.1 ≤ F_m ≤ 0.5.

---

## License

This project is developed as part of DHI Singapore research. Contact the repository owner for licensing terms.
