# Dash Rendering Spike — Study Plan

## Context

Before writing the spec for the upcoming interactive ship-track Dash app, we need empirical evidence for which Plotly/Dash rendering technique handles this project's data scale (~0.5–2 M wave-impact points per AIS day, plus vessel positions and segmented tracks). Today's viz is matplotlib-only static PNG; Dash 4.1 is in `pyproject.toml` but unused. The existing 1‑m coastline-binned top‑N downsampler (`aiswakepy/viz/wave_map.py::_downsample`, doc in `docs/VIZ_DOWNSAMPLING.md`) is the obvious server-side primitive to reuse, but we need to know which client renderer it should feed and where the breaking points are.

The user has fixed the interaction model:

1. **Wave-impact points on coastline** — must support hover at all zoom levels (full ~0.5–2 M scale).
2. **Vessel positions + tracks (densest, unfiltered overview)** — no hover required; density display is sufficient.
3. **Tracks are only drawn after the user filters wave points** — i.e. selecting a subset of wave events first, then plotting the responsible vessel tracks.
4. **After filtering, hover is required on vessel points and tracks** (point count is now small).
5. **Hover is gated by zoom level** — if a technique cannot keep up with hover at a given density, the spike must specify the visible-points threshold below which hover is enabled (clientside-toggled), and disable it above that threshold instead of letting the browser stall.

This staging means we don't need a single technique to do everything — we need a *layered* design where the densest case uses one approach and the filtered case switches to another.

## Primary success criterion: smoothness during pan/zoom

The user's central concern is that scrolling/zooming must feel *smooth* even while the underlying data is being re-aggregated. Server-side bbox-downsample on every `relayoutData` event will blow this out — every wheel tick triggers a Python round-trip, JSON serialisation, and full figure re-render. The spike must therefore not only measure raw render cost but also evaluate **perceived smoothness** under several mitigation strategies (Scenario E below). The architectural insight to test: Mapbox/Leaflet-based traces (`go.Scattermap`, datashader as a Mapbox image layer) get *native* browser zoom animation — the map continues moving immediately, the data layer fills in afterwards. Cartesian Scattergl has no such native animation, so every pan is a hard callback. This shapes the recommendation.

## Deliverable

A single Jupyter notebook: `dash_rendering_spike.ipynb` (at repo root). Run inline, measure cost on real or synthetic 2 M-point data, print numbers and screenshots into the notebook. No new package, no Dash app skeleton yet — that comes after the study informs the spec.

## Scope of the spike

Five scenarios, mapped to the user's interaction model. Each scenario tests 1–3 candidate techniques and records concrete metrics.

### Scenario A — Wave-impact points on map (hover required, up to 2 M)
- **A.1** `go.Scattermap` raw, no downsampling — establish failure baseline.
- **A.2** `go.Scattermap` fed by bbox-aware top-N downsampler, re-aggregated on `relayoutData` callback. Reuses `_downsample()` from `aiswakepy/viz/wave_map.py:22–76` adapted to take a viewport bbox. Includes a **clientside callback** that reads current trace length and toggles `hovermode` off above a threshold (e.g. > 50 k visible points), re-enabling hover only once the user zooms in enough to drop below it. The exact threshold is one of the spike's outputs.
- **A.3** `go.Scattergl` on Cartesian (lon, lat) — WebGL-accelerated; loses basemap but useful as upper-bound on raw browser capacity.

### Scenario B — Wave-impact points along the 1-D coastline distance axis (the "points aligned on a line" view)
- **B.1** `go.Scattergl` with all points (distance-along-coastline x, wave-height y).
- **B.2** `go.Scattergl` wrapped in `plotly-resampler` `FigureResampler` (MinMaxLTTB aggregator). This is the one case where plotly-resampler *applies* — it doesn't support `scatter_mapbox`, but a 1-D Cartesian coastline strip is its native use case.

### Scenario C — Vessel positions + tracks, dense unfiltered overview (no hover)
- **C.1** Datashader rasterise → Mapbox image-layer overlay, re-rasterised on `relayoutData`. Pattern from NVIDIA RAPIDS census demo and the "Visualizing a Billion Points" Databricks walkthrough.
- **C.2** `dash-deck` `ScatterplotLayer` + `PathLayer`. Note dash-deck is officially "proof-of-concept" — stability is part of what we're testing.

### Scenario D — Filtered subset (post-selection, ~10–50 k points, hover required)
- **D.1** `go.Scattermap` plain — confirm this works comfortably at the post-filter scale.
- **D.2** Same data via `dash-deck` with picking enabled — comparison for visual quality and click latency.

### Scenario E — Smoothness mitigations (the user's central concern)

This scenario directly targets perceived smoothness during continuous wheel-zoom and drag-pan, layered on top of Scenario A.2 (the realistic production candidate).

- **E.1 — Naïve baseline:** server callback fires on every `relayoutData` event. Expected outcome: callback storm, visible chunkiness.
- **E.2 — Debounced callback:** clientside debouncer (~250 ms idle window) before the Python callback fires. Map keeps panning natively; data updates after the user stops moving.
- **E.3 — Two-resolution layering:** an always-on coarse base layer (e.g. 30 k pre-aggregated points covering the whole AOI) is rendered once and never updated. A second high-res trace is filled by the debounced bbox callback. The base layer guarantees the viewport is never empty during pan/zoom.
- **E.4 — Pre-tiled aggregations:** precompute top-N aggregations at 4–5 zoom levels (quadtree-like). On `relayoutData`, callback does an O(1) lookup of the closest precomputed level rather than re-running the downsample. Heavier server prep, near-zero callback latency.
- **E.5 — Partial property updates** (Dash 2.9+): use `Patch()` to update only the trace `lat`/`lon`/`marker.color` arrays, not the entire figure. Reduces JSON size and re-render cost.

Expected synthesis: E.2 + E.3 (debounce + always-on coarse layer) is the minimum viable combo for smooth feel; E.4 is the upgrade if E.2 + E.3 is still not enough.

## Metrics to record per cell

| Metric | How to measure |
|---|---|
| Data prep time (server) | `time.perf_counter()` around the downsample/aggregate call |
| Initial figure JSON size | `len(fig.to_json())` |
| Initial render time | Manual: stopwatch from "Run cell" to first paint, plus browser DevTools Performance tab for the Dash variants |
| Pan/zoom callback round-trip | Dash callback `n_clicks` + `dcc.Store` of `time.time()` at start/end |
| Perceived smoothness during continuous wheel-zoom | DevTools Performance tab — count dropped frames over a 5 s zoom; record subjective rating (smooth / chunky / unusable) |
| Callback storm | Number of `relayoutData` events fired per 5 s pan; debounced count vs raw count |
| Hover responsiveness | Qualitative — does hover label appear within ~100 ms? Note the visible-point threshold at which hover stops being usable. |
| Browser memory | DevTools Memory tab, after load and after 5 pan/zoom cycles |

Numbers will be approximate (single machine, single browser). Goal is order-of-magnitude separation, not microbenchmarks.

## Background hypotheses to verify (from research)

- Scattergl breaks down for interaction around 100–200 k points (Plotly docs); community reports at 180 k.
- Scattermap (the new replacement for deprecated scattermapbox) is **not** WebGL-accelerated — expect failure well below 100 k.
- Plotly-resampler does **not** support `scatter_mapbox` (confirmed in its docs); only `go.Scatter`/`go.Scattergl`. So it is only viable for Scenario B.
- Datashader → Mapbox image layer scales to billions but loses per-point hover — fits Scenario C, *not* A.
- dash-leaflet GeoJSON + supercluster claims "millions of points" but starts to struggle past ~1 000 raw markers without clustering (issue #24). Could be added as an optional A.4 if time permits.

## Critical files referenced (read-only)

- `aiswakepy/viz/wave_map.py:22–76` — `_downsample()` is the bbox-aware top-N primitive to reuse.
- `aiswakepy/viz/wave_map.py:130` — call site showing how downsample feeds matplotlib today.
- `docs/VIZ_DOWNSAMPLING.md` — algorithm reference; benchmarks (100 k → 10 s, 2 M → 25 s on existing matplotlib pipeline) are the "before" numbers to beat.
- `aiswakepy/stages/wave_impact.py` — structure of the wave-impact dataframe (ShLongitude, ShLatitude, WaveHeight, WavePeriod) the notebook will load.
- `aiswakepy/pipeline.py` — how to drive the pipeline on real `examples/` data so the notebook works against a representative dataset.
- `pyproject.toml` — Dash and spike dependencies now declared.

## Notebook outline

1. **Setup & data load** — load real pipeline output if available (`examples/`), else synthesise a representative 2 M-row dataframe shaped like the wave-impact output. Produce a vessel-track frame too (raw AIS interpolated).
2. **Reference table** — markdown cell summarising each technique with hyperlinks to the sources gathered in this study.
3. **Scenario A** cells — 3 sub-cells, one per candidate, each launching Dash inline. Record metrics into a small dict.
4. **Scenario B** cells — 2 sub-cells.
5. **Scenario C** cells — 2 sub-cells.
6. **Scenario D** cells — 2 sub-cells.
7. **Scenario E** cells — 5 sub-cells, each layering one mitigation on top of A.2.
8. **Synthesis cell** — markdown table comparing all measured metrics + a recommended architecture for the Dash app spec, including:
   - which renderer per interaction state,
   - the visible-points threshold above which hover is auto-disabled,
   - the smoothness-mitigation stack (debounce ms, coarse-layer point count, whether pre-tiling is needed).
   Likely shape: server-side bbox-aware top-N feeding Scattermap for waves with debounced callback + always-on coarse base layer; datashader image layer for unfiltered vessel density; plain Scattermap for the filtered drill-down.

## New dependencies added

Via `uv add`:
- `plotly-resampler` — for Scenario B.2.
- `datashader` — for Scenario C.1.
- `dash-deck` *and* `pydeck` — for Scenarios C.2 and D.2.
- `pytz` — transitive dependency of plotly-resampler.

`dash` itself is already in `pyproject.toml`.

## How to run the spike

1. `uv run jupyter lab dash_rendering_spike.ipynb`
2. Run cells top-to-bottom against the synthesised 2 M-row wave-impact dataset.
3. Each scenario cell has `app.run(...)` commented out on its own dedicated port (8061–8072) — uncomment when you want to launch that variant.
4. Pan/zoom for 5 s with DevTools Performance recording. Fill in the manual smoothness rating in `record(...)`.
5. Run the synthesis cell at the bottom to print the metrics table and architecture recommendation.

**Smoothness check (primary success criterion):** in Scenario E, continuous wheel-zoom over 5 s must feel smooth (no dropped frames visible) under at least one of the mitigation combinations. If none pass, escalate to E.4 pre-tiling.

## Out of scope

- Building the Dash app itself.
- Streaming/server-side multi-user concerns (single-user desktop is the deployment target).
- Mobile/touch performance.
- Tile-server architecture (Terracotta etc.) — overkill for a single-user analysis tool.
