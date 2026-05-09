# Visualisation Downsampling — Coastline-Binned Top-N

**Implementation**: `aiswakepy/viz/wave_map.py::_downsample`
**Used by**: `plot_wave_height_map`, `plot_wave_period_map`
**Default cap**: `max_points = 100_000` (configurable via `OutputConfig.plot_max_points`)

---

## Problem

After processing a typical AIS day in Singapore waters, the shore-impact stage emits **0.5–2 M** wake events along the coastline. A direct `ax.scatter()` call:

- takes 2–5 minutes to render at 150 DPI;
- produces visually opaque blobs in dense areas (worst-case overplotting);
- hides the highest waves under a sea of low-wave noise (scatter z-order is insertion order).

The downsampler caps the rendered point count while preserving the highest impact at every location along the coast.

---

## Algorithm

```
input:  df_impact (N rows: ShLongitude, ShLatitude, WaveHeight, WavePeriod, ...),
        coastline_shp,
        max_points (int)

if N <= max_points:
    return df sorted ascending by WaveHeight       # nothing to drop, just z-order

1. Reproject coastline → UTM (estimate_utm_crs).         # metric distances
2. Merge the polygon boundary into one LineString
   (unary_union → linemerge if MultiLineString).
3. Reproject the impact points to the same UTM CRS.
4. For each point, compute dist_along = coastline_line.project(point)
   → distance in metres along the merged coastline.
5. Bin: bin_id = int(dist_along)                          # 1-metre bins
6. n_bins = nunique(bin_id)
   top_n  = ceil(max_points / n_bins)                     # adaptive per-bin cap
7. Sort by WaveHeight desc, group by bin_id, head(top_n).
8. Sort the survivors ascending by WaveHeight             # render order
   so the highest values are drawn last (top z-order).
return df_plot   (≤ max_points rows)
```

### Key properties

| Property | Value |
|---|---|
| Spatial bin width | 1 metre (in UTM) |
| Per-bin point cap | adaptive: `ceil(max_points / n_occupied_bins)` |
| Output cardinality | `≤ max_points` |
| Selection criterion | top-N by `WaveHeight` (only this column) |
| Render order | ascending by `WaveHeight` (highest on top) |
| Activation | only when `N > max_points` |

### Why these choices

- **1-m bins**: at 150 DPI on a 10-inch figure spanning ~10 km of coast, one display pixel is ≈ 6.7 m. 1-m bins are below pixel resolution, so the bin grid never aliases the pixel grid — visually the result looks continuous. Smaller bins (sub-metre) just mean fewer points share a bin and more total points survive; larger bins (10 m) start to drop visible detail.
- **Adaptive `top_n`**: a fixed per-bin N would either waste budget on sparse stretches of coast or starve dense ones. The adaptive form spends the budget proportionally to bin occupancy.
- **`WaveHeight` only**: the period map (`plot_wave_period_map`) also downsamples by `WaveHeight`, not by `WavePeriod`. Rationale: the high-impact events are what matters operationally, regardless of which variable colours the dot.
- **`int(dist_along)` truncation**: not rounding — points at 1.6 m and 1.9 m share bin `1`, but a point at 2.0 m goes to bin `2`. This is consistent (no rounding bias) and avoids edge effects at integer boundaries.

---

## Performance

| Pipeline size | Rendering before | Rendering after |
|---|---|---|
| 100 K points | ~10 s | ~10 s (no-op, under cap) |
| 500 K points | ~2 min | ~15 s |
| 2 M points | ~4–5 min | ~25 s |

The expensive operation inside `_downsample` itself is the per-point `LineString.project()` call (step 4) — currently a Python list comprehension. For 2 M points this takes ~30–60 s; STRtree-accelerated projection or a vectorised reimplementation would help if downsampling becomes a bottleneck.

---

## Limitations to know before reusing in a DASH app

1. **Map-extent agnostic**: the algorithm bins along the *entire* coastline, not the visible viewport. In an interactive map, panning to a small region still computes bins over the full coast — the downsampler would correctly return the highest-N visible-region points, but the work scales with the full dataset. **DASH fix**: filter `df_impact` by viewport bounds *before* calling `_downsample`.

2. **Single ranking variable**: hard-coded to `WaveHeight`. If users want to inspect "top events by period" or "top events by energy", the function needs a `rank_col` parameter. Trivial change — `df.sort_values(rank_col, ascending=False)`.

3. **No per-vessel grouping**: for ship-track analysis, users typically want "show all impacts from MMSI X". The downsampler would drop most of vessel X's impacts in dense areas. **DASH fix**: when an MMSI filter is active, either skip downsampling or group the head() by `(bin_id, MMSI)` so each vessel keeps its peak in each bin.

4. **No time dimension**: all of a day's events are pooled. A time-slider in DASH should pre-filter by `obstime` before downsampling — otherwise bins are dominated by the busiest hour.

5. **CRS recomputation**: `estimate_utm_crs()` and `to_crs()` are called every render. For a long-running DASH app, cache:
   - the merged metric coastline `LineString`,
   - and the pre-computed `(bin_id, dist_along)` arrays per impact point (these only change when the source data does, not when the user pans).

6. **Static z-order**: matplotlib's "ascending sort then scatter" trick does not translate to Plotly. In Plotly Scattermapbox, all markers in one trace are drawn together — to get the "highest on top" effect, split into two traces (low-value background, high-value foreground) or rely on marker opacity.

7. **`WaveHeight` colour scale fixed at vmax = 0.5 m**: in `plot_wave_height_map`, vmax is hard-coded so different runs are visually comparable. For an interactive app, expose this as a slider — the value range varies by site and dataset.

---

## Extension points (DASH-friendly API sketch)

```python
def downsample_for_viewport(
    df: pd.DataFrame,
    coastline_line_m: LineString,    # pre-merged, cached at startup
    bbox: tuple[float, float, float, float],  # (lon0, lat0, lon1, lat1)
    max_points: int = 50_000,        # smaller for interactive
    rank_col: str = "WaveHeight",
    bin_size_m: float = 1.0,         # exposed for zoom-aware tuning
    group_keys: list[str] | None = None,  # e.g. ["MMSI"] when a vessel filter is active
) -> pd.DataFrame:
    ...
```

Suggested defaults for an interactive app:
- `max_points`: 30 K–50 K (Plotly Scattermapbox renders ~100 K smoothly but interactivity drops above 50 K)
- `bin_size_m`: scale with current zoom (e.g. `1 m` at street zoom, `25 m` at city zoom)
- Pre-filter by viewport `bbox` and any active vessel/time filter *before* this call

---

## File references

- `aiswakepy/viz/wave_map.py:22` — `_downsample` implementation
- `aiswakepy/viz/wave_map.py:130` — call site in `_plot_impact_map`
- `aiswakepy/config.py:68` — `OutputConfig.plot_max_points`
- `docs/PERFORMANCE.md` Fix 5 — origin of the design
