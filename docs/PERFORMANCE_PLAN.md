# Performance Optimization Plan: 2M AIS Records

## Context

The current pipeline was developed and tested on ~2,300 AIS records (AIS_2563.csv).
This plan estimates the cost of scaling to **2 million records** and identifies
which parts of the code need optimization.

---

## Time Estimate: Current Code on 2M Records

| Stage | Operation | Bottleneck Type | Est. Time (2M rows) |
|-------|-----------|-----------------|---------------------|
| 1a. load_ais | CSV read | I/O | ~10 s |
| 1b. segment_trajectories | sort + diff + cumsum | Vectorized | ~3 s |
| 1c. **validate_speed** | **Python for-loop**, scalar `geodetic_distance` | **Row loop** | **~200–400 s** |
| 1d. **interpolate_trajectories** | Nested loop, per-gap DataFrame concat | **Row loop + alloc** | **~500–1000 s** |
| 1e. **mask_land** | `[Point(lon,lat) for ...]` per row | **List comp** | **~60–120 s** |
| 2. depth | KDTree query (vectorized, workers=-1) | Vectorized | ~15 s |
| 3. wave_params + kriebel | All vectorized numpy/pandas | Vectorized | ~5 s |
| 4. **shore_impact** | **`iterrows()` x 2 rays x Shapely intersection, no spatial index** | **Row loop x O(C)** | **~30–120 min** |
| 5. **viz** | **scatter() with millions of points** | **Rendering** | **~2–5 min per plot** |
| | | **TOTAL (current)** | **~45 min – 2.5 hours** |

Note: After interpolation (20m spacing), 2M input rows likely expand to **5–10M rows**.
After wave-param filtering, ~500K–2M rows enter Stage 4.

---

## Optimized Time Estimate

| Stage | Est. Time (optimised) | Change |
|-------|-----------------------|--------|
| 1. filter (all steps) | ~30–60 s | vectorise speed, mask_land, reduce allocs |
| 2. depth | ~15 s | (already fast) |
| 3. wave_params | ~5 s | (already fast) |
| 4. shore_impact | ~2–10 min | STRtree + prepared geometry |
| 5. viz | ~10–30 s | coastline-binned top-N selection |
| **TOTAL** | **~3–12 min** | **10–20x speedup** |

---

## Depth Query Algorithm (confirmed OK — no changes needed)

`bathymetry.py` uses `scipy.spatial.KDTree`:

- **Build**: O(N log N) on mesh node coordinates — one-time cost
- **Query**: O(log N) per point, parallelised with `workers=-1`
- **Coordinate system**: raw (lon, lat) Euclidean. At Singapore (1.3 deg N), anisotropy < 1% — nearest-neighbour ordering is correct.
- **10M points**: ~2–5 seconds. **No optimisation needed.**

---

## Fix 1: Vectorize validate_speed — `aiswakepy/stages/filter.py`

**Problem**: Python for-loop calling scalar `geodetic_distance` per row (lines ~95–101).

**Fix**: Replace with a single vectorized call across all row pairs in the same segment:
```python
same_seg = segs[1:] == segs[:-1]
d = geodetic_distance(lons[:-1], lats[:-1], lons[1:], lats[1:])  # array call
dt = (times[1:] - times[:-1]) / np.timedelta64(1, "s")
dist_m[1:] = np.where(same_seg, d, np.nan)
v_calc[1:] = np.where(same_seg & (dt > 0), d / dt / _KNOTS_TO_MS, np.nan)
```

**Speedup**: ~200–400 s → ~2–5 s

---

## Fix 2: Vectorize mask_land — `aiswakepy/stages/filter.py`

**Problem**: `[Point(lon, lat) for ...]` list comprehension creates one Shapely object per row.

**Fix**: Replace with:
```python
points = gpd.GeoSeries(
    gpd.points_from_xy(df["longitude"], df["latitude"]),
    crs="EPSG:4326"
)
```

**Speedup**: ~60–120 s → ~5–10 s

---

## Fix 3: Reduce allocations in interpolate_trajectories — `aiswakepy/stages/filter.py`

**Problem**: Nested loop creates a new DataFrame per gap, then `pd.concat` accumulates O(N²) copies.

**Fix**: Pre-compute numpy arrays per segment gap, build column arrays with `np.concatenate`, construct one DataFrame per segment, single `pd.concat` at end.

**Speedup**: ~500–1000 s → ~20–40 s

---

## Fix 4: STRtree + Rich progress bar — `aiswakepy/stages/shore_impact.py` + `aiswakepy/geo/coastline.py`

**Problem**: `iterrows()` over all wave events x 2 rays x O(C) linear scan of coastline segments.

**Fix**:
1. **`coastline.py`**: Add function to build Shapely `STRtree` from coastline boundary segments. Use `shapely.prepared.prep()` on coastline for fast containment checks.
2. **`shore_impact.py`**:
   - Vectorize ray endpoint computation with array `forward_point`
   - Use STRtree `query()` per ray to get candidate segments (reduces O(C) to O(log C + k))
   - Wrap the ray loop with `rich.progress.track()`

**Speedup**: ~30–120 min → ~2–10 min

---

## Fix 5: Coastline-binned top-N visualisation — `aiswakepy/viz/wave_map.py`

**Problem**: Scatter plot with millions of points takes 2–5 min to render; dense areas are illegible.

**Algorithm**:
1. Extract coastline boundary as a single `LineString` (or `MultiLineString` merged via `shapely.ops.linemerge`).
2. For each shore impact point, compute its position along the coastline: `dist_along = coastline_line.project(Point(sh_lon, sh_lat))`.
3. Assign to a 1-metre bin: `bin_idx = int(dist_along)`.
4. Within each bin, sort by `WaveHeight` descending, keep the **top N** (default: 10).
5. Sort the selected points by `WaveHeight` ascending (so the highest wave is plotted last = rendered on top with highest z-order).
6. Plot with small markers (`s=4–6`). Dense segments naturally show overlapping/crowded points.

**Configuration**: Add `plot_top_n_per_bin: int = 10` to the `output` section of `aiswakepy/config.py`.

**Why 1m bins / top 10**:
- At 150 DPI on a 10-inch figure spanning ~10 km of coastline: 1 pixel ~ 6.7 m.
- 1m bins give fine coastal resolution. Top 10 per bin caps total points (e.g. 20 km coast x 10 = 200K max, but most bins will have 0 hits so actual count is much less).
- Highest wave at each location is always visible on top (ascending sort before scatter).

**Speedup**: ~2–5 min → ~10–30 s

---

## Fix 6: Rich console + per-stage timing — `aiswakepy/pipeline.py`

**Fix**:
- Replace `print()` with `rich.console.Console` for stage logging
- Add elapsed time per stage using `time.perf_counter()`
- Add `rich` to `pyproject.toml` dependencies

---

## Files to Modify

| File | Fix(es) |
|------|---------|
| `aiswakepy/stages/filter.py` | #1 (validate_speed), #2 (mask_land), #3 (interpolate_trajectories) |
| `aiswakepy/stages/shore_impact.py` | #4 (STRtree, Rich progress) |
| `aiswakepy/geo/coastline.py` | #4 (STRtree builder) |
| `aiswakepy/viz/wave_map.py` | #5 (coastline-binned top-N) |
| `aiswakepy/pipeline.py` | #6 (Rich console + timing) |
| `aiswakepy/config.py` | #5 (`plot_top_n_per_bin` field) |
| `pyproject.toml` | #6 (add `rich` dependency) |

## Tests to Add/Update

| Test File | What to Verify |
|-----------|---------------|
| `tests/test_filter.py` | Vectorized speed validation gives same results as original |
| `tests/test_shore_impact.py` | STRtree intersection matches brute-force results |
| `tests/test_viz.py` | Binned plotting produces output file without error |

---

## Verification Steps

After each fix:
1. `uv run pytest tests/ -q` — all 102 tests pass
2. After all fixes: `uv run python validate_pipeline.py` — numerical results match previous
3. Time AIS_2563.csv before/after, log per-stage timings
4. Visually compare wave height map: old scatter vs new binned top-N
