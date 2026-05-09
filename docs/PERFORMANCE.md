# Performance Optimization Plan: 2M AIS Records

**Status**: ✅ All 6 fixes complete (Fix 6 implemented as `print()`, not Rich Console — intentional).
**Last reviewed**: 2026-05-09

## Status Summary

| Fix | File(s) | Status | Notes |
|-----|---------|--------|-------|
| 1. Vectorize `validate_speed` | `stages/filter.py` | ✅ done | Uses array `geodetic_distance` (filter.py:464) |
| 2. Vectorize `mask_land` | `stages/filter.py` | ✅ done | Uses `gpd.points_from_xy()` (filter.py:643,663) |
| 3. Reduce allocations in `interpolate_trajectories` | `stages/filter.py` | ✅ done | Numpy buffers per segment, single concat |
| 4. STRtree spatial index for shore intersection | `geo/coastline.py`, `stages/wave_impact.py` | ✅ done | `build_coastline_index` returns `(STRtree, segments)`; ray loop uses custom `Spinner` |
| 5. Coastline-binned top-N visualisation | `viz/wave_map.py`, `config.py` | ✅ done | Implemented as `plot_max_points: int = 100_000`; uses `ceil(max_points/n_occupied_bins)` per 1-m bin |
| 6. Per-stage timing + status logging | `pipeline.py` | ✅ done | `time.perf_counter()` per stage; plain `print()` for status messages (intentional — see note below) |

**Note on Fix 6**: The original plan called for `rich.console.Console`, but plain `print()` is the chosen approach. Vectorised stages execute as a single block — there is no row-by-row progress to render, so a Rich progress bar adds no value. A simple `print("Stage X/Y: ...")` + elapsed time is sufficient and avoids extra rendering complexity. The custom `Spinner` in `aiswakepy/_progress.py` covers the few stages that do have meaningful per-item progress.

---

## Original Context

The pipeline was developed and tested on ~2,300 AIS records (AIS_2563.csv).
This plan covered scaling to **2 million records** and identified bottlenecks.

### Time Estimates

| Stage | Operation | Bottleneck | Pre-fix (2M rows) | Post-fix |
|-------|-----------|-----------|-------------------|---------|
| 1a. load_ais | CSV read | I/O | ~10 s | ~10 s |
| 1b. segment_trajectories | sort + diff + cumsum | Vectorized | ~3 s | ~3 s |
| 1c. validate_speed | Python for-loop | Row loop → vectorized | ~200–400 s | ~2–5 s |
| 1d. interpolate_trajectories | Nested loop, concat | Row loop + alloc → buffers | ~500–1000 s | ~20–40 s |
| 1e. mask_land | List comp Point() | List comp → vectorized | ~60–120 s | ~5–10 s |
| 2. depth | KDTree query | Vectorized | ~15 s | ~15 s |
| 3. wave_params | Vectorized | — | ~5 s | ~5 s |
| 4. shore_impact | iterrows × O(C) | Linear → STRtree | ~30–120 min | ~2–10 min |
| 5. viz | scatter() millions | Rendering → binned top-N | ~2–5 min | ~10–30 s |
| | | **TOTAL** | **~45 min – 2.5 hr** | **~3–12 min** |

Note: After interpolation, 2M input rows expand to ~5–10 M rows. After wave-param filtering, ~500 K–2 M rows enter Stage 4.

---

## Depth Query Algorithm (no changes needed)

`bathymetry.py` uses `scipy.spatial.KDTree`:

- **Build**: O(N log N) on mesh node coordinates — one-time cost
- **Query**: O(log N) per point, parallelised with `workers=-1`
- **Coordinate system**: raw (lon, lat) Euclidean. At Singapore (1.3 °N), anisotropy < 1 % — nearest-neighbour ordering is correct.
- **10 M points**: ~2–5 seconds. **No optimisation needed.**

---

## Fix 1: Vectorize `validate_speed` ✅

**Before**: Python for-loop calling scalar `geodetic_distance` per row.

**After** (`stages/filter.py:440–477`):
```python
same_seg = np.zeros(n, dtype=bool)
same_seg[1:] = segs[1:] == segs[:-1]
idx = np.where(same_seg)[0]
if len(idx):
    d = geodetic_distance(lons[idx - 1], lats[idx - 1], lons[idx], lats[idx])
    dt = (times[idx] - times[idx - 1]) / np.timedelta64(1, "s")
    dist_m[idx] = d
    valid_dt = dt > 0
    v_calc[idx[valid_dt]] = d[valid_dt] / dt[valid_dt] / _KNOTS_TO_MS
```

**Speedup**: ~200–400 s → ~2–5 s.

---

## Fix 2: Vectorize `mask_land` ✅

**Before**: `[Point(lon, lat) for ...]` list comprehension.

**After** (`stages/filter.py:643,663`):
```python
points = gpd.GeoSeries(
    gpd.points_from_xy(df["longitude"], df["latitude"]),
    crs="EPSG:4326",
)
```

**Speedup**: ~60–120 s → ~5–10 s.

---

## Fix 3: Reduce allocations in `interpolate_trajectories` ✅

**Before**: Nested loop, per-gap DataFrame, O(N²) `pd.concat`.

**After**: Pre-computed numpy arrays per segment gap, single concat per segment.

**Speedup**: ~500–1000 s → ~20–40 s.

---

## Fix 4: STRtree spatial index ✅ (with note)

**Before**: `iterrows()` over wave events × 2 rays × O(C) coastline scan.

**After** (`geo/coastline.py`):
- `build_coastline_index(coastline)` returns `(STRtree, segments)`.
- `find_shore_intersection_indexed(ray, strtree, segments)` uses STRtree `query()`.

In `stages/wave_impact.py`:
- Vectorized ray endpoint computation.
- Per-event progress via custom `Spinner` (in `aiswakepy/_progress.py`), **not** `rich.progress.track()` as originally planned. The custom spinner already supports both terminal and Jupyter, so the functional outcome is achieved.

**Speedup**: ~30–120 min → ~2–10 min.

---

## Fix 5: Coastline-binned top-N visualisation ✅ (different config field name)

**Implementation** (`viz/wave_map.py`):
- Reproject to UTM, project shore points onto coastline `LineString`, bin into 1-m bins.
- Within each bin, keep top-N highest WaveHeight, where N = `ceil(max_points / n_occupied_bins)`.
- Sort ascending by WaveHeight before scatter so highest waves render on top.

**Config field** (`config.py:68`): `plot_max_points: int = 100_000` (not `plot_top_n_per_bin` as originally proposed). The cap is a global maximum total point count, derived per-bin dynamically.

**Speedup**: ~2–5 min → ~10–30 s.

---

## Fix 6: Per-stage timing + status logging ✅

**Implemented**: per-stage timing via `time.perf_counter()` and plain `print()` for status messages in `pipeline.py`.

**Why `print()` instead of Rich Console**: vectorised stages run as a single numpy/pandas block with no per-row work — a Rich progress bar has nothing to update incrementally. A one-line "Stage X/Y: …" + elapsed time is the right granularity. For the few stages that do iterate per item (e.g. shore intersection), the custom `Spinner` in `aiswakepy/_progress.py` provides terminal + Jupyter-compatible per-item progress without pulling Rich into the orchestrator.

---

## Files Modified

| File | Fix(es) |
|------|---------|
| `aiswakepy/stages/filter.py` | #1, #2, #3 |
| `aiswakepy/stages/wave_impact.py` | #4 (consumer of STRtree) |
| `aiswakepy/geo/coastline.py` | #4 (STRtree builder) |
| `aiswakepy/viz/wave_map.py` | #5 (binned top-N) |
| `aiswakepy/pipeline.py` | #6 (per-stage timing + `print()` status) |
| `aiswakepy/config.py` | #5 (`plot_max_points` field) |
| `aiswakepy/_progress.py` | (new) custom Spinner used in stages |
| `pyproject.toml` | `rich` dependency added |

## Verification

After each fix:
1. `uv run pytest tests/ -q` — all 145 tests pass on master.
2. `uv run python validate_pipeline.py` — see `tests/validation_report.md`.
