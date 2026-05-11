"""Dash + raw deck.gl + Apache Arrow performance test on real AIS data,
with an in-page two-step pipeline runner and per-input previews.

Run with:  uv run python dash_app.py
Then open: http://localhost:8050   (or  http://<lan-ip>:8050  from another host)
"""
from __future__ import annotations

import io
import sys
import threading
import time
from pathlib import Path

import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from dash import Dash, dcc, html, Input, Output, State, no_update
from flask import Response, jsonify, request

from aiswakepy.config import load_config
from aiswakepy.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent
OUT = REPO / 'output'
PORT = 8050
RASTER_W, RASTER_H = 1024, 768
RASTER_AOI = (103.55, 1.20, 103.78, 1.32)  # west, south, east, north
ZOOM_RASTER_THRESHOLD = 11
PREVIEW_AIS_MAX_POINTS = 5000          # subsample raw AIS to this many points for preview
PREVIEW_BATHY_MAX_TRIANGLES = 30000     # cap mesh wireframe size for preview

# ---------------------------------------------------------------------------
# Module-level caches (mutated by _build_* helpers and the pipeline thread)
# ---------------------------------------------------------------------------
df_vessels: pd.DataFrame
df_waves: pd.DataFrame
seg_meta: list[dict] = []
IPC_VESSELS: bytes = b''
IPC_WAVES: bytes = b''
IPC_TRACK_COORDS: bytes = b''
IPC_TRACK_META: bytes = b''
IPC_TRACK_OFFSETS: bytes = b''
PNG_BYTES: bytes = b''
LAST_RESULTS: dict = {}                # {'df_filtered': df, 'df_vessel': df, ...}


def _ipc(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    with ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    return sink.getvalue()


def _ensure_vessel_columns(df_w: pd.DataFrame, df_v: pd.DataFrame) -> pd.DataFrame:
    """Backward-compat join: fill new vessel-side columns into older wave CSVs
    that lack them (DT_04_impact.csv predates the wave_impact.py schema fix)."""
    needed = {'VesselLongitude', 'VesselLatitude', 'VesselCOG',
              'VesselDraught', 'segment_id'}
    missing = needed - set(df_w.columns)
    if not missing:
        return df_w
    print(f'  [compat] wave CSV missing {sorted(missing)} - joining from vessel CSV')
    join = df_v[['mmsi', 'obstime', 'longitude', 'latitude',
                 'segment_id', 'draught', 'cog']].rename(columns={
        'mmsi': 'MMSI', 'obstime': 'DateTime',
        'longitude': 'VesselLongitude', 'latitude': 'VesselLatitude',
        'cog': 'VesselCOG', 'draught': 'VesselDraught',
    })
    out = df_w.merge(join, on=['MMSI', 'DateTime'], how='left')
    n_unmatched = out['VesselLongitude'].isna().sum()
    if n_unmatched:
        print(f'  [compat] WARN: {n_unmatched} wave records have no matching vessel row')
    return out


def _build_vessel_caches(df_v: pd.DataFrame) -> None:
    """(Re)compute vessel Arrow + track-segment Arrow + datashader PNG.

    Vectorised segment encoding: sort by (mmsi, segment_id), use np.diff to find
    boundaries, then build flat coords + offsets without a Python-level groupby loop.
    """
    global df_vessels, seg_meta
    global IPC_VESSELS, IPC_TRACK_COORDS, IPC_TRACK_META, IPC_TRACK_OFFSETS, PNG_BYTES

    print('  casting types...')
    df_v = df_v.astype({
        'mmsi': 'int64', 'segment_id': 'int32', 'typecargo': 'float32',
        'longitude': 'float32', 'latitude': 'float32',
        'sog': 'float32', 'cog': 'float32',
        'width': 'float32', 'length': 'float32', 'draught': 'float32',
    }, errors='ignore')
    df_vessels = df_v

    print(f'  encoding vessel Arrow ({len(df_v):,} rows)...')
    arrow_vessels = pa.Table.from_pandas(
        df_v[['longitude', 'latitude', 'mmsi', 'sog', 'cog', 'typecargo']],
        preserve_index=False,
    )
    IPC_VESSELS = _ipc(arrow_vessels)

    print('  building track segments (vectorised)...')
    if len(df_v) == 0:
        flat_arr = np.zeros((0, 2), dtype=np.float32)
        offsets_arr = np.array([0], dtype=np.int32)
        meta_mmsi = np.array([], dtype=np.int64)
        meta_seg  = np.array([], dtype=np.int32)
        meta_n    = np.array([], dtype=np.int32)
    else:
        df_sorted = df_v.sort_values(['mmsi', 'segment_id'], kind='stable').reset_index(drop=True)
        mmsi_arr = df_sorted['mmsi'].to_numpy(dtype=np.int64)
        sid_arr  = df_sorted['segment_id'].to_numpy(dtype=np.int64)
        # Pack (mmsi, segment_id) into a single int for boundary detection.
        key = mmsi_arr * np.int64(2 ** 32) + sid_arr
        diffs = np.diff(key)
        boundaries = np.concatenate(([0], np.where(diffs != 0)[0] + 1, [len(key)])).astype(np.int64)
        starts = boundaries[:-1]
        sizes  = (boundaries[1:] - boundaries[:-1]).astype(np.int32)
        keep   = sizes >= 2
        if not keep.any():
            flat_arr = np.zeros((0, 2), dtype=np.float32)
            offsets_arr = np.array([0], dtype=np.int32)
            meta_mmsi = np.array([], dtype=np.int64)
            meta_seg  = np.array([], dtype=np.int32)
            meta_n    = np.array([], dtype=np.int32)
        else:
            kept_starts = starts[keep]
            kept_sizes  = sizes[keep]
            kept_ends   = kept_starts + kept_sizes.astype(np.int64)
            # Indices of rows belonging to kept segments — concatenated ranges.
            row_idx = np.concatenate([np.arange(s, e) for s, e in zip(kept_starts, kept_ends)])
            sel_lon = df_sorted['longitude'].to_numpy(dtype=np.float32)[row_idx]
            sel_lat = df_sorted['latitude'].to_numpy(dtype=np.float32)[row_idx]
            flat_arr = np.column_stack([sel_lon, sel_lat]).astype(np.float32)
            offsets_arr = np.concatenate(([0], np.cumsum(kept_sizes))).astype(np.int32)
            meta_mmsi = mmsi_arr[kept_starts]
            meta_seg  = sid_arr[kept_starts].astype(np.int32)
            meta_n    = kept_sizes.astype(np.int32)

    seg_meta = [{'mmsi': int(m), 'segment_id': int(s), 'n_points': int(n)}
                for m, s, n in zip(meta_mmsi, meta_seg, meta_n)]
    arrow_coords = pa.table({
        'lon': pa.array(flat_arr[:, 0], type=pa.float32()),
        'lat': pa.array(flat_arr[:, 1], type=pa.float32()),
    })
    arrow_meta = pa.table({
        'mmsi':       pa.array(meta_mmsi.astype(np.int64), type=pa.int64()),
        'segment_id': pa.array(meta_seg,                   type=pa.int32()),
        'n_points':   pa.array(meta_n,                     type=pa.int32()),
    })
    arrow_offsets = pa.table({'offset': pa.array(offsets_arr, type=pa.int32())})
    IPC_TRACK_COORDS = _ipc(arrow_coords)
    IPC_TRACK_META = _ipc(arrow_meta)
    IPC_TRACK_OFFSETS = _ipc(arrow_offsets)

    print('  rasterising vessel density...')
    canvas = ds.Canvas(plot_width=RASTER_W, plot_height=RASTER_H,
                       x_range=(RASTER_AOI[0], RASTER_AOI[2]),
                       y_range=(RASTER_AOI[1], RASTER_AOI[3]))
    agg = canvas.points(df_v, x='longitude', y='latitude')
    img = tf.shade(agg, cmap=['#330033', '#ff6600', '#ffff80'], how='log')
    buf = io.BytesIO()
    img.to_pil().save(buf, format='PNG')
    PNG_BYTES = buf.getvalue()


def _build_wave_caches(df_w: pd.DataFrame) -> None:
    """(Re)compute the wave Arrow with the full enriched schema."""
    global df_waves, IPC_WAVES

    cast = {
        'ShLongitude': 'float32', 'ShLatitude': 'float32',
        'VesselLongitude': 'float32', 'VesselLatitude': 'float32',
        'MMSI': 'int64', 'segment_id': 'int32',
        'WaveHeight': 'float32', 'WavePeriod': 'float32',
        'DistLoc_km': 'float32', 'SOG': 'float32',
        'VesselLength': 'float32', 'VesselWidth': 'float32',
        'VesselCOG': 'float32', 'VesselDraught': 'float32',
    }
    cols = ['ShLongitude', 'ShLatitude', 'MMSI', 'WaveHeight', 'WavePeriod',
            'Side', 'DistLoc_km', 'SOG', 'VesselLength', 'VesselWidth',
            'DateTime', 'VesselLongitude', 'VesselLatitude',
            'segment_id', 'VesselDraught', 'VesselCOG']
    for c, t in cast.items():
        if c not in df_w.columns:
            df_w[c] = pd.Series([np.nan] * len(df_w), dtype='float32')
    df_w = df_w[cols].copy()
    # Integer columns can't hold NaN — fill missing with sentinel before casting.
    for int_col in ('MMSI', 'segment_id'):
        if int_col in df_w.columns:
            df_w[int_col] = df_w[int_col].fillna(-1)
    df_w = df_w.astype({k: v for k, v in cast.items() if k in df_w.columns})
    df_w['DateTime'] = df_w['DateTime'].astype(str)
    df_w['Side'] = df_w['Side'].astype(str)

    df_waves = df_w
    IPC_WAVES = _ipc(pa.Table.from_pandas(df_w, preserve_index=False))


# ---------------------------------------------------------------------------
# Initial state: empty caches. The page boots showing only the basemap.
# Tracks appear after Step 1 (Filter AIS); waves after Step 2 (Calculate waves);
# AIS preview points appear when the AIS preview checkbox is ticked.
# ---------------------------------------------------------------------------
df_vessels = pd.DataFrame(columns=[
    'mmsi', 'longitude', 'latitude', 'sog', 'cog', 'typecargo',
    'segment_id', 'obstime', 'width', 'length', 'draught',
])
df_waves = pd.DataFrame(columns=[
    'ShLongitude', 'ShLatitude', 'MMSI', 'WaveHeight', 'WavePeriod',
    'Side', 'DistLoc_km', 'SOG', 'VesselLength', 'VesselWidth',
    'DateTime', 'VesselLongitude', 'VesselLatitude',
    'segment_id', 'VesselDraught', 'VesselCOG',
])
IPC_VESSELS = _ipc(pa.table({
    'longitude': pa.array([], pa.float32()), 'latitude': pa.array([], pa.float32()),
    'mmsi': pa.array([], pa.int64()), 'sog': pa.array([], pa.float32()),
    'cog': pa.array([], pa.float32()), 'typecargo': pa.array([], pa.float32()),
}))
IPC_TRACK_COORDS = _ipc(pa.table({
    'lon': pa.array([], pa.float32()), 'lat': pa.array([], pa.float32()),
}))
IPC_TRACK_META = _ipc(pa.table({
    'mmsi': pa.array([], pa.int64()), 'segment_id': pa.array([], pa.int32()),
    'n_points': pa.array([], pa.int32()),
}))
IPC_TRACK_OFFSETS = _ipc(pa.table({'offset': pa.array([0], pa.int32())}))
# Build wave Arrow with the empty df so the schema matches
_build_wave_caches(df_waves.copy())
# 1x1 transparent PNG placeholder so /api/raster.png never 500s before data exists
PNG_BYTES = bytes.fromhex(
    '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4'
    '890000000d49444154789c63000100000005000100200001ad6f0e0000000049'
    '454e44ae426082'
)
print('caches initialised empty - tracks/waves appear after the corresponding pipeline step.')


# ---------------------------------------------------------------------------
# Examples directory inventory
# ---------------------------------------------------------------------------
def _scan_examples() -> dict:
    rel = lambda p: p.relative_to(REPO).as_posix()
    return {
        'ais': sorted(rel(p) for p in (REPO / 'examples/ais').glob('*.csv')),
        'bathymetry': sorted(rel(p) for ext in ('mesh', 'dfs2', 'dfsu')
                             for p in (REPO / 'examples/bathymetry').glob(f'*.{ext}')),
        'coastline': sorted(rel(p) for p in (REPO / 'examples/coastline').glob('*.shp')),
        'tide': sorted(rel(p) for p in (REPO / 'examples/tide').glob('*.dfs0')),
    }


EXAMPLE_FILES = _scan_examples()
WAVE_FORMULAE = ['kriebel', 'pianc', 'sorensen', 'gates', 'blaauw', 'bhowmik', 'maynord']
CB_METHODS = ['L_Le', 'B_Le', 'table']


# ---------------------------------------------------------------------------
# Pipeline runner — background thread with stdout capture (\r-aware)
# ---------------------------------------------------------------------------
# RLock so the worker can `print` (which acquires the lock via _LineCapture)
# while it already holds the lock for state updates. A plain Lock deadlocks here.
_pipeline_lock = threading.RLock()
PIPELINE_STATE = {
    'running': False,
    'log': [],            # committed lines (static)
    'started_at': None,
    'finished_at': None,
    'error': None,
    'wave_version': 0,
    'track_version': 0,
    'n_waves': None,
    'n_filtered': None,
    'last_step': None,    # 'filter' or 'waves'
}


class _LineCapture(io.TextIOBase):
    """sys.stdout shim: capture every line as static text in PIPELINE_STATE['log'].

    Treats both \\r and \\n as line separators (no in-place spinner updates) — each
    write that produces a line commits it. Simpler than \\r-aware in-place updates,
    and immune to the deadlock + interpretation issues that approach suffered from.
    """

    def __init__(self, original):
        self._orig = original
        self._buf = ''

    def write(self, s: str) -> int:
        # Split incoming chunk on either CR or LF (universal newlines).
        self._buf += s
        # splitlines(keepends=False) handles \r, \n, \r\n.
        parts = self._buf.splitlines()
        # If the buffer ends with a line terminator, all parts are complete lines;
        # otherwise the last part is still in progress and stays in the buffer.
        if self._buf and self._buf[-1] in '\r\n':
            complete = parts
            self._buf = ''
        else:
            complete = parts[:-1]
            self._buf = parts[-1] if parts else ''
        new_lines = [ln for ln in complete if ln.strip()]
        if new_lines:
            with _pipeline_lock:
                PIPELINE_STATE['log'].extend(new_lines)
        try:
            self._orig.write(s)
            self._orig.flush()
        except Exception:
            pass
        return len(s)

    def flush(self) -> None:
        try:
            self._orig.flush()
        except Exception:
            pass


def _pipeline_thread(config_dict: dict, stages: list[str], step_label: str) -> None:
    """Worker thread. Runs the requested stages and refreshes only the affected caches.

    Cache builds run *outside* the pipeline lock so the polling tick can keep
    reading PIPELINE_STATE['log']/['live'] (and therefore the sidebar log keeps
    updating) while the slow groupby + Arrow encoding is in progress.
    """
    global LAST_RESULTS
    old_stdout = sys.stdout
    sys.stdout = _LineCapture(old_stdout)
    try:
        cfg = load_config(config_dict)
        seed = {k: v for k, v in LAST_RESULTS.items() if k.startswith('df_')}
        results = run_pipeline(cfg, stages=stages, seed_results=seed)
        LAST_RESULTS.update(results)

        # ---- Cache rebuild (no lock held) ----
        if 'df_filtered' in results and 'filter' in stages:
            print('Refreshing track caches...')
            t0 = time.perf_counter()
            _build_vessel_caches(results['df_filtered'])
            print(f'  -> {len(results["df_filtered"]):,} rows, '
                  f'{len(seg_meta):,} segments  ({time.perf_counter()-t0:.1f}s)')
            with _pipeline_lock:
                PIPELINE_STATE['track_version'] += 1
                PIPELINE_STATE['n_filtered'] = len(results['df_filtered'])

        if 'df_wave_impact' in results and 'wave_impact' in stages:
            print('Refreshing wave caches...')
            t0 = time.perf_counter()
            vessels_for_join = results.get('df_filtered', LAST_RESULTS.get('df_filtered'))
            if vessels_for_join is not None:
                enriched = _ensure_vessel_columns(results['df_wave_impact'], vessels_for_join)
            else:
                enriched = results['df_wave_impact']
            _build_wave_caches(enriched)
            print(f'  -> {len(results["df_wave_impact"]):,} wave events '
                  f'({time.perf_counter()-t0:.1f}s)')
            with _pipeline_lock:
                PIPELINE_STATE['wave_version'] += 1
                PIPELINE_STATE['n_waves'] = len(results['df_wave_impact'])

        with _pipeline_lock:
            PIPELINE_STATE['finished_at'] = time.time()
            PIPELINE_STATE['last_step'] = step_label
    except Exception as exc:
        import traceback
        traceback.print_exc()
        with _pipeline_lock:
            PIPELINE_STATE['error'] = f'{type(exc).__name__}: {exc}'
            PIPELINE_STATE['finished_at'] = time.time()
    finally:
        sys.stdout = old_stdout
        with _pipeline_lock:
            PIPELINE_STATE['running'] = False


# ---------------------------------------------------------------------------
# Preview helpers
# ---------------------------------------------------------------------------
def _preview_ais_arrow(path: Path, max_points: int = PREVIEW_AIS_MAX_POINTS) -> bytes:
    """Subsample an AIS CSV to ~max_points rows and return Arrow IPC of (lon,lat)."""
    # First pass: count rows. For perf, just stride-read.
    df = pd.read_csv(path, usecols=['longitude', 'latitude'])
    n = len(df)
    if n > max_points:
        stride = max(1, n // max_points)
        df = df.iloc[::stride].reset_index(drop=True)
    df = df.astype({'longitude': 'float32', 'latitude': 'float32'})
    return _ipc(pa.Table.from_pandas(df, preserve_index=False))


def _preview_ais_bbox(path: Path) -> tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) — used to fit the camera."""
    df = pd.read_csv(path, usecols=['longitude', 'latitude'])
    return (float(df['longitude'].min()), float(df['latitude'].min()),
            float(df['longitude'].max()), float(df['latitude'].max()))


def _preview_coast_geojson(path: Path) -> dict:
    """Load a shapefile and return a simplified GeoJSON FeatureCollection."""
    import geopandas as gpd
    gdf = gpd.read_file(str(path))
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    # Simplify to keep payload small; tolerance ~10 m at WGS84.
    gdf = gdf.copy()
    gdf['geometry'] = gdf.geometry.simplify(0.0001, preserve_topology=True)
    bbox = list(gdf.total_bounds)  # [minx, miny, maxx, maxy]
    return {
        'type': 'FeatureCollection',
        'bbox': bbox,
        'features': [
            {'type': 'Feature', 'properties': {}, 'geometry': geom.__geo_interface__}
            for geom in gdf.geometry if geom is not None and not geom.is_empty
        ],
    }


def _preview_tide(path: Path) -> dict:
    """Read a DFS0 tide file and return summary metadata."""
    import mikeio
    ds = mikeio.read(str(path))
    items = []
    for i, da in enumerate(ds):
        try:
            vals = np.asarray(da.values)
            v_min = float(np.nanmin(vals)) if vals.size else None
            v_max = float(np.nanmax(vals)) if vals.size else None
        except Exception:
            v_min = v_max = None
        items.append({
            'name': str(da.name),
            'unit': str(getattr(da.item, 'unit', '') or ''),
            'value_min': v_min,
            'value_max': v_max,
        })
    times = pd.to_datetime(ds.time)
    return {
        'items': items,
        'time_min': str(times[0]) if len(times) else None,
        'time_max': str(times[-1]) if len(times) else None,
        'n_steps': int(len(times)),
    }


def _preview_bathy_arrow(path: Path) -> bytes | None:
    """Load a .mesh/.dfsu and return an Arrow table with line segments for a wireframe.

    Layout: two Float32 columns (lon, lat). Every successive pair of rows is one
    line segment. Triangles contribute 3 segments each. Returns None for .dfs2.
    """
    suffix = path.suffix.lower()
    if suffix == '.dfs2':
        return None  # gridded — no triangulation to draw
    import mikeio
    if suffix == '.mesh':
        geom = mikeio.Mesh(str(path)).geometry
    else:  # .dfsu
        geom = mikeio.open(str(path)).geometry
    nodes = np.asarray(geom.node_coordinates)[:, :2]            # (N, 2) lon,lat
    elements = np.asarray(geom.element_table)                    # list of node-index lists
    # Normalize to 2D int array; mikeio is 1-based for some builds — guard against that.
    elements = [list(e) for e in elements]
    if elements and min(min(e) for e in elements) >= 1:
        elements = [[i - 1 for i in e] for e in elements]
    n_tris = len(elements)
    if n_tris > PREVIEW_BATHY_MAX_TRIANGLES:
        stride = max(1, n_tris // PREVIEW_BATHY_MAX_TRIANGLES)
        elements = elements[::stride]
    # Emit edges: for each element, every consecutive pair + closing pair.
    seg_lon = []
    seg_lat = []
    for e in elements:
        k = len(e)
        for j in range(k):
            a, b = e[j], e[(j + 1) % k]
            seg_lon.append(nodes[a, 0]); seg_lat.append(nodes[a, 1])
            seg_lon.append(nodes[b, 0]); seg_lat.append(nodes[b, 1])
    arr = pa.table({
        'lon': pa.array(np.asarray(seg_lon, dtype=np.float32)),
        'lat': pa.array(np.asarray(seg_lat, dtype=np.float32)),
    })
    return _ipc(arr)


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------
INDEX_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>aiswakepy - deck.gl spike</title>
    {%favicon%}
    {%css%}
    <script src="https://unpkg.com/deck.gl@9.1.13/dist.min.js"></script>
    <script type="importmap">
    {
      "imports": {
        "apache-arrow": "https://cdn.jsdelivr.net/npm/apache-arrow@21.0.0/+esm"
      }
    }
    </script>
    <script type="module">
        import { tableFromIPC } from 'apache-arrow';
        window.tableFromIPC = tableFromIPC;
        window.dispatchEvent(new Event('arrow-ready'));
    </script>
    <style>
        html, body { margin: 0; padding: 0; height: 100%; overflow: hidden;
                     font-family: system-ui, sans-serif; }
        #status-banner { position: fixed; top: 0; left: 0; right: 0; height: 28px;
                         padding: 4px 12px; box-sizing: border-box; z-index: 5;
                         font: 12px monospace; background: #eef; border-bottom: 1px solid #ccd; }
        #sidebar { position: fixed; top: 28px; left: 0; bottom: 0; width: 340px;
                   overflow-y: auto; padding: 12px; box-sizing: border-box;
                   background: #f8f8fb; border-right: 1px solid #ddd;
                   font: 12px system-ui, sans-serif; z-index: 4; }
        #sidebar h4 { margin: 0 0 8px; font-size: 13px; }
        #sidebar label { display: block; font-size: 11px; color: #555;
                         margin: 8px 0 2px; font-weight: 600; }
        .row-with-preview { display: flex; gap: 6px; align-items: center; }
        .row-with-preview > :first-child { flex: 1; min-width: 0; }
        .preview-box label { font-weight: normal !important; font-size: 10px !important;
                             color: #888; margin: 0 !important; white-space: nowrap; }
        .preview-info { font-size: 10px; color: #555; margin: 2px 0 0;
                        max-height: 60px; overflow: auto; white-space: pre-wrap; line-height: 1.3; }
        .preview-info.error { color: #c33; }
        #sidebar .row-buttons { display: flex; gap: 6px; margin-top: 12px; }
        #sidebar button { flex: 1; padding: 8px 6px;
                          border: 1px solid #58a; background: #69b; color: white;
                          border-radius: 3px; font-weight: bold; font-size: 11px; cursor: pointer; }
        #sidebar button:hover { background: #58a; }
        #sidebar button:disabled { background: #aaa; border-color: #888; cursor: wait; }
        #progress-log { background: #1e1e1e; color: #ddd; font: 11px ui-monospace, monospace;
                        padding: 8px; max-height: 45vh; overflow: auto; white-space: pre-wrap;
                        border-radius: 3px; margin: 4px 0; }
        #deck-container { position: fixed; top: 28px; left: 340px; right: 0; bottom: 0;
                          z-index: 1; overflow: hidden; }
        #tooltip { position: fixed; pointer-events: none; padding: 6px 10px;
                   background: rgba(0,0,0,0.85); color: white; font: 12px monospace;
                   border-radius: 4px; z-index: 100; display: none; white-space: nowrap; }
        #progress-overlay { position: fixed; top: 50%; left: 50%;
            transform: translate(-50%, -50%); background: white;
            border: 1px solid #aac; box-shadow: 0 4px 20px rgba(0,0,0,0.25);
            padding: 18px 24px; z-index: 200; min-width: 360px; font: 13px monospace; }
        #progress-title { font-weight: bold; margin-bottom: 10px; }
        .progress-row { display: flex; justify-content: space-between;
                        font-size: 11px; color: #444; margin: 2px 0; }
        .progress-row .name { font-weight: bold; }
        .progress-row .pct { color: #888; font-variant-numeric: tabular-nums; }
        #progress-bar { width: 100%; height: 10px; background: #eee;
                        border-radius: 5px; overflow: hidden; margin: 12px 0 6px; }
        #progress-fill { height: 100%; background: linear-gradient(90deg, #58c, #4ad);
                         width: 0%; transition: width 0.1s linear; }
        #progress-elapsed { font-size: 11px; color: #666; text-align: right; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""

app = Dash(__name__, suppress_callback_exceptions=True)
app.index_string = INDEX_TEMPLATE


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
def _bytes_response(b: bytes) -> Response:
    return Response(b, mimetype='application/vnd.apache.arrow.stream',
                    headers={'Cache-Control': 'no-store'})


@app.server.route('/api/vessels.arrow')
def _r_vessels(): return _bytes_response(IPC_VESSELS)


@app.server.route('/api/waves.arrow')
def _r_waves(): return _bytes_response(IPC_WAVES)


@app.server.route('/api/track_coords.arrow')
def _r_track_coords(): return _bytes_response(IPC_TRACK_COORDS)


@app.server.route('/api/track_meta.arrow')
def _r_track_meta(): return _bytes_response(IPC_TRACK_META)


@app.server.route('/api/track_offsets.arrow')
def _r_track_offsets(): return _bytes_response(IPC_TRACK_OFFSETS)


@app.server.route('/api/raster.png')
def _r_raster(): return Response(PNG_BYTES, mimetype='image/png')


@app.server.route('/api/inventory')
def _r_inventory(): return jsonify(EXAMPLE_FILES)


def _safe_repo_path(rel: str) -> Path:
    """Resolve a user-supplied relative path to an absolute path under REPO,
    raising if it tries to escape the repo root (defence against path traversal)."""
    if not rel:
        raise ValueError('empty path')
    abs_path = (REPO / rel).resolve()
    repo_root = REPO.resolve()
    try:
        abs_path.relative_to(repo_root)
    except ValueError:
        raise ValueError(f'path {rel!r} escapes repo root')
    if not abs_path.exists():
        raise FileNotFoundError(rel)
    return abs_path


@app.server.route('/api/preview/ais.arrow')
def _r_preview_ais():
    try:
        p = _safe_repo_path(request.args.get('path', ''))
        return _bytes_response(_preview_ais_arrow(p))
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.server.route('/api/preview/ais.bbox')
def _r_preview_ais_bbox():
    try:
        p = _safe_repo_path(request.args.get('path', ''))
        bbox = _preview_ais_bbox(p)
        return jsonify(bbox=bbox)
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.server.route('/api/preview/coast.geojson')
def _r_preview_coast():
    try:
        p = _safe_repo_path(request.args.get('path', ''))
        return jsonify(_preview_coast_geojson(p))
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.server.route('/api/preview/tide')
def _r_preview_tide():
    try:
        p = _safe_repo_path(request.args.get('path', ''))
        return jsonify(_preview_tide(p))
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.server.route('/api/preview/bathy.arrow')
def _r_preview_bathy():
    try:
        p = _safe_repo_path(request.args.get('path', ''))
        b = _preview_bathy_arrow(p)
        if b is None:
            return jsonify(error='dfs2 grid preview not implemented'), 400
        return _bytes_response(b)
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.server.route('/api/pipeline/status')
def _r_pipeline_status():
    with _pipeline_lock:
        s = dict(PIPELINE_STATE)
    if s['started_at']:
        end = s['finished_at'] or time.time()
        s['elapsed_s'] = round(end - s['started_at'], 1)
    else:
        s['elapsed_s'] = 0
    return jsonify(s)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def _opt_list(items: list[str]) -> list[dict]:
    return [{'label': Path(p).name, 'value': p} for p in items]


_default_ais = next((p for p in EXAMPLE_FILES['ais'] if 'AIS_2563' in p),
                    EXAMPLE_FILES['ais'][0] if EXAMPLE_FILES['ais'] else None)
_default_bathy = next((p for p in EXAMPLE_FILES['bathymetry'] if '61803960_WestCoast' in p),
                      EXAMPLE_FILES['bathymetry'][0] if EXAMPLE_FILES['bathymetry'] else None)
_default_coast = next((p for p in EXAMPLE_FILES['coastline'] if 'Coast_P1' in p),
                      EXAMPLE_FILES['coastline'][0] if EXAMPLE_FILES['coastline'] else None)
_default_tide = next((p for p in EXAMPLE_FILES['tide'] if '_6min' in p),
                     EXAMPLE_FILES['tide'][0] if EXAMPLE_FILES['tide'] else None)


def _picker_with_preview(label: str, dropdown_id: str, preview_id: str,
                         info_id: str, options: list, value, clearable=False):
    return html.Div([
        html.Label(label),
        html.Div([
            dcc.Dropdown(id=dropdown_id, options=_opt_list(options),
                         value=value, clearable=clearable),
            html.Div(
                dcc.Checklist(id=preview_id, options=[{'label': 'preview', 'value': '1'}],
                              value=[]),
                className='preview-box',
            ),
        ], className='row-with-preview'),
        html.Div(id=info_id, className='preview-info'),
    ])


app.layout = html.Div([
    html.Div([
        html.Span(id='cnt-vessels', children=f'vessels {len(df_vessels):,}'),
        ' | ', html.Span(id='cnt-segs',    children=f'segments {len(seg_meta):,}'),
        ' | ', html.Span(id='cnt-waves',   children=f'waves {len(df_waves):,}'),
        ' | ', html.Span(id='status', children='loading...'),
        ' | ', html.Span(id='click-info', style={'fontWeight': 'bold'}),
    ], id='status-banner'),

    html.Div([
        html.H4('Run pipeline'),
        _picker_with_preview('AIS CSV', 'sel-ais', 'pv-ais', 'pv-ais-info',
                             EXAMPLE_FILES['ais'], _default_ais),
        _picker_with_preview('Bathymetry', 'sel-bathy', 'pv-bathy', 'pv-bathy-info',
                             EXAMPLE_FILES['bathymetry'], _default_bathy),
        _picker_with_preview('Coastline', 'sel-coast', 'pv-coast', 'pv-coast-info',
                             EXAMPLE_FILES['coastline'], _default_coast),
        _picker_with_preview('Tide DFS0 (optional)', 'sel-tide', 'pv-tide', 'pv-tide-info',
                             EXAMPLE_FILES['tide'], _default_tide, clearable=True),
        html.Label('Cb method (Le determination)'),
        dcc.Dropdown(id='sel-cb',
                     options=[{'label': m, 'value': m} for m in CB_METHODS],
                     value='L_Le', clearable=False),
        html.Label('Wave formula'),
        dcc.Dropdown(id='sel-formula',
                     options=[{'label': f, 'value': f} for f in WAVE_FORMULAE],
                     value='kriebel', clearable=False),
        html.Div([
            html.Button('1. Filter AIS',     id='btn-filter', n_clicks=0),
            html.Button('2. Calculate waves', id='btn-waves',  n_clicks=0),
        ], className='row-buttons'),
        html.Hr(),
        html.Div('Progress', style={'fontWeight': 'bold'}),
        html.Pre(id='progress-log', children='(idle)'),
        html.Div(id='progress-elapsed-side',
                 style={'fontSize': '11px', 'color': '#666', 'marginTop': '4px'}),
    ], id='sidebar'),

    html.Div(id='deck-container'),
    dcc.Interval(id='boot', max_intervals=1, interval=200),
    dcc.Interval(id='poll', interval=400, disabled=True),
    dcc.Store(id='_init'),
    dcc.Store(id='_wave_version', data=0),
    dcc.Store(id='_track_version', data=0),
    # Preview state Stores: each carries {visible: bool, path: str|None}
    dcc.Store(id='_pv_ais',   data={'visible': False, 'path': None}),
    dcc.Store(id='_pv_bathy', data={'visible': False, 'path': None}),
    dcc.Store(id='_pv_coast', data={'visible': False, 'path': None}),
    dcc.Store(id='_pv_tide',  data={'visible': False, 'path': None}),
])


# ---------------------------------------------------------------------------
# Server-side callbacks: run buttons + polling
# ---------------------------------------------------------------------------
def _build_config(ais, bathy, coast, tide, formula, cb_method) -> dict:
    cfg = {
        'ais': {'raw_csv': ais},
        'vessel': {'cb_method': cb_method} if cb_method else {},
        'bathymetry': {'source': bathy},
        'coastline': {'shapefile': coast},
        'wave': {'formula': formula} if formula else {},
        'output': {'directory': 'output/', 'save_stage_csv': True},
    }
    if tide:
        cfg['bathymetry']['tide_dfs0'] = tide
        cfg['bathymetry']['tide_item'] = 'Predicted tidal elevation'
    return cfg


def _kick(config_dict, stages, label):
    with _pipeline_lock:
        if PIPELINE_STATE['running']:
            return False
        PIPELINE_STATE.update({
            'running': True, 'log': [], 'started_at': time.time(),
            'finished_at': None, 'error': None,
        })
    threading.Thread(target=_pipeline_thread,
                     args=(config_dict, stages, label), daemon=True).start()
    return True


@app.callback(
    Output('poll', 'disabled', allow_duplicate=True),
    Output('btn-filter', 'disabled', allow_duplicate=True),
    Output('btn-waves',  'disabled', allow_duplicate=True),
    Input('btn-filter', 'n_clicks'),
    State('sel-ais', 'value'), State('sel-bathy', 'value'),
    State('sel-coast', 'value'), State('sel-tide', 'value'),
    State('sel-formula', 'value'), State('sel-cb', 'value'),
    prevent_initial_call=True,
)
def kick_filter(n, ais, bathy, coast, tide, formula, cb):
    if not n or not ais or not coast:
        return no_update, no_update, no_update
    cfg = _build_config(ais, bathy, coast, tide, formula, cb)
    if _kick(cfg, ['filter'], 'filter'):
        return False, True, True
    return no_update, no_update, no_update


@app.callback(
    Output('poll', 'disabled', allow_duplicate=True),
    Output('btn-filter', 'disabled', allow_duplicate=True),
    Output('btn-waves',  'disabled', allow_duplicate=True),
    Input('btn-waves', 'n_clicks'),
    State('sel-ais', 'value'), State('sel-bathy', 'value'),
    State('sel-coast', 'value'), State('sel-tide', 'value'),
    State('sel-formula', 'value'), State('sel-cb', 'value'),
    prevent_initial_call=True,
)
def kick_waves(n, ais, bathy, coast, tide, formula, cb):
    if not n or not ais or not coast or not bathy:
        return no_update, no_update, no_update
    # If filter hasn't been run, run all stages so users can also do a one-shot run.
    stages = ['depth', 'vessel', 'wave_impact'] if 'df_filtered' in LAST_RESULTS \
             else ['filter', 'depth', 'vessel', 'wave_impact']
    cfg = _build_config(ais, bathy, coast, tide, formula, cb)
    if _kick(cfg, stages, 'waves'):
        return False, True, True
    return no_update, no_update, no_update


@app.callback(
    Output('progress-log', 'children'),
    Output('progress-elapsed-side', 'children'),
    Output('poll', 'disabled'),
    Output('btn-filter', 'disabled'),
    Output('btn-waves',  'disabled'),
    Output('_wave_version',  'data'),
    Output('_track_version', 'data'),
    Output('cnt-waves',   'children'),
    Output('cnt-segs',    'children'),
    Output('cnt-vessels', 'children'),
    Input('poll', 'n_intervals'),
    State('_wave_version', 'data'), State('_track_version', 'data'),
    prevent_initial_call=True,
)
def tick(_, prev_wave_v, prev_track_v):
    with _pipeline_lock:
        s = dict(PIPELINE_STATE)
    log_lines = list(s['log'][-300:])
    log_text = '\n'.join(log_lines) or '(no output yet)'
    elapsed = f"elapsed: {s.get('elapsed_s', 0):.1f}s"
    counts = (f'waves {len(df_waves):,}', f'segments {len(seg_meta):,}',
              f'vessels {len(df_vessels):,}')
    if s['error']:
        return (
            f"{log_text}\n\nERROR: {s['error']}",
            elapsed, True, False, False,
            prev_wave_v, prev_track_v, *counts,
        )
    if s['running']:
        return (
            log_text, elapsed, False, True, True,
            prev_wave_v, prev_track_v, no_update, no_update, no_update,
        )
    # Finished — push fresh versions so client refetches Arrows.
    return (
        log_text, elapsed, True, False, False,
        s['wave_version'], s['track_version'], *counts,
    )


# ---------------------------------------------------------------------------
# Preview callbacks: maintain Stores fed from dropdown + checkbox state
# ---------------------------------------------------------------------------
def _make_pv_callback(store_id, sel_id, pv_id):
    @app.callback(
        Output(store_id, 'data'),
        Input(sel_id, 'value'),
        Input(pv_id, 'value'),
        prevent_initial_call=False,
    )
    def _pv(path, pv_val):
        return {'visible': bool(pv_val), 'path': path}
    return _pv


_make_pv_callback('_pv_ais',   'sel-ais',   'pv-ais')
_make_pv_callback('_pv_bathy', 'sel-bathy', 'pv-bathy')
_make_pv_callback('_pv_coast', 'sel-coast', 'pv-coast')
_make_pv_callback('_pv_tide',  'sel-tide',  'pv-tide')


# ---------------------------------------------------------------------------
# Clientside JS
# ---------------------------------------------------------------------------
INIT_JS = r"""
function(n) {
    if (!n || window.__deck_initialized) return window.dash_clientside.no_update;
    const container = document.getElementById('deck-container');
    if (!container) return window.dash_clientside.no_update;
    if (typeof deck === 'undefined') {
        document.getElementById('status').textContent = 'waiting for deck.gl...';
        setTimeout(() => { window.__deck_initialized = false; }, 100);
        return window.dash_clientside.no_update;
    }
    window.__deck_initialized = true;

    // ---------- Tooltip ----------
    const tip = document.createElement('div');
    tip.id = 'tooltip';
    document.body.appendChild(tip);
    const showTip = (x, y, html) => {
        tip.innerHTML = html;
        tip.style.left = (x + 14) + 'px';
        tip.style.top = (y + 14) + 'px';
        tip.style.display = 'block';
    };
    const hideTip = () => { tip.style.display = 'none'; };

    // ---------- Reusable progress overlay (used post-pipeline for Arrow transfer) ----------
    const fmt = (b) => b > 1e6 ? (b/1e6).toFixed(1)+' MB' : (b/1e3).toFixed(0)+' KB';
    async function fetchAssetsWithProgress(assets, title) {
        const overlay = document.createElement('div');
        overlay.id = 'progress-overlay';
        overlay.innerHTML = `
            <div id="progress-title">${title}</div>
            <div id="progress-rows"></div>
            <div id="progress-bar"><div id="progress-fill"></div></div>
            <div id="progress-elapsed">0.0 s</div>
        `;
        document.body.appendChild(overlay);
        const rowsEl = overlay.querySelector('#progress-rows');
        const fillEl = overlay.querySelector('#progress-fill');
        const elapsedEl = overlay.querySelector('#progress-elapsed');
        const state = {};
        assets.forEach(a => {
            state[a.key] = { received: 0, total: 0, done: false };
            const row = document.createElement('div');
            row.className = 'progress-row';
            row.innerHTML = `<span class="name">${a.label}</span><span class="pct" data-key="${a.key}">queued</span>`;
            rowsEl.appendChild(row);
        });
        const t0 = performance.now();
        const tick = setInterval(() => {
            elapsedEl.textContent = ((performance.now() - t0) / 1000).toFixed(1) + ' s';
        }, 100);
        const render = () => {
            let totalRecv = 0, totalSize = 0;
            assets.forEach(a => {
                const s = state[a.key];
                const cell = rowsEl.querySelector(`[data-key="${a.key}"]`);
                if (cell) {
                    if (s.done) cell.textContent = fmt(s.received) + ' done';
                    else if (s.total) cell.textContent = `${fmt(s.received)} / ${fmt(s.total)}`;
                    else if (s.received) cell.textContent = `${fmt(s.received)} ...`;
                    else cell.textContent = 'queued';
                }
                totalRecv += s.received;
                if (s.total) totalSize += s.total;
            });
            if (totalSize > 0) {
                fillEl.style.width = Math.min(100, (totalRecv / totalSize) * 100).toFixed(1) + '%';
            }
        };
        async function fetchOne(asset) {
            const r = await fetch(asset.url);
            state[asset.key].total = +r.headers.get('content-length') || 0;
            const reader = r.body.getReader();
            const chunks = [];
            let received = 0;
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
                received += value.length;
                state[asset.key].received = received;
                render();
            }
            const buf = new Uint8Array(received);
            let off = 0;
            for (const c of chunks) { buf.set(c, off); off += c.length; }
            state[asset.key].done = true;
            render();
            return buf;
        }
        try {
            const buffers = await Promise.all(assets.map(fetchOne));
            return buffers;
        } finally {
            clearInterval(tick);
            overlay.remove();
        }
    }
    window.__fetchAssetsWithProgress = fetchAssetsWithProgress;

    // ---------- Helpers ----------
    const debounce = (fn, ms) => {
        let h = null;
        return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); };
    };
    const waitArrow = window.tableFromIPC
        ? Promise.resolve()
        : new Promise(r => window.addEventListener('arrow-ready', r, { once: true }));

    const status = document.getElementById('status');
    status.textContent = 'initialising...';

    waitArrow.then(() => {
        // ---- Track data (populated after Step 1) ----
        let cPos = new Float32Array(0);
        let startIndices = new Int32Array([0]);
        let tMMSI = new Int32Array(0);
        let tSeg  = new Int32Array(0);
        let tN    = new Int32Array(0);
        let segLookup = new Map();
        const initTrackArrays = (cT, mT, oT) => {
            const cLon = cT.getChild('lon').toArray();
            const cLat = cT.getChild('lat').toArray();
            cPos = new Float32Array(cLon.length * 2);
            for (let i = 0; i < cLon.length; i++) { cPos[i*2]=cLon[i]; cPos[i*2+1]=cLat[i]; }
            startIndices = oT.getChild('offset').toArray();
            tMMSI = mT.getChild('mmsi').toArray();
            tSeg  = mT.getChild('segment_id').toArray();
            tN    = mT.getChild('n_points').toArray();
            segLookup = new Map();
            for (let i = 0; i < tMMSI.length; i++) segLookup.set(`${tMMSI[i]}|${tSeg[i]}`, i);
        };

        // ---- Wave data (populated after Step 2) ----
        let wPos = new Float32Array(0);
        let wMMSI = new Int32Array(0);
        let wH = new Float32Array(0), wTp = new Float32Array(0);
        let wSide = (_i) => '', wTime = (_i) => '';
        let wSog = new Float32Array(0), wCog = new Float32Array(0);
        let wDraught = new Float32Array(0), wLen = new Float32Array(0);
        let wWid = new Float32Array(0), wDist = new Float32Array(0);
        let wVesselLon = new Float32Array(0), wVesselLat = new Float32Array(0);
        let wSegId = new Int32Array(0);
        function rebuildWaveArrays(wT) {
            const wLon = wT.getChild('ShLongitude').toArray();
            const wLat = wT.getChild('ShLatitude').toArray();
            wMMSI = wT.getChild('MMSI').toArray();
            wH    = wT.getChild('WaveHeight').toArray();
            wTp   = wT.getChild('WavePeriod').toArray();
            const sideCol = wT.getChild('Side');
            wSide = (i) => sideCol ? sideCol.get(i) : '';
            const timeCol = wT.getChild('DateTime');
            wTime = (i) => timeCol ? timeCol.get(i) : '';
            const get = (n) => wT.getChild(n) ? wT.getChild(n).toArray() : new Float32Array(wLon.length);
            wSog = get('SOG'); wCog = get('VesselCOG'); wDraught = get('VesselDraught');
            wLen = get('VesselLength'); wWid = get('VesselWidth'); wDist = get('DistLoc_km');
            wVesselLon = get('VesselLongitude'); wVesselLat = get('VesselLatitude');
            wSegId = wT.getChild('segment_id') ? wT.getChild('segment_id').toArray() : new Int32Array(wLon.length);
            wPos = new Float32Array(wLon.length * 2);
            for (let i = 0; i < wLon.length; i++) { wPos[i*2]=wLon[i]; wPos[i*2+1]=wLat[i]; }
        }

        // ---- Preview state (set by clientside callbacks) ----
        window.__previews = { ais: null, bathy: null, coast: null, tide: null };
        // Singapore-wide initial view: bbox roughly 103.55–104.05 / 1.20–1.50.
        window._currentZoom = 10;
        window._hoveredWave = null;
        // Track whether we have data so layers gate themselves.
        window.__hasTracks = false;
        window.__hasWaves = false;

        const buildLayers = (zoom, hoveredIdx) => {
            const useRaster = zoom < """ + str(ZOOM_RASTER_THRESHOLD) + r""";
            const layers = [
                new deck.TileLayer({
                    id: 'basemap',
                    data: 'https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}@2x.png',
                    minZoom: 0, maxZoom: 19, tileSize: 256,
                    renderSubLayers: (props) => {
                        const bbox = props.tile.boundingBox || props.tile.bbox;
                        const [[west, south], [east, north]] = bbox;
                        return new deck.BitmapLayer(props, {
                            data: null, image: props.data, bounds: [west, south, east, north],
                        });
                    },
                }),
            ];

            if (window.__hasTracks && useRaster) {
                layers.push(new deck.BitmapLayer({
                    id: 'raster',
                    image: '/api/raster.png',
                    bounds: [""" + f"{RASTER_AOI[0]}, {RASTER_AOI[1]}, {RASTER_AOI[2]}, {RASTER_AOI[3]}" + r"""],
                    pickable: false, opacity: 0.65,
                }));
            }
            if (window.__hasTracks && !useRaster && tMMSI.length > 0) {
                layers.push(new deck.PathLayer({
                    id: 'tracks',
                    data: { length: tMMSI.length, startIndices,
                            attributes: { getPath: { value: cPos, size: 2 } } },
                    pickable: true, _pathType: 'open',
                    getColor: [200, 30, 0, 110],
                    getWidth: 1, widthUnits: 'pixels', widthMinPixels: 1,
                    onHover: ({x, y, index}) => {
                        if (index < 0) { hideTip(); return; }
                        showTip(x, y, `<b>TRACK</b><br>MMSI: ${tMMSI[index]}<br>seg: ${tSeg[index]}<br>n: ${tN[index]}`);
                    },
                }));
            }
            if (window.__hasWaves && wMMSI.length > 0) {
                layers.push(new deck.ScatterplotLayer({
                    id: 'waves',
                    data: { length: wMMSI.length,
                            attributes: { getPosition: { value: wPos, size: 2 } } },
                    pickable: true, stroked: true,
                    getRadius: 30, radiusUnits: 'meters',
                    radiusMinPixels: 4, radiusMaxPixels: 12,
                    getFillColor: [255, 140, 0, 230],
                    getLineColor: [255, 255, 255, 220],
                    lineWidthMinPixels: 1,
                    onHover: ({x, y, index}) => {
                        if (index < 0) {
                            hideTip();
                            if (window._hoveredWave !== null) {
                                window._hoveredWave = null;
                                window.deckInstance.setProps({ layers: buildLayers(window._currentZoom, null) });
                            }
                            return;
                        }
                        const f = (v, d) => (v == null || isNaN(v)) ? '?' : v.toFixed(d);
                        showTip(x, y,
                            `<b>WAVE -> MMSI ${wMMSI[index]}</b><br>` +
                            `<b>H</b>: ${f(wH[index], 3)} m &nbsp;<b>T</b>: ${f(wTp[index], 2)} s &nbsp;<b>Side</b>: ${wSide(index)}<br>` +
                            `<b>SOG</b>: ${f(wSog[index], 1)} kn &nbsp;<b>COG</b>: ${f(wCog[index], 0)}°<br>` +
                            `<b>L×W×T</b>: ${f(wLen[index], 0)}×${f(wWid[index], 0)}×${f(wDraught[index], 1)} m<br>` +
                            `<b>shore dist</b>: ${f((wDist[index]||0)*1000, 0)} m<br><b>${wTime(index)}</b>`
                        );
                        if (window._hoveredWave !== index) {
                            window._hoveredWave = index;
                            window.deckInstance.setProps({ layers: buildLayers(window._currentZoom, index) });
                        }
                    },
                }));
            }

            // Wave hover highlights (cyan)
            if (window.__hasWaves && hoveredIdx != null && hoveredIdx >= 0 && hoveredIdx < wMMSI.length) {
                const segIdx = segLookup.get(`${wMMSI[hoveredIdx]}|${wSegId[hoveredIdx]}`);
                if (segIdx != null && segIdx < startIndices.length - 1) {
                    const segStart = startIndices[segIdx];
                    const segEnd   = startIndices[segIdx + 1];
                    layers.push(new deck.PathLayer({
                        id: 'track-highlight',
                        data: {
                            length: 1,
                            startIndices: [0, segEnd - segStart],
                            attributes: { getPath: { value: cPos.subarray(segStart * 2, segEnd * 2), size: 2 } },
                        },
                        _pathType: 'open',
                        getColor: [0, 220, 255, 240],
                        getWidth: 4, widthUnits: 'pixels', widthMinPixels: 3,
                        pickable: false,
                    }));
                }
                const vp = [wVesselLon[hoveredIdx], wVesselLat[hoveredIdx]];
                const wp = [wPos[hoveredIdx*2], wPos[hoveredIdx*2+1]];
                if (!isNaN(vp[0])) {
                    layers.push(new deck.ScatterplotLayer({
                        id: 'vessel-highlight',
                        data: [{ position: vp }],
                        getPosition: d => d.position,
                        getRadius: 100, radiusUnits: 'meters',
                        radiusMinPixels: 8, radiusMaxPixels: 14,
                        stroked: true, filled: false,
                        getLineColor: [0, 220, 255, 255],
                        lineWidthMinPixels: 3, pickable: false,
                    }));
                    layers.push(new deck.LineLayer({
                        id: 'wave-connector',
                        data: [{ from: vp, to: wp }],
                        getSourcePosition: d => d.from,
                        getTargetPosition: d => d.to,
                        getColor: [0, 220, 255, 200],
                        getWidth: 2, widthMinPixels: 1, pickable: false,
                    }));
                }
            }

            // ---- Preview layers ----
            const pv = window.__previews;
            if (pv.bathy && pv.bathy.lon && pv.bathy.lon.length > 0) {
                layers.push(new deck.LineLayer({
                    id: 'pv-bathy',
                    data: { length: pv.bathy.lon.length / 2, attributes: {
                        getSourcePosition: { value: pv.bathy.src, size: 2 },
                        getTargetPosition: { value: pv.bathy.tgt, size: 2 },
                    }},
                    getColor: [0, 130, 0, 110],
                    getWidth: 1, widthMinPixels: 1, pickable: false,
                }));
            }
            if (pv.coast && pv.coast.paths && pv.coast.paths.length > 0) {
                layers.push(new deck.PathLayer({
                    id: 'pv-coast',
                    data: pv.coast.paths,
                    getPath: d => d, _pathType: 'open',
                    getColor: [0, 80, 200, 220],
                    getWidth: 2, widthMinPixels: 2,
                    pickable: false,
                }));
            }
            if (pv.ais && pv.ais.pos && pv.ais.pos.length > 0) {
                layers.push(new deck.ScatterplotLayer({
                    id: 'pv-ais',
                    data: { length: pv.ais.pos.length / 2,
                            attributes: { getPosition: { value: pv.ais.pos, size: 2 } } },
                    getRadius: 4, radiusUnits: 'pixels', radiusMinPixels: 2,
                    getFillColor: [50, 150, 255, 200],
                    pickable: false,
                }));
            }
            return layers;
        };
        window.__buildLayers = buildLayers;

        // Singapore-wide initial view (covers ~103.55-104.05 / 1.20-1.50).
        const initialZoom = 10;
        window.deckInstance = new deck.Deck({
            parent: container,
            width: '100%', height: '100%',
            initialViewState: { longitude: 103.82, latitude: 1.32, zoom: initialZoom, pitch: 0, bearing: 0 },
            controller: true,
            layers: buildLayers(initialZoom, null),
            onClick: ({layer, index}) => {
                if (!layer || index < 0) {
                    document.getElementById('click-info').textContent = '';
                    return;
                }
                let msg = `${layer.id}#${index}`;
                if (layer.id === 'waves') msg = `wave MMSI=${wMMSI[index]} H=${wH[index].toFixed(3)}`;
                else if (layer.id === 'tracks') msg = `track MMSI=${tMMSI[index]} seg=${tSeg[index]}`;
                document.getElementById('click-info').textContent = '| ' + msg;
            },
            onViewStateChange: (params) => {
                window._currentZoom = params.viewState.zoom;
                rebuildOnView(params.viewState.zoom);
                return params.viewState;
            },
        });
        const rebuildOnView = debounce((z) => {
            window.deckInstance.setProps({ layers: buildLayers(z, window._hoveredWave) });
            status.textContent = `zoom=${z.toFixed(1)}`;
        }, 250);
        window.__rebuild = () => window.deckInstance.setProps({ layers: buildLayers(window._currentZoom, window._hoveredWave) });

        // Post-pipeline refresh hooks: show the same progress overlay as before, then rebuild layers.
        window.__refreshWaveLayer = async (version) => {
            const [buf] = await fetchAssetsWithProgress([
                { key: 'waves', url: `/api/waves.arrow?v=${version}`, label: 'wave impacts' },
            ], 'Loading wave impacts');
            rebuildWaveArrays(window.tableFromIPC(buf));
            window.__hasWaves = wMMSI.length > 0;
            window._hoveredWave = null;
            window.__rebuild();
        };
        window.__refreshTrackCaches = async (version) => {
            const [c, m, o] = await fetchAssetsWithProgress([
                { key: 'track_coords',  url: `/api/track_coords.arrow?v=${version}`,  label: 'track coords' },
                { key: 'track_meta',    url: `/api/track_meta.arrow?v=${version}`,    label: 'track metadata' },
                { key: 'track_offsets', url: `/api/track_offsets.arrow?v=${version}`, label: 'track offsets' },
            ], 'Loading vessel tracks');
            initTrackArrays(
                window.tableFromIPC(c),
                window.tableFromIPC(m),
                window.tableFromIPC(o),
            );
            window.__hasTracks = tMMSI.length > 0;
            window.__rebuild();
        };

        // Preview setters — invoked by Dash clientside callbacks below.
        window.__setPreviewAis = async (state) => {
            if (!state || !state.visible || !state.path) {
                window.__previews.ais = null; window.__rebuild(); return null;
            }
            try {
                const buf = await fetch('/api/preview/ais.arrow?path=' + encodeURIComponent(state.path))
                    .then(r => r.arrayBuffer());
                const t = window.tableFromIPC(new Uint8Array(buf));
                const lon = t.getChild('longitude').toArray();
                const lat = t.getChild('latitude').toArray();
                const pos = new Float32Array(lon.length * 2);
                for (let i = 0; i < lon.length; i++) { pos[i*2]=lon[i]; pos[i*2+1]=lat[i]; }
                window.__previews.ais = { pos };
                // Fit camera to bbox
                const bbResp = await fetch('/api/preview/ais.bbox?path=' + encodeURIComponent(state.path)).then(r => r.json());
                if (bbResp.bbox) {
                    const [w, s, e, n] = bbResp.bbox;
                    const lonC = (w + e) / 2, latC = (s + n) / 2;
                    const span = Math.max(e - w, n - s);
                    const z = Math.max(8, Math.min(15, 10 - Math.log2(span)));
                    window.deckInstance.setProps({ initialViewState: { longitude: lonC, latitude: latC, zoom: z, pitch: 0, bearing: 0 } });
                    window._currentZoom = z;
                }
                window.__rebuild();
                return `${lon.length.toLocaleString()} preview points`;
            } catch (e) {
                window.__previews.ais = null; window.__rebuild();
                return 'ERROR: ' + e.message;
            }
        };
        window.__setPreviewBathy = async (state) => {
            if (!state || !state.visible || !state.path) {
                window.__previews.bathy = null; window.__rebuild(); return null;
            }
            try {
                const buf = await fetch('/api/preview/bathy.arrow?path=' + encodeURIComponent(state.path))
                    .then(r => { if (!r.ok) return r.json().then(j => Promise.reject(new Error(j.error || 'preview failed'))); return r.arrayBuffer(); });
                const t = window.tableFromIPC(new Uint8Array(buf));
                const lon = t.getChild('lon').toArray();
                const lat = t.getChild('lat').toArray();
                // Edges are pairs of consecutive rows; split into source/target.
                const nEdges = Math.floor(lon.length / 2);
                const src = new Float32Array(nEdges * 2);
                const tgt = new Float32Array(nEdges * 2);
                for (let i = 0; i < nEdges; i++) {
                    src[i*2] = lon[i*2]; src[i*2+1] = lat[i*2];
                    tgt[i*2] = lon[i*2+1]; tgt[i*2+1] = lat[i*2+1];
                }
                window.__previews.bathy = { lon, src, tgt };
                window.__rebuild();
                return `${nEdges.toLocaleString()} mesh edges`;
            } catch (e) {
                window.__previews.bathy = null; window.__rebuild();
                return 'ERROR: ' + e.message;
            }
        };
        window.__setPreviewCoast = async (state) => {
            if (!state || !state.visible || !state.path) {
                window.__previews.coast = null; window.__rebuild(); return null;
            }
            try {
                const gj = await fetch('/api/preview/coast.geojson?path=' + encodeURIComponent(state.path)).then(r => r.json());
                if (gj.error) throw new Error(gj.error);
                const paths = [];
                const flatten = (geom) => {
                    if (!geom) return;
                    const t = geom.type, c = geom.coordinates;
                    if (t === 'LineString')      { paths.push(c); }
                    else if (t === 'Polygon')    { c.forEach(ring => paths.push(ring)); }
                    else if (t === 'MultiLineString' || t === 'MultiPolygon') {
                        c.forEach(part => flatten({ type: t === 'MultiLineString' ? 'LineString' : 'Polygon', coordinates: part }));
                    } else if (t === 'GeometryCollection') {
                        geom.geometries.forEach(flatten);
                    }
                };
                gj.features.forEach(f => flatten(f.geometry));
                window.__previews.coast = { paths };
                window.__rebuild();
                return `${gj.features.length} feature(s), ${paths.length} ring(s)`;
            } catch (e) {
                window.__previews.coast = null; window.__rebuild();
                return 'ERROR: ' + e.message;
            }
        };
        window.__setPreviewTide = async (state) => {
            if (!state || !state.visible || !state.path) return null;
            try {
                const j = await fetch('/api/preview/tide?path=' + encodeURIComponent(state.path)).then(r => r.json());
                if (j.error) throw new Error(j.error);
                const lines = [];
                lines.push(`${j.n_steps} steps, ${j.time_min} -> ${j.time_max}`);
                j.items.forEach(it => {
                    const r = (it.value_min != null && it.value_max != null)
                        ? ` [${it.value_min.toFixed(2)} .. ${it.value_max.toFixed(2)}${it.unit ? ' ' + it.unit : ''}]`
                        : '';
                    lines.push(`- ${it.name}${r}`);
                });
                return lines.join('\n');
            } catch (e) { return 'ERROR: ' + e.message; }
        };

        status.textContent = `ready - zoom=${initialZoom} (Singapore)`;
    }).catch(err => {
        document.getElementById('status').textContent = 'ERROR: ' + err.message;
        console.error(err);
    });

    return 'init';
}
"""

app.clientside_callback(INIT_JS, Output('_init', 'data'), Input('boot', 'n_intervals'))


# Wave / track refresh after a pipeline run.
WAVE_RELOAD_JS = r"""
async function(version) {
    if (!version || version === window.__lastWaveVersion) return window.dash_clientside.no_update;
    window.__lastWaveVersion = version;
    if (typeof window.__refreshWaveLayer === 'function') {
        try { await window.__refreshWaveLayer(version); } catch (e) { console.error(e); }
    }
    return null;
}
"""

TRACK_RELOAD_JS = r"""
async function(version) {
    if (!version || version === window.__lastTrackVersion) return window.dash_clientside.no_update;
    window.__lastTrackVersion = version;
    if (typeof window.__refreshTrackCaches === 'function') {
        try { await window.__refreshTrackCaches(version); } catch (e) { console.error(e); }
    }
    return null;
}
"""

app.clientside_callback(WAVE_RELOAD_JS,
    Output('_init', 'data', allow_duplicate=True),
    Input('_wave_version', 'data'),
    prevent_initial_call=True)

app.clientside_callback(TRACK_RELOAD_JS,
    Output('_init', 'data', allow_duplicate=True),
    Input('_track_version', 'data'),
    prevent_initial_call=True)


# ---- Preview clientside hooks: each updates a preview-info span and the deck.gl layers
def _make_preview_clientside(store_id, info_id, fn_name):
    js = r"""
async function(state) {
    if (typeof window.""" + fn_name + r""" !== 'function') return '';
    const result = await window.""" + fn_name + r"""(state);
    return result || '';
}
"""
    app.clientside_callback(js, Output(info_id, 'children'),
                            Input(store_id, 'data'),
                            prevent_initial_call=True)


_make_preview_clientside('_pv_ais',   'pv-ais-info',   '__setPreviewAis')
_make_preview_clientside('_pv_bathy', 'pv-bathy-info', '__setPreviewBathy')
_make_preview_clientside('_pv_coast', 'pv-coast-info', '__setPreviewCoast')
_make_preview_clientside('_pv_tide',  'pv-tide-info',  '__setPreviewTide')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _lan_ips() -> list[str]:
    import socket
    ips: list[str] = []
    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None, family=socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith(('127.', '169.254.')) and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80)); ip = s.getsockname()[0]; s.close()
        if ip and not ip.startswith(('127.', '169.254.')) and ip not in ips:
            ips.insert(0, ip)
    except OSError:
        pass
    return ips


if __name__ == '__main__':
    print(f'\n=== aiswakepy deck.gl spike ===')
    print(f'Local       : http://127.0.0.1:{PORT}')
    for ip in _lan_ips():
        print(f'LAN         : http://{ip}:{PORT}')
    print(f'(bind 0.0.0.0:{PORT} - accessible from any host that can reach this machine)\n')
    app.run(debug=False, host='0.0.0.0', port=PORT, threaded=True)
