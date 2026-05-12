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
    'log': [],            # committed lines (one per print '\n')
    'live': '',           # current in-progress spinner line (replaced on each '\r')
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
    """sys.stdout shim with proper carriage-return handling for in-place spinners.

    Char-by-char state machine:
      - '\\r' resets the in-progress buffer (cursor return — Spinner is about
        to overwrite); does NOT commit to log.
      - '\\n' commits the in-progress buffer as a single log line.
      - any other char appends to the buffer.
    The in-progress buffer is also surfaced as PIPELINE_STATE['live'] after
    every write call, so the UI can render it as a single replaceable line
    below the committed log — this is the "spinning in place" effect.
    """

    def __init__(self, original):
        self._orig = original
        self._buf = ''

    def write(self, s: str) -> int:
        new_lines: list[str] = []
        for ch in s:
            if ch == '\r':
                self._buf = ''
            elif ch == '\n':
                if self._buf.strip():
                    new_lines.append(self._buf)
                self._buf = ''
            else:
                self._buf += ch
        live = self._buf if self._buf.strip() else ''
        with _pipeline_lock:
            if new_lines:
                PIPELINE_STATE['log'].extend(new_lines)
            PIPELINE_STATE['live'] = live
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

        # ---- Filter freshness check ----
        # If wave-side stages are requested but no df_filtered is in memory,
        # try the on-disk cache; if that's stale (AIS file newer than the
        # cached filter CSV) or missing, prepend 'filter' to the stage list.
        wave_stages = {'depth', 'vessel', 'wave_impact'}
        requesting_waves = bool(set(stages) & wave_stages)
        if requesting_waves and 'df_filtered' not in seed:
            ais_path = Path(cfg.ais.raw_csv)
            ais_stem = ais_path.stem
            filtered_path = Path(cfg.output.directory) / f'{ais_stem}_01_filtered.csv'
            if filtered_path.exists() and (
                filtered_path.stat().st_mtime > ais_path.stat().st_mtime
            ):
                print(f'Loading cached filter output: {filtered_path.name}')
                df_cached = pd.read_csv(filtered_path)
                if 'obstime' in df_cached.columns:
                    df_cached['obstime'] = pd.to_datetime(df_cached['obstime'])
                seed['df_filtered'] = df_cached
                LAST_RESULTS['df_filtered'] = df_cached
                print(f'  -> {len(df_cached):,} rows loaded from disk cache')
            else:
                if filtered_path.exists():
                    print(f'Cached filter is older than AIS file - running filter first')
                else:
                    print(f'No cached filter found - running filter first')
                if 'filter' not in stages:
                    stages = ['filter'] + list(stages)

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
            PIPELINE_STATE['live'] = ''


# ---------------------------------------------------------------------------
# Preview helpers
# ---------------------------------------------------------------------------
def _preview_ais_arrow(path: Path) -> bytes:
    """Return Arrow IPC of all (lon,lat) rows from the AIS CSV (no subsampling).

    Note: large AIS files (millions of rows) produce tens of MB of Arrow IPC.
    The client UI gates this behind an explicit 'Import' button so the user
    chooses when to pay the cost.
    """
    df = pd.read_csv(path, usecols=['longitude', 'latitude'])
    df = df.dropna(subset=['longitude', 'latitude'])
    df = df.astype({'longitude': 'float32', 'latitude': 'float32'})
    return _ipc(pa.Table.from_pandas(df, preserve_index=False))


def _preview_ais_bbox(path: Path) -> tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) — used to fit the camera."""
    df = pd.read_csv(path, usecols=['longitude', 'latitude'])
    return (float(df['longitude'].min()), float(df['latitude'].min()),
            float(df['longitude'].max()), float(df['latitude'].max()))


def _preview_coast_geojson(path: Path) -> dict:
    """Load a shapefile with fiona (no shapely, no simplification) and return
    a GeoJSON FeatureCollection in WGS84. Reprojects with pyproj if the source
    CRS is not already lon/lat.

    Earlier versions used geopandas + shapely.simplify(0.0001) which collapsed
    small rectangles into triangles (tolerance ~10 m, comparable to edge length).
    """
    import fiona
    from pyproj import CRS, Transformer

    with fiona.open(str(path)) as src:
        src_crs = src.crs
        transformer = None
        try:
            if src_crs:
                src_obj = CRS.from_user_input(src_crs)
                if src_obj.to_epsg() != 4326:
                    transformer = Transformer.from_crs(src_obj, 'EPSG:4326', always_xy=True)
        except Exception:
            transformer = None

        def reproject(coords):
            """Recursively transform leaf [x, y] (or [x, y, z]) pairs to lists."""
            if not coords:
                return coords
            if isinstance(coords[0], (int, float)):
                x, y = transformer.transform(coords[0], coords[1])
                return [x, y]
            return [reproject(c) for c in coords]

        def to_lists(c):
            """Recursively coerce fiona's tuple-of-tuples into plain JSON-safe lists."""
            if hasattr(c, '__iter__') and not isinstance(c, (str, bytes)):
                return [to_lists(x) for x in c]
            return c

        features = []
        for f in src:
            geom = f.get('geometry')
            if not geom:
                continue
            # fiona.Geometry isn't directly JSON-serialisable — always rebuild as a
            # plain dict with nested lists so flask.jsonify accepts it.
            g_type = geom['type']
            g_coords = geom['coordinates']
            if transformer is not None:
                g_coords = reproject(g_coords)
            else:
                g_coords = to_lists(g_coords)
            features.append({
                'type': 'Feature', 'properties': {},
                'geometry': {'type': g_type, 'coordinates': g_coords},
            })

    # Compute bbox by walking all coordinates ourselves.
    def walk(coords):
        if not coords:
            return
        if isinstance(coords[0], (int, float)):
            yield coords
        else:
            for c in coords:
                yield from walk(c)

    xs, ys = [], []
    for f in features:
        for x, y, *_ in walk(f['geometry']['coordinates']):
            xs.append(x); ys.append(y)
    bbox = [min(xs), min(ys), max(xs), max(ys)] if xs else [0.0, 0.0, 0.0, 0.0]

    return {'type': 'FeatureCollection', 'bbox': bbox, 'features': features}


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
        .refresh-btn { flex: 0 0 auto !important; width: 26px !important;
                       padding: 4px 0 !important; font-size: 13px !important;
                       background: #ddd !important; color: #333 !important;
                       border: 1px solid #bbb !important; }
        .refresh-btn:hover { background: #c8c8c8 !important; }
        .ais-actions { display: flex; gap: 6px; align-items: center;
                       margin: 4px 0 0; }
        .ais-actions > button { flex: 0 0 auto !important; }
        .ais-actions .preview-box { flex: 1; }
        .secondary-btn { background: #ddd !important; color: #333 !important;
                         border: 1px solid #bbb !important;
                         padding: 6px 10px !important; font-size: 11px !important; }
        .secondary-btn:hover { background: #c8c8c8 !important; }
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
        /* Small bottom-right pill for "Rendering..." / "Ready" between transfer and first paint */
        #render-status { position: fixed; bottom: 20px; right: 20px;
                         background: #4ad; color: white; padding: 8px 16px;
                         border-radius: 4px; font: 12px monospace;
                         box-shadow: 0 2px 8px rgba(0,0,0,0.25); z-index: 150;
                         transition: opacity 0.3s; }
        #render-status.done { background: #4a4; }
        #render-status.fade { opacity: 0; }
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

# update_title=None disables Dash's default "Updating..." tab-title swap during callbacks.
app = Dash(__name__, suppress_callback_exceptions=True, update_title=None)
app.title = 'aiswakepy'
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


def _picker_with_refresh(label: str, dropdown_id: str, preview_id: str,
                         info_id: str, refresh_id: str,
                         options: list, value, clearable=False):
    """Dropdown + small ↻ button + preview tickbox. The refresh button forces a
    re-fetch even when the dropdown value didn't change (workaround for Dash's
    no-callback-on-same-value behaviour)."""
    return html.Div([
        html.Label(label),
        html.Div([
            dcc.Dropdown(id=dropdown_id, options=_opt_list(options),
                         value=value, clearable=clearable),
            html.Button('↻', id=refresh_id, n_clicks=0,
                        title='Re-apply selection',
                        className='refresh-btn'),
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

        # ---- AIS CSV: dropdown + Import + Preview tickbox + Filter button ----
        html.Label('AIS CSV'),
        dcc.Dropdown(id='sel-ais', options=_opt_list(EXAMPLE_FILES['ais']),
                     value=_default_ais, clearable=False),
        html.Div([
            html.Button('Import', id='btn-import-ais', n_clicks=0,
                        title='Load the selected AIS CSV onto the map',
                        className='secondary-btn'),
            html.Div(
                dcc.Checklist(id='pv-ais',
                              options=[{'label': 'preview', 'value': '1'}],
                              value=[]),
                className='preview-box',
            ),
            html.Button('Filter', id='btn-filter', n_clicks=0,
                        title='Run Stage 1: AIS cleaning + interpolation',
                        className='secondary-btn'),
        ], className='ais-actions'),
        html.Div(id='pv-ais-info', className='preview-info'),

        # ---- Other inputs: dropdown + refresh + preview ----
        _picker_with_refresh('Bathymetry', 'sel-bathy', 'pv-bathy',
                             'pv-bathy-info', 'btn-pv-bathy-refresh',
                             EXAMPLE_FILES['bathymetry'], _default_bathy),
        _picker_with_refresh('Coastline', 'sel-coast', 'pv-coast',
                             'pv-coast-info', 'btn-pv-coast-refresh',
                             EXAMPLE_FILES['coastline'], _default_coast),
        _picker_with_refresh('Tide DFS0 (optional)', 'sel-tide', 'pv-tide',
                             'pv-tide-info', 'btn-pv-tide-refresh',
                             EXAMPLE_FILES['tide'], _default_tide, clearable=True),

        html.Label('Interpolation method'),
        dcc.Dropdown(id='sel-interp',
                     options=[
                         {'label': 'linear (straight-line)', 'value': 'linear'},
                         {'label': 'hermite (cubic spline)', 'value': 'hermite'},
                     ],
                     value='linear', clearable=False),

        html.Label('Cb method (Le determination)'),
        dcc.Dropdown(id='sel-cb',
                     options=[{'label': m, 'value': m} for m in CB_METHODS],
                     value='L_Le', clearable=False),
        html.Label('Wave formula'),
        dcc.Dropdown(id='sel-formula',
                     options=[{'label': f, 'value': f} for f in WAVE_FORMULAE],
                     value='kriebel', clearable=False),

        html.Div([
            html.Button('Calculate waves', id='btn-waves', n_clicks=0,
                        title='Run depth + vessel + wave_impact stages '
                              '(auto-runs filter first if needed)'),
        ], className='row-buttons'),

        # ---- Debug filter: isolate a single track segment for spike inspection ----
        html.Hr(),
        html.Div('Debug: isolate one track segment',
                 style={'fontWeight': 'bold', 'fontSize': '11px', 'color': '#555'}),
        html.Div([
            html.Div([
                html.Label('MMSI', style={'margin': '4px 0 2px'}),
                dcc.Input(id='inp-debug-mmsi', type='number',
                          placeholder='e.g. 372490000', debounce=True,
                          style={'width': '100%', 'fontSize': '11px',
                                 'padding': '4px', 'boxSizing': 'border-box'}),
            ], style={'flex': 1}),
            html.Div([
                html.Label('segment', style={'margin': '4px 0 2px'}),
                dcc.Input(id='inp-debug-seg', type='number',
                          placeholder='e.g. 5730', debounce=True,
                          style={'width': '100%', 'fontSize': '11px',
                                 'padding': '4px', 'boxSizing': 'border-box'}),
            ], style={'flex': 1, 'marginLeft': '6px'}),
        ], style={'display': 'flex'}),
        html.Div([
            html.Button('Isolate', id='btn-debug-apply', n_clicks=0,
                        className='secondary-btn',
                        title='Hide all other tracks; show this segment + each AIS point'),
            html.Button('Show all', id='btn-debug-clear', n_clicks=0,
                        className='secondary-btn',
                        title='Restore the full track view'),
        ], className='row-buttons'),
        html.Div(id='debug-info', className='preview-info'),

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
    dcc.Store(id='_ais_import', data={'path': None, 'nonce': 0}),
    # Preview state Stores: {visible, path, nonce}; nonce bumps on refresh-btn click
    dcc.Store(id='_pv_ais',   data={'visible': False, 'path': None}),
    dcc.Store(id='_pv_bathy', data={'visible': False, 'path': None, 'nonce': 0}),
    dcc.Store(id='_pv_coast', data={'visible': False, 'path': None, 'nonce': 0}),
    dcc.Store(id='_pv_tide',  data={'visible': False, 'path': None, 'nonce': 0}),
    dcc.Store(id='_debug_filter', data={'mmsi': None, 'segment_id': None, 'nonce': 0}),
])


# ---------------------------------------------------------------------------
# Server-side callbacks: run buttons + polling
# ---------------------------------------------------------------------------
def _build_config(ais, bathy, coast, tide, formula, cb_method, interp_method='linear') -> dict:
    ais_cfg = {'raw_csv': ais}
    if interp_method:
        ais_cfg['interp_method'] = interp_method
    cfg = {
        'ais': ais_cfg,
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
            'running': True, 'log': [], 'live': '',
            'started_at': time.time(),
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
    State('sel-interp', 'value'),
    prevent_initial_call=True,
)
def kick_filter(n, ais, bathy, coast, tide, formula, cb, interp):
    if not n or not ais or not coast:
        return no_update, no_update, no_update
    cfg = _build_config(ais, bathy, coast, tide, formula, cb, interp)
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
    State('sel-interp', 'value'),
    prevent_initial_call=True,
)
def kick_waves(n, ais, bathy, coast, tide, formula, cb, interp):
    """Run depth+vessel+wave_impact.

    The worker checks (in order): in-memory LAST_RESULTS['df_filtered'],
    then disk-cached `{stem}_01_filtered.csv` newer than the AIS file,
    then falls back to running filter first.
    """
    if not n or not ais or not coast or not bathy:
        return no_update, no_update, no_update
    cfg = _build_config(ais, bathy, coast, tide, formula, cb, interp)
    # Always request the wave stages — the worker decides whether to
    # prepend 'filter' based on cache freshness checks.
    if _kick(cfg, ['depth', 'vessel', 'wave_impact'], 'waves'):
        return False, True, True
    return no_update, no_update, no_update


# ---- AIS import button: triggers clientside fetch with progress overlay ----
@app.callback(
    Output('_ais_import', 'data'),
    Output('pv-ais', 'value', allow_duplicate=True),
    Input('btn-import-ais', 'n_clicks'),
    State('sel-ais', 'value'),
    State('_ais_import', 'data'),
    prevent_initial_call=True,
)
def trigger_ais_import(n, path, prev):
    if not n or not path:
        return no_update, no_update
    nonce = (prev or {}).get('nonce', 0) + 1
    # Auto-tick the preview tickbox so the imported data shows on map.
    return {'path': path, 'nonce': nonce}, ['1']


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
    if s.get('live'):
        log_lines.append(s['live'])  # in-progress spinner line, replaced each tick
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
# Preview callbacks: bathy / coast / tide listen to dropdown + tickbox + refresh.
# The refresh button bumps a `nonce`, so even if the dropdown value is unchanged
# the Store data changes and the clientside callback fires (fixes the "clicking
# the already-selected dropdown option does nothing" UX).
# ---------------------------------------------------------------------------
def _make_pv_callback(store_id, sel_id, pv_id, refresh_id):
    @app.callback(
        Output(store_id, 'data'),
        Input(sel_id, 'value'),
        Input(pv_id, 'value'),
        Input(refresh_id, 'n_clicks'),
        State(store_id, 'data'),
        prevent_initial_call=False,
    )
    def _pv(path, pv_val, refresh_n, prev):
        prev_nonce = (prev or {}).get('nonce', 0)
        # Bump nonce on any refresh-button click so the clientside re-fetches.
        nonce = (refresh_n or 0) + prev_nonce * 0  # use refresh_n directly
        return {'visible': bool(pv_val), 'path': path, 'nonce': refresh_n or 0}
    return _pv


_make_pv_callback('_pv_bathy', 'sel-bathy', 'pv-bathy', 'btn-pv-bathy-refresh')
_make_pv_callback('_pv_coast', 'sel-coast', 'pv-coast', 'btn-pv-coast-refresh')
_make_pv_callback('_pv_tide',  'sel-tide',  'pv-tide',  'btn-pv-tide-refresh')


# AIS preview tickbox is decoupled from the dropdown — it just toggles
# visibility of the already-imported AIS data.
@app.callback(
    Output('_pv_ais', 'data'),
    Input('pv-ais', 'value'),
    prevent_initial_call=False,
)
def _pv_ais_toggle(pv_val):
    return {'visible': bool(pv_val)}


# ---- Debug filter: isolate one (mmsi, segment_id) on the map ----
@app.callback(
    Output('_debug_filter', 'data'),
    Input('btn-debug-apply', 'n_clicks'),
    Input('btn-debug-clear', 'n_clicks'),
    State('inp-debug-mmsi', 'value'),
    State('inp-debug-seg', 'value'),
    State('_debug_filter', 'data'),
    prevent_initial_call=True,
)
def _set_debug_filter(apply_n, clear_n, mmsi, seg, prev):
    from dash import ctx
    nonce = ((apply_n or 0) + (clear_n or 0))
    if ctx.triggered_id == 'btn-debug-clear':
        return {'mmsi': None, 'segment_id': None, 'nonce': nonce}
    # Coerce empty strings to None (number inputs return '' until typed).
    return {
        'mmsi': int(mmsi) if mmsi not in (None, '') else None,
        'segment_id': int(seg) if seg not in (None, '') else None,
        'nonce': nonce,
    }


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

    // ---------- Render-state pill (shown after fetch, until first paint) ----------
    function setRenderStatus(text, doneFlag) {
        let el = document.getElementById('render-status');
        if (!el) {
            el = document.createElement('div');
            el.id = 'render-status';
            document.body.appendChild(el);
        }
        el.textContent = text;
        el.className = doneFlag ? 'done' : '';
    }
    function clearRenderStatus(delay) {
        const el = document.getElementById('render-status');
        if (!el) return;
        if (delay) {
            el.classList.add('fade');
            setTimeout(() => { if (el.parentNode) el.remove(); }, delay);
        } else {
            el.remove();
        }
    }
    window.__setRenderStatus = setRenderStatus;
    window.__clearRenderStatus = clearRenderStatus;
    // Yields until deck.gl has painted at least one frame after rebuild.
    function waitForPaint() {
        return new Promise(r => requestAnimationFrame(() =>
            requestAnimationFrame(r)));   // two RAFs = at least one full paint cycle
    }
    window.__waitForPaint = waitForPaint;

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
        // Debug filter: when set to {mmsi, segment_id}, hide full tracks layer
        // and show only the matching segment + each vertex as a pickable dot.
        window.__debugFilter = { mmsi: null, segment_id: null };

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
            const df = window.__debugFilter || {};
            const debugActive = df.mmsi != null && df.segment_id != null;

            if (window.__hasTracks && !useRaster && tMMSI.length > 0 && !debugActive) {
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

            // Debug filter: isolate one (mmsi, segment_id) — show segment line + every vertex as a pickable dot.
            if (debugActive && window.__hasTracks) {
                const segIdx = segLookup.get(`${df.mmsi}|${df.segment_id}`);
                if (segIdx != null && segIdx < startIndices.length - 1) {
                    const segStart = startIndices[segIdx];
                    const segEnd   = startIndices[segIdx + 1];
                    const segCoords = cPos.subarray(segStart * 2, segEnd * 2);
                    layers.push(new deck.PathLayer({
                        id: 'debug-segment-line',
                        data: {
                            length: 1,
                            startIndices: [0, segEnd - segStart],
                            attributes: { getPath: { value: segCoords, size: 2 } },
                        },
                        _pathType: 'open',
                        getColor: [255, 180, 0, 240],
                        getWidth: 2, widthUnits: 'pixels', widthMinPixels: 2,
                        pickable: false,
                    }));
                    layers.push(new deck.ScatterplotLayer({
                        id: 'debug-segment-points',
                        data: { length: segEnd - segStart,
                                attributes: { getPosition: { value: segCoords, size: 2 } } },
                        pickable: true, stroked: true, filled: true,
                        getRadius: 5, radiusUnits: 'pixels',
                        radiusMinPixels: 4, radiusMaxPixels: 8,
                        getFillColor: [255, 60, 0, 220],
                        getLineColor: [255, 255, 255, 240],
                        lineWidthMinPixels: 1,
                        onHover: ({x, y, index}) => {
                            if (index < 0) { hideTip(); return; }
                            const lon = segCoords[index*2];
                            const lat = segCoords[index*2+1];
                            showTip(x, y,
                                `<b>POINT ${index}/${segEnd - segStart - 1}</b><br>` +
                                `MMSI ${df.mmsi}  seg ${df.segment_id}<br>` +
                                `lon: ${lon.toFixed(6)}<br>lat: ${lat.toFixed(6)}`);
                        },
                    }));
                }
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
            if (pv.coast && pv.coast.geojson) {
                // GeoJsonLayer handles Polygon/MultiPolygon/LineString natively.
                layers.push(new deck.GeoJsonLayer({
                    id: 'pv-coast',
                    data: pv.coast.geojson,
                    stroked: true, filled: true,
                    getFillColor: [60, 130, 220, 40],
                    getLineColor: [20, 80, 180, 230],
                    getLineWidth: 2, lineWidthMinPixels: 2,
                    pickable: false,
                }));
            }
            // AIS preview: rendered only when visible flag is on (import is separate).
            if (pv.ais && pv.ais.visible && pv.ais.pos && pv.ais.pos.length > 0) {
                layers.push(new deck.ScatterplotLayer({
                    id: 'pv-ais',
                    data: { length: pv.ais.pos.length / 2,
                            attributes: { getPosition: { value: pv.ais.pos, size: 2 } } },
                    getRadius: 3, radiusUnits: 'pixels', radiusMinPixels: 1,
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

        // Post-pipeline refresh hooks: show the same progress overlay as before, then rebuild layers,
        // then show a "Rendering..." pill until deck.gl has painted at least one frame.
        window.__refreshWaveLayer = async (version) => {
            const [buf] = await fetchAssetsWithProgress([
                { key: 'waves', url: `/api/waves.arrow?v=${version}`, label: 'wave impacts' },
            ], 'Loading wave impacts');
            setRenderStatus('Rendering waves...', false);
            rebuildWaveArrays(window.tableFromIPC(buf));
            window.__hasWaves = wMMSI.length > 0;
            window._hoveredWave = null;
            window.__rebuild();
            await waitForPaint();
            setRenderStatus(`Waves ready (${wMMSI.length.toLocaleString()})`, true);
            clearRenderStatus(1500);
        };
        window.__refreshTrackCaches = async (version) => {
            const [c, m, o] = await fetchAssetsWithProgress([
                { key: 'track_coords',  url: `/api/track_coords.arrow?v=${version}`,  label: 'track coords' },
                { key: 'track_meta',    url: `/api/track_meta.arrow?v=${version}`,    label: 'track metadata' },
                { key: 'track_offsets', url: `/api/track_offsets.arrow?v=${version}`, label: 'track offsets' },
            ], 'Loading vessel tracks');
            setRenderStatus('Rendering tracks...', false);
            initTrackArrays(
                window.tableFromIPC(c),
                window.tableFromIPC(m),
                window.tableFromIPC(o),
            );
            window.__hasTracks = tMMSI.length > 0;
            window.__rebuild();
            await waitForPaint();
            setRenderStatus(`Tracks ready (${tMMSI.length.toLocaleString()} segments)`, true);
            clearRenderStatus(1500);
        };

        // ---- AIS import (slow, explicit button) + preview toggle (cheap) ----
        window.__importedAisPath = null;
        window.__importAis = async (path) => {
            if (!path) return 'no file selected';
            try {
                const [buf] = await fetchAssetsWithProgress([
                    { key: 'ais',  url: '/api/preview/ais.arrow?path=' + encodeURIComponent(path),
                      label: 'AIS positions' },
                ], 'Importing AIS data');
                setRenderStatus('Rendering AIS points...', false);
                const t = window.tableFromIPC(buf);
                const lon = t.getChild('longitude').toArray();
                const lat = t.getChild('latitude').toArray();
                const pos = new Float32Array(lon.length * 2);
                for (let i = 0; i < lon.length; i++) { pos[i*2]=lon[i]; pos[i*2+1]=lat[i]; }
                window.__previews.ais = { pos, visible: true };
                window.__importedAisPath = path;
                try {
                    const bb = await fetch('/api/preview/ais.bbox?path=' + encodeURIComponent(path)).then(r => r.json());
                    if (bb.bbox) {
                        const [w, s, e, n] = bb.bbox;
                        const lonC = (w + e) / 2, latC = (s + n) / 2;
                        const span = Math.max(e - w, n - s);
                        const z = Math.max(8, Math.min(15, 10 - Math.log2(span)));
                        window.deckInstance.setProps({ initialViewState: { longitude: lonC, latitude: latC, zoom: z, pitch: 0, bearing: 0 } });
                        window._currentZoom = z;
                    }
                } catch (e) { /* bbox fit failure is non-fatal */ }
                window.__rebuild();
                await waitForPaint();
                setRenderStatus(`AIS ready (${lon.length.toLocaleString()} points)`, true);
                clearRenderStatus(1500);
                return `imported ${lon.length.toLocaleString()} points`;
            } catch (e) {
                clearRenderStatus(0);
                return 'ERROR: ' + e.message;
            }
        };
        window.__togglePreviewAis = (visible) => {
            if (window.__previews.ais) {
                window.__previews.ais.visible = !!visible;
                window.__rebuild();
                return window.__previews.ais.visible
                    ? `showing ${(window.__previews.ais.pos.length/2).toLocaleString()} points`
                    : 'hidden';
            }
            return visible ? 'import first' : '';
        };
        // Legacy entry point kept for backward compat — now just toggles visibility.
        window.__setPreviewAis = async (state) => {
            if (!state) return '';
            return window.__togglePreviewAis(state.visible) || '';
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
                window.__previews.coast = null; window.__rebuild(); return '';
            }
            try {
                const gj = await fetch('/api/preview/coast.geojson?path=' + encodeURIComponent(state.path)).then(r => r.json());
                if (gj.error) throw new Error(gj.error);
                window.__previews.coast = { geojson: gj };
                // Fit camera to the shapefile bbox.
                if (gj.bbox && gj.bbox.length === 4) {
                    const [w, s, e, n] = gj.bbox;
                    const lonC = (w + e) / 2, latC = (s + n) / 2;
                    const span = Math.max(e - w, n - s, 0.001);
                    const z = Math.max(8, Math.min(15, 10 - Math.log2(span)));
                    window.deckInstance.setProps({ initialViewState: { longitude: lonC, latitude: latC, zoom: z, pitch: 0, bearing: 0 } });
                    window._currentZoom = z;
                }
                window.__rebuild();
                const nFeat = gj.features ? gj.features.length : 0;
                return `${nFeat} feature(s)`;
            } catch (e) {
                window.__previews.coast = null; window.__rebuild();
                return 'ERROR: ' + e.message;
            }
        };
        // ---- Debug filter setter ----
        window.__setDebugFilter = (state) => {
            window.__debugFilter = state || { mmsi: null, segment_id: null };
            const df = window.__debugFilter;
            if (df.mmsi == null || df.segment_id == null) {
                window.__rebuild();
                return 'showing all tracks';
            }
            if (!segLookup) return 'no track data yet — run Filter first';
            const segIdx = segLookup.get(`${df.mmsi}|${df.segment_id}`);
            if (segIdx == null || segIdx >= startIndices.length - 1) {
                window.__rebuild();
                return `not found: MMSI=${df.mmsi}, segment=${df.segment_id}`;
            }
            const segStart = startIndices[segIdx];
            const segEnd   = startIndices[segIdx + 1];
            const n = segEnd - segStart;
            // Fit camera to the segment's bbox.
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            for (let i = segStart; i < segEnd; i++) {
                const x = cPos[i*2], y = cPos[i*2+1];
                if (x < minX) minX = x; if (x > maxX) maxX = x;
                if (y < minY) minY = y; if (y > maxY) maxY = y;
            }
            if (isFinite(minX)) {
                const lonC = (minX + maxX) / 2, latC = (minY + maxY) / 2;
                const span = Math.max(maxX - minX, maxY - minY, 0.0005);
                const z = Math.max(12, Math.min(18, 11 - Math.log2(span)));
                window.deckInstance.setProps({
                    initialViewState: { longitude: lonC, latitude: latC,
                                        zoom: z, pitch: 0, bearing: 0 },
                });
                window._currentZoom = z;
            }
            window.__rebuild();
            return `MMSI ${df.mmsi} seg ${df.segment_id}: ${n} vertices  (hover dots for lon/lat)`;
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


# AIS import clientside callback — triggered by the Import button via _ais_import Store.
app.clientside_callback(
    r"""
    async function(state) {
        if (!state || !state.path || !state.nonce) return '';
        if (typeof window.__importAis !== 'function') return 'init pending...';
        const result = await window.__importAis(state.path);
        return result || '';
    }
    """,
    Output('pv-ais-info', 'children', allow_duplicate=True),
    Input('_ais_import', 'data'),
    prevent_initial_call=True,
)


# Debug filter clientside callback — calls __setDebugFilter and surfaces the info text.
app.clientside_callback(
    r"""
    function(state) {
        if (typeof window.__setDebugFilter !== 'function') return 'init pending...';
        return window.__setDebugFilter(state) || '';
    }
    """,
    Output('debug-info', 'children'),
    Input('_debug_filter', 'data'),
    prevent_initial_call=True,
)


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
