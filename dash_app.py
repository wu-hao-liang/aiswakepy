"""Dash + raw deck.gl + Apache Arrow performance test on real AIS data,
with an in-page two-step pipeline runner and per-input previews.

Run with:  uv run python dash_app.py
Then open: http://localhost:8050   (or  http://<lan-ip>:8050  from another host)
"""
from __future__ import annotations

import functools
import io
import json
import os
import shutil
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
from aiswakepy.viz.report import (
    plot_wave_height_report,
    plot_wave_period_report,
    top_vessels_table,
    plot_vessel_track_scatter,
)

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
PREVIEW_BATHY_MAX_TRIANGLES = 500000    # cap mesh element count for preview

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
    }).copy()
    # Normalise both join keys to second-precision string to avoid dtype mismatches
    df_w = df_w.copy()
    df_w['_dt_key'] = pd.to_datetime(df_w['DateTime'], errors='coerce').dt.floor('s').astype(str)
    join['_dt_key'] = pd.to_datetime(join['DateTime'], errors='coerce').dt.floor('s').astype(str)
    join = join.drop(columns=['DateTime'])
    out = df_w.merge(join, on=['MMSI', '_dt_key'], how='left').drop(columns=['_dt_key'])
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
    _obstime_ns = df_v['obstime'].to_numpy(dtype='datetime64[ns]').astype(np.int64) if 'obstime' in df_v.columns else np.zeros(len(df_v), dtype=np.int64)
    arrow_vessels = pa.table({
        'longitude': pa.array(df_v['longitude'].to_numpy(dtype=np.float32), type=pa.float32()),
        'latitude':  pa.array(df_v['latitude'].to_numpy(dtype=np.float32),  type=pa.float32()),
        'mmsi':      pa.array(df_v['mmsi'].to_numpy(dtype=np.int64),         type=pa.int64()),
        'sog':       pa.array(df_v['sog'].to_numpy(dtype=np.float32),        type=pa.float32()),
        'cog':       pa.array(df_v['cog'].to_numpy(dtype=np.float32),        type=pa.float32()),
        'typecargo': pa.array(df_v['typecargo'].to_numpy(dtype=np.float32) if 'typecargo' in df_v.columns else np.full(len(df_v), np.nan, dtype=np.float32), type=pa.float32()),
        'obstime':   pa.array(_obstime_ns, type=pa.int64()),
    })
    IPC_VESSELS = _ipc(arrow_vessels)

    print('  building track segments (vectorised)...')
    if len(df_v) == 0:
        flat_arr = np.zeros((0, 2), dtype=np.float32)
        offsets_arr = np.array([0], dtype=np.int32)
        meta_mmsi = np.array([], dtype=np.int64)
        meta_seg  = np.array([], dtype=np.int32)
        meta_n    = np.array([], dtype=np.int32)
        meta_type = np.array([], dtype=np.int32)
        sel_sog = sel_cog = np.array([], dtype=np.float32)
        sel_time_ns = np.array([], dtype=np.int64)
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
            meta_type = np.array([], dtype=np.int32)
            sel_sog = sel_cog = np.array([], dtype=np.float32)
            sel_time_ns = np.array([], dtype=np.int64)
        else:
            kept_starts = starts[keep]
            kept_sizes  = sizes[keep]
            kept_ends   = kept_starts + kept_sizes.astype(np.int64)
            # Indices of rows belonging to kept segments — concatenated ranges.
            row_idx = np.concatenate([np.arange(s, e) for s, e in zip(kept_starts, kept_ends)])
            sel_lon  = df_sorted['longitude'].to_numpy(dtype=np.float32)[row_idx]
            sel_lat  = df_sorted['latitude'].to_numpy(dtype=np.float32)[row_idx]
            sel_sog  = df_sorted['sog'].to_numpy(dtype=np.float32)[row_idx]
            sel_cog  = df_sorted['cog'].to_numpy(dtype=np.float32)[row_idx]
            # obstime as int64 ns for Arrow transport
            sel_time = df_sorted['obstime'].to_numpy(dtype='datetime64[ns]')[row_idx]
            sel_time_ns = sel_time.astype(np.int64)
            flat_arr = np.column_stack([sel_lon, sel_lat]).astype(np.float32)
            offsets_arr = np.concatenate(([0], np.cumsum(kept_sizes))).astype(np.int32)
            meta_mmsi = mmsi_arr[kept_starts]
            meta_seg  = sid_arr[kept_starts].astype(np.int32)
            meta_n    = kept_sizes.astype(np.int32)
            # typecargo: use first point of each segment (fill NaN → -1)
            type_arr = df_sorted['typecargo'].fillna(-1).to_numpy(dtype=np.float32) if 'typecargo' in df_sorted.columns else np.full(len(df_sorted), -1.0)
            meta_type = type_arr[kept_starts].astype(np.int32)

    seg_meta = [{'mmsi': int(m), 'segment_id': int(s), 'n_points': int(n)}
                for m, s, n in zip(meta_mmsi, meta_seg, meta_n)]
    arrow_coords = pa.table({
        'lon':      pa.array(flat_arr[:, 0], type=pa.float32()),
        'lat':      pa.array(flat_arr[:, 1], type=pa.float32()),
        'sog':      pa.array(sel_sog, type=pa.float32()),
        'cog':      pa.array(sel_cog, type=pa.float32()),
        'obstime':  pa.array(sel_time_ns, type=pa.int64()),
    })
    arrow_meta = pa.table({
        'mmsi':       pa.array(meta_mmsi.astype(np.int64), type=pa.int64()),
        'segment_id': pa.array(meta_seg,                   type=pa.int32()),
        'n_points':   pa.array(meta_n,                     type=pa.int32()),
        'typecargo':  pa.array(meta_type,                  type=pa.int32()),
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


def _write_wave_track_link(df_w: pd.DataFrame, out_dir: Path) -> None:
    """Write a slim external sidecar mapping each wave row → (MMSI, segment_id).

    Always writes a header even if the frame is empty so downstream consumers
    can rely on the file existing whenever waves output exists.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(df_w) and {'MMSI', 'segment_id'} <= set(df_w.columns):
        link = df_w[['MMSI', 'segment_id']].reset_index(drop=True)
        link.insert(0, 'wave_row', link.index)
    else:
        link = pd.DataFrame(columns=['wave_row', 'MMSI', 'segment_id'])
    link.to_csv(out_dir / 'wave_track_link.csv', index=False)


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
    'sog': pa.array([], pa.float32()), 'cog': pa.array([], pa.float32()),
    'obstime': pa.array([], pa.int64()),
}))
IPC_TRACK_META = _ipc(pa.table({
    'mmsi': pa.array([], pa.int64()), 'segment_id': pa.array([], pa.int32()),
    'n_points': pa.array([], pa.int32()), 'typecargo': pa.array([], pa.int32()),
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
# Server deployment config (UNC paths, host-specific settings)
# ---------------------------------------------------------------------------
_srv_cfg_path = REPO / 'server_config.json'
_srv_cfg: dict = json.loads(_srv_cfg_path.read_text()) if _srv_cfg_path.exists() else {}
DATA_UNC_ROOT: str = _srv_cfg.get('data_unc_root', '')

# ---------------------------------------------------------------------------
# Data directory inventory
# ---------------------------------------------------------------------------
DATA_ROOT = REPO / 'data'
DATA_ROOT.mkdir(exist_ok=True)


def _scan_data_subdirs() -> list[dict]:
    """Return working-directory dropdown options from data/ subdirectories."""
    dirs = sorted(
        p.relative_to(REPO).as_posix()
        for p in DATA_ROOT.iterdir() if p.is_dir()
    )
    return [{'label': d, 'value': d} for d in dirs]


def _scan_working_dir(workdir: str | None) -> dict:
    """Scan standard subdirs under the working directory.

    Expected layout::

        data/<workdir>/
            ais/*.csv
            coastline/*.shp
            land/*.shp
            bathymetry/*.{mesh,dfs2,dfsu}
            tide/*.dfs0
    """
    empty = {'ais': [], 'bathymetry': [], 'coastline': [], 'land': [], 'tide': []}
    if not workdir:
        return empty
    base = REPO / workdir
    if not base.exists():
        return empty
    rel = lambda p: p.relative_to(REPO).as_posix()
    return {
        'ais':        sorted(rel(p) for p in (base / 'ais').glob('*.csv')),
        'bathymetry': sorted(rel(p) for ext in ('mesh', 'dfs2', 'dfsu')
                             for p in (base / 'bathymetry').glob(f'*.{ext}')),
        'coastline':  sorted(rel(p) for p in (base / 'coastline').glob('*.shp')),
        'land':       sorted(rel(p) for p in (base / 'land').glob('*.shp')),
        'tide':       sorted(rel(p) for p in (base / 'tide').glob('*.dfs0')),
    }


def _ais_time_range_str() -> str:
    if 'obstime' not in df_vessels.columns or len(df_vessels) == 0:
        return ''
    ts = df_vessels['obstime']
    lo, hi = ts.min(), ts.max()
    if pd.isna(lo):
        return ''
    return f"{lo.strftime('%Y-%m-%d')} – {hi.strftime('%Y-%m-%d')} |"


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
    'cfg': None,          # last config_dict used (for report plots in _export_filtered)
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


def _generate_report_plots(
    cfg_dict: dict,
    df_vessel: pd.DataFrame,
    df_wave_impact: pd.DataFrame,
    out_dir: Path | None = None,
) -> None:
    """Generate the four report outputs and save to out_dir.

    Called after pipeline completes (full results) and after export (filtered
    results).  Failures are caught and printed so they never crash the caller.
    """
    if out_dir is None:
        out_dir = Path(cfg_dict.get('output', {}).get('directory', 'output/'))
    out_dir = Path(out_dir)
    coastline_shp = cfg_dict.get('coastline', {}).get('shapefile', '')

    try:
        tbl = top_vessels_table(df_wave_impact, n=10,
                                output_path=out_dir / 'top10_vessels.csv')
        print(f'  ✓ report: top10_vessels.csv  ({len(tbl)} rows)')
    except Exception as e:
        print(f'  WARN: top_vessels_table failed: {e}')

    try:
        plot_vessel_track_scatter(df_vessel, df_wave_impact,
                                  output_path=out_dir / 'vessel_track_scatter.png')
        print(f'  ✓ report: vessel_track_scatter.png')
    except Exception as e:
        print(f'  WARN: plot_vessel_track_scatter failed: {e}')

    if not coastline_shp:
        print('  WARN: coastline_shp not set — skipping wave maps')
        return

    wave_height_name = cfg_dict.get('output', {}).get('wave_height_map_name', 'WaveHeightMap.png')
    wave_period_name = cfg_dict.get('output', {}).get('wave_period_map_name', 'WavePeriodMap.png')

    try:
        plot_wave_height_report(df_wave_impact, coastline_shp,
                                output_path=out_dir / wave_height_name)
        print(f'  ✓ report: {wave_height_name}')
    except Exception as e:
        print(f'  WARN: plot_wave_height_report failed: {e}')

    try:
        plot_wave_period_report(df_wave_impact, coastline_shp,
                                output_path=out_dir / wave_period_name)
        print(f'  ✓ report: {wave_period_name}')
    except Exception as e:
        print(f'  WARN: plot_wave_period_report failed: {e}')


def _pipeline_thread(config_dict: dict, stages: list[str], step_label: str) -> None:
    """Worker thread. Runs the requested stages and refreshes only the affected caches.

    Cache builds run *outside* the pipeline lock so the polling tick can keep
    reading PIPELINE_STATE['log']/['live'] (and therefore the sidebar log keeps
    updating) while the slow groupby + Arrow encoding is in progress.
    """
    global LAST_RESULTS
    PIPELINE_STATE['cfg'] = config_dict
    old_stdout = sys.stdout
    sys.stdout = _LineCapture(old_stdout)
    try:
        cfg = load_config(config_dict)

        # The unified pipeline always runs filter+vessel+wave_impact from scratch;
        # no seed-results / filter-cache shortcut. Filter is cheap relative to
        # wave_impact and the new bathy/tide params have to flow through it.
        results = run_pipeline(cfg, stages=stages)
        LAST_RESULTS.update(results)

        # ---- Cache rebuild (no lock held) ----
        out_dir = Path(cfg.output.directory)
        # Track-display source: df_vessel — its rows are exactly those that
        # produced waves (post depth + SOG + BLratio trims). segment_ids align
        # with df_wave_impact because both inherit from the single final
        # segment_trajectories call inside filter_ais.
        # Mirror df_vessel into LAST_RESULTS['df_filtered'] so the "Export
        # filtered" path and any consumers reading LAST_RESULTS see the same
        # segment_id space as the displayed tracks and waves.
        vessels_for_tracks = results.get('df_vessel')
        if vessels_for_tracks is not None:
            LAST_RESULTS['df_filtered'] = vessels_for_tracks
            print('Refreshing track caches from df_vessel...')
            t0 = time.perf_counter()
            _build_vessel_caches(vessels_for_tracks)
            print(f'  -> {len(vessels_for_tracks):,} rows, '
                  f'{len(seg_meta):,} segments  ({time.perf_counter()-t0:.1f}s)')
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                vessels_for_tracks.to_parquet(out_dir / 'vessels.parquet', index=False)
                print(f'  ✓ saved results: vessels.parquet')
            except Exception as e:
                print(f'  WARN: could not save vessels.parquet: {e}')
            with _pipeline_lock:
                PIPELINE_STATE['track_version'] += 1
                PIPELINE_STATE['n_filtered'] = len(vessels_for_tracks)

        if 'df_wave_impact' in results and 'wave_impact' in stages:
            print('Refreshing wave caches...')
            t0 = time.perf_counter()
            # Fresh runs already have the columns _ensure_vessel_columns would
            # back-fill, so no join needed here. The helper is kept around for
            # _load_results also handles legacy CSV imports.
            _build_wave_caches(results['df_wave_impact'])
            print(f'  -> {len(results["df_wave_impact"]):,} wave events '
                  f'({time.perf_counter()-t0:.1f}s)')
            # Save wave parquet for dev-loading
            try:
                results['df_wave_impact'].to_parquet(out_dir / 'waves.parquet', index=False)
                print(f'  ✓ saved results: waves.parquet')
            except Exception as e:
                print(f'  WARN: could not save waves.parquet: {e}')
            try:
                _write_wave_track_link(results['df_wave_impact'], out_dir)
                print(f'  ✓ saved wave_track_link.csv')
            except Exception as e:
                print(f'  WARN: could not save wave_track_link.csv: {e}')
            with _pipeline_lock:
                PIPELINE_STATE['wave_version'] += 1
                PIPELINE_STATE['n_waves'] = len(results['df_wave_impact'])

            print('Generating report plots...')
            _generate_report_plots(
                config_dict,
                vessels_for_tracks if vessels_for_tracks is not None else pd.DataFrame(),
                results['df_wave_impact'],
                out_dir=out_dir,
            )

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
    """Return Arrow IPC of (lon, lat, sog, cog, obstime) from the AIS CSV.

    Called automatically when the user selects a new AIS CSV in the dropdown.
    """
    cols = ['longitude', 'latitude']
    try:
        df = pd.read_csv(path, nrows=1)
        for c in ['sog', 'cog']:
            if c in df.columns or c.lower() in (x.lower() for x in df.columns):
                cols.append(c)
        if 'obstime' in df.columns or 'obstime' in (x.lower() for x in df.columns):
            cols.append('obstime')
    except Exception:
        pass
    df = pd.read_csv(path, usecols=cols)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=['longitude', 'latitude'])
    cast = {'longitude': 'float32', 'latitude': 'float32'}
    for c in ['sog', 'cog']:
        if c in df.columns:
            cast[c] = 'float32'
    df = df.astype(cast)
    # Convert obstime string → int64 ns so the client can decode it with
    # new Date(ns / 1e6).  Sentinel 0 for unparseable / missing values.
    time_col = None
    for c in df.columns:
        if c.lower() == 'obstime':
            time_col = c
            break
    if time_col:
        # Parse as UTC, strip tz, store as int64 epoch ns.
        # Client reconstructs local display via new Date(ns / 1e6).
        t = pd.to_datetime(df[time_col], utc=True, errors='coerce')
        t = t.dt.tz_localize(None)
        df[time_col + '_ns'] = t.astype('datetime64[ns]').astype('int64').fillna(0)
        df = df.drop(columns=[time_col])
    return _ipc(pa.Table.from_pandas(df, preserve_index=False))


def _preview_ais_bbox(path: Path) -> dict:
    """Return bbox + time range — used to fit the camera and show data info."""
    df = pd.read_csv(path, usecols=['longitude', 'latitude', 'obstime'])
    lons = df['longitude']; lats = df['latitude']
    times = pd.to_datetime(df['obstime'], utc=False, errors='coerce').dropna()
    return {
        'bbox': [float(lons.min()), float(lats.min()),
                 float(lons.max()), float(lats.max())],
        'time_min': str(times.iloc[0]) if len(times) else None,
        'time_max': str(times.iloc[-1]) if len(times) else None,
        'n_rows': len(df),
    }


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


@functools.lru_cache(maxsize=8)
def _preview_bathy_arrow(path: Path, bbox: tuple | None = None) -> tuple[bytes, bytes] | None:
    """Load a .mesh/.dfsu and return (coords_ipc, offsets_ipc) for polygon rendering.

    coords_ipc: Arrow table {lon: f32, lat: f32} — flat vertex ring for all elements.
    offsets_ipc: Arrow table {offset: i32, z: f32} — start index per element (padded
        to length = n_elements + 1) with per-element mean z value.
    bbox: optional (west, south, east, north) to restrict elements to those whose
        centroid falls within the box.  Pass the AIS bbox padded 2× on the client.
    Returns None for .dfs2.
    """
    suffix = path.suffix.lower()
    if suffix == '.dfs2':
        return None  # gridded — no triangulation to draw
    import mikeio
    if suffix == '.mesh':
        geom = mikeio.Mesh(str(path)).geometry
    else:  # .dfsu
        geom = mikeio.open(str(path)).geometry
    nodes_xy = np.asarray(geom.node_coordinates, dtype=np.float64)[:, :2]
    elem_coords = np.asarray(geom.element_coordinates, dtype=np.float64)  # (n_elem, 3) centroid lon,lat,z
    elements = [np.asarray(e, dtype=np.int64) for e in geom.element_table]

    if bbox is not None:
        bw, bs, be, bn = bbox
        cx, cy = elem_coords[:, 0], elem_coords[:, 1]
        mask = (cx >= bw) & (cx <= be) & (cy >= bs) & (cy <= bn)
        keep_idx = np.flatnonzero(mask)
        elements = [elements[i] for i in keep_idx]
        elem_z = elem_coords[keep_idx, 2].astype(np.float32)
    else:
        elem_z = elem_coords[:, 2].astype(np.float32)

    sizes = np.array([len(e) for e in elements], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int32)
    flat = np.concatenate([nodes_xy[e, :] for e in elements]).astype(np.float32)

    coords_ipc = _ipc(pa.table({
        'lon': pa.array(flat[:, 0], type=pa.float32()),
        'lat': pa.array(flat[:, 1], type=pa.float32()),
    }))
    # Pad z to match offsets length (Arrow requires equal-length columns).
    z_padded = np.concatenate([elem_z, [np.float32('nan')]]).astype(np.float32)
    offsets_ipc = _ipc(pa.table({
        'offset': pa.array(offsets, type=pa.int32()),
        'z':      pa.array(z_padded, type=pa.float32()),
    }))
    return coords_ipc, offsets_ipc


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
    <script>
        window.__copyText = function(text) {
            function fallback() {
                var ta = document.createElement('textarea');
                ta.value = text;
                ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;';
                document.body.appendChild(ta);
                ta.focus(); ta.select();
                try { document.execCommand('copy'); } catch(e) {}
                document.body.removeChild(ta);
            }
            if (navigator.clipboard) {
                navigator.clipboard.writeText(text).catch(fallback);
            } else {
                fallback();
            }
        };
        window.__showCopyToast = function(mmsi, clientX, clientY) {
            var t = document.getElementById('copy-toast');
            if (!t) return;
            t.textContent = 'MMSI ' + mmsi + ' copied to clipboard';
            var tx = Math.min((clientX || 0) + 14, window.innerWidth - 220);
            var ty = Math.max((clientY || 0) - 38, 50);
            t.style.left = tx + 'px';
            t.style.top  = ty + 'px';
            t.classList.add('visible');
            clearTimeout(window.__copyToastTimer);
            window.__copyToastTimer = setTimeout(function() {
                t.classList.remove('visible');
            }, 2000);
        };
        (function initRSB() {
            var btn  = document.getElementById('btn-rsb-toggle');
            var rsb  = document.getElementById('right-sidebar');
            var deck = document.getElementById('deck-container');
            if (!btn || !rsb) { setTimeout(initRSB, 80); return; }
            btn.addEventListener('click', function() {
                var open = rsb.classList.toggle('open');
                btn.classList.toggle('open', open);
                var arrowChr = document.getElementById('rsb-arrow-chr');
                if (arrowChr) arrowChr.textContent = open ? '▶' : '◀';
                var leg = document.getElementById('map-legend');
                if (leg) leg.style.right = open ? '530px' : '50px';
            });
        })();
        (function initApplyBtn() {
            // Dash 4 renders the actions row asynchronously via Radix popper;
            // poll briefly after each trigger click to catch the row regardless of timing.
            function setup() {
                var trigger = document.getElementById('fil-type');
                if (!trigger) { setTimeout(setup, 200); return; }
                function attempt() {
                    var controls = trigger.getAttribute('aria-controls');
                    if (!controls) return;
                    var content = document.getElementById(controls);
                    if (!content) return;
                    var actions = content.querySelector('.dash-dropdown-actions');
                    if (!actions || actions.querySelector('.apply-action-btn')) return;
                    var b = document.createElement('button');
                    b.type = 'button';
                    b.className = 'dash-dropdown-action-button apply-action-btn';
                    b.textContent = 'Apply';
                    b.addEventListener('click', function(e) {
                        e.preventDefault(); e.stopPropagation();
                        trigger.click();
                    });
                    actions.appendChild(b);
                }
                trigger.addEventListener('click', function() {
                    setTimeout(attempt, 0);
                    setTimeout(attempt, 60);
                    setTimeout(attempt, 200);
                });
            }
            setup();
        })();
    </script>
    <style>
        html, body { margin: 0; padding: 0; height: 100%; overflow: hidden;
                     font-family: system-ui, sans-serif; }
        #status-banner { position: fixed; top: 0; left: 0; right: 0; height: 40px;
                         padding: 0 10px 0 12px; box-sizing: border-box; z-index: 25;
                         font: 12px monospace; background: #eef; border-bottom: 1px solid #ccd;
                         display: flex; align-items: center; gap: 10px; }
        #workdir-wrapper { min-width: 160px; flex: 0 1 auto; display: flex; align-items: center; gap: 4px; }
        #workdir-wrapper .Select, #workdir-wrapper .dash-dropdown { flex: 1; min-width: 0; }
        #workdir-wrapper .dash-dropdown-wrapper { min-height: 26px !important; }
        #btn-rescan-workdir { flex-shrink: 0; width: 22px; height: 26px; padding: 0;
                              font-size: 13px; line-height: 1; cursor: pointer;
                              background: #e8eef8; border: 1px solid #a8c0d8;
                              border-radius: 4px; color: #3a6080; box-sizing: border-box; }
        #btn-rescan-workdir:hover { background: #d4e4f4; }
        #workdir-unc-display { font: 11px monospace; color: #446; background: #eef;
                               border: 1px solid #ccd; border-radius: 3px;
                               padding: 3px 6px; margin-bottom: 8px;
                               user-select: all; cursor: text; word-break: break-all;
                               display: none; }
        #workdir-unc-display.visible { display: block; }
        /* Amber speech bubble (left edge aligned with Load Results centerline)
           with an upward-pointing tail at the top-left, drawing attention to
           the workdir/Load-Results area. */
        #workdir-hint { position: fixed; top: 52px; left: 419px;
                        padding: 16px 22px; background: #ffd54f; color: #2a1a05;
                        border: 2px solid #c98800; border-radius: 10px;
                        font-size: 16px; font-weight: 700; line-height: 1.4;
                        box-shadow: 0 6px 24px rgba(0,0,0,0.55);
                        z-index: 30; max-width: 380px;
                        animation: workdir-hint-pulse 1.6s ease-in-out infinite; }
        #workdir-hint::before { content: ''; position: absolute; top: -12px; left: 36px;
                                 border-left: 12px solid transparent;
                                 border-right: 12px solid transparent;
                                 border-bottom: 12px solid #c98800; }
        #workdir-hint::after  { content: ''; position: absolute; top: -9px; left: 38px;
                                 border-left: 10px solid transparent;
                                 border-right: 10px solid transparent;
                                 border-bottom: 10px solid #ffd54f; }
        @keyframes workdir-hint-pulse {
            0%, 100% { transform: translateY(0); }
            50%      { transform: translateY(-4px); }
        }
        #banner-meta { white-space: nowrap; overflow: hidden; flex-shrink: 0;
                       text-overflow: ellipsis; text-align: right; font-size: 11px;
                       margin-left: auto; }
        #banner-title { position: absolute; left: 50%; transform: translateX(-50%);
                        font-weight: 800; font-size: 13px; letter-spacing: 2.5px;
                        color: #334; pointer-events: none; white-space: nowrap;
                        font-family: system-ui, sans-serif; }
        @media (max-width: 1000px) { #banner-title { display: none; } }
        @media (max-width: 700px)  { #banner-meta  { display: none; } }
        #sidebar { position: fixed; top: 40px; left: 0; bottom: 0; width: 340px;
                   overflow-y: auto; padding: 12px; box-sizing: border-box;
                   background: #f8f8fb; border-right: 1px solid #ddd;
                   font: 12px system-ui, sans-serif; z-index: 4; }
        #right-sidebar { position: fixed; top: 40px; right: 0; bottom: 0; width: 480px;
                         overflow-y: auto; padding: 10px 14px; box-sizing: border-box;
                         background: #f4f5f9; border-left: 1px solid #d4d8e4;
                         font: 12px system-ui, sans-serif; z-index: 4;
                         transform: translateX(100%); transition: transform 0.2s ease; }
        #right-sidebar.open { transform: translateX(0); }
        #right-sidebar hr { border: none; border-top: 1px solid #d0d8e8; margin: 8px -14px; }
        #btn-rsb-toggle { position: fixed; top: calc(40px + (100vh - 40px) / 2); transform: translateY(-50%); right: 0;
                          width: 22px; background: #d4dae8; color: #446;
                          border: 1px solid #b8c2d4; border-right: none;
                          border-radius: 5px 0 0 5px; padding: 18px 3px;
                          cursor: pointer; font-size: 11px; line-height: 1;
                          z-index: 6; transition: right 0.2s ease, background 0.15s;
                          text-align: center; }
        #btn-rsb-toggle:hover { background: #c4cade; }
        #btn-rsb-toggle.open { right: 480px; }
        #sidebar h4 { margin: 0 0 8px; font-size: 13px; }
        #sidebar label { display: block; font-size: 10px; color: #555;
                         margin: 6px 0 1px; font-weight: 600; }
        .row-with-preview { display: flex; gap: 6px; align-items: center; }
        .row-with-preview > :first-child { flex: 1; min-width: 0; }
        .secondary-btn { background: #eef2f7 !important; color: #3a6080 !important;
                         border: 1px solid #a8c0d8 !important;
                         padding: 6px 10px !important; font-size: 11px !important;
                         border-radius: 5px !important;
                         transition: background 0.15s, box-shadow 0.15s !important; }
        .secondary-btn:hover { background: #dce8f2 !important;
                               box-shadow: 0 2px 5px rgba(0,80,140,0.15) !important; }
        .secondary-btn:active { background: #c8daea !important;
                                box-shadow: inset 0 1px 3px rgba(0,0,0,0.12) !important; }
        .preview-box label { font-weight: normal !important; font-size: 10px !important;
                             color: #888; margin: 0 !important; white-space: nowrap; }
        .preview-info { font-size: 10px; color: #555; margin: 2px 0 0;
                        max-height: 60px; overflow: auto; white-space: pre-wrap; line-height: 1.3; }
        .preview-info.error { color: #c33; }
        #sidebar .row-buttons { display: flex; gap: 6px; margin-top: 12px; }
        #filter-section-wrap label { margin: 8px 0 4px; }
        #filter-section-wrap .row-buttons { margin-top: 4px; }
        #filter-section-wrap button { text-align: left; padding-left: 10px; }
        #filter-section-wrap button:disabled {
            background: linear-gradient(180deg, #6aabda, #4a85b5) !important;
            border-color: #4a85b5 !important; box-shadow: none !important;
            cursor: default !important; text-shadow: 0 1px 1px rgba(0,0,0,0.18) !important; }
        #sidebar button { flex: 1; padding: 8px 6px;
                          border: 1px solid #4a85b5;
                          background: linear-gradient(180deg, #6aabda, #4a85b5);
                          color: white; border-radius: 5px; font-weight: 600;
                          font-size: 11px; cursor: pointer; letter-spacing: 0.2px;
                          text-shadow: 0 1px 1px rgba(0,0,0,0.18);
                          transition: background 0.15s, box-shadow 0.15s; }
        #sidebar button:hover { background: linear-gradient(180deg, #5a9bca, #3a75a5);
                                box-shadow: 0 2px 6px rgba(0,80,140,0.25); }
        #sidebar button:active { background: linear-gradient(180deg, #3a75a5, #2a6595);
                                 box-shadow: inset 0 1px 3px rgba(0,0,0,0.2); }
        #sidebar button:disabled { background: #b8c8d8 !important; border-color: #a0b4c4 !important;
                                   box-shadow: none !important; cursor: wait; text-shadow: none; }
        #btn-waves { background: linear-gradient(180deg, #5abaaa, #3a9a8a) !important;
                     border-color: #3a9a8a !important; }
        #btn-waves:hover { background: linear-gradient(180deg, #4aaa9a, #2a8a7a) !important;
                           box-shadow: 0 2px 6px rgba(0,120,100,0.3) !important; }
        #btn-waves:active { background: linear-gradient(180deg, #2a8a7a, #1a7a6a) !important; }
        #btn-waves:disabled { background: #b8c8d8 !important; border-color: #a0b4c4 !important; }
        #new-folder-form { position: fixed; top: 46px; left: 4px; z-index: 30;
                           background: white; border: 1px solid #b0c4d8; border-radius: 6px;
                           padding: 8px 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.18);
                           display: flex; align-items: center; gap: 6px; }
        #setup-overlay { position: fixed; top: 40px; left: 0; right: 0; bottom: 0;
                         background: rgba(0,0,0,0.45); z-index: 20; cursor: not-allowed;
                         display: flex; align-items: center; justify-content: center; }
        #progress-log { background: #1e1e1e; color: #ddd; font: 11px ui-monospace, monospace;
                        padding: 8px; max-height: 160px; overflow: auto; white-space: pre-wrap;
                        border-radius: 3px; margin: 4px 0; }
        #deck-container { position: fixed; top: 40px; left: 340px; right: 0; bottom: 0;
                          z-index: 1; overflow: hidden; transition: right 0.2s ease; }
        #copy-toast { position: fixed; top: 0; left: 0;
                      background: rgba(10,20,40,0.88); color: #8df;
                      border: 1px solid rgba(80,180,240,0.35); border-radius: 6px;
                      padding: 7px 16px; font: 11px ui-monospace, monospace; z-index: 50;
                      pointer-events: none; white-space: nowrap;
                      opacity: 0; transition: opacity 0.25s ease; }
        #copy-toast.visible { opacity: 1; }
        #ctrl-hint { position: fixed; bottom: 20px; left: 350px; z-index: 5;
                     pointer-events: none; background: rgba(16,18,28,0.88);
                     border-radius: 8px; padding: 10px 14px;
                     border: 1px solid rgba(255,255,255,0.08);
                     box-shadow: 0 2px 12px rgba(0,0,0,0.45); }
        #ctrl-hint-title { font-size: 13px; font-weight: 800;
                           color: #eee; letter-spacing: 0.4px;
                           line-height: 1.2; }
        #ctrl-hint-body { font-size: 10px; color: rgba(200,220,255,0.75);
                          margin-top: 4px; line-height: 1.7; }
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
        /* Centred pill for "Rendering..." / "Ready" between transfer and first paint */
        #render-status { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                         background: #4ad; color: white; padding: 12px 28px;
                         border-radius: 8px; font: 13px monospace;
                         box-shadow: 0 4px 16px rgba(0,0,0,0.35); z-index: 150;
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


def compute_similar_tracks(df: 'pd.DataFrame', seed_mmsi: int, seed_seg: int,
                           buffer_m: float, min_coverage: float) -> list[list[int]]:
    """Return [[mmsi, segment_id], ...] for tracks that pass within buffer_m of the seed
    and have at least min_coverage fraction of their points inside the buffer polygon."""
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon
    from pyproj import Transformer

    seed_df = df[(df['mmsi'] == seed_mmsi) & (df['segment_id'] == seed_seg)]
    if len(seed_df) < 2:
        return []

    lon_c = float(seed_df['longitude'].mean())
    lat_c = float(seed_df['latitude'].mean())
    zone = int((lon_c + 180) / 6) + 1
    epsg = 32600 + zone if lat_c >= 0 else 32700 + zone

    to_utm = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    to_wgs = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)

    seed_xy = [to_utm.transform(lon, lat)
               for lon, lat in zip(seed_df['longitude'], seed_df['latitude'])]
    buf_utm = LineString(seed_xy).buffer(buffer_m)
    buf_wgs_coords = [to_wgs.transform(x, y) for x, y in buf_utm.exterior.coords]
    buf_poly = Polygon(buf_wgs_coords)

    other = df[~((df['mmsi'] == seed_mmsi) & (df['segment_id'] == seed_seg))].copy()
    if len(other) == 0:
        return []

    gdf = gpd.GeoDataFrame(
        other[['mmsi', 'segment_id']],
        geometry=gpd.points_from_xy(other['longitude'], other['latitude']),
        crs='EPSG:4326',
    )
    gdf_buf = gpd.GeoDataFrame(geometry=[buf_poly], crs='EPSG:4326')
    joined = gpd.sjoin(gdf, gdf_buf, how='left', predicate='within')
    joined['_in'] = ~joined['index_right'].isna()
    fractions = joined.groupby(['mmsi', 'segment_id'])['_in'].mean()
    return [[int(m), int(s)] for (m, s), frac in fractions.items() if frac >= min_coverage]


@app.server.route('/api/similar_tracks', methods=['POST'])
def _r_similar_tracks():
    body = request.get_json(force=True, silent=True) or {}
    seed_mmsi = body.get('mmsi')
    seed_seg  = body.get('segment_id')
    buffer_m  = float(body.get('buffer_m', 200))
    min_cov   = float(body.get('min_coverage', 0.5))
    if seed_mmsi is None or seed_seg is None:
        return jsonify({'error': 'mmsi and segment_id required'}), 400
    ref_df = df_vessels if len(df_vessels) > 0 else None
    if ref_df is None:
        return jsonify({'error': 'no track data — run Filter first'}), 400
    try:
        result = compute_similar_tracks(ref_df, int(seed_mmsi), int(seed_seg), buffer_m, min_cov)
        return jsonify({'mmsi_segs': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.server.route('/api/raster.png')
def _r_raster(): return Response(PNG_BYTES, mimetype='image/png')


# ---------------------------------------------------------------------------
# Load Results: scan for output dirs and load pre-computed results
# ---------------------------------------------------------------------------
def _scan_output_dirs() -> list[str]:
    """Return relative paths to directories that contain loadable results (parquet files)."""
    candidates = []
    # Standard output/ at repo root
    if (REPO / 'output').is_dir():
        candidates.append('output')
    # data/*/output  (one level of project subdirectory)
    data_root = REPO / 'data'
    if data_root.is_dir():
        for p in sorted(data_root.glob('*/output')):
            if p.is_dir():
                candidates.append(str(p.relative_to(REPO)).replace('\\', '/'))
    return candidates


def _load_results(directory: str) -> dict:
    """Load pre-computed vessel/wave results from *directory* and rebuild IPC caches.

    Resolution order per asset:
      Tracks → vessels.parquet → *_03_vessel.csv → *_01_filtered.csv
      Waves  → waves.parquet   → *_04_wave_impact.csv → shore_impact.csv
    """
    global df_vessels, df_waves, IPC_VESSELS, IPC_WAVES
    global IPC_TRACK_COORDS, IPC_TRACK_META, IPC_TRACK_OFFSETS, PNG_BYTES, seg_meta

    p = (REPO / directory).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f'directory not found: {directory}')

    # ---- Vessel / track data ----
    df_v = None
    for src in [
        p / 'vessels.parquet',
        *sorted(p.glob('*_03_vessel.csv')),
        *sorted(p.glob('*_01_filtered.csv')),
    ]:
        if src.exists():
            print(f'  loading vessels: {src.name}')
            df_v = pd.read_parquet(src) if src.suffix == '.parquet' else pd.read_csv(src)
            if 'obstime' in df_v.columns:
                df_v['obstime'] = pd.to_datetime(df_v['obstime'])
            break
    if df_v is None:
        raise FileNotFoundError('No vessel data found in directory')

    # ---- Wave / impact data ----
    df_w = None
    for src in [
        p / 'waves.parquet',
        *sorted(p.glob('*_04_wave_impact.csv')),
        p / 'shore_impact.csv',
    ]:
        if src.exists():
            print(f'  loading waves: {src.name}')
            df_w = pd.read_parquet(src) if src.suffix == '.parquet' else pd.read_csv(src)
            break

    _build_vessel_caches(df_v)
    # Seed LAST_RESULTS so a subsequent "Calculate Waves" run skips the filter stage.
    LAST_RESULTS['df_filtered'] = df_v
    if df_w is not None:
        # Normalise DateTime to datetime64 so the obstime join key matches
        if 'DateTime' in df_w.columns and df_w['DateTime'].dtype == object:
            df_w['DateTime'] = pd.to_datetime(df_w['DateTime'], errors='coerce')
        enriched = _ensure_vessel_columns(df_w, df_v) if len(df_v) > 0 else df_w
        _build_wave_caches(enriched)
        LAST_RESULTS['df_wave_impact'] = enriched
    else:
        # Reset wave caches to empty
        _build_wave_caches(df_waves.iloc[:0].copy())
        LAST_RESULTS.pop('df_wave_impact', None)

    with _pipeline_lock:
        PIPELINE_STATE['track_version'] += 1
        if df_w is not None:
            PIPELINE_STATE['wave_version'] += 1
        tv = PIPELINE_STATE['track_version']
        wv = PIPELINE_STATE['wave_version']

    bbox = None
    if len(df_v) > 0 and {'longitude', 'latitude'} <= set(df_v.columns):
        lon = df_v['longitude'].to_numpy()
        lat = df_v['latitude'].to_numpy()
        mask = np.isfinite(lon) & np.isfinite(lat)
        if mask.any():
            bbox = [float(lon[mask].min()), float(lat[mask].min()),
                    float(lon[mask].max()), float(lat[mask].max())]
    return {
        'track_version': tv,
        'wave_version': wv,
        'n_segs': len(seg_meta),
        'n_waves': len(df_waves),
        'source': directory,
        'bbox': bbox,
    }


@app.server.route('/api/load_results', methods=['POST'])
def _r_load_results():
    body = request.get_json(force=True, silent=True) or {}
    directory = body.get('directory', '')
    if not directory:
        return jsonify({'error': 'directory required'}), 400
    try:
        result = _load_results(directory)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Export filtered tracks/waves/AIS subset → new data/ subfolder (rerun-ready)
# ---------------------------------------------------------------------------
_FORBIDDEN_NAME_CHARS = ('/', '\\', '..', ':', '\0')


def _export_filtered(body: dict) -> dict:
    """Export a rerun-ready slice of the current pipeline state.

    Layout written under ``data/<dest_name>/``::

        ais/<original_stem>.csv      ← cleaned filtered points (no segment_id)
        coastline/                   ← every file copied from source workdir
        land/                        ← every file copied from source workdir
        bathymetry/                  ← every file copied from source workdir
        tide/                        ← every file copied from source workdir
        output/vessels.parquet       ← filtered tracks
        output/waves.parquet         ← filtered waves
        output/wave_track_link.csv   ← wave_row → MMSI/segment_id mapping
    """
    dest_name = (body.get('dest_name') or '').strip()
    if not dest_name:
        raise ValueError('destination folder name is required')
    if any(c in dest_name for c in _FORBIDDEN_NAME_CHARS):
        raise ValueError(f'invalid characters in folder name: {dest_name!r}')
    workdir = (body.get('workdir') or '').strip()
    if not workdir:
        raise ValueError('source workdir is required')
    seg_keys = body.get('seg_keys') or []
    if not seg_keys:
        raise ValueError('no filter is active — nothing to export')
    seg_key_set = {(int(m), int(s)) for m, s in seg_keys}
    wave_idxs = body.get('wave_idxs')  # may be None or list[int]
    sel_ais = body.get('sel_ais') or ''

    base = DATA_ROOT / dest_name
    if base.exists():
        raise FileExistsError(f'destination already exists: data/{dest_name}')
    src_root = REPO / workdir
    if not src_root.is_dir():
        raise FileNotFoundError(f'source workdir not found: {workdir}')

    created = False
    try:
        # Layout
        for sub in ('ais', 'coastline', 'land', 'bathymetry', 'tide', 'output'):
            (base / sub).mkdir(parents=True, exist_ok=True)
        created = True

        # ---- 1. AIS subset (cleaned post-filter, drop segment_id) ----
        df_f = LAST_RESULTS.get('df_filtered')
        if df_f is None or len(df_f) == 0:
            raise RuntimeError('no filtered AIS available — run Filter or Load Results first')
        keys = list(zip(df_f['mmsi'].astype(int), df_f['segment_id'].astype(int)))
        mask = pd.Series([k in seg_key_set for k in keys], index=df_f.index)
        ais_subset = df_f.loc[mask].copy()
        ais_cols = ['mmsi', 'width', 'length', 'draught', 'obstime',
                    'longitude', 'latitude', 'sog', 'cog', 'typecargo']
        keep_cols = [c for c in ais_cols if c in ais_subset.columns]
        ais_subset = ais_subset[keep_cols]
        ais_stem = Path(sel_ais).stem if sel_ais else 'ais_filtered'
        ais_out = base / 'ais' / f'{ais_stem}.csv'
        ais_subset.to_csv(ais_out, index=False)
        n_ais = len(ais_subset)

        # ---- 2. Bulk-copy input directories (coastline/land/bathymetry/tide) ----
        copied = {}
        for sub in ('coastline', 'land', 'bathymetry', 'tide'):
            src_sub = src_root / sub
            dst_sub = base / sub
            count = 0
            if src_sub.is_dir():
                for f in src_sub.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dst_sub / f.name)
                        count += 1
            copied[sub] = count

        # ---- 3. Rerun-ready cached results in output/ ----
        out_dir = base / 'output'
        tracks_mask = pd.Series(
            [k in seg_key_set for k in zip(df_vessels['mmsi'].astype(int),
                                           df_vessels['segment_id'].astype(int))],
            index=df_vessels.index,
        ) if len(df_vessels) > 0 else pd.Series([], dtype=bool)
        df_tracks_out = df_vessels.loc[tracks_mask].reset_index(drop=True)
        df_tracks_out.to_parquet(out_dir / 'vessels.parquet', index=False)
        n_tracks_out = len(df_tracks_out)

        n_waves_out = 0
        if len(df_waves) > 0:
            if wave_idxs is not None:
                idxs = [int(i) for i in wave_idxs if 0 <= int(i) < len(df_waves)]
                df_waves_out = df_waves.iloc[idxs].reset_index(drop=True)
            else:
                wmask = pd.Series(
                    [(int(m), int(s)) in seg_key_set for m, s in
                     zip(df_waves['MMSI'], df_waves['segment_id'])],
                    index=df_waves.index,
                )
                df_waves_out = df_waves.loc[wmask].reset_index(drop=True)
            df_waves_out.to_parquet(out_dir / 'waves.parquet', index=False)
            n_waves_out = len(df_waves_out)
        else:
            df_waves_out = df_waves.iloc[:0].copy()
            df_waves_out.to_parquet(out_dir / 'waves.parquet', index=False)

        # ---- 4. Wave↔track link sidecar ----
        _write_wave_track_link(df_waves_out, out_dir)

        # ---- 5. Report plots for filtered results ----
        cfg_snap = PIPELINE_STATE.get('cfg')
        if cfg_snap and n_waves_out > 0 and n_tracks_out > 0:
            try:
                _generate_report_plots(cfg_snap, df_tracks_out, df_waves_out, out_dir=out_dir)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'WARN: report plots failed for export: {e}')

        return {
            'workdir': f'data/{dest_name}',
            'n_tracks': int(n_tracks_out),
            'n_waves': int(n_waves_out),
            'n_ais': int(n_ais),
            'copied': copied,
        }
    except Exception:
        if created:
            shutil.rmtree(base, ignore_errors=True)
        raise


@app.server.route('/api/export/filtered', methods=['POST'])
def _r_export_filtered():
    body = request.get_json(force=True, silent=True) or {}
    try:
        result = _export_filtered(body)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _safe_repo_path(rel: str) -> Path:
    """Resolve a user-supplied relative path to an absolute path under REPO,
    raising if it tries to escape the repo root (defence against path traversal).
    Uses normpath rather than resolve so symlinked subdirectories (e.g. data/)
    are allowed — resolve() would follow the symlink outside the repo root."""
    if not rel:
        raise ValueError('empty path')
    norm = Path(os.path.normpath(REPO / rel))
    try:
        norm.relative_to(REPO)
    except ValueError:
        raise ValueError(f'path {rel!r} escapes repo root')
    if not norm.exists():
        raise FileNotFoundError(rel)
    return norm


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
        return jsonify(_preview_ais_bbox(p))
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.server.route('/api/preview/coast.geojson')
def _r_preview_coast():
    try:
        p = _safe_repo_path(request.args.get('path', ''))
        return jsonify(_preview_coast_geojson(p))
    except Exception as exc:
        return jsonify(error=str(exc)), 400


def _parse_bbox(bbox_str: str) -> tuple | None:
    if not bbox_str:
        return None
    try:
        parts = [float(x) for x in bbox_str.split(',')]
        return tuple(parts) if len(parts) == 4 else None
    except ValueError:
        return None


@app.server.route('/api/preview/bathy.arrow')
def _r_preview_bathy():
    try:
        p = _safe_repo_path(request.args.get('path', ''))
        bbox = _parse_bbox(request.args.get('bbox', ''))
        result = _preview_bathy_arrow(p, bbox)
        if result is None:
            return jsonify(error='dfs2 grid preview not implemented'), 400
        coords_ipc, _ = result
        return _bytes_response(coords_ipc)
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.server.route('/api/preview/bathy_offsets.arrow')
def _r_preview_bathy_offsets():
    try:
        p = _safe_repo_path(request.args.get('path', ''))
        bbox = _parse_bbox(request.args.get('bbox', ''))
        result = _preview_bathy_arrow(p, bbox)
        if result is None:
            return jsonify(error='dfs2 grid preview not implemented'), 400
        _, offsets_ipc = result
        return _bytes_response(offsets_ipc)
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


# All defaults start blank — user must select files explicitly.


def _picker_with_preview(label: str, dropdown_id: str, preview_id: str,
                         info_id: str, options: list, value, clearable=False,
                         placeholder='', preview_disabled=False):
    """Dropdown row with a preview tickbox on the side. Returns a flat list for * unpacking."""
    preview_opts = [{'label': 'preview', 'value': '1', 'disabled': preview_disabled}]
    return [
        html.Label(label),
        html.Div([
            dcc.Dropdown(id=dropdown_id, options=_opt_list(options),
                         value=value, clearable=clearable, placeholder=placeholder,
                         className='compact-dropdown', style={'fontSize': '10px'}),
            html.Div(
                dcc.Checklist(id=preview_id, options=preview_opts, value=[]),
                className='preview-box',
            ),
        ], className='row-with-preview'),
        html.Div(id=info_id, className='preview-info'),
    ]


_RSB_SECTION = {'fontSize': '11px', 'fontWeight': '700', 'color': '#8899bb',
                'textTransform': 'uppercase', 'letterSpacing': '0.8px',
                'margin': '8px 0 3px', 'borderBottom': '1px solid #d8dfe8',
                'paddingBottom': '2px'}
_RSB_INPUT = {'width': '76px', 'fontSize': '11px', 'padding': '2px 5px',
              'boxSizing': 'border-box', 'border': '1px solid #c8d4e0',
              'borderRadius': '3px', 'background': '#f6f8fb', 'color': '#223',
              'fontFamily': 'ui-monospace, Consolas, monospace',
              'flexShrink': '0', 'textAlign': 'right'}
_RSB_DESC = {'fontSize': '10px', 'color': '#8899aa', 'margin': '0',
             'lineHeight': '1.25'}
_RSB_LABEL = {'fontWeight': '700', 'fontSize': '11px', 'color': '#445',
              'lineHeight': '1.2', 'marginBottom': '1px'}
_RSB_ROW = {'display': 'flex', 'alignItems': 'flex-start',
            'marginBottom': '2px', 'gap': '8px'}


def _rsb_num(label, pid, default, desc):
    return html.Div([
        html.Div([
            html.Div(label, style=_RSB_LABEL),
            html.Div(desc, style=_RSB_DESC),
        ], style={'flex': 1, 'minWidth': 0}),
        dcc.Input(id=pid, type='number', value=default, debounce=True, style=_RSB_INPUT),
    ], style=_RSB_ROW)


def _rsb_dd(label, pid, options, default, desc):
    return html.Div([
        html.Div([
            html.Div(label, style=_RSB_LABEL),
            html.Div(desc, style=_RSB_DESC),
        ], style={'flex': 1, 'minWidth': 0}),
        html.Div(
            dcc.Dropdown(id=pid, options=options, value=default, clearable=False,
                         className='compact-dropdown', style={'fontSize': '10px'}),
            style={'width': '150px', 'flexShrink': '0'},
        ),
    ], style=_RSB_ROW)

app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(id='sel-workdir', options=[], value=None, clearable=False,
                         placeholder='Select a data/ subfolder...',
                         className='compact-dropdown',
                         style={'fontSize': '10px'}),
            html.Button('↻', id='btn-rescan-workdir', n_clicks=0,
                        title='Re-scan working directory for new files'),
        ], id='workdir-wrapper'),
        html.Button('New Folder', id='btn-new-folder', n_clicks=0,
                    title='Create a new project folder in data/',
                    style={'padding': '3px 10px', 'fontSize': '11px', 'background': '#e8f0f8',
                           'border': '1px solid #a8c0d8', 'borderRadius': '4px',
                           'cursor': 'pointer', 'color': '#3a6080', 'whiteSpace': 'nowrap',
                           'flexShrink': '0', 'fontWeight': '600',
                           'lineHeight': '1.4', 'height': '26px', 'boxSizing': 'border-box'}),
        html.Button('Load Results', id='btn-load-results', n_clicks=0,
                    title='Load pre-computed output from the selected working directory',
                    style={'padding': '3px 10px', 'fontSize': '11px', 'background': '#f2ede4',
                           'border': '1px solid #c8b488', 'borderRadius': '4px',
                           'cursor': 'pointer', 'color': '#7a6040', 'whiteSpace': 'nowrap',
                           'flexShrink': '0', 'fontWeight': '600',
                           'lineHeight': '1.4', 'height': '26px', 'boxSizing': 'border-box'}),
        html.Span('', id='load-results-status',
                  style={'fontSize': '11px', 'color': '#c44', 'marginLeft': '2px'}),
        html.Div('AISWAKEPY', id='banner-title'),
        html.Div([
            html.Span(id='ais-time-range', style={'color': '#558', 'marginRight': '4px'}),
            html.Span(id='cnt-vessels', children=f'vessels {len(df_vessels):,}'),
            ' | ', html.Span(id='cnt-segs',    children=f'segments {len(seg_meta):,}'),
            ' | ', html.Span(id='cnt-waves',   children=f'waves {len(df_waves):,}'),
            ' | ', html.Span(id='status', children='loading...'),
            ' | ', html.Span(id='click-info', style={'fontWeight': 'bold'}),
        ], id='banner-meta'),
    ], id='status-banner'),

    html.Div([

        html.Div(id='workdir-unc-display'),

        # ---- AIS Data (always first — everything else requires it) ----
        html.Label('AIS Data'),
        html.Div([
            dcc.Dropdown(id='sel-ais', options=[], value=None, clearable=False,
                         placeholder='Select AIS CSV...',
                         className='compact-dropdown', style={'fontSize': '10px'}),
            html.Div(
                dcc.Checklist(id='pv-ais',
                              options=[{'label': 'preview', 'value': '1'}],
                              value=[]),
                className='preview-box',
            ),
        ], className='row-with-preview'),
        html.Div(id='pv-ais-info', className='preview-info'),

        # ---- Remaining pickers + run button (disabled until AIS is selected) ----
        html.Div(id='pickers-need-ais',
                 style={'pointerEvents': 'none', 'opacity': '0.5'}, children=[

            # ---- Coastline & Land mask ----
            *_picker_with_preview('Coastline (block/calculate waves)', 'sel-coast', 'pv-coast',
                                  'pv-coast-info', [], None, clearable=False,
                                  placeholder='Select shapefile...'),
            *_picker_with_preview('Land (filter AIS data)', 'sel-land', 'pv-land',
                                  'pv-land-info', [], None, clearable=False,
                                  placeholder='Select shapefile...'),

            # ---- Bathymetry ----
            *_picker_with_preview('Bathymetry', 'sel-bathy', 'pv-bathy',
                                  'pv-bathy-info', [], None, clearable=False,
                                  placeholder='Select .dfsu or .mesh file...',
                                  preview_disabled=True),

            # ---- Tide DFS0 ----
            html.Label('Tide'),
            html.Div([
                html.Div('No tide file', id='cascade-tide-trigger', className='cascade-trigger'),
                html.Div(id='cascade-tide-panel', className='cascade-panel',
                         style={'display': 'none'}),
            ], id='cascade-tide', style={'position': 'relative', 'marginBottom': '4px'}),

            html.Div([
                html.Button('Calculate Waves', id='btn-waves', n_clicks=0, disabled=True,
                            title='Run AIS filter + interpolate + vessel params '
                                  '+ wave impact (requires bathymetry)'),
            ], className='row-buttons'),

        ]),

        # Hidden Dash components for tide — outside the pointer-events wrapper so
        # JS-driven value bumps are never blocked.
        html.Div(style={'display': 'none'}, children=[
            dcc.Dropdown(id='sel-tide', options=[], value=None, clearable=True),
            dcc.Dropdown(id='sel-tide-item', options=[], value=None, clearable=False),
            html.Button(id='_tide-file-btn', n_clicks=0),
            html.Button(id='_tide-item-btn', n_clicks=0),
            dcc.Dropdown(id='sel-results-dir', options=[], value=None, clearable=True),
        ]),
        dcc.Store(id='_tide_file_pick', data={'value': None, 'nonce': 0}),
        dcc.Store(id='_tide_item_pick', data={'value': None, 'nonce': 0}),

        # ---- Track visualization filter (disabled until waves are loaded) ----
        html.Div(id='filter-section-wrap',
                 style={'pointerEvents': 'none', 'opacity': '0.5'}, children=[

            html.Hr(),
            html.Div('Vessel Track Visualization Filter',
                     style={'fontWeight': 'bold', 'fontSize': '11px', 'color': '#555'}),
            # Hidden Dash dropdowns for callback compatibility (driven by JS cascade widget)
            dcc.Dropdown(id='fil-mmsi', options=[], value=None, clearable=True,
                         style={'display': 'none'}),
            dcc.Dropdown(id='fil-segs', options=[], value=None, multi=True,
                         style={'display': 'none'}),
            # Cascading MMSI → Segment picker (populated by JS from track data)
            html.Label('MMSI / Track Segment'),
            html.Div([
                html.Div('All tracks', id='cascade-mmsi-trigger', className='cascade-trigger'),
                html.Div(id='cascade-mmsi-panel', className='cascade-panel',
                         style={'display': 'none'}),
            ], id='cascade-mmsi-seg', style={'position': 'relative', 'marginBottom': '4px'}),
            html.Label('Vessel Type'),
            dcc.Dropdown(id='fil-type', options=[], value=None, multi=True, clearable=True,
                         placeholder='All types', searchable=True,
                         className='compact-dropdown', style={'fontSize': '10px'}),
            html.Label('Free-hand selection'),
            html.Div([
                html.Button('Draw line across tracks', id='btn-freehand', n_clicks=0,
                            title='Draw a line on the map — selects all crossed tracks'),
            ], className='row-buttons'),
            html.Label('Similar selection'),
            html.Div([
                html.Button('Select one representative track', id='btn-similar', n_clicks=0,
                            title='Pick a track then find tracks with similar routing'),
            ], className='row-buttons'),
            # Inline Similar panel (hidden until a track is picked)
            html.Div([
                html.Div('', id='sim-picked-label',
                         style={'fontSize': '10px', 'color': '#445', 'marginBottom': '5px',
                                'fontWeight': '600'}),
                html.Div([
                    html.Span('Buffer (m):', style={'fontSize': '10px', 'marginRight': '4px',
                                                    'whiteSpace': 'nowrap'}),
                    dcc.Input(id='sim-buffer-m', type='number', value=200, debounce=True,
                              style={'width': '70px', 'fontSize': '10px', 'padding': '2px 4px',
                                     'border': '1px solid #c8d4e0', 'borderRadius': '3px'}),
                    html.Span('Coverage:', style={'fontSize': '10px', 'margin': '0 4px',
                                                  'whiteSpace': 'nowrap'}),
                    dcc.Input(id='sim-coverage', type='number', value=0.5, debounce=True,
                              style={'width': '60px', 'fontSize': '10px', 'padding': '2px 4px',
                                     'border': '1px solid #c8d4e0', 'borderRadius': '3px'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px',
                          'flexWrap': 'wrap', 'gap': '3px'}),
                html.Div([
                    html.Button('Confirm', id='btn-sim-confirm', n_clicks=0,
                                className='secondary-btn'),
                    html.Button('Cancel', id='btn-sim-cancel', n_clicks=0,
                                className='secondary-btn'),
                ], className='row-buttons'),
            ], id='sim-panel',
               style={'display': 'none', 'background': '#eef2f7', 'borderRadius': '4px',
                      'padding': '7px 8px', 'margin': '4px 0',
                      'border': '1px solid #bcd0e4'}),
            html.Label('Wave arrival area'),
            html.Div([
                html.Button('Drag box on the map', id='btn-wavebox', n_clicks=0,
                            title='Drag a rectangle on the map — keeps only waves landing '
                                  'inside, plus the tracks that produced them'),
            ], className='row-buttons'),
            html.Div([
                html.Button('Clear all filters', id='btn-fil-clear', n_clicks=0),
                html.Button('Export filtered →', id='btn-fil-export', n_clicks=0,
                            title='Save filtered tracks, waves, AIS subset and input file '
                                  'copies into a new data/ subfolder'),
            ], className='row-buttons'),
            html.Div(id='export-dest-form', style={'display': 'none'}, children=[
                dcc.Input(id='inp-export-dest', type='text',
                          placeholder='new folder name (under data/)',
                          debounce=False,
                          style={'fontSize': '10px', 'width': '100%',
                                 'padding': '3px 6px', 'marginTop': '4px',
                                 'border': '1px solid #c8d4e0', 'borderRadius': '3px',
                                 'boxSizing': 'border-box'}),
            ]),
            html.Div('', id='export-status',
                     style={'fontSize': '10px', 'color': '#556', 'marginTop': '3px',
                            'minHeight': '14px', 'whiteSpace': 'pre-wrap'}),
            html.Div('', id='fil-status',
                     style={'fontSize': '10px', 'color': '#556', 'marginTop': '3px',
                            'minHeight': '14px'}),

        ]),

        html.Hr(),
        html.Div('Progress', style={'fontWeight': 'bold'}),
        html.Pre(id='progress-log', children='(idle)'),
        html.Div(id='progress-elapsed-side',
                 style={'fontSize': '11px', 'color': '#666', 'marginTop': '4px'}),

    ], id='sidebar'),

    html.Div(id='deck-container'),
    # New folder form — fixed panel below the workdir dropdown
    html.Div(id='new-folder-form', style={'display': 'none'}, children=[
        dcc.Input(id='inp-new-folder', type='text', placeholder='folder name',
                  debounce=False,
                  style={'fontSize': '11px', 'padding': '4px 8px', 'borderRadius': '4px',
                         'border': '1px solid #b0c4d8', 'width': '160px',
                         'outline': 'none'}),
        html.Button('Create', id='btn-create-folder', n_clicks=0,
                    style={'fontSize': '11px', 'padding': '4px 10px', 'background': '#4a85b5',
                           'color': 'white', 'border': '1px solid #3a75a5', 'borderRadius': '4px',
                           'cursor': 'pointer', 'fontWeight': '600'}),
        html.Button('✕', id='btn-cancel-folder', n_clicks=0,
                    style={'fontSize': '11px', 'padding': '4px 8px', 'background': '#eef2f7',
                           'border': '1px solid #a8c0d8', 'borderRadius': '4px',
                           'cursor': 'pointer', 'color': '#556'}),
        html.Span(id='new-folder-status', style={'fontSize': '10px', 'color': '#c44'}),
    ]),
    # Setup overlay — blocks UI until a working directory is selected
    html.Div(id='setup-overlay'),
    # Speech bubble pointing to the workdir dropdown when no folder selected.
    html.Div('Select or create a data folder to begin', id='workdir-hint'),
    # MMSI copy toast — appears briefly after Ctrl+click copies MMSI
    html.Div('', id='copy-toast'),
    # Ctrl hint — permanent floating label at bottom-left of canvas
    html.Div([
        html.Div('Hold Ctrl:', id='ctrl-hint-title'),
        html.Div([
            html.Div('+ hover for vessel / track / wave'),
            html.Div('+ click to pin & copy MMSI'),
        ], id='ctrl-hint-body'),
    ], id='ctrl-hint'),
    # Freehand draw canvas (overlaid on deck-container, managed by JS)
    html.Canvas(id='freehand-canvas', style={
        'position': 'fixed', 'top': '40px', 'left': '340px', 'right': '0', 'bottom': '0',
        'pointerEvents': 'none', 'zIndex': '3', 'display': 'none',
    }),
    # Floating legend (bottom-right of map area)
    html.Div(id='map-legend', style={
        'position': 'fixed', 'bottom': '20px', 'right': '50px',
        'zIndex': '10', 'background': 'rgba(16,18,28,0.88)',
        'borderRadius': '8px', 'padding': '10px 14px',
        'color': '#eee', 'fontSize': '10px', 'fontFamily': 'system-ui, sans-serif',
        'minWidth': '160px', 'display': 'none',
        'boxShadow': '0 2px 12px rgba(0,0,0,0.45)',
        'border': '1px solid rgba(255,255,255,0.08)',
        'lineHeight': '1.4',
    }),

    # ---- Collapsible right sidebar: pipeline configuration ----
    html.Div([
        html.Div('⚙ Pipeline Config',
                 style={'fontWeight': '700', 'fontSize': '15px', 'color': '#334',
                        'marginBottom': '2px', 'letterSpacing': '0.3px',
                        'fontFamily': 'system-ui, sans-serif'}),
        html.Div('Parameters read at run time — changes take effect on next Filter / Calculate.',
                 style={'fontSize': '9px', 'color': '#99a', 'marginBottom': '6px',
                        'lineHeight': '1.3'}),

        # ── AIS Cleaning ──────────────────────────────────────────────────
        html.Div('AIS Cleaning', style=_RSB_SECTION),
        _rsb_num('Minimum Speed Filter (min_speed_knots)', 'rsb-min-speed', 0.0,
                 'Strip AIS fixes where SOG is below this threshold'),
        _rsb_num('Time Gap to Split Segments (traj_gap_s)', 'rsb-traj-gap', 180.0,
                 'Split a track into separate segments when AIS silence exceeds this'),
        _rsb_num('Max Position Jump Speed (max_velocity_knots)', 'rsb-max-velocity', 36.0,
                 'Flag fix as positional error if implied displacement speed exceeds this'),
        _rsb_num('Max Implied Acceleration (max_acceleration_ms2)', 'rsb-max-accel', 10.0,
                 'Flag SOG/COG as erroneous if implied acceleration between fixes exceeds this'),
        _rsb_num('Max Draught-to-Beam Ratio (max_draught_to_width)', 'rsb-max-dw', 1.0,
                 'Discard vessel if reported draught divided by beam exceeds this ratio'),
        _rsb_num('Stationary Speed Threshold (low_sog_threshold_ms)', 'rsb-low-sog', 1.0,
                 'SOG below this is treated as stationary — affects mixed interpolation and speed checks'),
        _rsb_num('Position vs. Reported Speed Ratio (velocity_ratio_threshold)', 'rsb-vel-ratio', 2.0,
                 'Flag fix if GPS-derived displacement speed exceeds this multiple of the reported SOG'),
        _rsb_num('Min Speed Consistency Ratio (speed_consistency_ratio)', 'rsb-spd-ratio', 0.5,
                 'Flag fix if GPS-derived speed falls below this fraction of the AIS velocity vector'),

        # ── Interpolation ─────────────────────────────────────────────────
        html.Div('Interpolation', style=_RSB_SECTION),
        _rsb_dd('Interpolation Method (interp_method)', 'rsb-interp',
                [{'label': 'linear — straight-line between raw fixes (default)', 'value': 'linear'},
                 {'label': 'hermite — cubic spline using SOG/COG as velocity', 'value': 'hermite'},
                 {'label': 'mixed — linear when both ends stationary', 'value': 'mixed'}],
                'linear',
                'Method used to fill gaps between consecutive raw AIS fixes'),
        _rsb_num('Resampling Time Step (interp_interval_s)', 'rsb-interp-interval', 30.0,
                 'Output time step after resampling the interpolated trajectory'),

        # ── Vessel ────────────────────────────────────────────────────────
        html.Div('Vessel', style=_RSB_SECTION),
        _rsb_dd('Block Coefficient Method (cb_method)', 'rsb-cb-method',
                [{'label': 'L_Le — waterline / bow-entry length ratio (default)', 'value': 'L_Le'},
                 {'label': 'B_Le — beam / bow-entry length ratio', 'value': 'B_Le'},
                 {'label': 'table — lookup Cb by vessel type', 'value': 'table'}],
                'L_Le',
                'Method used to estimate block coefficient Cb from vessel dimensions'),
        _rsb_num('Waterline Length Factor (waterline_factor)', 'rsb-waterline', 0.8,
                 'Waterline length = LOA × this factor (Kriebel requires Lw, not LOA)'),

        # ── Wave Model ────────────────────────────────────────────────────
        html.Div('Wave Model', style=_RSB_SECTION),
        _rsb_dd('Wake Height Formula (formula)', 'rsb-formula',
                [{'label': 'kriebel (2005) — default', 'value': 'kriebel'},
                 {'label': 'bhowmik (1982)', 'value': 'bhowmik'},
                 {'label': 'blaauw (1984)', 'value': 'blaauw'},
                 {'label': 'gates (1977)', 'value': 'gates'},
                 {'label': 'maynord (1987)', 'value': 'maynord'},
                 {'label': 'pianc (modified)', 'value': 'pianc'},
                 {'label': 'sorensen (1967)', 'value': 'sorensen'}],
                'kriebel',
                'Empirical formula used to compute wake wave height at the shore'),
        _rsb_num('Local Gravity (gravity)', 'rsb-gravity', 9.78,
                 'Gravitational acceleration — Singapore ≈ 9.78, international standard 9.81'),
        _rsb_num('Max Vessel Speed for Model (max_sog_knots)', 'rsb-max-sog', 12.0,
                 'Discard vessel events with SOG above this — outside the formula valid range'),
        _rsb_num('Max Beam-to-Length Ratio (max_bl_ratio)', 'rsb-max-bl', 0.3,
                 'Discard vessel if beam divided by length exceeds this — atypical hull shape'),

        # ── Kriebel Applicability Limits ──────────────────────────────────
        html.Div('Kriebel Limits', style=_RSB_SECTION),
        _rsb_num('Min Modified Froude Number (min_Froude_M)', 'rsb-min-froude', 0.1,
                 'Lower bound on Fr* — vessel must be in the wave-making regime'),
        _rsb_num('Max Modified Froude Number (max_Froude_M)', 'rsb-max-froude', 0.5,
                 'Upper bound on Fr* — avoids supercritical / planing regime'),
        _rsb_num('Max BF Coefficient (max_bf)', 'rsb-max-bf', 0.4,
                 'Upper bound on BF = β·(Fr*−0.1)² — no Kriebel calibration data exceeds 0.4'),

        # ── Wave Impact ───────────────────────────────────────────────────
        html.Div('Wave Impact', style=_RSB_SECTION),
        _rsb_num('Max Wake Ray Distance (max_propagation_m)', 'rsb-max-prop', 2000.0,
                 'Wake rays extending beyond this distance from the vessel are discarded'),
        _rsb_num('Min Recorded Wave Height (wake_cutoff_m)', 'rsb-wake-cutoff', 0.01,
                 'Minimum shore wave height to record — smaller events are dropped'),
    ], id='right-sidebar'),

    html.Button([
        html.Span('◀', id='rsb-arrow-chr'),
        html.Span('Pipeline Config', style={
            'display': 'block', 'writingMode': 'vertical-rl', 'fontSize': '9px',
            'letterSpacing': '1.2px', 'marginTop': '8px', 'opacity': '0.70',
            'textTransform': 'uppercase', 'textAlign': 'center',
        }),
    ], id='btn-rsb-toggle', n_clicks=0, title='Toggle pipeline configuration panel'),

    dcc.Store(id='_rescan_count', data=0),
    dcc.Store(id='_log_scroll', data=0),
    dcc.Interval(id='boot', max_intervals=1, interval=200),
    dcc.Interval(id='poll', interval=400, disabled=True),
    dcc.Store(id='_init'),
    dcc.Store(id='_wave_version', data=0),
    dcc.Store(id='_track_version', data=0),
    dcc.Store(id='_ais_import', data={'path': None, 'nonce': 0}),
    # Preview state Stores: {visible, path}
    dcc.Store(id='_pv_ais',   data={'visible': False, 'path': None}),
    dcc.Store(id='_pv_bathy', data={'visible': False, 'path': None}),
    dcc.Store(id='_pv_coast', data={'visible': False, 'path': None}),
    dcc.Store(id='_pv_land',  data={'visible': False, 'path': None}),
    dcc.Store(id='_filter_structural', data={'mmsi': None, 'seg_ids': [], 'types': [], 'nonce': 0}),
    dcc.Store(id='_load_result'),
    dcc.Store(id='_tide_items_meta', data=[]),
    dcc.Store(id='_tide_files_meta', data=[]),
    dcc.Store(id='_wave_n', data=0),
    dcc.Store(id='_any_filter_active', data=False),
])


# ---------------------------------------------------------------------------
# Working-directory callback: populates all file dropdowns
# ---------------------------------------------------------------------------
@app.callback(
    Output('sel-workdir', 'options'),
    Input('boot', 'n_intervals'),
    prevent_initial_call=False,
)
def _populate_workdir_dirs(_):
    return _scan_data_subdirs()


app.clientside_callback(
    """
    function(children) {
        var el = document.getElementById('progress-log');
        if (el) requestAnimationFrame(function(){ el.scrollTop = el.scrollHeight; });
        return window.dash_clientside.no_update;
    }
    """,
    Output('_log_scroll', 'data'),
    Input('progress-log', 'children'),
    prevent_initial_call=True,
)


@app.callback(
    Output('sel-ais',   'options'),
    Output('sel-bathy', 'options'),
    Output('sel-coast', 'options'),
    Output('sel-land',  'options'),
    Output('sel-tide',  'options'),
    Input('sel-workdir', 'value'),
    Input('_rescan_count', 'data'),
    prevent_initial_call=False,
)
def _update_file_dropdowns(workdir, _rescan):
    files = _scan_working_dir(workdir)
    return (
        _opt_list(files['ais']),
        _opt_list(files['bathymetry']),
        _opt_list(files['coastline']),
        _opt_list(files['land']),
        _opt_list(files['tide']),
    )


# When the user switches workdir, reset every downstream picker + active filters
# so the new folder is configured from a clean slate.
@app.callback(
    Output('sel-ais',       'value', allow_duplicate=True),
    Output('sel-bathy',     'value', allow_duplicate=True),
    Output('sel-coast',     'value', allow_duplicate=True),
    Output('sel-land',      'value', allow_duplicate=True),
    Output('sel-tide',      'value', allow_duplicate=True),
    Output('sel-tide-item', 'value', allow_duplicate=True),
    Output('pv-ais',        'value', allow_duplicate=True),
    Output('load-results-status', 'children', allow_duplicate=True),
    Output('export-status',   'children', allow_duplicate=True),
    Output('inp-export-dest', 'value', allow_duplicate=True),
    Output('export-dest-form', 'style', allow_duplicate=True),
    Output('_filter_structural', 'data', allow_duplicate=True),
    Output('fil-type', 'value', allow_duplicate=True),
    Input('sel-workdir', 'value'),
    State('_filter_structural', 'data'),
    prevent_initial_call=True,
)
def _reset_pickers_on_workdir_change(_workdir, prev_filter):
    nonce = ((prev_filter or {}).get('nonce', 0) + 1)
    return (None, None, None, None, None, None,
            [], '', '', '', {'display': 'none'},
            {'mmsi': None, 'seg_ids': [], 'types': [], 'nonce': nonce, '_clear': True},
            None)


@app.callback(
    Output('_rescan_count', 'data'),
    Input('btn-rescan-workdir', 'n_clicks'),
    State('_rescan_count', 'data'),
    prevent_initial_call=True,
)
def _rescan_workdir(_, count):
    return (count or 0) + 1


@app.callback(
    Output('workdir-unc-display', 'children'),
    Output('workdir-unc-display', 'className'),
    Input('sel-workdir', 'value'),
    Input('_rescan_count', 'data'),
    prevent_initial_call=False,
)
def _update_unc_display(workdir, _rescan):
    if not workdir or not DATA_UNC_ROOT:
        return '', ''
    subdir = workdir.removeprefix(f'{DATA_ROOT.name}/')
    unc = DATA_UNC_ROOT.rstrip('\\') + '\\' + subdir.replace('/', '\\')
    return unc, 'visible'


@app.callback(
    Output('setup-overlay', 'style'),
    Output('workdir-hint', 'style'),
    Input('sel-workdir', 'value'),
    prevent_initial_call=False,
)
def _toggle_setup_overlay(workdir):
    overlay_shown = {'position': 'fixed', 'top': '40px', 'left': 0, 'right': 0, 'bottom': 0,
                     'background': 'rgba(0,0,0,0.45)', 'zIndex': 20, 'cursor': 'not-allowed',
                     'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}
    if workdir:
        return {'display': 'none'}, {'display': 'none'}
    return overlay_shown, {}


@app.callback(
    Output('new-folder-form', 'style'),
    Input('btn-new-folder', 'n_clicks'),
    Input('btn-cancel-folder', 'n_clicks'),
    prevent_initial_call=True,
)
def _toggle_folder_form(n_new, n_cancel):
    from dash import ctx
    if ctx.triggered_id == 'btn-new-folder' and n_new:
        return {'display': 'flex'}
    return {'display': 'none'}


@app.callback(
    Output('sel-workdir', 'options', allow_duplicate=True),
    Output('sel-workdir', 'value', allow_duplicate=True),
    Output('new-folder-form', 'style', allow_duplicate=True),
    Output('new-folder-status', 'children'),
    Input('btn-create-folder', 'n_clicks'),
    State('inp-new-folder', 'value'),
    prevent_initial_call=True,
)
def _create_folder(n, name):
    if not n or not name or not name.strip():
        return no_update, no_update, no_update, 'Enter a folder name'
    name = name.strip()
    if any(c in name for c in ('/', '\\', '..', ':')):
        return no_update, no_update, no_update, 'Invalid name'
    base = DATA_ROOT / name
    for sub in ('ais', 'coastline', 'land', 'bathymetry', 'tide', 'output'):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return _scan_data_subdirs(), f'data/{name}', {'display': 'none'}, ''


# ---------------------------------------------------------------------------
# Server-side callbacks: run buttons + polling
# ---------------------------------------------------------------------------
def _build_config(ais, land, bathy, coast, tide, tide_item=None,
                  min_speed=0.0, traj_gap=180.0, interp='linear', interp_interval=30.0,
                  cb_method='L_Le', max_prop=2000.0, max_sog=12.0,
                  max_velocity=36.0, max_accel=10.0, max_dw=1.0,
                  low_sog=1.0, vel_ratio=2.0, spd_ratio=0.5,
                  waterline=0.8, formula='kriebel', gravity=9.78,
                  max_bl=0.3, min_froude=0.1, max_froude=0.5, max_bf=0.4,
                  wake_cutoff=0.01, workdir=None) -> dict:
    cfg = {
        'ais': {'raw_csv': ais, 'land_shp': land,
                'min_speed_knots': float(min_speed or 0.0),
                'traj_gap_s': float(traj_gap or 180.0),
                'interp_method': interp or 'linear',
                'interp_interval_s': float(interp_interval or 30.0),
                'max_velocity_knots': float(max_velocity or 36.0),
                'max_acceleration_ms2': float(max_accel or 10.0),
                'max_draught_to_width': float(max_dw or 1.0),
                'low_sog_threshold_ms': float(low_sog or 1.0),
                'velocity_ratio_threshold': float(vel_ratio or 2.0),
                'speed_consistency_ratio': float(spd_ratio or 0.5)},
        'vessel': {'cb_method': cb_method or 'L_Le',
                   'waterline_factor': float(waterline or 0.8)},
        'bathymetry': {'source': bathy or 'placeholder.mesh'},
        'coastline': {'shapefile': coast},
        'wave': {'max_sog_knots': float(max_sog or 12.0),
                 'formula': formula or 'kriebel',
                 'gravity': float(gravity or 9.78),
                 'max_bl_ratio': float(max_bl or 0.3),
                 'min_Froude_M': float(min_froude or 0.1),
                 'max_Froude_M': float(max_froude or 0.5),
                 'max_bf': float(max_bf or 0.4)},
        'impact': {'max_propagation_m': float(max_prop or 2000.0),
                   'wake_cutoff_m': float(wake_cutoff or 0.01)},
        'output': {'directory': f'{workdir}/output/' if workdir else 'output/',
                   'save_stage_csv': True},
    }
    if tide:
        cfg['bathymetry']['tide_dfs0'] = tide
        if tide_item:
            cfg['bathymetry']['tide_item'] = tide_item
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
    Output('btn-waves',  'disabled', allow_duplicate=True),
    Output('progress-log', 'children', allow_duplicate=True),
    Input('btn-waves', 'n_clicks'),
    State('sel-ais', 'value'), State('sel-land', 'value'),
    State('sel-bathy', 'value'), State('sel-coast', 'value'),
    State('sel-tide', 'value'), State('sel-tide-item', 'value'),
    State('rsb-min-speed', 'value'), State('rsb-traj-gap', 'value'),
    State('rsb-interp', 'value'), State('rsb-interp-interval', 'value'),
    State('rsb-cb-method', 'value'),
    State('rsb-max-prop', 'value'), State('rsb-max-sog', 'value'),
    State('rsb-max-velocity', 'value'), State('rsb-max-accel', 'value'),
    State('rsb-max-dw', 'value'), State('rsb-low-sog', 'value'),
    State('rsb-vel-ratio', 'value'), State('rsb-spd-ratio', 'value'),
    State('rsb-waterline', 'value'), State('rsb-formula', 'value'),
    State('rsb-gravity', 'value'), State('rsb-max-bl', 'value'),
    State('rsb-min-froude', 'value'), State('rsb-max-froude', 'value'),
    State('rsb-max-bf', 'value'), State('rsb-wake-cutoff', 'value'),
    State('sel-workdir', 'value'),
    prevent_initial_call=True,
)
def kick_waves(n, ais, land, bathy, coast, tide, tide_item,
               min_speed, traj_gap, interp, interp_interval,
               cb_method, max_prop, max_sog,
               max_velocity, max_accel, max_dw, low_sog,
               vel_ratio, spd_ratio, waterline, formula,
               gravity, max_bl, min_froude, max_froude, max_bf, wake_cutoff,
               workdir):
    if not n:
        return no_update, no_update, no_update
    missing = []
    if not ais:   missing.append('AIS data file')
    if not land:  missing.append('Land mask shapefile')
    if not coast: missing.append('Coastline shapefile')
    if not bathy: missing.append('Bathymetry file (required for depth check)')
    if missing:
        warn = '⚠ Cannot calculate waves — missing required inputs:\n  • ' + '\n  • '.join(missing)
        return no_update, no_update, warn
    cfg = _build_config(ais, land, bathy, coast, tide, tide_item,
                        min_speed, traj_gap, interp, interp_interval,
                        cb_method, max_prop, max_sog,
                        max_velocity, max_accel, max_dw, low_sog,
                        vel_ratio, spd_ratio, waterline, formula,
                        gravity, max_bl, min_froude, max_froude, max_bf, wake_cutoff,
                        workdir=workdir)
    if _kick(cfg, ['filter', 'vessel', 'wave_impact'], 'waves'):
        return False, True, no_update
    return no_update, no_update, no_update


# ---- AIS auto-import: fires whenever the user picks a new AIS file ----
@app.callback(
    Output('_ais_import', 'data'),
    Output('pv-ais', 'value', allow_duplicate=True),
    Input('sel-ais', 'value'),
    State('_ais_import', 'data'),
    prevent_initial_call=True,
)
def trigger_ais_import(path, prev):
    if not path:
        return no_update, no_update
    nonce = (prev or {}).get('nonce', 0) + 1
    # Auto-tick the preview tickbox so the imported data shows on map.
    return {'path': path, 'nonce': nonce}, ['1']


@app.callback(
    Output('pv-bathy', 'options'),
    Input('_ais_import', 'data'),
    prevent_initial_call=True,
)
def _enable_bathy_preview(data):
    if data and data.get('path'):
        return [{'label': 'preview', 'value': '1'}]
    return no_update


# btn-waves enabled only when all 5 input files are selected (including tide file + item).
app.clientside_callback(
    "function(a,c,l,b,t,ti){ return !(a&&c&&l&&b&&t&&ti); }",
    Output('btn-waves', 'disabled', allow_duplicate=True),
    Input('sel-ais', 'value'), Input('sel-coast', 'value'),
    Input('sel-land', 'value'), Input('sel-bathy', 'value'),
    Input('sel-tide', 'value'), Input('sel-tide-item', 'value'),
    prevent_initial_call='initial_duplicate',
)


@app.callback(
    Output('progress-log', 'children'),
    Output('progress-elapsed-side', 'children'),
    Output('poll', 'disabled'),
    Output('btn-waves',  'disabled'),
    Output('_wave_version',  'data'),
    Output('_track_version', 'data'),
    Output('cnt-waves',   'children'),
    Output('cnt-segs',    'children'),
    Output('cnt-vessels', 'children'),
    Output('ais-time-range', 'children'),
    Output('pv-bathy', 'value', allow_duplicate=True),
    Output('pv-coast', 'value', allow_duplicate=True),
    Output('pv-land',  'value', allow_duplicate=True),
    Output('pv-ais',   'value', allow_duplicate=True),
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
    elapsed = ''
    counts = (f'waves {len(df_waves):,}', f'segments {len(seg_meta):,}',
              f'vessels {len(df_vessels):,}')
    _no_pv = (no_update,) * 4
    if s['error']:
        return (
            f"{log_text}\n\nERROR: {s['error']}",
            elapsed, True, False,
            prev_wave_v, prev_track_v, *counts, no_update, *_no_pv,
        )
    if s['running']:
        return (
            log_text, elapsed, False, True,
            prev_wave_v, prev_track_v, no_update, no_update, no_update, no_update,
            *_no_pv,
        )
    # Finished — push fresh versions and clear all preview checkboxes.
    return (
        log_text, elapsed, True, False,
        s['wave_version'], s['track_version'], *counts, _ais_time_range_str(),
        [], [], [], [],
    )


# ---------------------------------------------------------------------------
# Preview callbacks: bathy / coast / tide listen to dropdown + tickbox.
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


_make_pv_callback('_pv_bathy', 'sel-bathy', 'pv-bathy')
_make_pv_callback('_pv_coast', 'sel-coast', 'pv-coast')
_make_pv_callback('_pv_land',  'sel-land',  'pv-land')


def _make_pv_auto_tick(sel_id, pv_id):
    @app.callback(
        Output(pv_id, 'value'),
        Input(sel_id, 'value'),
        prevent_initial_call=True,
    )
    def _auto(path):
        return ['1'] if path else []


_make_pv_auto_tick('sel-bathy', 'pv-bathy')
_make_pv_auto_tick('sel-coast', 'pv-coast')
_make_pv_auto_tick('sel-land',  'pv-land')


@app.callback(
    Output('sel-tide-item', 'options'),
    Output('sel-tide-item', 'value'),
    Output('_tide_items_meta', 'data'),
    Input('sel-tide', 'value'),
    prevent_initial_call=False,
)
def _populate_tide_items(path):
    if not path:
        return [], None, []
    try:
        data = _preview_tide(REPO / path)
        opts, items_meta = [], []
        for it in data['items']:
            rng = ''
            if it['value_min'] is not None and it['value_max'] is not None:
                rng = f" [{it['value_min']:.2f}..{it['value_max']:.2f} {it['unit']}]"
            opts.append({'label': f"{it['name']}{rng}", 'value': it['name']})
            items_meta.append({'name': it['name'], 'unit': it.get('unit', ''),
                                'label': f"{it['name']}{rng}"})
        val = opts[0]['value'] if len(opts) == 1 else None
        return opts, val, items_meta
    except Exception:
        return [], None, []


# AIS preview tickbox is decoupled from the dropdown — it just toggles
# visibility of the already-imported AIS data.


@app.callback(
    Output('sel-results-dir', 'options'),
    Input('_init', 'data'),
    prevent_initial_call=False,
)
def _populate_results_dirs(_):
    dirs = _scan_output_dirs()
    return [{'label': d, 'value': d} for d in dirs]


@app.callback(
    Output('_load_result', 'data'),
    Output('btn-load-results', 'disabled'),
    Input('btn-load-results', 'n_clicks'),
    State('sel-workdir', 'value'),
    prevent_initial_call=True,
)
def load_results_click(n, workdir):
    if not n or not workdir:
        return no_update, no_update
    directory = f'{workdir}/output'
    out_path = REPO / directory
    if not (out_path / 'vessels.parquet').exists():
        return {'error': 'Cannot find results — run Calculate Waves first'}, False
    try:
        result = _load_results(directory)
        return result, False
    except Exception as e:
        return {'error': str(e)}, False


@app.callback(
    Output('_pv_ais', 'data'),
    Input('pv-ais', 'value'),
    prevent_initial_call=False,
)
def _pv_ais_toggle(pv_val):
    return {'visible': bool(pv_val)}


# ---- Track filter: MMSI/segment/type selectors ----
_VESSEL_CATEGORIES = [
    ('tanker',         'Tanker'),
    ('cargo',          'Cargo'),
    ('passenger',      'Passenger'),
    ('tug',            'Tug'),
    ('pleasure_craft', 'Pleasure Craft'),
    ('pilot_vessel',   'Pilot Vessel'),
    ('other',          'Other'),
    ('unknown',        'Unknown'),
]


@app.callback(
    Output('fil-mmsi', 'options'),
    Output('fil-type', 'options'),
    Input('_track_version', 'data'),
    prevent_initial_call=False,
)
def _populate_filter_options(_):
    mmsi_opts = []
    if seg_meta:
        mmsis = sorted(set(s['mmsi'] for s in seg_meta))
        mmsi_opts = [{'label': str(m), 'value': m} for m in mmsis]
    type_opts = [{'label': label, 'value': cat} for cat, label in _VESSEL_CATEGORIES]
    return mmsi_opts, type_opts


@app.callback(
    Output('fil-segs', 'options'),
    Output('fil-segs', 'value'),
    Input('fil-mmsi', 'value'),
    prevent_initial_call=False,
)
def _populate_seg_options(mmsi):
    if mmsi is None:
        return [], None
    segs = sorted(s['segment_id'] for s in seg_meta if s['mmsi'] == int(mmsi))
    return [{'label': f'seg {s}', 'value': s} for s in segs], None


@app.callback(
    Output('_filter_structural', 'data', allow_duplicate=True),
    Output('fil-type', 'value', allow_duplicate=True),
    Output('btn-fil-export', 'disabled', allow_duplicate=True),
    Input('btn-fil-clear', 'n_clicks'),
    State('_filter_structural', 'data'),
    prevent_initial_call=True,
)
def _clear_track_filter(_, prev):
    nonce = ((prev or {}).get('nonce', 0) + 1)
    return {'mmsi': None, 'seg_ids': [], 'types': [], 'nonce': nonce, '_clear': True}, None, True


# Vessel type applies immediately on dropdown change
app.clientside_callback(
    """
    function(types, prev) {
        var next = types || [];
        var prevTypes = (prev || {}).types || [];
        if (next.length === prevTypes.length &&
                next.every(function(t, i) { return t === prevTypes[i]; })) {
            return window.dash_clientside.no_update;
        }
        var nonce = ((prev || {}).nonce || 0) + 1;
        return {mmsi: null, seg_ids: [], types: next, nonce: nonce, _clear: false};
    }
    """,
    Output('_filter_structural', 'data', allow_duplicate=True),
    Input('fil-type', 'value'),
    State('_filter_structural', 'data'),
    prevent_initial_call=True,
)



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
            if (!r.ok) {
                let msg = `HTTP ${r.status}`;
                try { const e = await r.json(); if (e.error) msg = e.error; } catch (_) {}
                throw new Error(`${asset.label}: ${msg}`);
            }
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
        let tType = new Int32Array(0);
        let segLookup = new Map();
        let mmsiToSegIdxs = new Map();
        let typeToSegIdxs = new Map();
        let catToSegIdxs  = new Map();
        let pointSog = new Float32Array(0);
        let pointCog = new Float32Array(0);
        let pointTime = new BigInt64Array(0);  // obstime as ns-since-epoch
        // Cached filtered arrays (rebuilt when visibility changes)
        let filteredCoords = new Float32Array(0);
        let filteredStarts = new Int32Array([0]);
        let filteredSegIdxs = [];  // segIdx for each entry in filteredStarts
        let filteredPointPos = new Float32Array(0);   // flat lon,lat per visible point
        let filteredPointSeg = new Int32Array(0);     // segIdx for each visible point
        let filteredPointRow = new Int32Array(0);     // original cPos row for sog/cog/time
        const MAX_FILTERED_POINTS = 200_000;

        const initTrackArrays = (cT, mT, oT) => {
            const cLon = cT.getChild('lon').toArray();
            const cLat = cT.getChild('lat').toArray();
            cPos = new Float32Array(cLon.length * 2);
            for (let i = 0; i < cLon.length; i++) { cPos[i*2]=cLon[i]; cPos[i*2+1]=cLat[i]; }
            pointSog  = cT.getChild('sog')     ? cT.getChild('sog').toArray()     : new Float32Array(cLon.length);
            pointCog  = cT.getChild('cog')     ? cT.getChild('cog').toArray()     : new Float32Array(cLon.length);
            pointTime = cT.getChild('obstime') ? cT.getChild('obstime').toArray() : new BigInt64Array(cLon.length);
            startIndices = oT.getChild('offset').toArray();
            tMMSI = mT.getChild('mmsi').toArray();
            tSeg  = mT.getChild('segment_id').toArray();
            tN    = mT.getChild('n_points').toArray();
            tType = mT.getChild('typecargo') ? mT.getChild('typecargo').toArray() : new Int32Array(tMMSI.length);
            segLookup = new Map();
            mmsiToSegIdxs = new Map();
            typeToSegIdxs = new Map();
            catToSegIdxs  = new Map();
            for (let i = 0; i < tMMSI.length; i++) {
                segLookup.set(`${tMMSI[i]}|${tSeg[i]}`, i);
                const m = Number(tMMSI[i]);
                if (!mmsiToSegIdxs.has(m)) mmsiToSegIdxs.set(m, []);
                mmsiToSegIdxs.get(m).push(i);
                const t = Number(tType[i]);
                if (!typeToSegIdxs.has(t)) typeToSegIdxs.set(t, []);
                typeToSegIdxs.get(t).push(i);
                const cat = _typeCategory(t);
                if (!catToSegIdxs.has(cat)) catToSegIdxs.set(cat, []);
                catToSegIdxs.get(cat).push(i);
            }
            // Rebuild wave→track mapping (tracks just changed; if waves loaded,
            // they need to be re-linked to the new track segment indices).
            if (typeof buildWaveSegMapping === 'function') buildWaveSegMapping();
            // Re-resolve any existing filter after data reload
            if (typeof window.__recomputeVisibility === 'function') window.__recomputeVisibility();
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
        // Precomputed mapping: wave index → track segment index (-1 = no match).
        // Rebuilt every time wave OR track data changes; values are Number-normalised
        // so BigInt (int64) vs Number (int32) representations never disagree.
        let waveToSegIdx = null;
        const buildWaveSegMapping = () => {
            if (wMMSI.length === 0 || tMMSI.length === 0) { waveToSegIdx = null; return; }
            const tMap = new Map();
            for (let si = 0; si < tMMSI.length; si++) {
                tMap.set(`${Number(tMMSI[si])}|${Number(tSeg[si])}`, si);
            }
            const arr = new Int32Array(wMMSI.length);
            let matched = 0;
            for (let i = 0; i < wMMSI.length; i++) {
                const si = tMap.get(`${Number(wMMSI[i])}|${Number(wSegId[i])}`);
                if (si != null) { arr[i] = si; matched++; } else { arr[i] = -1; }
            }
            waveToSegIdx = arr;
            console.log(`[wave-seg-map] ${matched}/${wMMSI.length} waves linked to tracks`);
        };
        window.__buildWaveSegMapping = buildWaveSegMapping;
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
            // Sort by H ascending so highest wave renders on top (deck.gl draws last index last = on top)
            if (wH.length > 1) {
                const n = wH.length;
                const perm = Array.from({length: n}, (_, i) => i)
                    .sort((a, b) => (wH[a] || 0) - (wH[b] || 0));
                const reorder = (arr) => {
                    const out = new arr.constructor(arr.length);
                    for (let i = 0; i < n; i++) out[i] = arr[perm[i]];
                    return out;
                };
                const newPos = new Float32Array(n * 2);
                for (let i = 0; i < n; i++) {
                    newPos[i*2] = wPos[perm[i]*2]; newPos[i*2+1] = wPos[perm[i]*2+1];
                }
                wPos = newPos;
                wMMSI = reorder(wMMSI); wH = reorder(wH); wTp = reorder(wTp);
                wSog = reorder(wSog); wCog = reorder(wCog); wDraught = reorder(wDraught);
                wLen = reorder(wLen); wWid = reorder(wWid); wDist = reorder(wDist);
                wVesselLon = reorder(wVesselLon); wVesselLat = reorder(wVesselLat);
                wSegId = reorder(wSegId);
                const _origSide = wSide, _origTime = wTime;
                wSide = (i) => _origSide(perm[i]);
                wTime = (i) => _origTime(perm[i]);
            }
            buildWaveSegMapping();
        }

        // ---- Preview state (set by clientside callbacks) ----
        window.__previews = { ais: null, bathy: null, coast: null, land: null, tide: null };
        // Singapore-wide initial view: bbox roughly 103.55–104.05 / 1.20–1.50.
        window._currentZoom = 10;
        window._hoveredWave = null;
        window._pinnedWave  = null;  // click-locked wave highlight; null = none
        // Track whether we have data so layers gate themselves.
        window.__hasTracks = false;
        window.__hasWaves = false;

        // ---- Track colour by vessel type ----
        const _typeCategory = (c) => {
            if (c <= 0 || (c <= 19) || (c === 38) || (c === 39) || (c >= 56 && c <= 57) || c >= 100) return 'unknown';
            if (c >= 80 && c < 90) return 'tanker';
            if (c >= 70 && c < 80) return 'cargo';
            if (c >= 60 && c < 70) return 'passenger';
            if (c === 52) return 'tug';
            if (c === 37) return 'pleasure_craft';
            if (c === 50) return 'pilot_vessel';
            return 'other';  // all remaining named categories (HSC, SAR, WIG, fishing, towing, etc.)
        };
        const _CAT_COLORS = {
            unknown:       [127, 127, 127],  // #7F7F7F
            tanker:        [165, 138, 255],  // #A58AFF
            cargo:         [248, 111, 101],  // #F86F65
            passenger:     [ 73, 176,   0],  // #49B000
            tug:           [251,  95, 215],  // #FB5FD7
            pleasure_craft:[  0, 178, 235],  // #00B2EB
            pilot_vessel:  [ 16, 195, 154],  // #10C39A
            other:         [196, 154,   0],  // #C49A00
        };
        const trackColor = (typeCode, alpha) => {
            const rgb = _CAT_COLORS[_typeCategory(typeCode)] || [160, 160, 160];
            return [rgb[0], rgb[1], rgb[2], alpha];
        };

        // ---- Wave colour by wave height -----------------------------------------
        const _lrp = (a, b, t) => Math.round(a + (b - a) * t);
        const waveColor = (i) => {
            const h = wH[i];
            if (isNaN(h) || h <= 0) return [80, 200, 120, 170];
            if (h < 0.15) {
                const t = h / 0.15;
                return [_lrp(80, 255, t), _lrp(200, 220, t), _lrp(120, 0, t), 170];
            }
            if (h < 0.4) {
                const t = (h - 0.15) / 0.25;
                return [255, _lrp(220, 100, t), 0, 170];
            }
            const t = Math.min((h - 0.4) / 0.3, 1);
            return [255, _lrp(100, 30, t), _lrp(0, 30, t), 170];
        };

        // ---- Multi-filter state ----
        window.__filterState = { mmsi: null, seg_ids: [], types: [], freehand: null, similar: null, waveBox: null };
        window.__visibleSegIdxs = null; // null = show all; Set<segIdx> = filtered
        window.__visibleWaveIdxs = null; // null = show all; Set<waveIdx> = filtered (post-sort idx)
        window.__visibleWaveIdxsArr = null; // sorted Array<waveIdx> matching visibility (preserves H-asc order)
        window.__filteredWavePos = null; // packed Float32Array of [lon,lat,...] for visible waves

        // Rebuild the cached filtered coord/offset arrays from a list of segIdxs
        const rebuildFilteredArrays = (visArr) => {
            if (!visArr || visArr.length === 0) {
                filteredCoords = new Float32Array(0);
                filteredStarts = new Int32Array([0]);
                filteredSegIdxs = [];
                filteredPointPos = new Float32Array(0);
                filteredPointSeg = new Int32Array(0);
                filteredPointRow = new Int32Array(0);
                return;
            }
            const totalPts = visArr.reduce((s, i) => s + (startIndices[i+1] - startIndices[i]), 0);
            filteredCoords = new Float32Array(totalPts * 2);
            filteredStarts = new Int32Array(visArr.length + 1);
            filteredPointPos = new Float32Array(totalPts * 2);
            filteredPointSeg = new Int32Array(totalPts);
            filteredPointRow = new Int32Array(totalPts);
            let ptr = 0;
            for (let k = 0; k < visArr.length; k++) {
                const si = visArr[k];
                const s = startIndices[si], e = startIndices[si + 1];
                const n = e - s;
                filteredCoords.set(cPos.subarray(s * 2, e * 2), ptr * 2);
                filteredPointPos.set(cPos.subarray(s * 2, e * 2), ptr * 2);
                filteredStarts[k] = ptr;
                for (let j = 0; j < n; j++) {
                    filteredPointSeg[ptr + j] = si;
                    filteredPointRow[ptr + j] = s + j;
                }
                ptr += n;
            }
            filteredStarts[visArr.length] = ptr;
            filteredSegIdxs = visArr;
        };

        // Rebuild the wave-side filtered arrays from the current __visibleSegIdxs
        // and __filterState.waveBox. Called by __recomputeVisibility.
        const rebuildFilteredWaveArrays = () => {
            const fs = window.__filterState;
            const M = wMMSI.length;
            const segVis = window.__visibleSegIdxs;
            // No waves loaded → reset.
            if (M === 0 || !window.__hasWaves) {
                window.__visibleWaveIdxs = null;
                window.__visibleWaveIdxsArr = null;
                window.__filteredWavePos = null;
                return;
            }
            // No filter active → reset.
            const noBox = fs.waveBox == null;
            const noTrackFilter = segVis === null;
            if (noBox && noTrackFilter) {
                window.__visibleWaveIdxs = null;
                window.__visibleWaveIdxsArr = null;
                window.__filteredWavePos = null;
                return;
            }
            // Build candidate set from waveBox if active (post-sort indices).
            const boxSet = (!noBox && fs.waveBox.waveIdxs) ? fs.waveBox.waveIdxs : null;
            // Iterate all waves (already sorted lowest H first).
            // Keep indices that pass both waveBox and segment-visibility filters.
            // Uses the precomputed waveToSegIdx mapping — built with
            // Number-normalised keys so BigInt(int64) vs Number(int32) never
            // disagree. With the unified pipeline (filter→vessel→wave_impact)
            // tracks and waves share one segment_id space, so every wave should
            // map to a valid track segment.
            const visSet = new Set();
            const visArr = [];
            for (let i = 0; i < M; i++) {
                if (boxSet && !boxSet.has(i)) continue;
                if (!noTrackFilter) {
                    const sIdx = waveToSegIdx ? waveToSegIdx[i] : -1;
                    if (sIdx < 0 || !segVis.has(sIdx)) continue;
                }
                visSet.add(i);
                visArr.push(i);
            }
            window.__visibleWaveIdxs = visSet;
            window.__visibleWaveIdxsArr = visArr;
            // Pack positions for the visible waves (lon/lat pairs).
            const pos = new Float32Array(visArr.length * 2);
            for (let k = 0; k < visArr.length; k++) {
                const i = visArr[k];
                pos[k * 2]     = wPos[i * 2];
                pos[k * 2 + 1] = wPos[i * 2 + 1];
            }
            window.__filteredWavePos = pos;
        };

        window.__recomputeVisibility = () => {
            const fs = window.__filterState;
            const N = tMMSI.length;
            const allNull = fs.mmsi == null &&
                            (!fs.seg_ids || !fs.seg_ids.length) &&
                            (!fs.types || !fs.types.length) &&
                            fs.freehand == null && fs.similar == null &&
                            fs.waveBox == null;
            if (N === 0 || allNull) {
                window.__visibleSegIdxs = null;
                rebuildFilteredArrays(null);
                rebuildFilteredWaveArrays();
                window.__rebuild();
                const stat = document.getElementById('fil-status');
                if (stat) stat.textContent = N > 0 ? `All ${N.toLocaleString()} tracks visible` : '';
                return;
            }
            const sets = [];
            // MMSI + segment filter
            if (fs.mmsi != null) {
                const candidates = mmsiToSegIdxs.get(Number(fs.mmsi));
                if (candidates) {
                    if (fs.seg_ids && fs.seg_ids.length > 0) {
                        const segSet = new Set(fs.seg_ids.map(Number));
                        sets.push(new Set(candidates.filter(i => segSet.has(Number(tSeg[i])))));
                    } else {
                        sets.push(new Set(candidates));
                    }
                } else {
                    sets.push(new Set());
                }
            }
            // Vessel category filter (category strings e.g. 'tanker', 'cargo')
            if (fs.types && fs.types.length > 0) {
                const typeSet = new Set();
                for (const t of fs.types) {
                    const idxs = catToSegIdxs.get(t);
                    if (idxs) idxs.forEach(i => typeSet.add(i));
                }
                sets.push(typeSet);
            }
            // Freehand filter: [[mmsi, seg_id], ...]
            if (fs.freehand != null) {
                const fhSet = new Set();
                for (const [m, s] of fs.freehand) {
                    const i = segLookup.get(`${m}|${s}`);
                    if (i != null) fhSet.add(i);
                }
                sets.push(fhSet);
            }
            // Similar filter: [[mmsi, seg_id], ...]
            if (fs.similar != null) {
                const simSet = new Set();
                for (const [m, s] of fs.similar) {
                    const i = segLookup.get(`${m}|${s}`);
                    if (i != null) simSet.add(i);
                }
                sets.push(simSet);
            }
            // Wave-arrival-area box: derives track segIdxs from the waves in the box.
            // Only push if non-empty; an empty segIdxs would zero-out the intersection.
            if (fs.waveBox != null && fs.waveBox.segIdxs && fs.waveBox.segIdxs.size > 0) {
                sets.push(new Set(fs.waveBox.segIdxs));
            }
            if (sets.length === 0) {
                window.__visibleSegIdxs = null;
                rebuildFilteredArrays(null);
            } else {
                sets.sort((a, b) => a.size - b.size);
                let result = sets[0];
                for (let k = 1; k < sets.length; k++) {
                    const next = new Set();
                    for (const v of result) { if (sets[k].has(v)) next.add(v); }
                    result = next;
                }
                window.__visibleSegIdxs = result;
                rebuildFilteredArrays(Array.from(result));
            }
            rebuildFilteredWaveArrays();
            window.__rebuild();
            // Update status div directly (also updated via Dash callback for structural changes)
            const stat = document.getElementById('fil-status');
            if (stat) {
                const vis = window.__visibleSegIdxs;
                stat.textContent = vis === null
                    ? (N > 0 ? `All ${N.toLocaleString()} tracks visible` : '')
                    : `${vis.size.toLocaleString()} of ${N.toLocaleString()} tracks visible`;
            }
            if (window.dash_clientside?.set_props) {
                const isActive = window.__visibleSegIdxs !== null && window.__visibleSegIdxs.size > 0;
                window.dash_clientside.set_props('_any_filter_active', {data: isActive});
            }
        };

        window.__applyStructuralFilter = (structural) => {
            if (!structural) return '';
            if (structural._clear) {
                // Full reset from "Clear all" button
                window.__filterState.mmsi    = null;
                window.__filterState.seg_ids = [];
                window.__filterState.types   = [];
                window.__filterState.freehand = null;
                window.__filterState.similar  = null;
                window.__filterState.waveBox  = null;
                window.__cascadeMMSI = null;
                window.__cascadeSegs = [];
                if (typeof window.__updateCascadeTrigger === 'function') window.__updateCascadeTrigger();
                const p = document.getElementById('sim-panel');
                if (p) p.style.display = 'none';
                // Reset armed states and restore button labels
                if (window.__freehandArmed || window.__freehandMode) {
                    if (typeof window.__cancelFreehandDraw === 'function') window.__cancelFreehandDraw();
                    window.__freehandArmed = false;
                    const bf = document.getElementById('btn-freehand');
                    if (bf) { bf.textContent = 'Draw line across tracks'; bf.style.opacity = ''; }
                }
                if (window.__waveBoxArmed || window.__waveBoxMode) {
                    if (typeof window.__cancelWaveBoxDraw === 'function') window.__cancelWaveBoxDraw();
                    window.__waveBoxArmed = false;
                    const bw = document.getElementById('btn-wavebox');
                    if (bw) { bw.textContent = 'Drag box on the map'; bw.style.opacity = ''; }
                }
                if (window.__similarArmed) {
                    window.__similarArmed = false;
                    const bs = document.getElementById('btn-similar');
                    if (bs) { bs.textContent = 'Select one representative track'; bs.style.opacity = ''; }
                }
                window.__updateDeckCursor();
            } else {
                // Partial update from "Apply filters" — only vessel types go through Dash.
                // MMSI/seg are managed client-side by the cascade widget; don't override them.
                window.__filterState.types = structural.types || [];
            }
            window.__recomputeVisibility();
            const vis = window.__visibleSegIdxs;
            const N = tMMSI.length;
            return vis === null
                ? (N > 0 ? `All ${N.toLocaleString()} tracks visible` : '')
                : `${vis.size.toLocaleString()} of ${N.toLocaleString()} tracks visible`;
        };

        // ---- Segment-segment intersection helpers (for free-hand) ----
        const ccw = (ax, ay, bx, by, cx, cy) => (cy-ay)*(bx-ax) > (by-ay)*(cx-ax);
        const segsIntersect = (ax, ay, bx, by, cx, cy, dx, dy) =>
            ccw(ax,ay,cx,cy,dx,dy) !== ccw(bx,by,cx,cy,dx,dy) &&
            ccw(ax,ay,bx,by,cx,cy) !== ccw(ax,ay,bx,by,dx,dy);

        // ---- Free-hand draw mode (with canvas trace overlay) ----
        // Canvas is a fixed overlay matching the deck area; we show/draw on it during
        // freehand mode and hide it when done.
        const getOrCreateCanvas = () => {
            let cv = document.getElementById('freehand-canvas');
            if (!cv) {
                cv = document.createElement('canvas');
                cv.id = 'freehand-canvas';
                cv.style.cssText = 'position:fixed;top:40px;left:340px;right:0;bottom:0;' +
                    'pointer-events:none;z-index:3;display:none;';
                document.body.appendChild(cv);
            }
            const container = document.getElementById('deck-container');
            const r = container.getBoundingClientRect();
            cv.width = r.width; cv.height = r.height;
            return cv;
        };
        window.__freehandArmed = false;
        window.__freehandMode  = false;
        window.__cancelFreehandDraw = null;

        window.__enterFreehandMode = () => {
            const btn = document.getElementById('btn-freehand');
            if (window.__freehandArmed) {
                window.__freehandArmed = false;
                if (btn) { btn.textContent = 'Draw line across tracks'; btn.style.opacity = ''; }
                window.__updateDeckCursor();
                return;
            }
            window.__freehandArmed = true;
            if (btn) { btn.textContent = 'Hold Ctrl to draw...'; btn.style.opacity = '0.65'; }
            window.__updateDeckCursor();
        };

        window.__activateFreehandDraw = () => {
            if (!window.__freehandArmed || window.__freehandMode) return;
            window.__freehandMode = true;
            window.deckInstance.setProps({ controller: false });
            window.__updateDeckCursor();
            const btn = document.getElementById('btn-freehand');
            if (btn) { btn.textContent = 'Drawing...'; btn.style.opacity = '0.65'; }
            const cv = getOrCreateCanvas();
            cv.style.display = 'block';
            const ctx2d = cv.getContext('2d');
            ctx2d.clearRect(0, 0, cv.width, cv.height);
            ctx2d.strokeStyle = 'rgba(80,200,255,0.85)';
            ctx2d.lineWidth = 2;
            ctx2d.lineCap = 'round';
            ctx2d.lineJoin = 'round';
            ctx2d.setLineDash([6, 4]);
            const container = document.getElementById('deck-container');
            let pencil = [], isDrawing = false;
            const rect = () => container.getBoundingClientRect();
            const onDown = (e) => {
                isDrawing = true;
                const r = rect();
                const px = e.clientX - r.left, py = e.clientY - r.top;
                pencil = [[px, py]];
                ctx2d.clearRect(0, 0, cv.width, cv.height);
                ctx2d.beginPath();
                ctx2d.moveTo(px, py);
            };
            const onMove = (e) => {
                if (!isDrawing) return;
                const r = rect();
                const px = e.clientX - r.left, py = e.clientY - r.top;
                pencil.push([px, py]);
                ctx2d.lineTo(px, py);
                ctx2d.stroke();
            };
            const onUp = () => {
                if (!isDrawing || pencil.length < 2) { cancel(); return; }
                isDrawing = false;
                const vp = window.deckInstance.getViewports()[0];
                const wp = pencil.map(([px, py]) => vp.unproject([px, py]));
                let pMinX=Infinity,pMaxX=-Infinity,pMinY=Infinity,pMaxY=-Infinity;
                for (const [wx,wy] of wp) {
                    if(wx<pMinX)pMinX=wx; if(wx>pMaxX)pMaxX=wx;
                    if(wy<pMinY)pMinY=wy; if(wy>pMaxY)pMaxY=wy;
                }
                const hits = [];
                for (let si = 0; si < tMMSI.length; si++) {
                    const s = startIndices[si], e = startIndices[si + 1];
                    let sMinX=Infinity,sMaxX=-Infinity,sMinY=Infinity,sMaxY=-Infinity;
                    for (let p = s; p < e; p++) {
                        const x=cPos[p*2],y=cPos[p*2+1];
                        if(x<sMinX)sMinX=x; if(x>sMaxX)sMaxX=x;
                        if(y<sMinY)sMinY=y; if(y>sMaxY)sMaxY=y;
                    }
                    if (sMaxX<pMinX||sMinX>pMaxX||sMaxY<pMinY||sMinY>pMaxY) continue;
                    let hit = false;
                    for (let p = s; p < e-1 && !hit; p++) {
                        const ax=cPos[p*2],ay=cPos[p*2+1],bx=cPos[(p+1)*2],by=cPos[(p+1)*2+1];
                        for (let q = 0; q < wp.length-1 && !hit; q++) {
                            if (segsIntersect(ax,ay,bx,by,wp[q][0],wp[q][1],wp[q+1][0],wp[q+1][1])) hit=true;
                        }
                    }
                    if (hit) hits.push([Number(tMMSI[si]), Number(tSeg[si])]);
                }
                window.__filterState.freehand = hits.length > 0 ? hits : null;
                window.__recomputeVisibility();
                finish();
            };
            const removeListeners = () => {
                container.removeEventListener('pointerdown', onDown);
                container.removeEventListener('pointermove', onMove);
                container.removeEventListener('pointerup', onUp);
                window.__cancelFreehandDraw = null;
            };
            const cancel = () => {
                removeListeners();
                window.__freehandMode = false;
                window.deckInstance.setProps({ controller: DEFAULT_CONTROLLER });
                window.__updateDeckCursor();
                if (btn) { btn.textContent = 'Hold Ctrl to draw...'; btn.style.opacity = '0.65'; }
                cv.style.display = 'none';
                ctx2d.clearRect(0, 0, cv.width, cv.height);
            };
            const finish = () => {
                removeListeners();
                window.__freehandArmed = false;
                window.__freehandMode  = false;
                window.deckInstance.setProps({ controller: DEFAULT_CONTROLLER });
                window.__updateDeckCursor();
                if (btn) { btn.textContent = 'Draw line across tracks'; btn.style.opacity = ''; }
                cv.style.display = 'none';
                ctx2d.clearRect(0, 0, cv.width, cv.height);
            };
            window.__cancelFreehandDraw = cancel;
            container.addEventListener('pointerdown', onDown);
            container.addEventListener('pointermove', onMove);
            container.addEventListener('pointerup',   onUp);
        };

        // ---- Wave-arrival-area box mode ----
        const getOrCreateWaveBoxCanvas = () => {
            let cv = document.getElementById('wavebox-canvas');
            if (!cv) {
                cv = document.createElement('canvas');
                cv.id = 'wavebox-canvas';
                cv.style.cssText = 'position:fixed;top:40px;left:340px;right:0;bottom:0;' +
                    'pointer-events:none;z-index:3;display:none;';
                document.body.appendChild(cv);
            }
            const container = document.getElementById('deck-container');
            const r = container.getBoundingClientRect();
            cv.width = r.width; cv.height = r.height;
            return cv;
        };
        window.__waveBoxArmed = false;
        window.__waveBoxMode  = false;
        window.__cancelWaveBoxDraw = null;

        window.__enterWaveBoxMode = () => {
            const btn = document.getElementById('btn-wavebox');
            if (window.__waveBoxArmed) {
                window.__waveBoxArmed = false;
                if (btn) { btn.textContent = 'Drag box on the map'; btn.style.opacity = ''; }
                window.__updateDeckCursor();
                return;
            }
            if (!window.__hasWaves || wMMSI.length === 0) return;
            window.__waveBoxArmed = true;
            if (btn) { btn.textContent = 'Hold Ctrl to drag...'; btn.style.opacity = '0.65'; }
            window.__updateDeckCursor();
        };

        window.__activateWaveBoxDraw = () => {
            if (!window.__waveBoxArmed || window.__waveBoxMode) return;
            if (!window.__hasWaves || wMMSI.length === 0) return;
            window.__waveBoxMode = true;
            window.deckInstance.setProps({ controller: false });
            window.__updateDeckCursor();
            const btn = document.getElementById('btn-wavebox');
            if (btn) { btn.textContent = 'Dragging...'; btn.style.opacity = '0.65'; }
            const container = document.getElementById('deck-container');
            const cv = getOrCreateWaveBoxCanvas();
            cv.style.display = 'block';
            const ctx2d = cv.getContext('2d');
            ctx2d.clearRect(0, 0, cv.width, cv.height);
            let isDragging = false;
            let p0 = null;
            const rect = () => container.getBoundingClientRect();
            const drawRect = (a, b) => {
                ctx2d.clearRect(0, 0, cv.width, cv.height);
                const x = Math.min(a[0], b[0]), y = Math.min(a[1], b[1]);
                const w = Math.abs(b[0] - a[0]), h = Math.abs(b[1] - a[1]);
                ctx2d.fillStyle = 'rgba(80,200,255,0.12)';
                ctx2d.fillRect(x, y, w, h);
                ctx2d.strokeStyle = 'rgba(80,200,255,0.85)';
                ctx2d.lineWidth = 2;
                ctx2d.setLineDash([6, 4]);
                ctx2d.strokeRect(x, y, w, h);
            };
            const onDown = (e) => {
                const r = rect();
                isDragging = true;
                p0 = [e.clientX - r.left, e.clientY - r.top];
            };
            const onMove = (e) => {
                if (!isDragging) return;
                const r = rect();
                const p = [e.clientX - r.left, e.clientY - r.top];
                drawRect(p0, p);
            };
            const onUp = (e) => {
                if (!isDragging) { cancel(); return; }
                isDragging = false;
                const r = rect();
                const p1 = [e.clientX - r.left, e.clientY - r.top];
                if (Math.abs(p1[0] - p0[0]) < 4 || Math.abs(p1[1] - p0[1]) < 4) {
                    cancel();
                    return;
                }
                const vp = window.deckInstance.getViewports()[0];
                const c1 = vp.unproject(p0);
                const c2 = vp.unproject(p1);
                const lonMin = Math.min(c1[0], c2[0]), lonMax = Math.max(c1[0], c2[0]);
                const latMin = Math.min(c1[1], c2[1]), latMax = Math.max(c1[1], c2[1]);
                const boxWaveIdxs = [];
                const boxSegIdxs = new Set();
                for (let i = 0; i < wMMSI.length; i++) {
                    const lon = wPos[i*2], lat = wPos[i*2+1];
                    if (lon >= lonMin && lon <= lonMax && lat >= latMin && lat <= latMax) {
                        boxWaveIdxs.push(i);
                        const si = waveToSegIdx ? waveToSegIdx[i] : -1;
                        if (si >= 0) boxSegIdxs.add(si);
                    }
                }
                if (boxWaveIdxs.length > 0) {
                    window.__filterState.waveBox = {
                        waveIdxs: new Set(boxWaveIdxs),
                        segIdxs: boxSegIdxs,
                    };
                    window.__recomputeVisibility();
                }
                finish();
            };
            const removeListeners = () => {
                container.removeEventListener('pointerdown', onDown);
                container.removeEventListener('pointermove', onMove);
                container.removeEventListener('pointerup', onUp);
                window.__cancelWaveBoxDraw = null;
            };
            const cancel = () => {
                removeListeners();
                window.__waveBoxMode = false;
                window.deckInstance.setProps({ controller: DEFAULT_CONTROLLER });
                window.__updateDeckCursor();
                if (btn) { btn.textContent = 'Hold Ctrl to drag...'; btn.style.opacity = '0.65'; }
                cv.style.display = 'none';
                ctx2d.clearRect(0, 0, cv.width, cv.height);
            };
            const finish = () => {
                removeListeners();
                window.__waveBoxArmed = false;
                window.__waveBoxMode  = false;
                window.deckInstance.setProps({ controller: DEFAULT_CONTROLLER });
                window.__updateDeckCursor();
                if (btn) { btn.textContent = 'Drag box on the map'; btn.style.opacity = ''; }
                cv.style.display = 'none';
                ctx2d.clearRect(0, 0, cv.width, cv.height);
            };
            window.__cancelWaveBoxDraw = cancel;
            container.addEventListener('pointerdown', onDown);
            container.addEventListener('pointermove', onMove);
            container.addEventListener('pointerup',   onUp);
        };

        // ---- Window-scope accessors for clientside callbacks outside this IIFE ----
        window.__getFilteredSegKeys = () => {
            const vis = window.__visibleSegIdxs;
            if (!vis || vis.size === 0) return [];
            const out = [];
            for (const si of vis) {
                out.push([Number(tMMSI[si]), Number(tSeg[si])]);
            }
            return out;
        };
        window.__getFilteredWaveIdxs = () => {
            if (!window.__hasWaves) return null;
            const arr = window.__visibleWaveIdxsArr;
            return arr ? Array.from(arr).map(Number) : null;
        };

        // ---- Reset all filters (used when waves are recalculated) ----
        window.__resetAllFilters = () => {
            window.__filterState.mmsi     = null;
            window.__filterState.seg_ids  = [];
            window.__filterState.types    = [];
            window.__filterState.freehand = null;
            window.__filterState.similar  = null;
            window.__filterState.waveBox  = null;
            if (typeof window.__cascadeMMSI !== 'undefined') {
                window.__cascadeMMSI = null;
                window.__cascadeSegs = [];
            }
            if (typeof window.__updateCascadeTrigger === 'function') {
                window.__updateCascadeTrigger();
            }
            const simPanel = document.getElementById('sim-panel');
            if (simPanel) simPanel.style.display = 'none';
            // Reset armed states and restore button labels
            if (window.__freehandArmed || window.__freehandMode) {
                if (typeof window.__cancelFreehandDraw === 'function') window.__cancelFreehandDraw();
                window.__freehandArmed = false;
                const bf = document.getElementById('btn-freehand');
                if (bf) { bf.textContent = 'Draw line across tracks'; bf.style.opacity = ''; }
            }
            if (window.__waveBoxArmed || window.__waveBoxMode) {
                if (typeof window.__cancelWaveBoxDraw === 'function') window.__cancelWaveBoxDraw();
                window.__waveBoxArmed = false;
                const bw = document.getElementById('btn-wavebox');
                if (bw) { bw.textContent = 'Drag box on the map'; bw.style.opacity = ''; }
            }
            if (window.__similarArmed) {
                window.__similarArmed = false;
                const bs = document.getElementById('btn-similar');
                if (bs) { bs.textContent = 'Select one representative track'; bs.style.opacity = ''; }
            }
            window.__updateDeckCursor();
            window.__recomputeVisibility();
        };

        // ---- Similar select mode ----
        window.__similarArmed  = false;
        window.__simPickedData = null;
        window.__enterSimilarMode = () => {
            const btn = document.getElementById('btn-similar');
            if (window.__similarArmed) {
                window.__similarArmed = false;
                if (btn) { btn.textContent = 'Select one representative track'; btn.style.opacity = ''; }
                window.__updateDeckCursor();
                return 'Similar mode cancelled';
            }
            window.__similarArmed = true;
            if (btn) { btn.textContent = 'Ctrl+click a track...'; btn.style.opacity = '0.65'; }
            window.__updateDeckCursor();
            return 'Hold Ctrl and click a track to pick the reference';
        };
        window.__runSimilar = async (buffer_m, min_coverage) => {
            const pick = window.__simPickedData;
            if (!pick || pick.mmsi == null) return 'No track picked';
            try {
                const resp = await fetch('/api/similar_tracks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mmsi: pick.mmsi, segment_id: pick.seg, buffer_m, min_coverage }),
                });
                const j = await resp.json();
                if (j.error) throw new Error(j.error);
                window.__filterState.similar = j.mmsi_segs;
                window.__recomputeVisibility();
                const panel = document.getElementById('sim-panel');
                if (panel) panel.style.display = 'none';
                return `Similar: ${j.mmsi_segs.length} tracks found`;
            } catch(e) { return 'Similar error: ' + e.message; }
        };

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
            // Bathy mesh sits under tracks/waves as a background reference
            if (window.__previews.bathy && window.__previews.bathy.pos
                && window.__previews.bathy.offsets && window.__previews.bathy.offsets.length > 1) {
                const pvb = window.__previews.bathy;
                layers.push(new deck.SolidPolygonLayer({
                    id: 'pv-bathy',
                    data: {
                        length: pvb.offsets.length - 1,
                        startIndices: pvb.offsets,
                        attributes: {
                            getPolygon:   { value: pvb.pos,     size: 2 },
                            getFillColor: { value: pvb.fillRgb, size: 4 },
                        },
                    },
                    _normalize: false,
                    filled: true,
                    pickable: false,
                }));
            }
            // Shapefile previews — rendered under tracks/waves
            const pv = window.__previews;
            if (pv.coast && pv.coast.geojson) {
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
            if (pv.land && pv.land.geojson) {
                layers.push(new deck.GeoJsonLayer({
                    id: 'pv-land',
                    data: pv.land.geojson,
                    stroked: true, filled: true,
                    getFillColor: [230, 150, 50, 50],
                    getLineColor: [200, 100, 20, 200],
                    getLineWidth: 1, lineWidthMinPixels: 1,
                    pickable: false,
                }));
            }
            // Filter-aware tracks rendering
            if (window.__hasTracks && !useRaster && tMMSI.length > 0) {
                const vis = window.__visibleSegIdxs;
                if (vis === null) {
                    // No filter → show all, colour by vessel type
                    layers.push(new deck.PathLayer({
                        id: 'tracks',
                        data: { length: tMMSI.length, startIndices,
                                attributes: { getPath: { value: cPos, size: 2 } } },
                        pickable: true, _pathType: 'open',
                        getColor: (_, {index}) => trackColor(tType[index], 80),
                        getWidth: 1.5, widthUnits: 'pixels', widthMinPixels: 1.5,
                        updateTriggers: { getColor: [tType] },
                        onHover: ({x, y, index}) => {
                            if (!window.__ctrlHeld) { hideTip(); return; }
                            if (index < 0) { hideTip(); return; }
                            showTip(x, y, `<b>TRACK</b><br>MMSI: ${tMMSI[index]}<br>seg: ${tSeg[index]}<br>n: ${tN[index]}<br>type: ${tType[index]}`);
                        },
                    }));
                } else if (vis.size > 0 && filteredSegIdxs.length > 0) {
                    // Show only visible segments, colour by vessel type
                    layers.push(new deck.PathLayer({
                        id: 'tracks',
                        data: { length: filteredSegIdxs.length, startIndices: filteredStarts,
                                attributes: { getPath: { value: filteredCoords, size: 2 } } },
                        pickable: true, _pathType: 'open',
                        getColor: (_, {index}) => trackColor(tType[filteredSegIdxs[index]], 160),
                        getWidth: 2.5, widthUnits: 'pixels', widthMinPixels: 2,
                        updateTriggers: { getColor: [filteredSegIdxs] },
                        onHover: ({x, y, index}) => {
                            if (!window.__ctrlHeld) { hideTip(); return; }
                            if (index < 0) { hideTip(); return; }
                            const si = filteredSegIdxs[index];
                            showTip(x, y, `<b>TRACK</b><br>MMSI: ${tMMSI[si]}<br>seg: ${tSeg[si]}<br>n: ${tN[si]}<br>type: ${tType[si]}`);
                        },
                    }));
                    // Per-point layer so individual AIS pings can be hovered
                    const nPts = filteredPointPos.length / 2;
                    if (nPts > 0 && nPts <= MAX_FILTERED_POINTS) {
                        layers.push(new deck.ScatterplotLayer({
                            id: 'tracks-pts',
                            data: { length: nPts,
                                    attributes: { getPosition: { value: filteredPointPos, size: 2 } } },
                            getRadius: 3.5, radiusUnits: 'pixels', radiusMinPixels: 2,
                            getFillColor: (_, {index}) => {
                                const si = filteredPointSeg[index];
                                return trackColor(tType[si], 160);
                            },
                            pickable: true,
                            onHover: ({x, y, index}) => {
                                if (!window.__ctrlHeld) { hideTip(); return; }
                                if (index < 0) { hideTip(); return; }
                                const row = filteredPointRow[index];
                                const si  = filteredPointSeg[index];
                                const timeNs = pointTime && pointTime[row] != null ? Number(pointTime[row]) : NaN;
                                let dt = '?';
                                try { if (!isNaN(timeNs)) dt = new Date(timeNs / 1e6).toISOString().replace('T', ' ').slice(0, 19); } catch (_) {}
                                const s = pointSog ? pointSog[row] : null;
                                const c = pointCog ? pointCog[row] : null;
                                const lon2 = filteredPointPos[index * 2];
                                const lat2 = filteredPointPos[index * 2 + 1];
                                showTip(x, y,
                                    `<b>TRACK POINT</b><br>` +
                                    `MMSI: ${tMMSI[si]}  seg: ${tSeg[si]}<br>` +
                                    (s != null ? `SOG: ${s.toFixed(1)} kn` : '') +
                                    (c != null ? `  COG: ${Math.round(c)}°` : '') + '<br>' +
                                    `time: ${dt}<br>` +
                                    `${lon2.toFixed(5)}, ${lat2.toFixed(5)}`
                                );
                            },
                        }));
                    }
                }
                // else: vis.size === 0 → nothing added → all hidden
            }
            // Wave layer — when a filter is active, render only visible waves and
            // remap the layer's local index back to the original wave index for
            // colour/hover lookups.
            const wvFilter = window.__visibleWaveIdxsArr;
            const wvLen = wvFilter !== null ? wvFilter.length : wMMSI.length;
            const wvPos = wvFilter !== null ? window.__filteredWavePos : wPos;
            const wvIdx = (i) => (wvFilter !== null ? wvFilter[i] : i);
            if (window.__hasWaves && wvLen > 0) {
                layers.push(new deck.ScatterplotLayer({
                    id: 'waves',
                    data: { length: wvLen,
                            attributes: { getPosition: { value: wvPos, size: 2 } } },
                    pickable: true,
                    getRadius: 12, radiusUnits: 'meters',
                    radiusMinPixels: 1.5, radiusMaxPixels: 5,
                    getFillColor: (_, {index}) => waveColor(wvIdx(index)),
                    updateTriggers: { getFillColor: [wH, wvFilter] },
                    onHover: ({x, y, index}) => {
                        if (!window.__ctrlHeld || index < 0) {
                            hideTip();
                            if (window._hoveredWave !== null) {
                                window._hoveredWave = null;
                                if (window._pinnedWave === null)
                                    window.deckInstance.setProps({ layers: buildLayers(window._currentZoom, null) });
                            }
                            return;
                        }
                        const origIdx = wvIdx(index);
                        const f = (v, d) => (v == null || isNaN(v)) ? '?' : v.toFixed(d);
                        const pinned = window._pinnedWave === origIdx ? ' 📌' : '';
                        showTip(x, y,
                            `<b>WAVE → MMSI ${wMMSI[origIdx]}${pinned}</b><br>` +
                            `<b>H</b>: ${f(wH[origIdx], 3)} m &nbsp;<b>T</b>: ${f(wTp[origIdx], 2)} s &nbsp;<b>Side</b>: ${wSide(origIdx)}<br>` +
                            `<b>SOG</b>: ${f(wSog[origIdx], 1)} kn &nbsp;<b>COG</b>: ${f(wCog[origIdx], 0)}°<br>` +
                            `<b>L×W×T</b>: ${f(wLen[origIdx], 0)}×${f(wWid[origIdx], 0)}×${f(wDraught[origIdx], 1)} m<br>` +
                            `<b>shore dist</b>: ${f((wDist[origIdx]||0)*1000, 0)} m<br><b>${wTime(origIdx)}</b>`
                        );
                        if (window._hoveredWave !== origIdx) {
                            window._hoveredWave = origIdx;
                            if (window._pinnedWave === null)
                                window.deckInstance.setProps({ layers: buildLayers(window._currentZoom, origIdx) });
                        }
                    },
                }));
            }

            // Wave highlight (cyan): pinned overrides hover
            const highlightIdx = window._pinnedWave != null ? window._pinnedWave : hoveredIdx;
            if (window.__hasWaves && highlightIdx != null && highlightIdx >= 0 && highlightIdx < wMMSI.length) {
                const segIdx = segLookup.get(`${wMMSI[highlightIdx]}|${wSegId[highlightIdx]}`);
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
                const vp = [wVesselLon[highlightIdx], wVesselLat[highlightIdx]];
                const wp = [wPos[highlightIdx*2], wPos[highlightIdx*2+1]];
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

            // ---- Preview layers (AIS points, rendered on top) ----
            // AIS preview: rendered only when visible flag is on (import is separate).
            if (pv.ais && pv.ais.visible && pv.ais.pos && pv.ais.pos.length > 0) {
                layers.push(new deck.ScatterplotLayer({
                    id: 'pv-ais',
                    data: { length: pv.ais.pos.length / 2,
                            attributes: { getPosition: { value: pv.ais.pos, size: 2 } } },
                    getRadius: 1.5, radiusUnits: 'pixels', radiusMinPixels: 0.5,
                    getFillColor: [50, 150, 255, 120],
                    pickable: true,
                    onHover: ({x, y, index}) => {
                        if (!window.__ctrlHeld) { hideTip(); return; }
                        if (index < 0) { hideTip(); return; }
                        const timeNs = pv.ais.obstimeNs && pv.ais.obstimeNs[index] != null ? Number(pv.ais.obstimeNs[index]) : NaN;
                        let dt = '?';
                        try { if (!isNaN(timeNs)) dt = new Date(timeNs / 1e6).toISOString().replace('T', ' ').slice(0, 19); } catch (_) {}
                        const s = pv.ais.sog    ? pv.ais.sog[index]    : null;
                        const c = pv.ais.cog    ? pv.ais.cog[index]    : null;
                        const lon = pv.ais.pos[index*2];
                        const lat = pv.ais.pos[index*2+1];
                        showTip(x, y,
                            `<b>AIS #${index.toLocaleString()}</b><br>` +
                            `SOG: ${s != null ? s.toFixed(1) : '?'} kn  COG: ${c != null ? c.toFixed(0) : '?'}°<br>` +
                            `${dt}<br>` +
                            `lon: ${lon.toFixed(6)}  lat: ${lat.toFixed(6)}`);
                    },
                }));
            }
            return layers;
        };
        window.__buildLayers = buildLayers;

        // Singapore-wide initial view (covers ~103.55-104.05 / 1.20-1.50).
        const initialZoom = 10;
        const CURSOR_PENCIL = `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='22' height='22' viewBox='0 0 22 22'%3E%3Cpath fill='%23ffffff' stroke='%23222222' stroke-width='1.3' stroke-linejoin='round' d='M3 19L15 2l4 4L5 21z'/%3E%3Cpath fill='%23cccccc' stroke='%23555555' stroke-width='1' d='M3 19l4 2-5-5z'/%3E%3Ccircle cx='18' cy='5' r='1' fill='%2380c0ff'/%3E%3C/svg%3E") 0 22, crosshair`;
        window.__ctrlHeld = false;
        // Pan + zoom always on. Ctrl key gates hover/click (inspect mode).
        const DEFAULT_CONTROLLER = { type: deck.MapController, dragPan: true,
            dragRotate: false, scrollZoom: true, doubleClickZoom: true, touchZoom: true };
        // Single source of truth for cursor — used by deck.gl getCursor AND manual updates.
        // isHovering comes from deck.gl's pick state; isDragging from mousedown/up.
        const getCursorForState = (isDragging = false, isHovering = false) => {
            if (window.__freehandMode) return CURSOR_PENCIL;
            if (window.__waveBoxMode)  return 'crosshair';
            if (window.__ctrlHeld) {
                if (window.__freehandArmed) return CURSOR_PENCIL;
                if (window.__waveBoxArmed)  return 'crosshair';
                if (window.__similarArmed)  return 'pointer';
                return isHovering ? 'pointer' : 'default';
            }
            return isDragging ? 'grabbing' : 'grab';
        };
        window.deckInstance = new deck.Deck({
            parent: container,
            width: '100%', height: '100%',
            initialViewState: { longitude: 103.82, latitude: 1.32, zoom: initialZoom, pitch: 0, bearing: 0 },
            controller: DEFAULT_CONTROLLER,
            layers: buildLayers(initialZoom, null),
            getCursor: ({isDragging, isHovering}) => getCursorForState(isDragging, isHovering),
            onClick: (info, event) => {
                const {layer, index} = info;
                // Sync __ctrlHeld from the real event so Ctrl+click works even
                // before a non-Ctrl click has occurred (first interaction edge case)
                const hasCtrl = window.__ctrlHeld || !!(event?.srcEvent?.ctrlKey);
                if (!hasCtrl) return;
                if (!window.__ctrlHeld) { window.__ctrlHeld = true; window.__updateDeckCursor(); }
                // Similar pick mode: capture the clicked track
                if (window.__similarArmed && layer && layer.id === 'tracks' && index >= 0) {
                    window.__similarArmed = false;
                    window.__updateDeckCursor();
                    const si = (window.__visibleSegIdxs !== null && filteredSegIdxs.length > 0)
                        ? filteredSegIdxs[index] : index;
                    const mmsi = Number(tMMSI[si]);
                    const seg  = Number(tSeg[si]);
                    window.__simPickedData = { mmsi, seg };
                    const lbl = document.getElementById('sim-picked-label');
                    if (lbl) lbl.textContent = `Picked: MMSI ${mmsi} / seg ${seg}`;
                    const panel = document.getElementById('sim-panel');
                    if (panel) panel.style.display = 'block';
                    const btn = document.getElementById('btn-similar');
                    if (btn) { btn.textContent = 'Select one representative track'; btn.style.opacity = ''; }
                    window.__copyText(String(mmsi));
                    if (typeof window.__showCopyToast === 'function') window.__showCopyToast(mmsi, event?.srcEvent?.clientX, event?.srcEvent?.clientY);
                    return;
                }
                if (!layer || index < 0) {
                    // Click on empty space: unpin wave highlight
                    if (window._pinnedWave !== null) {
                        window._pinnedWave = null;
                        window.deckInstance.setProps({ layers: buildLayers(window._currentZoom, window._hoveredWave) });
                    }
                    document.getElementById('click-info').textContent = '';
                    return;
                }
                let msg = `${layer.id}#${index}`;
                let copyMmsi = null;
                if (layer.id === 'waves') {
                    // Convert layer-local index → original wave index (needed when a
                    // wave filter is active and the layer was built from a subset).
                    const wvMap = window.__visibleWaveIdxsArr;
                    const origIdx = wvMap ? wvMap[index] : index;
                    // Toggle pin: click same wave to unpin, click different to pin
                    if (window._pinnedWave === origIdx) {
                        window._pinnedWave = null;
                        msg = `unpinned wave MMSI=${wMMSI[origIdx]}`;
                    } else {
                        window._pinnedWave = origIdx;
                        msg = `📌 wave MMSI=${wMMSI[origIdx]} H=${wH[origIdx].toFixed(3)}m`;
                    }
                    copyMmsi = wMMSI[origIdx];
                    window.deckInstance.setProps({ layers: buildLayers(window._currentZoom, window._hoveredWave) });
                } else if (layer.id === 'tracks') {
                    const si = (window.__visibleSegIdxs !== null && filteredSegIdxs.length > 0)
                        ? filteredSegIdxs[index] : index;
                    copyMmsi = tMMSI[si];
                    msg = `track MMSI=${tMMSI[si]} seg=${tSeg[si]}`;
                }
                if (copyMmsi != null) {
                    window.__copyText(String(copyMmsi));
                    if (typeof window.__showCopyToast === 'function') window.__showCopyToast(copyMmsi, event?.srcEvent?.clientX, event?.srcEvent?.clientY);
                }
                document.getElementById('click-info').textContent = '| ' + msg;
            },
            onViewStateChange: (params) => {
                window._currentZoom = params.viewState.zoom;
                rebuildOnView(params.viewState.zoom);
                return params.viewState;
            },
        });
        container.style.cursor = 'grab';
        window.__updateDeckCursor = (isDragging = false) => {
            container.style.cursor = getCursorForState(isDragging);
        };
        // Ctrl held = inspect mode (hover tooltips + click picking enabled).
        // No INPUT/TEXTAREA guard — __ctrlHeld only affects the map, not DOM inputs;
        // those handle Ctrl+A/C/V natively regardless of this flag.
        window.addEventListener('keydown', (e) => {
            if (e.key !== 'Control' || window.__ctrlHeld) return;
            window.__ctrlHeld = true;
            window.__updateDeckCursor();
            if (typeof window.__rebuild === 'function') window.__rebuild();
            if (window.__freehandArmed && !window.__freehandMode) window.__activateFreehandDraw();
            if (window.__waveBoxArmed  && !window.__waveBoxMode)  window.__activateWaveBoxDraw();
        });
        window.addEventListener('keyup', (e) => {
            if (e.key !== 'Control' || !window.__ctrlHeld) return;
            window.__ctrlHeld = false;
            hideTip();
            if (window.__freehandMode && typeof window.__cancelFreehandDraw === 'function') window.__cancelFreehandDraw();
            if (window.__waveBoxMode  && typeof window.__cancelWaveBoxDraw  === 'function') window.__cancelWaveBoxDraw();
            window.__updateDeckCursor();
        });
        // Clear inspect mode if window loses focus while Ctrl is held (otherwise
        // ctrl-tabbing away leaves the flag stuck on with no key event to clear it)
        window.addEventListener('blur', () => {
            if (!window.__ctrlHeld) return;
            window.__ctrlHeld = false;
            hideTip();
            if (window.__freehandMode && typeof window.__cancelFreehandDraw === 'function') window.__cancelFreehandDraw();
            if (window.__waveBoxMode  && typeof window.__cancelWaveBoxDraw  === 'function') window.__cancelWaveBoxDraw();
            window.__updateDeckCursor();
        });
        container.addEventListener('mousedown', (e) => {
            // Sync Ctrl state from the real event so first-interaction Ctrl+click works
            if (e.ctrlKey && !window.__ctrlHeld) {
                window.__ctrlHeld = true;
                window.__updateDeckCursor();
                // Activate armed modes in case keydown didn't fire before Ctrl was held
                if (window.__freehandArmed && !window.__freehandMode) window.__activateFreehandDraw();
                if (window.__waveBoxArmed  && !window.__waveBoxMode)  window.__activateWaveBoxDraw();
            }
            if (!window.__freehandMode && !window.__waveBoxMode) window.__updateDeckCursor(true);
        });
        window.addEventListener('mouseup', () => {
            if (!window.__freehandMode && !window.__waveBoxMode) window.__updateDeckCursor(false);
        });
        const rebuildOnView = debounce((z) => {
            window.deckInstance.setProps({ layers: buildLayers(z, window._hoveredWave) });
            status.textContent = `zoom=${z.toFixed(1)}`;
        }, 250);
        window.__rebuild = () => window.deckInstance.setProps({ layers: buildLayers(window._currentZoom, window._hoveredWave) });

        // ---- Floating legend ----
        const updateLegend = () => {
            const leg = document.getElementById('map-legend');
            if (!leg) return;
            const hasBathy = !!(window.__previews && window.__previews.bathy);
            const hasTracks = !!window.__hasTracks;
            const rows = [];
            if (hasTracks) {
                rows.push('<div style="font-weight:700;font-size:10px;color:#aac;margin-bottom:5px;letter-spacing:.4px">VESSEL TYPE</div>');
                const categories = [
                    ['tanker',         'Tanker'],
                    ['cargo',          'Cargo'],
                    ['passenger',      'Passenger'],
                    ['tug',            'Tug'],
                    ['pleasure_craft', 'Pleasure Craft'],
                    ['pilot_vessel',   'Pilot Vessel'],
                    ['other',          'Other'],
                    ['unknown',        'Unknown'],
                ];
                categories.forEach(([cat, label]) => {
                    const c = _CAT_COLORS[cat] || [160,160,160];
                    const hex = `#${c.map(x=>x.toString(16).padStart(2,'0')).join('')}`;
                    rows.push(
                        `<div style="display:flex;align-items:center;gap:6px;margin:2px 0">` +
                        `<span style="width:14px;height:6px;border-radius:2px;background:${hex};flex-shrink:0"></span>` +
                        `<span style="font-size:10px">${label}</span></div>`
                    );
                });
            }
            if (window.__hasWaves) {
                rows.push('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:7px 0">');
                rows.push('<div style="font-weight:700;font-size:10px;color:#aac;margin-bottom:5px;letter-spacing:.4px">WAVE HEIGHT (m)</div>');
                // Continuous gradient, max (red) on top. Labels = threshold values only.
                const wThresholds = [0.4, 0.15, 0.05, 0];  // top to bottom
                const wGrad = 'linear-gradient(to bottom, #ff1e1e, #ff6400, #ffdc00, #50c878)';
                const wBarH = 100;
                const wStep = wBarH / (wThresholds.length - 1);
                rows.push(
                    `<div style="display:flex;gap:6px;align-items:flex-start">` +
                    `<div style="position:relative;width:14px;height:${wBarH}px;flex-shrink:0">` +
                    `<div style="width:100%;height:100%;border-radius:3px;background:${wGrad}"></div>` +
                    wThresholds.slice(1,-1).map((_, i) =>
                        `<div style="position:absolute;top:${Math.round((i+1)*wStep)}px;left:0;right:0;height:1px;background:rgba(255,255,255,0.25)"></div>`
                    ).join('') +
                    `</div>` +
                    `<div style="position:relative;height:${wBarH}px;font-size:9px;color:#ccc;overflow:visible">` +
                    wThresholds.map((v, i) => {
                        const tr = i === 0 ? 'translateY(0)' : i === wThresholds.length-1 ? 'translateY(-100%)' : 'translateY(-50%)';
                        return `<span style="position:absolute;top:${Math.round(i*wStep)}px;transform:${tr};white-space:nowrap">${v}</span>`;
                    }).join('') +
                    `</div></div>`
                );
            }
            if (hasBathy) {
                rows.push('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:7px 0">');
                rows.push('<div style="font-weight:700;font-size:10px;color:#aac;margin-bottom:5px;letter-spacing:.4px">BATHYMETRY (mCD)</div>');
                const b = window.__previews.bathy;
                // Max (shallowest, zMax) on top; gradient shallow→deep
                const bGrad = 'linear-gradient(to bottom, rgb(175,220,235), rgb(12,35,95))';
                const bBarH = 90;
                const bRange = b.zMax - b.zMin;
                // Nice round intermediate ticks
                const niceTicks = (lo, hi, n) => {
                    const range = Math.abs(hi - lo);
                    if (range === 0) return [];
                    const rawStep = range / n;
                    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
                    const niceStep = [1,2,5,10].map(f => f*mag).find(s => s >= rawStep) || mag*10;
                    const first = Math.ceil(lo / niceStep) * niceStep;
                    const ticks = [];
                    for (let v = first; v <= hi + niceStep*0.01; v += niceStep)
                        ticks.push(Math.round(v * 1e6) / 1e6);
                    return ticks;
                };
                const interTicks = niceTicks(b.zMin, b.zMax, 5).filter(t =>
                    (b.zMax - t) / bRange > 0.08 && (t - b.zMin) / bRange > 0.08
                );
                // Ticks: zMax at top, then intermediate values only (no zMin label)
                const allTicks = [b.zMax, ...interTicks];
                const tickY = t => Math.round((b.zMax - t) / bRange * bBarH);
                rows.push(
                    `<div style="display:flex;gap:6px;align-items:flex-start">` +
                    `<div style="position:relative;width:14px;height:${bBarH}px;flex-shrink:0">` +
                    `<div style="width:100%;height:100%;border-radius:3px;background:${bGrad}"></div>` +
                    interTicks.map(t =>
                        `<div style="position:absolute;top:${tickY(t)}px;left:0;right:0;height:1px;background:rgba(255,255,255,0.3)"></div>`
                    ).join('') +
                    `</div>` +
                    `<div style="position:relative;height:${bBarH}px;font-size:9px;color:#ccc;min-width:40px;overflow:visible">` +
                    allTicks.map((t, i) => {
                        const tr = i === 0 ? 'translateY(0)' : 'translateY(-50%)';
                        return `<span style="position:absolute;top:${tickY(t)}px;transform:${tr}">${t.toFixed(0)}</span>`;
                    }).join('') +
                    `</div></div>`
                );
            }
            if (rows.length === 0) { leg.style.display = 'none'; return; }
            leg.style.display = 'block';
            leg.innerHTML = rows.join('');
        };
        window.__updateLegend = updateLegend;

        // ---- Cascading MMSI → Segment picker ----
        // State is module-level so clear button and track reload can reset it.
        window.__cascadeMMSI = null;
        window.__cascadeSegs = [];

        const updateCascadeTrigger = () => {
            const trigger = document.getElementById('cascade-mmsi-trigger');
            if (!trigger) return;
            if (window.__cascadeMMSI === null) {
                trigger.textContent = 'All tracks';
                trigger.className = 'cascade-trigger';
            } else if (window.__cascadeSegs.length === 0) {
                trigger.textContent = `MMSI ${window.__cascadeMMSI} · all segs`;
                trigger.className = 'cascade-trigger cascade-active';
            } else {
                const s = window.__cascadeSegs.length === 1
                    ? `seg ${window.__cascadeSegs[0]}`
                    : `${window.__cascadeSegs.length} segs`;
                trigger.textContent = `MMSI ${window.__cascadeMMSI} · ${s}`;
                trigger.className = 'cascade-trigger cascade-active';
            }
        };

        const applyCascadeToFilter = () => {
            window.__filterState.mmsi    = window.__cascadeMMSI;
            window.__filterState.seg_ids = [...window.__cascadeSegs];
            window.__recomputeVisibility();
            updateCascadeTrigger();
        };

        const buildCascadePanel = () => {
            const panel = document.getElementById('cascade-mmsi-panel');
            if (!panel) return;
            panel.innerHTML = '';
            // Left column: search + MMSI list
            const leftCol = document.createElement('div');
            leftCol.className = 'cascade-col';
            // Right column: segments (populated on MMSI hover)
            const rightCol = document.createElement('div');
            rightCol.className = 'cascade-col cascade-seg-col';
            rightCol.style.display = 'none';

            const mmsiArr = [];
            mmsiToSegIdxs.forEach((_, mmsi) => mmsiArr.push(Number(mmsi)));
            mmsiArr.sort((a, b) => a - b);

            if (mmsiArr.length === 0) {
                leftCol.innerHTML = '<div class="cascade-empty">No tracks loaded</div>';
                panel.appendChild(leftCol);
                return;
            }

            // Search input
            const searchWrap = document.createElement('div');
            searchWrap.style.cssText = 'padding:3px 5px;border-bottom:1px solid rgba(255,255,255,0.08);position:sticky;top:0;background:#1e2d3d;z-index:1';
            const searchInput = document.createElement('input');
            searchInput.type = 'text';
            searchInput.placeholder = 'Search MMSI…';
            searchInput.style.cssText = 'width:100%;box-sizing:border-box;font-size:10px;padding:2px 5px;border:1px solid #3a5a78;border-radius:3px;outline:none;background:#162230;color:#b8cede;caret-color:#7aaace';
            searchInput.addEventListener('click', e => e.stopPropagation());
            searchWrap.appendChild(searchInput);
            leftCol.appendChild(searchWrap);

            // "All tracks" row (always visible, not filtered by search)
            const allRow = document.createElement('div');
            allRow.className = 'cascade-item' + (window.__cascadeMMSI === null ? ' cascade-selected' : '');
            allRow.textContent = 'All tracks';
            allRow.addEventListener('click', (e) => {
                e.stopPropagation();
                window.__cascadeMMSI = null;
                window.__cascadeSegs = [];
                applyCascadeToFilter();
                panel.style.display = 'none';
            });
            leftCol.appendChild(allRow);

            // MMSI rows container (filtered by search)
            const mmsiListEl = document.createElement('div');
            leftCol.appendChild(mmsiListEl);

            const renderMMSIRows = (query) => {
                mmsiListEl.innerHTML = '';
                const filtered = query ? mmsiArr.filter(m => String(m).includes(query)) : mmsiArr;
                if (filtered.length === 0) {
                    mmsiListEl.innerHTML = '<div class="cascade-empty">No match</div>';
                    rightCol.style.display = 'none';
                    return;
                }
                filtered.forEach(mmsi => {
                    const segsForMMSI = (mmsiToSegIdxs.get(mmsi) || []).map(i => Number(tSeg[i])).sort((a,b)=>a-b);
                    const row = document.createElement('div');
                    const isSelMMSI = window.__cascadeMMSI === mmsi;
                    row.className = 'cascade-item' + (isSelMMSI ? ' cascade-selected' : '');
                    row.innerHTML = `<span>${mmsi}</span><span class="cascade-arrow">${segsForMMSI.length > 1 ? '▶' : ''}</span>`;
                    row.title = `${segsForMMSI.length} segment${segsForMMSI.length !== 1 ? 's' : ''}`;

                    const showSegs = () => {
                        rightCol.innerHTML = '';
                        rightCol.style.display = 'block';
                        // "All" option — show all segments for this MMSI
                        const allSeg = document.createElement('div');
                        const isAllSeg = isSelMMSI && window.__cascadeSegs.length === 0;
                        allSeg.className = 'cascade-item' + (isAllSeg ? ' cascade-selected' : '');
                        allSeg.textContent = 'All';
                        allSeg.title = `All ${segsForMMSI.length} segments`;
                        allSeg.addEventListener('click', (e) => {
                            e.stopPropagation();
                            window.__cascadeMMSI = mmsi;
                            window.__cascadeSegs = [];
                            applyCascadeToFilter();
                            panel.style.display = 'none';
                        });
                        rightCol.appendChild(allSeg);
                        segsForMMSI.forEach(seg => {
                            const sRow = document.createElement('div');
                            const isSelSeg = isSelMMSI && window.__cascadeSegs.includes(seg);
                            sRow.className = 'cascade-item' + (isSelSeg ? ' cascade-selected' : '');
                            sRow.textContent = `seg ${seg}`;
                            sRow.addEventListener('click', (e) => {
                                e.stopPropagation();
                                window.__cascadeMMSI = mmsi;
                                window.__cascadeSegs = [seg];
                                applyCascadeToFilter();
                                panel.style.display = 'none';
                            });
                            rightCol.appendChild(sRow);
                        });
                    };

                    row.addEventListener('mouseenter', showSegs);
                    row.addEventListener('click', (e) => {
                        e.stopPropagation();
                        window.__cascadeMMSI = mmsi;
                        window.__cascadeSegs = [];
                        applyCascadeToFilter();
                        panel.style.display = 'none';
                    });
                    mmsiListEl.appendChild(row);
                });
            };

            searchInput.addEventListener('input', () => renderMMSIRows(searchInput.value.trim()));
            renderMMSIRows('');

            panel.appendChild(leftCol);
            panel.appendChild(rightCol);
        };

        const initCascade = () => {
            const trigger = document.getElementById('cascade-mmsi-trigger');
            const panel = document.getElementById('cascade-mmsi-panel');
            if (!trigger || !panel) return;
            trigger.addEventListener('click', (e) => {
                e.stopPropagation();
                if (panel.style.display === 'none') {
                    buildCascadePanel();
                    panel.style.display = 'flex';
                } else {
                    panel.style.display = 'none';
                }
            });
            document.addEventListener('click', () => {
                if (panel) panel.style.display = 'none';
            });
        };
        initCascade();
        window.__buildCascadeMMSI = buildCascadePanel;
        window.__updateCascadeTrigger = updateCascadeTrigger;

        // ---- Tide DFS0 file × item cascade (same UX as MMSI / Segment) ----
        window.__tideFiles = window.__tideFiles || [];
        window.__tideItems = window.__tideItems || [];
        window.__tideFilePick = null;   // value pushed into _tide_file_pick on next button click
        window.__tideItemPick = null;
        window.__tideSelFile = null;    // currently selected file (path)
        window.__tideSelItem = null;    // currently selected item name
        window.__tideAwaitingItems = false; // true while waiting for server to return items after file pick

        const _fileLabel = (path) => {
            if (!path) return null;
            const f = (window.__tideFiles || []).find(o => o.value === path);
            return f ? f.label : path.split(/[\\/]/).pop();
        };
        const updateCascadeTideTrigger = () => {
            const trig = document.getElementById('cascade-tide-trigger');
            if (!trig) return;
            if (!window.__tideSelFile) {
                trig.textContent = 'No tide file';
                trig.className = 'cascade-trigger';
            } else if (!window.__tideSelItem) {
                trig.textContent = `${_fileLabel(window.__tideSelFile)} · pick item`;
                trig.className = 'cascade-trigger cascade-active';
            } else {
                trig.textContent = `${_fileLabel(window.__tideSelFile)} · ${window.__tideSelItem}`;
                trig.className = 'cascade-trigger cascade-active';
            }
        };

        const buildCascadeTidePanel = () => {
            const panel = document.getElementById('cascade-tide-panel');
            if (!panel) return;
            panel.innerHTML = '';
            const leftCol = document.createElement('div');
            leftCol.className = 'cascade-col';
            const rightCol = document.createElement('div');
            rightCol.className = 'cascade-col cascade-seg-col';
            rightCol.style.display = window.__tideSelFile ? 'block' : 'none';

            const files = window.__tideFiles || [];
            if (files.length === 0) {
                leftCol.innerHTML = '<div class="cascade-empty">No .dfs0 files</div>';
                panel.appendChild(leftCol);
                return;
            }

            files.forEach(f => {
                const row = document.createElement('div');
                const isSel = window.__tideSelFile === f.value;
                row.className = 'cascade-item' + (isSel ? ' cascade-selected' : '');
                row.innerHTML = `<span>${f.label}</span><span class="cascade-arrow">▶</span>`;
                row.addEventListener('click', (e) => {
                    e.stopPropagation();
                    window.__tideSelFile = f.value;
                    window.__tideSelItem = null;
                    window.__tideItems = [];      // clear stale items; will repopulate on server response
                    window.__tideFilePick = f.value;
                    window.__tideAwaitingItems = true;
                    // __tideKeepPanel prevents the document click listener from closing the
                    // panel when the programmatic btn.click() event bubbles up to document.
                    window.__tideKeepPanel = true;
                    document.getElementById('_tide-file-btn').click();
                    window.__tideKeepPanel = false;
                    buildCascadeTidePanel();      // render with "Loading..." in right col
                    updateCascadeTideTrigger();
                });
                leftCol.appendChild(row);
            });

            if (window.__tideSelFile) {
                const items = window.__tideItems || [];
                if (items.length === 0) {
                    rightCol.innerHTML = '<div class="cascade-empty">Loading items…</div>';
                } else {
                    items.forEach(it => {
                        const sRow = document.createElement('div');
                        const isSelI = window.__tideSelItem === it.name;
                        sRow.className = 'cascade-item' + (isSelI ? ' cascade-selected' : '');
                        sRow.textContent = it.label || it.name;
                        sRow.title = it.label || it.name;
                        sRow.addEventListener('click', (e) => {
                            e.stopPropagation();
                            window.__tideSelItem = it.name;
                            window.__tideItemPick = it.name;
                            document.getElementById('_tide-item-btn').click();
                            updateCascadeTideTrigger();
                            panel.style.display = 'none';
                        });
                        rightCol.appendChild(sRow);
                    });
                }
            }
            panel.appendChild(leftCol);
            panel.appendChild(rightCol);
        };

        const initCascadeTide = () => {
            const trig = document.getElementById('cascade-tide-trigger');
            const panel = document.getElementById('cascade-tide-panel');
            if (!trig || !panel) return;
            trig.addEventListener('click', (e) => {
                e.stopPropagation();
                if (panel.style.display === 'none') {
                    buildCascadeTidePanel();
                    panel.style.display = 'flex';
                } else {
                    panel.style.display = 'none';
                }
            });
            document.addEventListener('click', () => {
                if (window.__tideKeepPanel) return;
                if (panel) panel.style.display = 'none';
            });
        };
        window.__tideKeepPanel = false;
        initCascadeTide();
        window.__rebuildCascadeTide = () => {
            const panel = document.getElementById('cascade-tide-panel');
            buildCascadeTidePanel();
            if (panel && window.__tideAwaitingItems && (window.__tideItems || []).length > 0) {
                panel.style.display = 'flex';
                window.__tideAwaitingItems = false;
            } else if (panel && panel.style.display === 'none') {
                // panel already closed by user — leave it closed
            }
            updateCascadeTideTrigger();
        };
        updateCascadeTideTrigger();

        // Post-pipeline refresh hooks: show the same progress overlay as before, then rebuild layers,
        // then show a "Rendering..." pill until deck.gl has painted at least one frame.
        window.__refreshWaveLayer = async (version) => {
            const [buf] = await fetchAssetsWithProgress([
                { key: 'waves', url: `/api/waves.arrow?v=${version}`, label: 'wave impacts' },
            ], 'Loading wave impacts');
            setRenderStatus('Rendering waves...', false);
            rebuildWaveArrays(window.tableFromIPC(buf));
            window.__hasWaves = wMMSI.length > 0;
            window.__waveCount = wMMSI.length;
            window._hoveredWave = null;
            window._pinnedWave  = null;
            window.__rebuild();
            if (typeof window.__updateLegend === 'function') window.__updateLegend();
            await waitForPaint();
            setRenderStatus(`Waves ready (${wMMSI.length.toLocaleString()})`, true);
            clearRenderStatus(1500);
            if (window.dash_clientside?.set_props) {
                window.dash_clientside.set_props('_wave_n', {data: wMMSI.length});
            }
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
            // Reset cascade selection display when new track data arrives
            if (typeof window.__updateCascadeTrigger === 'function') window.__updateCascadeTrigger();
            window.__rebuild();
            if (typeof window.__updateLegend === 'function') window.__updateLegend();
            await waitForPaint();
            setRenderStatus(`Tracks ready (${tMMSI.length.toLocaleString()} segments)`, true);
            clearRenderStatus(1500);
        };

        // ---- Load Results: fetch tracks + waves under one overlay, then zoom-to-fit
        window.__loadResults = async (result) => {
            if (!result) return;
            const tv = result.track_version || 0;
            const wv = result.wave_version || 0;
            const hasWaves = (result.n_waves || 0) > 0;
            const assets = [
                { key: 'track_coords',  url: `/api/track_coords.arrow?v=${tv}`,  label: 'track coords' },
                { key: 'track_meta',    url: `/api/track_meta.arrow?v=${tv}`,    label: 'track metadata' },
                { key: 'track_offsets', url: `/api/track_offsets.arrow?v=${tv}`, label: 'track offsets' },
            ];
            if (hasWaves) assets.push({ key: 'waves', url: `/api/waves.arrow?v=${wv}`, label: 'wave impacts' });
            try {
                const buffers = await fetchAssetsWithProgress(assets, 'Loading saved results');
                setRenderStatus('Rendering...', false);
                initTrackArrays(
                    window.tableFromIPC(buffers[0]),
                    window.tableFromIPC(buffers[1]),
                    window.tableFromIPC(buffers[2]),
                );
                window.__hasTracks = tMMSI.length > 0;
                if (hasWaves) {
                    rebuildWaveArrays(window.tableFromIPC(buffers[3]));
                    window.__hasWaves = wMMSI.length > 0;
                    window.__waveCount = wMMSI.length;
                } else {
                    // Empty wave caches so cross-filter logic stays consistent
                    window.__hasWaves = false;
                    window.__waveCount = 0;
                }
                window._hoveredWave = null;
                window._pinnedWave  = null;
                if (typeof window.__updateCascadeTrigger === 'function') window.__updateCascadeTrigger();
                // Mark versions as "already handled" so the per-version refresh callbacks
                // short-circuit when we bump the stores at the end of the load-results handler.
                window.__lastTrackVersion = tv;
                window.__lastWaveVersion  = wv;
                // Zoom-to-fit using the server-supplied bbox.
                if (result.bbox && Array.isArray(result.bbox) && result.bbox.length === 4) {
                    const [w, s, e, n] = result.bbox;
                    const lonC = (w + e) / 2, latC = (s + n) / 2;
                    const span = Math.max(e - w, n - s, 1e-4);
                    const z = Math.max(8, Math.min(15, 10 - Math.log2(span)));
                    if (window.deckInstance) {
                        window.deckInstance.setProps({
                            initialViewState: { longitude: lonC, latitude: latC, zoom: z, pitch: 0, bearing: 0 },
                        });
                        window._currentZoom = z;
                    }
                }
                window.__rebuild();
                if (typeof window.__updateLegend === 'function') window.__updateLegend();
                await waitForPaint();
                const segN = tMMSI.length, waveN = hasWaves ? wMMSI.length : 0;
                setRenderStatus(`Loaded ${segN.toLocaleString()} segments` +
                                (hasWaves ? ` + ${waveN.toLocaleString()} waves` : ''), true);
                clearRenderStatus(1500);
                if (window.dash_clientside?.set_props) {
                    window.dash_clientside.set_props('_wave_n', {data: window.__waveCount || 0});
                }
            } catch (e) {
                clearRenderStatus(0);
                console.error('load-results failed:', e);
            }
        };

        // ---- AIS import (slow, explicit button) + preview toggle (cheap) ----
        window.__importedAisPath = null;
        window.__aisBbox = null;
        const interleaveLonLat = (lon, lat) => {
            const p = new Float32Array(lon.length * 2);
            for (let i = 0; i < lon.length; i++) { p[i*2] = lon[i]; p[i*2+1] = lat[i]; }
            return p;
        };
        window.__importAis = async (path) => {
            if (!path) return 'no file selected';
            try {
                const [buf] = await window.__fetchAssetsWithProgress(
                    [{ key: 'ais', label: 'AIS preview', url: '/api/preview/ais.arrow?path=' + encodeURIComponent(path) }],
                    'Importing AIS data',
                );
                setRenderStatus('Rendering AIS points...', false);
                const t = window.tableFromIPC(buf);
                const lon = t.getChild('longitude').toArray();
                const lat = t.getChild('latitude').toArray();
                const pos = interleaveLonLat(lon, lat);
                const pvSog  = t.getChild('sog')     ? t.getChild('sog').toArray()     : new Float32Array(lon.length);
                const pvCog  = t.getChild('cog')     ? t.getChild('cog').toArray()     : new Float32Array(lon.length);
                const pvTime = t.getChild('obstime_ns') ? t.getChild('obstime_ns').toArray() : new BigInt64Array(lon.length);
                window.__previews.ais = { pos, visible: true,
                    sog: pvSog, cog: pvCog, obstimeNs: pvTime };
                window.__importedAisPath = path;
                try {
                    const bb = await fetch('/api/preview/ais.bbox?path=' + encodeURIComponent(path)).then(r => r.json());
                    if (bb.bbox) {
                        const [w, s, e, n] = bb.bbox;
                        window.__aisBbox = [w, s, e, n];
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
                try {
                    const meta = await fetch('/api/preview/ais.bbox?path=' + encodeURIComponent(path)).then(r => r.json());
                    if (meta.time_min) {
                        window.__previews.ais.timeMin = meta.time_min;
                        window.__previews.ais.timeMax = meta.time_max;
                    }
                } catch (e) { /* nonfatal */ }
                const range = window.__previews.ais.timeMin
                    ? `\ntime: ${window.__previews.ais.timeMin}  →  ${window.__previews.ais.timeMax}`
                    : '';
                return `imported ${lon.length.toLocaleString()} points${range}`;
            } catch (e) {
                clearRenderStatus(0);
                return 'ERROR: ' + e.message;
            }
        };
        // Replaces the AIS preview buffer with the server-side filtered points.
        // Called after track_version bumps (filter stage completes).
        window.__refreshFilteredAisPoints = async () => {
            try {
                const resp = await fetch('/api/vessels.arrow');
                if (!resp.ok) { console.warn('__refreshFilteredAisPoints: HTTP', resp.status); return; }
                const buf = await resp.arrayBuffer();
                const t = window.tableFromIPC(new Uint8Array(buf));
                const lon = t.getChild('longitude') ? t.getChild('longitude').toArray() : null;
                const lat = t.getChild('latitude')  ? t.getChild('latitude').toArray()  : null;
                if (!lon || lon.length === 0) { console.warn('__refreshFilteredAisPoints: empty table'); return; }
                const pos = interleaveLonLat(lon, lat);
                const sog  = t.getChild('sog')     ? t.getChild('sog').toArray()     : new Float32Array(lon.length);
                const cog  = t.getChild('cog')     ? t.getChild('cog').toArray()     : new Float32Array(lon.length);
                const time = t.getChild('obstime') ? t.getChild('obstime').toArray() : new BigInt64Array(lon.length);
                const prev = window.__previews.ais;
                const visible   = prev ? prev.visible   : true;
                const timeMin   = prev ? prev.timeMin   : undefined;
                const timeMax   = prev ? prev.timeMax   : undefined;
                window.__previews.ais = { pos, sog, cog, obstimeNs: time,
                    visible, timeMin, timeMax, filtered: true };
                if (typeof window.__rebuild === 'function') window.__rebuild();
            } catch (e) { console.warn('__refreshFilteredAisPoints error:', e); }
        };
        window.__togglePreviewAis = (visible) => {
            if (window.__previews.ais) {
                window.__previews.ais.visible = !!visible;
                window.__rebuild();
                if (!window.__previews.ais.visible) return 'hidden';
                const range = window.__previews.ais.timeMin
                    ? `\ntime: ${window.__previews.ais.timeMin}  →  ${window.__previews.ais.timeMax}`
                    : '';
                return `showing ${(window.__previews.ais.pos.length/2).toLocaleString()} points${range}`;
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
                window.__previews.bathy = null;
                window.__rebuild();
                if (typeof window.__updateLegend === 'function') window.__updateLegend();
                return null;
            }
            const enc = encodeURIComponent(state.path);
            // Server filters mesh elements to AIS bbox expanded 4× in each dimension.
            let bboxParam = '';
            if (window.__aisBbox) {
                const [bw, bs, be, bn] = window.__aisBbox;
                const dLon = (be - bw), dLat = (bn - bs);
                const pw = bw - dLon, pe = be + dLon;
                const ps = bs - dLat, pn = bn + dLat;
                bboxParam = `&bbox=${pw},${ps},${pe},${pn}`;
            }
            try {
                const [bufC, bufO] = await Promise.all([
                    fetch('/api/preview/bathy.arrow?path=' + enc + bboxParam)
                        .then(r => { if (!r.ok) return r.json().then(j => Promise.reject(new Error(j.error || 'preview failed'))); return r.arrayBuffer(); }),
                    fetch('/api/preview/bathy_offsets.arrow?path=' + enc + bboxParam)
                        .then(r => { if (!r.ok) return r.json().then(j => Promise.reject(new Error(j.error || 'preview failed'))); return r.arrayBuffer(); }),
                ]);
                const tC = window.tableFromIPC(new Uint8Array(bufC));
                const tO = window.tableFromIPC(new Uint8Array(bufO));
                const lon = tC.getChild('lon').toArray();
                const lat = tC.getChild('lat').toArray();
                const pos = interleaveLonLat(lon, lat);
                const offsets = tO.getChild('offset').toArray();
                const zArr    = tO.getChild('z').toArray();
                const nElems = offsets.length - 1;
                // Find z range (skip padding NaN at index nElems)
                let zMin = Infinity, zMax = -Infinity;
                for (let i = 0; i < nElems; i++) {
                    const v = zArr[i];
                    if (Number.isFinite(v)) {
                        if (v < zMin) zMin = v;
                        if (v > zMax) zMax = v;
                    }
                }
                if (!Number.isFinite(zMin)) { zMin = 0; zMax = 1; }
                const zSpan = (zMax - zMin) || 1;
                // Per-vertex RGBA colour buffer (dark blue=deep → light cyan=shallow)
                const totalVerts = lon.length;
                const fillRgb = new Uint8Array(totalVerts * 4);
                const DEEP = [12, 35, 95];
                const SHAL = [175, 220, 235];
                for (let i = 0; i < nElems; i++) {
                    const z = Number.isFinite(zArr[i]) ? zArr[i] : zMin;
                    // t=0 → deepest (zMin), t=1 → shallowest (zMax)
                    const t = (z - zMin) / zSpan;
                    const r = (DEEP[0] + (SHAL[0] - DEEP[0]) * t) | 0;
                    const g = (DEEP[1] + (SHAL[1] - DEEP[1]) * t) | 0;
                    const b = (DEEP[2] + (SHAL[2] - DEEP[2]) * t) | 0;
                    const vS = offsets[i], vE = offsets[i + 1];
                    for (let v = vS; v < vE; v++) {
                        fillRgb[v*4]   = r;
                        fillRgb[v*4+1] = g;
                        fillRgb[v*4+2] = b;
                        fillRgb[v*4+3] = 200;
                    }
                }
                window.__previews.bathy = { pos, offsets, fillRgb, zMin, zMax };
                window.__rebuild();
                if (typeof window.__updateLegend === 'function') window.__updateLegend();
                return `${nElems.toLocaleString()} mesh elements  (z: ${zMin.toFixed(2)} → ${zMax.toFixed(2)})`;
            } catch (e) {
                window.__previews.bathy = null;
                window.__rebuild();
                if (typeof window.__updateLegend === 'function') window.__updateLegend();
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
                return '';
            } catch (e) {
                window.__previews.coast = null; window.__rebuild();
                return 'ERROR: ' + e.message;
            }
        };
        window.__setPreviewLand = async (state) => {
            if (!state || !state.visible || !state.path) {
                window.__previews.land = null; window.__rebuild(); return '';
            }
            try {
                const gj = await fetch('/api/preview/coast.geojson?path=' + encodeURIComponent(state.path)).then(r => r.json());
                if (gj.error) throw new Error(gj.error);
                window.__previews.land = { geojson: gj };
                // No auto-zoom — land mask covers all of Singapore and zooming
                // out that far would lose the project-area context.
                window.__rebuild();
                return '';
            } catch (e) {
                window.__previews.land = null; window.__rebuild();
                return 'ERROR: ' + e.message;
            }
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
# When the version actually changes (i.e. waves were just recalculated, not a dev-load
# which sets __lastWaveVersion upfront), also reset any active track filters so the
# user inspects the fresh wave dataset against the full track set.
WAVE_RELOAD_JS = r"""
async function(version) {
    if (!version || version === window.__lastWaveVersion) return window.dash_clientside.no_update;
    window.__lastWaveVersion = version;
    if (typeof window.__resetAllFilters === 'function' &&
            window.__visibleSegIdxs !== null) {
        window.__resetAllFilters();
    }
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
        try {
            await window.__refreshTrackCaches(version);
            if (window.__previews?.ais && typeof window.__refreshFilteredAisPoints === 'function') {
                await window.__refreshFilteredAisPoints();
            }
        } catch (e) { console.error(e); }
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
_make_preview_clientside('_pv_land',  'pv-land-info',  '__setPreviewLand')


# Mirror sel-tide options + items meta into JS globals so the cascade
# widget can render from them without going through the DOM.
app.clientside_callback(
    r"""
    function(files, items) {
        window.__tideFiles = files || [];
        window.__tideItems = items || [];
        if (typeof window.__rebuildCascadeTide === 'function') window.__rebuildCascadeTide();
        return files || [];
    }
    """,
    Output('_tide_files_meta', 'data'),
    Input('sel-tide', 'options'),
    Input('_tide_items_meta', 'data'),
    prevent_initial_call=False,
)


# JS-clicked file or item bumps the matching hidden button; these two
# callbacks capture window.__tideFilePick / __tideItemPick into Stores.
app.clientside_callback(
    r"""
    function(n, prev) {
        if (!n) return window.dash_clientside.no_update;
        const nonce = ((prev || {}).nonce || 0) + 1;
        return { value: window.__tideFilePick || null, nonce };
    }
    """,
    Output('_tide_file_pick', 'data'),
    Input('_tide-file-btn', 'n_clicks'),
    State('_tide_file_pick', 'data'),
    prevent_initial_call=True,
)

app.clientside_callback(
    r"""
    function(n, prev) {
        if (!n) return window.dash_clientside.no_update;
        const nonce = ((prev || {}).nonce || 0) + 1;
        return { value: window.__tideItemPick || null, nonce };
    }
    """,
    Output('_tide_item_pick', 'data'),
    Input('_tide-item-btn', 'n_clicks'),
    State('_tide_item_pick', 'data'),
    prevent_initial_call=True,
)


@app.callback(
    Output('sel-tide', 'value', allow_duplicate=True),
    Input('_tide_file_pick', 'data'),
    prevent_initial_call=True,
)
def _sync_tide_file_from_pick(data):
    if data and data.get('value') is not None:
        return data['value']
    return no_update


@app.callback(
    Output('sel-tide-item', 'value', allow_duplicate=True),
    Input('_tide_item_pick', 'data'),
    prevent_initial_call=True,
)
def _sync_tide_item_from_pick(data):
    if data and data.get('value'):
        return data['value']
    return no_update


# Load Results result → single combined progress overlay + zoom-to-fit, then bump stores
# (sentinel-guarded so the per-version refresh callbacks short-circuit).
app.clientside_callback(
    r"""
    async function(result) {
        const nu = window.dash_clientside.no_update;
        if (!result) return [nu, nu, ''];
        if (result.error) return [nu, nu, result.error];
        if (typeof window.__loadResults === 'function') {
            await window.__loadResults(result);
        }
        const status = `Loaded: ${(result.n_segs||0).toLocaleString()} segments, `
                     + `${(result.n_waves||0).toLocaleString()} waves  ← ${result.source}`;
        return [result.track_version, result.wave_version, status];
    }
    """,
    Output('_track_version', 'data', allow_duplicate=True),
    Output('_wave_version',  'data', allow_duplicate=True),
    Output('load-results-status', 'children'),
    Input('_load_result', 'data'),
    prevent_initial_call=True,
)


# AIS import clientside callback — triggered by file selection via _ais_import Store.
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


# Structural filter (MMSI / segment IDs / vessel types) → recompute visibility
app.clientside_callback(
    r"""
    function(structural) {
        if (typeof window.__applyStructuralFilter !== 'function') return window.dash_clientside.no_update;
        return window.__applyStructuralFilter(structural) || '';
    }
    """,
    Output('fil-status', 'children'),
    Input('_filter_structural', 'data'),
    prevent_initial_call=True,
)

# Freehand button → enter draw mode
app.clientside_callback(
    r"""
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        if (typeof window.__enterFreehandMode === 'function') window.__enterFreehandMode();
        return window.dash_clientside.no_update;
    }
    """,
    Output('fil-status', 'children', allow_duplicate=True),
    Input('btn-freehand', 'n_clicks'),
    prevent_initial_call=True,
)

# Wave-arrival-area button click → enter box-drag mode.
app.clientside_callback(
    r"""
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        if (typeof window.__enterWaveBoxMode === 'function') window.__enterWaveBoxMode();
        return window.dash_clientside.no_update;
    }
    """,
    Output('fil-status', 'children', allow_duplicate=True),
    Input('btn-wavebox', 'n_clicks'),
    prevent_initial_call=True,
)

# Export button: enabled when _any_filter_active store is True (set by __recomputeVisibility).
app.clientside_callback(
    r"function(active) { return !active; }",
    Output('btn-fil-export', 'disabled', allow_duplicate=True),
    Input('_any_filter_active', 'data'),
    prevent_initial_call='initial_duplicate',
)

# Enable/disable the dependent-picker section based on AIS selection.
app.clientside_callback(
    "function(v) { return v ? {} : {pointerEvents: 'none', opacity: '0.5'}; }",
    Output('pickers-need-ais', 'style'),
    Input('sel-ais', 'value'),
    prevent_initial_call=False,
)

# Enable/disable the track-filter section based on whether waves are loaded.
app.clientside_callback(
    "function(n) { return (n && n > 0) ? {} : {pointerEvents: 'none', opacity: '0.5'}; }",
    Output('filter-section-wrap', 'style'),
    Input('_wave_n', 'data'),
    prevent_initial_call=False,
)

# Reset the cascade-tide-trigger label when sel-tide is cleared (e.g. on workdir
# change). Without this the trigger keeps showing the previous workdir's pick.
app.clientside_callback(
    r"""
    function(tide_val) {
        if (tide_val) return window.dash_clientside.no_update;
        window.__tideSelFile = null;
        window.__tideSelItem = null;
        const trig = document.getElementById('cascade-tide-trigger');
        if (trig) {
            trig.textContent = 'No tide file';
            trig.className = 'cascade-trigger';
        }
        const panel = document.getElementById('cascade-tide-panel');
        if (panel) panel.style.display = 'none';
        return window.dash_clientside.no_update;
    }
    """,
    Output('cascade-tide-trigger', 'children'),
    Input('sel-tide', 'value'),
    prevent_initial_call=True,
)


# Export button click → show destination form on first click, POST on second click.
app.clientside_callback(
    r"""
    async function(n, dest, workdir, sel_ais) {
        const nu = window.dash_clientside.no_update;
        const showForm = {'display': 'block'};
        const hideForm = {'display': 'none'};
        if (!n) return [nu, nu, nu];
        const cleanDest = (dest || '').trim();
        if (!cleanDest) {
            // First click or no name yet: reveal the folder input.
            return ['Enter a destination folder name above', nu, showForm];
        }
        if (!workdir) {
            return ['Pick a source workdir first', nu, showForm];
        }
        const seg_keys = (typeof window.__getFilteredSegKeys === 'function')
            ? window.__getFilteredSegKeys() : [];
        if (seg_keys.length === 0) {
            return ['Apply a filter first', nu, showForm];
        }
        const wave_idxs = (typeof window.__getFilteredWaveIdxs === 'function')
            ? window.__getFilteredWaveIdxs() : null;
        const body = JSON.stringify({
            dest_name: cleanDest, workdir, seg_keys, wave_idxs, sel_ais: sel_ais || '',
        });
        try {
            const resp = await fetch('/api/export/filtered',
                { method: 'POST', headers: {'Content-Type': 'application/json'}, body });
            const j = await resp.json();
            if (!resp.ok || j.error) {
                return [`Error: ${j.error || resp.statusText}`, nu, showForm];
            }
            const cp = j.copied || {};
            const cpStr = ['coastline','land','bathymetry','tide']
                .map(k => `${k}:${cp[k]||0}`).join(' ');
            const msg = `Exported to ${j.workdir}\n`
                      + `  tracks=${j.n_tracks}  waves=${j.n_waves}  ais=${j.n_ais}\n`
                      + `  copied  ${cpStr}`;
            // Bump the workdir rescan trigger so the new folder appears.
            return [msg, Date.now(), hideForm];
        } catch (e) {
            return [`Error: ${e.message}`, nu, showForm];
        }
    }
    """,
    Output('export-status', 'children'),
    Output('_rescan_count', 'data', allow_duplicate=True),
    Output('export-dest-form', 'style'),
    Input('btn-fil-export', 'n_clicks'),
    State('inp-export-dest', 'value'),
    State('sel-workdir', 'value'),
    State('sel-ais', 'value'),
    prevent_initial_call=True,
)

# Similar button → enter pick mode
app.clientside_callback(
    r"""
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        if (typeof window.__enterSimilarMode === 'function') {
            const msg = window.__enterSimilarMode();
            return msg || window.dash_clientside.no_update;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('fil-status', 'children', allow_duplicate=True),
    Input('btn-similar', 'n_clicks'),
    prevent_initial_call=True,
)

# Similar confirm → POST to server, apply result
app.clientside_callback(
    r"""
    async function(n, buffer_m, min_cov) {
        if (!n) return window.dash_clientside.no_update;
        if (typeof window.__runSimilar === 'function') {
            const result = await window.__runSimilar(buffer_m || 200, min_cov || 0.5);
            return result || '';
        }
        return 'init pending';
    }
    """,
    Output('fil-status', 'children', allow_duplicate=True),
    Input('btn-sim-confirm', 'n_clicks'),
    State('sim-buffer-m', 'value'),
    State('sim-coverage', 'value'),
    prevent_initial_call=True,
)

# Similar cancel
app.clientside_callback(
    r"""
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        const p = document.getElementById('sim-panel');
        if (p) p.style.display = 'none';
        if (window.__filterState) { window.__filterState.similar = null; }
        if (window.__similarArmed) {
            window.__similarArmed = false;
            const btn = document.getElementById('btn-similar');
            if (btn) { btn.textContent = 'Select one representative track'; btn.style.opacity = ''; }
            if (typeof window.__updateDeckCursor === 'function') window.__updateDeckCursor();
        }
        if (typeof window.__recomputeVisibility === 'function') window.__recomputeVisibility();
        return 'Similar filter cleared';
    }
    """,
    Output('fil-status', 'children', allow_duplicate=True),
    Input('btn-sim-cancel', 'n_clicks'),
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
    from waitress import serve

    THREADS = 4
    print(f'\n=== aiswakepy deck.gl spike ===')
    print(f'Local       : http://127.0.0.1:{PORT}')
    for ip in _lan_ips():
        print(f'LAN         : http://{ip}:{PORT}')
    print(f'(waitress, threads={THREADS}, bind 0.0.0.0:{PORT} - accessible from any host that can reach this machine)\n')
    serve(app.server, host='0.0.0.0', port=PORT, threads=THREADS)
