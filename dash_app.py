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
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from dash import Dash, dcc, html, Input, Output, State, no_update
from flask import Response, jsonify, request, send_file
from werkzeug.utils import secure_filename

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
# Empty runtime templates
# ---------------------------------------------------------------------------
VESSEL_COLUMNS = [
    'mmsi', 'longitude', 'latitude', 'sog', 'cog', 'typecargo',
    'segment_id', 'obstime', 'width', 'length', 'draught',
]
WAVE_COLUMNS = [
    'ShLongitude', 'ShLatitude', 'MMSI', 'WaveHeight', 'WavePeriod',
    'Side', 'DistLoc_km', 'SOG', 'VesselLength', 'VesselWidth',
    'DateTime', 'VesselLongitude', 'VesselLatitude',
    'segment_id', 'VesselDraught', 'VesselCOG',
]
ANIMATION_RAY_COLUMNS = [
    'MMSI', 'segment_id', 'SourceLongitude', 'SourceLatitude',
    'EndLongitude', 'EndLatitude', 'SourceTime', 'Side',
    'Distance_m', 'ReachedShore', 'WakeDirection_deg', 'Theta_deg',
    'SOGms', 'PhaseSpeed_mps', 'GroupSpeed_mps',
    'CuspAngle_deg', 'TransverseSpeed_mps', 'CuspDirection_deg',
    'CuspEndLongitude', 'CuspEndLatitude', 'CuspDistance_m',
    'CuspReachedShore',
]


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


def _build_vessel_caches(state: 'SessionState', df_v: pd.DataFrame) -> None:
    """(Re)compute vessel Arrow + track-segment Arrow + datashader PNG.

    Vectorised segment encoding: sort by (mmsi, segment_id), use np.diff to find
    boundaries, then build flat coords + offsets without a Python-level groupby loop.
    """
    print('  casting types...')
    df_v = df_v.astype({
        'mmsi': 'int64', 'segment_id': 'int32', 'typecargo': 'float32',
        'longitude': 'float32', 'latitude': 'float32',
        'sog': 'float32', 'cog': 'float32',
        'width': 'float32', 'length': 'float32', 'draught': 'float32',
    }, errors='ignore')
    state.df_vessels = df_v

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
    state.ipc_vessels = _ipc(arrow_vessels)

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

    state.seg_meta = [{'mmsi': int(m), 'segment_id': int(s), 'n_points': int(n)}
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
    state.ipc_track_coords = _ipc(arrow_coords)
    state.ipc_track_meta = _ipc(arrow_meta)
    state.ipc_track_offsets = _ipc(arrow_offsets)

    print('  rasterising vessel density...')
    canvas = ds.Canvas(plot_width=RASTER_W, plot_height=RASTER_H,
                       x_range=(RASTER_AOI[0], RASTER_AOI[2]),
                       y_range=(RASTER_AOI[1], RASTER_AOI[3]))
    agg = canvas.points(df_v, x='longitude', y='latitude')
    img = tf.shade(agg, cmap=['#330033', '#ff6600', '#ffff80'], how='log')
    buf = io.BytesIO()
    img.to_pil().save(buf, format='PNG')
    state.png_bytes = buf.getvalue()


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


def _build_wave_caches(state: 'SessionState', df_w: pd.DataFrame) -> None:
    """(Re)compute the wave Arrow with the full enriched schema."""
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

    state.df_waves = df_w
    state.ipc_waves = _ipc(pa.Table.from_pandas(df_w, preserve_index=False))


def _build_animation_ray_cache(state: 'SessionState', df_rays: pd.DataFrame) -> None:
    """Encode exact wave propagation rays for browser animation."""
    df = df_rays.reindex(columns=ANIMATION_RAY_COLUMNS).copy()
    numeric = [
        'SourceLongitude', 'SourceLatitude', 'EndLongitude', 'EndLatitude',
        'Distance_m', 'WakeDirection_deg', 'Theta_deg', 'SOGms',
        'PhaseSpeed_mps', 'GroupSpeed_mps', 'CuspAngle_deg',
        'TransverseSpeed_mps', 'CuspDirection_deg', 'CuspEndLongitude',
        'CuspEndLatitude', 'CuspDistance_m',
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    df['MMSI'] = pd.to_numeric(df['MMSI'], errors='coerce').fillna(-1).astype('int64')
    df['segment_id'] = (
        pd.to_numeric(df['segment_id'], errors='coerce').fillna(-1).astype('int32')
    )
    df['SourceTime'] = pd.to_datetime(df['SourceTime'], errors='coerce').astype('int64')
    df['Side'] = df['Side'].fillna('').astype(str)
    df['ReachedShore'] = df['ReachedShore'].fillna(False).astype(bool)
    df['CuspReachedShore'] = df['CuspReachedShore'].fillna(False).astype(bool)
    state.df_animation_rays = df
    state.ipc_animation_rays = _ipc(pa.Table.from_pandas(df, preserve_index=False))


# ---------------------------------------------------------------------------
# Initial state: empty caches. The page boots showing only the basemap.
# Tracks appear after Step 1 (Filter AIS); waves after Step 2 (Calculate waves);
# AIS preview points appear when the AIS preview checkbox is ticked.
# ---------------------------------------------------------------------------
EMPTY_IPC_VESSELS = _ipc(pa.table({
    'longitude': pa.array([], pa.float32()), 'latitude': pa.array([], pa.float32()),
    'mmsi': pa.array([], pa.int64()), 'sog': pa.array([], pa.float32()),
    'cog': pa.array([], pa.float32()), 'typecargo': pa.array([], pa.float32()),
}))
EMPTY_IPC_TRACK_COORDS = _ipc(pa.table({
    'lon': pa.array([], pa.float32()), 'lat': pa.array([], pa.float32()),
    'sog': pa.array([], pa.float32()), 'cog': pa.array([], pa.float32()),
    'obstime': pa.array([], pa.int64()),
}))
EMPTY_IPC_TRACK_META = _ipc(pa.table({
    'mmsi': pa.array([], pa.int64()), 'segment_id': pa.array([], pa.int32()),
    'n_points': pa.array([], pa.int32()), 'typecargo': pa.array([], pa.int32()),
}))
EMPTY_IPC_TRACK_OFFSETS = _ipc(pa.table({'offset': pa.array([0], pa.int32())}))
_empty_wave_state = type('_EmptyWaveState', (), {})()
_build_wave_caches(_empty_wave_state, pd.DataFrame(columns=WAVE_COLUMNS))
EMPTY_IPC_WAVES = _empty_wave_state.ipc_waves
_build_animation_ray_cache(
    _empty_wave_state, pd.DataFrame(columns=ANIMATION_RAY_COLUMNS)
)
EMPTY_IPC_ANIMATION_RAYS = _empty_wave_state.ipc_animation_rays
# 1x1 transparent PNG placeholder so /api/raster.png never 500s before data exists
EMPTY_PNG_BYTES = bytes.fromhex(
    '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4'
    '890000000d49444154789c63000100000005000100200001ad6f0e0000000049'
    '454e44ae426082'
)
print('caches initialised empty - tracks/waves appear after the corresponding pipeline step.')


def _new_pipeline_state() -> dict:
    return {
        'running': False, 'log': [], 'live': '',
        'started_at': None, 'finished_at': None, 'error': None,
        'wave_version': 0, 'track_version': 0,
        'n_waves': None, 'n_filtered': None, 'last_step': None, 'cfg': None,
    }


@dataclass
class SessionState:
    session_id: str
    root: Path
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    files: dict[str, Path] = field(default_factory=dict)
    original_names: dict[str, str] = field(default_factory=dict)
    df_vessels: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=VESSEL_COLUMNS))
    df_waves: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=WAVE_COLUMNS))
    df_animation_rays: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=ANIMATION_RAY_COLUMNS))
    seg_meta: list[dict] = field(default_factory=list)
    ipc_vessels: bytes = EMPTY_IPC_VESSELS
    ipc_waves: bytes = EMPTY_IPC_WAVES
    ipc_animation_rays: bytes = EMPTY_IPC_ANIMATION_RAYS
    ipc_track_coords: bytes = EMPTY_IPC_TRACK_COORDS
    ipc_track_meta: bytes = EMPTY_IPC_TRACK_META
    ipc_track_offsets: bytes = EMPTY_IPC_TRACK_OFFSETS
    png_bytes: bytes = EMPTY_PNG_BYTES
    last_results: dict = field(default_factory=dict)
    downloads: dict[str, Path] = field(default_factory=dict)
    pipeline: dict = field(default_factory=_new_pipeline_state)
    lock: threading.RLock = field(default_factory=threading.RLock)


# ---------------------------------------------------------------------------
# Server deployment config (UNC paths, host-specific settings)
# ---------------------------------------------------------------------------
_srv_cfg_path = REPO / 'server_config.json'
_srv_cfg: dict = json.loads(_srv_cfg_path.read_text()) if _srv_cfg_path.exists() else {}
DATA_UNC_ROOT: str = _srv_cfg.get('data_unc_root', '')

# ---------------------------------------------------------------------------
# Data directory inventory
# ---------------------------------------------------------------------------
_data_root_value = os.environ.get('DATA_ROOT') or _srv_cfg.get('data_root') or 'data'
DATA_ROOT = Path(_data_root_value).expanduser()
if not DATA_ROOT.is_absolute():
    DATA_ROOT = REPO / DATA_ROOT
DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _data_path_value(path: Path) -> str:
    """Return a stable UI path rooted at data/, independent of DATA_ROOT."""
    return f"data/{path.relative_to(DATA_ROOT).as_posix()}"


def _safe_app_path(value: str, *, must_exist: bool = True) -> Path:
    """Resolve a UI path under either DATA_ROOT or the repository root."""
    if not value:
        raise ValueError('empty path')
    rel = Path(value)
    if rel.is_absolute():
        raise ValueError(f'absolute path not allowed: {value!r}')

    parts = rel.parts
    if parts and parts[0] == 'data':
        root = DATA_ROOT.resolve()
        suffix = Path(*parts[1:])
    else:
        root = REPO.resolve()
        suffix = rel

    resolved = (root / suffix).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f'path {value!r} escapes its allowed root')
    if must_exist and not resolved.exists():
        raise FileNotFoundError(value)
    return resolved




def _ais_time_range_str(state: SessionState) -> str:
    if 'obstime' not in state.df_vessels.columns or len(state.df_vessels) == 0:
        return ''
    ts = state.df_vessels['obstime']
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
_active_pipeline_session: str | None = None


class _LineCapture(io.TextIOBase):
    """sys.stdout shim with proper carriage-return handling for in-place spinners.

    Char-by-char state machine:
      - '\\r' resets the in-progress buffer (cursor return — Spinner is about
        to overwrite); does NOT commit to log.
      - '\\n' commits the in-progress buffer as a single log line.
      - any other char appends to the buffer.
    The in-progress buffer is also surfaced as the session pipeline ``live`` after
    every write call, so the UI can render it as a single replaceable line
    below the committed log — this is the "spinning in place" effect.
    """

    def __init__(self, original, state: SessionState):
        self._orig = original
        self._state = state
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
        with self._state.lock:
            if new_lines:
                self._state.pipeline['log'].extend(new_lines)
            self._state.pipeline['live'] = live
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


def _pipeline_thread(
    state: SessionState,
    config_dict: dict,
    stages: list[str],
    step_label: str,
) -> None:
    """Worker thread. Runs the requested stages and refreshes only the affected caches.

    Cache builds run *outside* the pipeline lock so the polling tick can keep
    reading the session pipeline log/live state (and therefore the sidebar log keeps
    updating) while the slow groupby + Arrow encoding is in progress.
    """
    global _active_pipeline_session
    state.pipeline['cfg'] = config_dict
    old_stdout = sys.stdout
    sys.stdout = _LineCapture(old_stdout, state)
    try:
        cfg = load_config(config_dict)

        # The unified pipeline always runs filter+vessel+wave_impact from scratch;
        # no seed-results / filter-cache shortcut. Filter is cheap relative to
        # wave_impact and the new bathy/tide params have to flow through it.
        results = run_pipeline(cfg, stages=stages)
        state.last_results.update(results)

        # ---- Cache rebuild (no lock held) ----
        out_dir = Path(cfg.output.directory)
        # Track-display source: df_vessel — its rows are exactly those that
        # produced waves (post depth + SOG + BLratio trims). segment_ids align
        # with df_wave_impact because both inherit from the single final
        # segment_trajectories call inside filter_ais.
        # Mirror df_vessel into session.last_results['df_filtered'] so the "Export
        # filtered" path and any consumers reading session.last_results see the same
        # segment_id space as the displayed tracks and waves.
        vessels_for_tracks = results.get('df_vessel')
        if vessels_for_tracks is not None:
            state.last_results['df_filtered'] = vessels_for_tracks
            print('Refreshing track caches from df_vessel...')
            t0 = time.perf_counter()
            _build_vessel_caches(state, vessels_for_tracks)
            print(f'  -> {len(vessels_for_tracks):,} rows, '
                  f'{len(state.seg_meta):,} segments  ({time.perf_counter()-t0:.1f}s)')
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                vessels_for_tracks.to_parquet(out_dir / 'vessels.parquet', index=False)
                print(f'  ✓ saved results: vessels.parquet')
            except Exception as e:
                print(f'  WARN: could not save vessels.parquet: {e}')
            with state.lock:
                state.pipeline['track_version'] += 1
                state.pipeline['n_filtered'] = len(vessels_for_tracks)

        if 'df_wave_impact' in results and 'wave_impact' in stages:
            print('Refreshing wave caches...')
            t0 = time.perf_counter()
            # Fresh runs already have the columns _ensure_vessel_columns would
            # back-fill, so no join needed here. The helper is kept around for
            # _load_results also handles legacy CSV imports.
            _build_wave_caches(state, results['df_wave_impact'])
            _build_animation_ray_cache(
                state, results.get(
                    'df_wave_animation',
                    pd.DataFrame(columns=ANIMATION_RAY_COLUMNS),
                )
            )
            print(f'  -> {len(results["df_wave_impact"]):,} wave events '
                  f'({time.perf_counter()-t0:.1f}s)')
            # Save wave parquet for dev-loading
            try:
                results['df_wave_impact'].to_parquet(out_dir / 'waves.parquet', index=False)
                print(f'  ✓ saved results: waves.parquet')
            except Exception as e:
                print(f'  WARN: could not save waves.parquet: {e}')
            try:
                results.get(
                    'df_wave_animation',
                    pd.DataFrame(columns=ANIMATION_RAY_COLUMNS),
                ).to_parquet(out_dir / 'wave_animation.parquet', index=False)
                print('  ✓ saved results: wave_animation.parquet')
            except Exception as e:
                print(f'  WARN: could not save wave_animation.parquet: {e}')
            try:
                _write_wave_track_link(results['df_wave_impact'], out_dir)
                print(f'  ✓ saved wave_track_link.csv')
            except Exception as e:
                print(f'  WARN: could not save wave_track_link.csv: {e}')
            with state.lock:
                state.pipeline['wave_version'] += 1
                state.pipeline['n_waves'] = len(results['df_wave_impact'])

            print('Generating report plots...')
            _generate_report_plots(
                config_dict,
                vessels_for_tracks if vessels_for_tracks is not None else pd.DataFrame(),
                results['df_wave_impact'],
                out_dir=out_dir,
            )

        with state.lock:
            state.pipeline['finished_at'] = time.time()
            state.pipeline['last_step'] = step_label
    except Exception as exc:
        import traceback
        traceback.print_exc()
        with state.lock:
            state.pipeline['error'] = f'{type(exc).__name__}: {exc}'
            state.pipeline['finished_at'] = time.time()
    finally:
        sys.stdout = old_stdout
        with state.lock:
            state.pipeline['running'] = False
            state.pipeline['live'] = ''
        with _pipeline_lock:
            if _active_pipeline_session == state.session_id:
                _active_pipeline_session = None


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
    <script src="/assets/animation_controller.js"></script>
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
        #btn-fil-export.export-busy { pointer-events: none; cursor: wait; opacity: 0.8; }
        #btn-fil-export.export-busy::before {
            content: ''; display: inline-block; width: 10px; height: 10px;
            margin-right: 6px; vertical-align: -1px; border: 2px solid rgba(255,255,255,0.45);
            border-top-color: white; border-radius: 50%;
            animation: export-spin 0.75s linear infinite;
        }
        @keyframes export-spin { to { transform: rotate(360deg); } }
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
                        line-height: 14px; padding: 5px 8px; height: 52px; overflow: auto;
                        white-space: pre-wrap; box-sizing: border-box;
                        border-radius: 3px; margin: 4px 0; }
        #deck-container { position: fixed; top: 40px; left: 340px; right: 0; bottom: 0;
                          z-index: 1; overflow: hidden; transition: right 0.2s ease; }
        #btn-animation { padding: 3px 11px; border-radius: 5px; font-size: 15px;
                         border: 1px solid #4a85b5; color: white; font-weight: 700;
                         background: linear-gradient(180deg, #6aabda, #4a85b5);
                         box-shadow: 0 1px 4px rgba(0,0,0,0.28); cursor: pointer;
                         flex-shrink: 0; line-height: 1; }
        #btn-animation.playing { background: linear-gradient(180deg, #5abaaa, #3a9a8a);
                                 border-color: #3a9a8a; }
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
        .status-highlight { animation: status-highlight 1.1s ease-out; }
        #ctrl-hint.ctrl-highlight { animation: ctrl-highlight 2.6s ease-out; }
        @keyframes status-highlight {
            0% { background: rgba(255,220,80,0.9); color: #17243a;
                 box-shadow: 0 0 0 3px rgba(255,220,80,0.35); }
            100% { background: transparent; box-shadow: none; }
        }
        @keyframes ctrl-highlight {
            0%, 55% { background: rgba(255,215,55,0.98);
                 color: #17243a; border-color: rgba(255,235,120,1);
                 box-shadow: 0 0 0 7px rgba(255,215,55,0.42), 0 2px 18px rgba(0,0,0,0.55);
                 transform: scale(1.08); }
            100% { border-color: rgba(255,255,255,0.08);
                   color: inherit;
                   background: rgba(16,18,28,0.88);
                   box-shadow: 0 2px 12px rgba(0,0,0,0.45); transform: scale(1); }
        }
        #ctrl-hint.ctrl-highlight #ctrl-hint-title,
        #ctrl-hint.ctrl-highlight #ctrl-hint-body { color: #17243a; }
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
server = app.server

_upload_folder_value = os.environ.get('UPLOAD_FOLDER')
UPLOAD_FOLDER = Path(
    _upload_folder_value or Path(tempfile.gettempdir()) / 'aiswakepy-uploads'
).expanduser()
if not UPLOAD_FOLDER.is_absolute():
    UPLOAD_FOLDER = REPO / UPLOAD_FOLDER
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Bundled example data (tracked in repo under example_data/).
EXAMPLE_DATA = REPO / 'example_data'

# Role specs: canonical mapping of upload role → file accept filter, example subdir, UI label.
# Backend role keys: 'ais', 'coast', 'land', 'bathy', 'tide' (coast/bathy differ from
# the config fields coastline/bathymetry — these are internal session keys only).
ROLE_SPECS: dict[str, dict] = {
    'ais':   {'accept': '.csv',        'example_dir': 'ais',        'label': 'AIS CSV'},
    'coast': {'accept': '.shp,.shx,.dbf,.prj,.xml,.sbn,.sbx,.cpg',
              'example_dir': 'coastline', 'label': 'Coastline Shapefile'},
    'land':  {'accept': '.shp,.shx,.dbf,.prj,.xml,.sbn,.sbx,.cpg',
              'example_dir': 'land', 'label': 'Land Mask Shapefile'},
    'bathy': {'accept': '.mesh,.dfsu', 'example_dir': 'bathymetry', 'label': 'Bathymetry'},
    'tide':  {'accept': '.dfs0',       'example_dir': 'tide',       'label': 'Tide DFS0 (optional)'},
}

server.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY'),
    UPLOAD_FOLDER=str(UPLOAD_FOLDER),
    MAX_CONTENT_LENGTH=int(os.environ.get('MAX_UPLOAD_FILE_BYTES', 100 * 1024 * 1024)),
)


@server.errorhandler(413)
def _upload_too_large(_exc):
    return jsonify({'error': 'file exceeds the per-file upload limit'}), 413
MAX_UPLOAD_FILE_BYTES = int(
    os.environ.get('MAX_UPLOAD_FILE_BYTES', 100 * 1024 * 1024))
MAX_SESSION_BYTES = int(
    os.environ.get('MAX_SESSION_BYTES', 300 * 1024 * 1024))
SESSION_TTL_SECONDS = int(os.environ.get('SESSION_TTL_SECONDS', 3600))

_sessions: dict[str, SessionState] = {}
_sessions_lock = threading.RLock()
_last_cleanup = 0.0


def _cleanup_sessions(*, force: bool = False) -> None:
    global _last_cleanup
    now = time.time()
    if not force and now - _last_cleanup < 60:
        return
    expired: list[SessionState] = []
    with _sessions_lock:
        for session_id, state in list(_sessions.items()):
            if state.pipeline['running']:
                continue
            if now - state.last_access > SESSION_TTL_SECONDS:
                expired.append(_sessions.pop(session_id))
        _last_cleanup = now
    for state in expired:
        shutil.rmtree(state.root, ignore_errors=True)


def _create_session() -> SessionState:
    _cleanup_sessions()
    session_id = uuid.uuid4().hex
    root = UPLOAD_FOLDER / session_id
    root.mkdir(parents=True, exist_ok=False)
    state = SessionState(session_id=session_id, root=root)
    with _sessions_lock:
        _sessions[session_id] = state
    return state


def _get_session(session_id: str | None = None) -> SessionState:
    _cleanup_sessions()
    sid = session_id or request.args.get('session_id') or request.headers.get('X-Session-ID')
    if not sid:
        raise ValueError('session_id required')
    with _sessions_lock:
        state = _sessions.get(sid)
    if state is None:
        raise FileNotFoundError('session expired or not found; refresh the page')
    state.last_access = time.time()
    return state


def _session_path(state: SessionState, value: str, *, must_exist: bool = True) -> Path:
    if not value:
        raise ValueError('empty upload path')
    candidate = (state.root / value).resolve()
    try:
        candidate.relative_to(state.root.resolve())
    except ValueError as exc:
        raise ValueError('upload path escapes session storage') from exc
    if must_exist and not candidate.exists():
        raise FileNotFoundError(value)
    return candidate


def _session_size(state: SessionState) -> int:
    return sum(p.stat().st_size for p in state.root.rglob('*') if p.is_file())


def _shapefile_base_name(path: Path) -> str:
    name = path.name.lower()
    return name[:-8] if name.endswith('.shp.xml') else path.stem.lower()


def _validate_loose_shapefile(directory: Path) -> Path:
    """Validate a directory already containing a shapefile bundle; return the .shp path."""
    shapefiles = sorted(directory.glob('*.shp'))
    if len(shapefiles) != 1:
        raise ValueError('shapefile directory must contain exactly one .shp file')
    stem = shapefiles[0].stem
    files_by_suffix = {p.suffix.lower(): p for p in directory.iterdir() if p.is_file()}
    missing = [ext for ext in ('.shx', '.dbf') if ext not in files_by_suffix]
    if missing:
        raise ValueError(f'shapefile is missing required sidecars: {", ".join(missing)}')
    import geopandas as gpd
    layer = gpd.read_file(shapefiles[0])
    if layer.empty:
        raise ValueError('shapefile contains no features')
    return shapefiles[0]


def _validate_uploaded_file(role: str, path: Path, state: SessionState) -> Path:
    suffix = path.suffix.lower()
    if role == 'ais':
        if suffix != '.csv':
            raise ValueError('AIS input must be a .csv file')
        cols = {str(c).strip().lower() for c in pd.read_csv(path, nrows=5).columns}
        required = {
            'mmsi', 'width', 'length', 'draught', 'obstime',
            'longitude', 'latitude', 'sog', 'cog', 'typecargo',
        }
        missing = sorted(required - cols)
        if missing:
            raise ValueError(f'AIS CSV is missing required columns: {missing}')
        return path
    if role in {'coast', 'land'}:
        raise ValueError('shapefiles must be uploaded with their sidecar files')
    if role == 'bathy':
        if suffix not in {'.mesh', '.dfsu'}:
            raise ValueError('bathymetry input must be .mesh or .dfsu')
        from aiswakepy.geo.bathymetry import load_bathymetry
        load_bathymetry(path)
        return path
    if role == 'tide':
        if suffix != '.dfs0':
            raise ValueError('tide input must be a .dfs0 file')
        _preview_tide(path)
        return path
    raise ValueError(f'unsupported upload role: {role}')


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.server.route('/api/session', methods=['POST'])
def _r_create_session():
    state = _create_session()
    return jsonify({
        'session_id': state.session_id,
        'expires_in_s': SESSION_TTL_SECONDS,
        'max_file_bytes': MAX_UPLOAD_FILE_BYTES,
        'max_session_bytes': MAX_SESSION_BYTES,
    })


@app.server.route('/api/upload/<role>', methods=['POST'])
def _r_upload(role: str):
    try:
        state = _get_session()
        if role not in ROLE_SPECS:
            return jsonify({'error': 'unknown upload role'}), 404
        uploads = request.files.getlist('files') if role in {'coast', 'land'} else []
        if not uploads:
            uploaded = request.files.get('file')
            uploads = [uploaded] if uploaded is not None else []
        uploads = [item for item in uploads if item and item.filename]
        if not uploads:
            return jsonify({'error': 'file upload is required'}), 400
        expected_len = request.content_length or 0
        if expected_len > MAX_UPLOAD_FILE_BYTES:
            return jsonify({'error': 'file exceeds the per-file upload limit'}), 413
        role_dir = state.root / 'uploads' / role
        state.files.pop(role, None)
        state.original_names.pop(role, None)
        shutil.rmtree(role_dir, ignore_errors=True)
        role_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        allowed_sidecars = {'.shp', '.shx', '.dbf', '.prj', '.xml', '.sbn', '.sbx', '.cpg'}
        for uploaded in uploads:
            filename = secure_filename(uploaded.filename)
            if not filename:
                raise ValueError('invalid filename')
            suffix = Path(filename).suffix.lower()
            if role in {'coast', 'land'} and suffix not in allowed_sidecars:
                continue
            raw_path = role_dir / filename
            uploaded.save(raw_path)
            if raw_path.stat().st_size > MAX_UPLOAD_FILE_BYTES:
                shutil.rmtree(role_dir, ignore_errors=True)
                return jsonify({'error': f'{filename} exceeds the per-file upload limit'}), 413
            saved.append(raw_path)
        if not saved:
            raise ValueError('no supported files were selected')
        if _session_size(state) > MAX_SESSION_BYTES:
            shutil.rmtree(role_dir, ignore_errors=True)
            return jsonify({'error': 'session upload storage limit exceeded'}), 413

        warning = None
        if role in {'coast', 'land'}:
            shapefiles = [p for p in saved if p.suffix.lower() == '.shp']
            if len(shapefiles) != 1:
                raise ValueError('select exactly one .shp file and its sidecars')
            shp_stem = shapefiles[0].stem.lower()
            mismatched = [p.name for p in saved if _shapefile_base_name(p) != shp_stem]
            if mismatched:
                raise ValueError(
                    'all shapefile sidecars must have the same base name: '
                    + ', '.join(mismatched)
                )
            path = _validate_loose_shapefile(role_dir)
            if not any(p.suffix.lower() == '.prj' for p in saved):
                warning = 'No .prj selected; assuming the coordinates are WGS84.'
        else:
            raw_path = saved[0]
            path = _validate_uploaded_file(role, raw_path, state)

        filename = path.name
        state.files[role] = path
        state.original_names[role] = filename
        resp: dict = {
            'role': role,
            'filename': filename,
            'path': str(path.relative_to(state.root)),
            'bytes': sum(p.stat().st_size for p in saved),
            'files': [p.name for p in saved],
        }
        if warning:
            resp['warning'] = warning
        return jsonify(resp)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 400


@app.server.route('/api/example', methods=['POST'])
def _r_example():
    """Load bundled example_data/ files into the current session as if uploaded."""
    import traceback
    try:
        state = _get_session()
        if not EXAMPLE_DATA.exists():
            return jsonify({'error': 'example_data/ directory not found in this deployment'}), 404

        loaded: dict[str, dict] = {}

        for role, spec in ROLE_SPECS.items():
            src_dir = EXAMPLE_DATA / spec['example_dir']
            if not src_dir.exists():
                return jsonify({'error': f'example_data/{spec["example_dir"]} not found'}), 404

            if role == 'ais':
                # Single CSV file
                csv_files = sorted(src_dir.glob('*.csv'))
                if not csv_files:
                    return jsonify({'error': 'No AIS CSV in example_data/ais/'}), 404
                src = csv_files[0]
                role_dir = state.root / 'uploads' / role
                shutil.rmtree(role_dir, ignore_errors=True)
                role_dir.mkdir(parents=True, exist_ok=True)
                dest = role_dir / src.name
                shutil.copy2(src, dest)
                path = _validate_uploaded_file(role, dest, state)
                state.files[role] = path
                state.original_names[role] = src.name
                loaded[role] = {
                    'path': str(path.relative_to(state.root)),
                    'filename': src.name,
                }

            elif role in ('coast', 'land'):
                # Loose shapefile bundle — copy all files to <root>/<role>/
                extracted = state.root / role
                shutil.rmtree(extracted, ignore_errors=True)
                extracted.mkdir(parents=True, exist_ok=True)
                for f in src_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, extracted / f.name)
                path = _validate_loose_shapefile(extracted)
                state.files[role] = path
                state.original_names[role] = path.name
                loaded[role] = {
                    'path': str(path.relative_to(state.root)),
                    'filename': path.name,
                }

            elif role == 'bathy':
                # .dfsu or .mesh file
                bathy_files = sorted(
                    f for f in src_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in {'.dfsu', '.mesh'}
                )
                if not bathy_files:
                    return jsonify({'error': 'No bathymetry file in example_data/bathymetry/'}), 404
                src = bathy_files[0]
                role_dir = state.root / 'uploads' / role
                shutil.rmtree(role_dir, ignore_errors=True)
                role_dir.mkdir(parents=True, exist_ok=True)
                dest = role_dir / src.name
                shutil.copy2(src, dest)
                path = _validate_uploaded_file(role, dest, state)
                state.files[role] = path
                state.original_names[role] = src.name
                loaded[role] = {
                    'path': str(path.relative_to(state.root)),
                    'filename': src.name,
                }

            elif role == 'tide':
                # .dfs0 file
                tide_files = sorted(src_dir.glob('*.dfs0'))
                if not tide_files:
                    # Tide is optional — skip silently
                    continue
                src = tide_files[0]
                role_dir = state.root / 'uploads' / role
                shutil.rmtree(role_dir, ignore_errors=True)
                role_dir.mkdir(parents=True, exist_ok=True)
                dest = role_dir / src.name
                shutil.copy2(src, dest)
                # Don't call _validate_uploaded_file for tide here (it calls _preview_tide which
                # we'll call anyway to get items). Just register the path directly.
                state.files[role] = dest
                state.original_names[role] = src.name
                loaded[role] = {
                    'path': str(dest.relative_to(state.root)),
                    'filename': src.name,
                }

        if _session_size(state) > MAX_SESSION_BYTES:
            return jsonify({'error': 'session storage limit exceeded by example data'}), 413

        return jsonify({'roles': loaded})

    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 400


def _bytes_response(b: bytes) -> Response:
    return Response(b, mimetype='application/vnd.apache.arrow.stream',
                    headers={'Cache-Control': 'no-store'})


@app.server.route('/api/vessels.arrow')
def _r_vessels():
    return _bytes_response(_get_session().ipc_vessels)


@app.server.route('/api/waves.arrow')
def _r_waves():
    return _bytes_response(_get_session().ipc_waves)


@app.server.route('/api/wave_animation.arrow')
def _r_wave_animation():
    return _bytes_response(_get_session().ipc_animation_rays)


@app.server.route('/api/track_coords.arrow')
def _r_track_coords():
    return _bytes_response(_get_session().ipc_track_coords)


@app.server.route('/api/track_meta.arrow')
def _r_track_meta():
    return _bytes_response(_get_session().ipc_track_meta)


@app.server.route('/api/track_offsets.arrow')
def _r_track_offsets():
    return _bytes_response(_get_session().ipc_track_offsets)


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
    try:
        state = _get_session()
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400
    ref_df = state.df_vessels if len(state.df_vessels) > 0 else None
    if ref_df is None:
        return jsonify({'error': 'no track data — run Filter first'}), 400
    try:
        result = compute_similar_tracks(ref_df, int(seed_mmsi), int(seed_seg), buffer_m, min_cov)
        return jsonify({'mmsi_segs': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.server.route('/api/raster.png')
def _r_raster():
    return Response(_get_session().png_bytes, mimetype='image/png')


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
    if DATA_ROOT.is_dir():
        for p in sorted(DATA_ROOT.glob('*/output')):
            if p.is_dir():
                candidates.append(_data_path_value(p))
    return candidates


def _load_results(directory: str) -> dict:
    """Load pre-computed vessel/wave results from *directory* and rebuild IPC caches.

    Resolution order per asset:
      Tracks → vessels.parquet → *_03_vessel.csv → *_01_filtered.csv
      Waves  → waves.parquet   → *_04_wave_impact.csv → shore_impact.csv
    """
    global df_vessels, df_waves, IPC_VESSELS, IPC_WAVES
    global IPC_TRACK_COORDS, IPC_TRACK_META, IPC_TRACK_OFFSETS, PNG_BYTES, seg_meta

    p = _safe_app_path(directory)
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
    return jsonify({'error': 'loading server-side result folders is disabled for temporary sessions'}), 410


# ---------------------------------------------------------------------------
# Export full or filtered tracks/waves and session inputs → downloadable rerun ZIP
# ---------------------------------------------------------------------------
_FORBIDDEN_NAME_CHARS = ('/', '\\', '..', ':', '\0')


def _export_filtered(state: SessionState, body: dict) -> Path:
    """Build a rerun-ready ZIP of the full session or its visible filtered slice."""
    dest_name = (body.get('dest_name') or '').strip()
    if not dest_name:
        dest_name = 'aiswakepy_export'
    if any(c in dest_name for c in _FORBIDDEN_NAME_CHARS):
        raise ValueError(f'invalid characters in folder name: {dest_name!r}')
    filtered = bool(body.get('filtered'))
    seg_keys = body.get('seg_keys') or []
    seg_key_set = {(int(m), int(s)) for m, s in seg_keys}
    wave_idxs = body.get('wave_idxs')  # may be None or list[int]
    sel_ais = body.get('sel_ais') or ''

    export_root = state.root / 'exports' / f'{dest_name}_{int(time.time())}'
    if export_root.exists():
        shutil.rmtree(export_root)
    out_dir = export_root / 'output'
    for sub in ('ais', 'coastline', 'land', 'bathymetry', 'tide', 'output'):
        (export_root / sub).mkdir(parents=True, exist_ok=True)

    df_f = state.last_results.get('df_filtered')
    if df_f is None or len(df_f) == 0:
        raise RuntimeError('no filtered AIS available — run Calculate Waves first')
    ais_source = state.files.get('ais')
    if not filtered and ais_source and ais_source.is_file():
        ais_name = state.original_names.get('ais') or ais_source.name
        shutil.copy2(ais_source, export_root / 'ais' / ais_name)
        ais_relpath = f'ais/{ais_name}'
    else:
        keys = list(zip(df_f['mmsi'].astype(int), df_f['segment_id'].astype(int)))
        ais_subset = (
            df_f.loc[pd.Series([k in seg_key_set for k in keys], index=df_f.index)].copy()
            if filtered else df_f.copy()
        )
        ais_cols = ['mmsi', 'width', 'length', 'draught', 'obstime',
                    'longitude', 'latitude', 'sog', 'cog', 'typecargo']
        ais_stem = Path(state.original_names.get('ais') or sel_ais or 'ais').stem
        ais_name = f'{ais_stem}.csv'
        ais_subset[[c for c in ais_cols if c in ais_subset.columns]].to_csv(
            export_root / 'ais' / ais_name, index=False)
        ais_relpath = f'ais/{ais_name}'

    df_v = state.df_vessels
    if filtered and len(df_v) > 0:
        tracks_mask = pd.Series(
            [k in seg_key_set for k in zip(
                df_v['mmsi'].astype(int), df_v['segment_id'].astype(int))],
            index=df_v.index,
        )
        df_tracks_out = df_v.loc[tracks_mask].reset_index(drop=True)
    else:
        df_tracks_out = df_v.reset_index(drop=True)
    df_tracks_out.to_parquet(out_dir / 'vessels.parquet', index=False)

    df_w = state.df_waves
    if filtered and len(df_w) > 0:
        if wave_idxs is not None:
            idxs = [int(i) for i in wave_idxs if 0 <= int(i) < len(df_w)]
            df_waves_out = df_w.iloc[idxs].reset_index(drop=True)
        else:
            wmask = pd.Series(
                [(int(m), int(s)) in seg_key_set for m, s in zip(df_w['MMSI'], df_w['segment_id'])],
                index=df_w.index,
            )
            df_waves_out = df_w.loc[wmask].reset_index(drop=True)
    else:
        df_waves_out = df_w.reset_index(drop=True)

    if not filtered:
        session_output = state.root / 'output'
        if session_output.is_dir():
            for path in session_output.iterdir():
                if path.is_file():
                    shutil.copy2(path, out_dir / path.name)
    df_waves_out.to_parquet(out_dir / 'waves.parquet', index=False)
    _write_wave_track_link(df_waves_out, out_dir)

    for role, sub in [('coast', 'coastline'), ('land', 'land'), ('bathy', 'bathymetry'), ('tide', 'tide')]:
        src = state.files.get(role)
        if src and src.exists():
            if role in {'coast', 'land'}:
                for f in src.parent.iterdir():
                    if f.is_file():
                        shutil.copy2(f, export_root / sub / f.name)
            else:
                shutil.copy2(src, export_root / sub / src.name)

    cfg_snap = json.loads(json.dumps(state.pipeline.get('cfg') or {}))
    if cfg_snap:
        cfg_snap.setdefault('output', {})['directory'] = 'output'
        cfg_snap['ais']['raw_csv'] = ais_relpath
        cfg_snap['ais']['land_shp'] = next(
            (f'land/{p.name}' for p in (export_root / 'land').glob('*.shp')), '')
        cfg_snap['coastline']['shapefile'] = next(
            (f'coastline/{p.name}' for p in (export_root / 'coastline').glob('*.shp')), '')
        bathy_files = list((export_root / 'bathymetry').iterdir())
        if bathy_files:
            cfg_snap['bathymetry']['source'] = f'bathymetry/{bathy_files[0].name}'
        tide_files = list((export_root / 'tide').iterdir())
        if tide_files:
            cfg_snap['bathymetry']['tide_dfs0'] = f'tide/{tide_files[0].name}'
        (export_root / 'config.json').write_text(json.dumps(cfg_snap, indent=2), encoding='utf-8')
        if filtered and len(df_waves_out) > 0:
            _generate_report_plots(
                state.pipeline.get('cfg') or {},
                df_tracks_out,
                df_waves_out,
                out_dir=out_dir,
            )

    zip_path = state.root / 'exports' / f'{dest_name}.zip'
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path in export_root.rglob('*'):
            if path.is_file():
                zf.write(path, path.relative_to(export_root))
    shutil.rmtree(export_root, ignore_errors=True)
    return zip_path


@app.server.route('/api/export/filtered', methods=['POST'])
def _r_export_filtered():
    body = request.get_json(force=True, silent=True) or {}
    try:
        state = _get_session()
        zip_path = _export_filtered(state, body)
        token = uuid.uuid4().hex
        with state.lock:
            state.downloads[token] = zip_path
        return jsonify({
            'filename': zip_path.name,
            'download_url': (
                f'/api/export/download/{token}?session_id={state.session_id}'
            ),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.server.route('/api/export/download/<token>')
def _r_download_export(token: str):
    try:
        state = _get_session()
        with state.lock:
            zip_path = state.downloads.pop(token, None)
        if zip_path is None:
            return jsonify({'error': 'download expired or already used'}), 404
        zip_path = zip_path.resolve()
        try:
            zip_path.relative_to(state.root.resolve())
        except ValueError:
            return jsonify({'error': 'invalid download path'}), 400
        if not zip_path.is_file():
            return jsonify({'error': 'download file not found'}), 404
        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_path.name,
            max_age=0,
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.server.route('/api/preview/ais.arrow')
def _r_preview_ais():
    try:
        state = _get_session()
        p = _session_path(state, request.args.get('path', ''))
        return _bytes_response(_preview_ais_arrow(p))
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.server.route('/api/preview/ais.bbox')
def _r_preview_ais_bbox():
    try:
        state = _get_session()
        p = _session_path(state, request.args.get('path', ''))
        return jsonify(_preview_ais_bbox(p))
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.server.route('/api/preview/coast.geojson')
def _r_preview_coast():
    try:
        state = _get_session()
        p = _session_path(state, request.args.get('path', ''))
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
        state = _get_session()
        p = _session_path(state, request.args.get('path', ''))
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
        state = _get_session()
        p = _session_path(state, request.args.get('path', ''))
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
    try:
        state = _get_session()
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400
    with state.lock:
        s = dict(state.pipeline)
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
        # load-results-status kept as hidden dummy — still used by the load-results
        # clientside callback (kept for future "resume from zip" feature).
        html.Span('', id='load-results-status', style={'display': 'none'}),
        html.Button('Run example', id='btn-run-example', n_clicks=0, disabled=True,
                    title='Load bundled example data (Singapore, JI channel)'),
        html.Span('', id='upload-status',
                  style={'fontSize': '11px', 'color': '#556', 'whiteSpace': 'nowrap',
                         'overflow': 'hidden', 'textOverflow': 'ellipsis',
                         'maxWidth': '260px', 'flexShrink': '1'}),
        html.Div('AISWAKEPY_PUBLIC', id='banner-title'),
        html.Div([
            html.Span(id='ais-time-range', style={'color': '#558', 'marginRight': '4px'}),
            html.Span(id='cnt-vessels', children='vessels 0'),
            ' | ', html.Span(id='cnt-segs',    children='segments 0'),
            ' | ', html.Span(id='cnt-waves',   children='waves 0'),
            ' | ', html.Span(id='status', children='loading...'),
            ' | ', html.Span(id='click-info', style={'fontWeight': 'bold'}),
        ], id='banner-meta'),
        html.Button('▶', id='btn-animation', disabled=True,
                    title='Ctrl+click a track, track point, or wave to select an animation',
                    style={'display': 'none'}),
    ], id='status-banner'),

    html.Div([

        html.Div([
            html.Div(id='upload-ais-host'),
            html.Div(id='pv-ais-info', className='preview-info'),
            html.Div(id='upload-coast-host'),
            html.Div(id='pv-coast-info', className='preview-info'),
            html.Div(id='upload-land-host'),
            html.Div(id='pv-land-info', className='preview-info'),
            html.Div(id='upload-bathy-host'),
            html.Div(id='pv-bathy-info', className='preview-info'),
            html.Div(id='upload-tide-host'),
        ], id='upload-panel'),

        # ---- Calculate Waves button ----
        html.Div([
            html.Button('Calculate Waves', id='btn-waves', n_clicks=0, disabled=True,
                        title='Run AIS filter + interpolate + vessel params '
                              '+ wave impact (requires AIS, coastline, land, bathymetry)'),
        ], className='row-buttons'),

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
                html.Button('Draw polygon on the map', id='btn-wavebox', n_clicks=0,
                            title='Draw a polygon on the map — keeps only waves landing '
                                  'inside, plus the tracks that produced them'),
            ], className='row-buttons'),
            html.Div([
                html.Button('Reset', id='btn-fil-clear', n_clicks=0),
                html.Button('Invert', id='btn-fil-invert', n_clicks=0,
                            title='Invert the final combined filter selection'),
                html.Button('Export', id='btn-fil-export', n_clicks=0,
                            title='Download all inputs and results, or only visible results '
                                  'when a visualization filter is active'),
            ], className='row-buttons'),
            html.Div([
                html.Div('', id='fil-status',
                         style={'fontSize': '10px', 'color': '#556',
                                'minHeight': '14px', 'flex': '1', 'minWidth': '0'}),
                html.Div('', id='export-status',
                         style={'fontSize': '10px', 'color': '#556',
                                'minHeight': '14px', 'flex': '1', 'minWidth': '0',
                                'textAlign': 'right', 'whiteSpace': 'pre-wrap'}),
            ], style={'display': 'flex', 'alignItems': 'flex-start', 'gap': '8px',
                      'marginTop': '3px'}),

        ]),

        html.Hr(),
        html.Div('Progress', style={'fontWeight': 'bold'}),
        html.Pre(id='progress-log', children='(idle)'),
        html.Div(id='progress-elapsed-side',
                 style={'fontSize': '11px', 'color': '#666', 'marginTop': '4px'}),

    ], id='sidebar'),

    html.Div(id='deck-container'),
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
    dcc.Store(id='_session', data=None),
    dcc.Store(id='_log_scroll', data=0),
    dcc.Interval(id='boot', max_intervals=1, interval=200),
    dcc.Interval(id='poll', interval=400, disabled=True),
    dcc.Store(id='_init'),
    dcc.Store(id='_wave_version', data=0),
    dcc.Store(id='_track_version', data=0),
    dcc.Store(id='_ais_import', data={'path': None, 'nonce': 0}),
    # Uploaded-files store: maps role → {path, filename}
    # This is the client-side source-of-truth for what has been uploaded.
    dcc.Store(id='_uploaded_files', data={}),
    # Preview state Stores: {visible, path}
    dcc.Store(id='_pv_ais',   data={'visible': False, 'path': None}),
    dcc.Store(id='_pv_bathy', data={'visible': False, 'path': None}),
    dcc.Store(id='_pv_coast', data={'visible': False, 'path': None}),
    dcc.Store(id='_pv_land',  data={'visible': False, 'path': None}),
    dcc.Store(id='_filter_structural', data={'mmsi': None, 'seg_ids': [], 'types': [], 'nonce': 0}),
    dcc.Store(id='_load_result'),
    dcc.Store(id='_wave_n', data=0),
    dcc.Store(id='_any_filter_active', data=False),
])


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


# ---------------------------------------------------------------------------
# Server-side callbacks: run buttons + polling
# ---------------------------------------------------------------------------
def _build_config(state: SessionState, ais, land, bathy, coast, tide,
                  min_speed=0.0, traj_gap=180.0, interp='linear', interp_interval=30.0,
                  cb_method='L_Le', max_prop=2000.0, max_sog=12.0,
                  max_velocity=36.0, max_accel=10.0, max_dw=1.0,
                  low_sog=1.0, vel_ratio=2.0, spd_ratio=0.5,
                  waterline=0.8, formula='kriebel', gravity=9.78,
                  max_bl=0.3, min_froude=0.1, max_froude=0.5, max_bf=0.4,
                  wake_cutoff=0.01) -> dict:
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
        'output': {'directory': str(state.root / 'output'),
                   'save_stage_csv': True},
    }
    cfg['ais']['raw_csv'] = str(_session_path(state, ais)) if ais else ais
    cfg['ais']['land_shp'] = str(_session_path(state, land)) if land else land
    cfg['bathymetry']['source'] = str(_session_path(state, bathy)) if bathy else 'placeholder.mesh'
    cfg['coastline']['shapefile'] = str(_session_path(state, coast)) if coast else coast
    if tide:
        cfg['bathymetry']['tide_dfs0'] = str(_session_path(state, tide))
    return cfg


def _kick(state: SessionState, config_dict, stages, label):
    global _active_pipeline_session
    with _pipeline_lock:
        if _active_pipeline_session is not None:
            return False
        _active_pipeline_session = state.session_id
    with state.lock:
        state.pipeline.update({
            'running': True, 'log': [], 'live': '',
            'started_at': time.time(),
            'finished_at': None, 'error': None,
        })
    threading.Thread(target=_pipeline_thread,
                     args=(state, config_dict, stages, label), daemon=True).start()
    return True


@app.callback(
    Output('poll', 'disabled', allow_duplicate=True),
    Output('btn-waves',  'disabled', allow_duplicate=True),
    Output('progress-log', 'children', allow_duplicate=True),
    Input('btn-waves', 'n_clicks'),
    State('_uploaded_files', 'data'),
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
    State('_session', 'data'),
    prevent_initial_call=True,
)
def kick_waves(n, uploaded_files,
               min_speed, traj_gap, interp, interp_interval,
               cb_method, max_prop, max_sog,
               max_velocity, max_accel, max_dw, low_sog,
               vel_ratio, spd_ratio, waterline, formula,
               gravity, max_bl, min_froude, max_froude, max_bf, wake_cutoff,
               session_data):
    if not n:
        return no_update, no_update, no_update
    try:
        state = _get_session((session_data or {}).get('session_id'))
    except Exception as exc:
        return no_update, no_update, f'⚠ {exc}'
    uf = uploaded_files or {}
    ais   = (uf.get('ais')   or {}).get('path')
    land  = (uf.get('land')  or {}).get('path')
    coast = (uf.get('coast') or {}).get('path')
    bathy = (uf.get('bathy') or {}).get('path')
    tide  = (uf.get('tide')  or {}).get('path')
    missing = []
    if not ais:   missing.append('AIS data file')
    if not land:  missing.append('Land mask shapefile')
    if not coast: missing.append('Coastline shapefile')
    if not bathy: missing.append('Bathymetry file (required for depth check)')
    if missing:
        warn = '⚠ Cannot calculate waves — missing required inputs:\n  • ' + '\n  • '.join(missing)
        return no_update, no_update, warn
    cfg = _build_config(state, ais, land, bathy, coast, tide,
                        min_speed, traj_gap, interp, interp_interval,
                        cb_method, max_prop, max_sog,
                        max_velocity, max_accel, max_dw, low_sog,
                        vel_ratio, spd_ratio, waterline, formula,
                        gravity, max_bl, min_froude, max_froude, max_bf, wake_cutoff)
    if _kick(state, cfg, ['filter', 'vessel', 'wave_impact'], 'waves'):
        return False, True, no_update
    return no_update, False, 'Server is busy calculating another session. Try again later.'


# btn-waves enabled when required inputs are present in _uploaded_files. Tide is optional.
app.clientside_callback(
    "function(uf){ uf=uf||{}; return !(uf.ais&&uf.coast&&uf.land&&uf.bathy); }",
    Output('btn-waves', 'disabled', allow_duplicate=True),
    Input('_uploaded_files', 'data'),
    prevent_initial_call='initial_duplicate',
)

# When _ais_import fires, make AIS layer visible in _pv_ais store.
app.clientside_callback(
    "function(s){ if(!s||!s.path) return window.dash_clientside.no_update; "
    "return {visible:true, path:s.path}; }",
    Output('_pv_ais', 'data', allow_duplicate=True),
    Input('_ais_import', 'data'),
    prevent_initial_call=True,
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
    Input('poll', 'n_intervals'),
    State('_wave_version', 'data'), State('_track_version', 'data'),
    State('_session', 'data'),
    prevent_initial_call=True,
)
def tick(_, prev_wave_v, prev_track_v, session_data):
    try:
        state = _get_session((session_data or {}).get('session_id'))
    except Exception as exc:
        return (
            f'ERROR: {exc}', '', True, True,
            prev_wave_v, prev_track_v,
            no_update, no_update, no_update, no_update,
        )
    with state.lock:
        s = dict(state.pipeline)
    log_lines = list(s['log'][-300:])
    if s.get('live'):
        log_lines.append(s['live'])  # in-progress spinner line, replaced each tick
    log_text = '\n'.join(log_lines) or '(no output yet)'
    elapsed = ''
    counts = (f'waves {len(state.df_waves):,}', f'segments {len(state.seg_meta):,}',
              f'vessels {len(state.df_vessels):,}')
    if s['error']:
        return (
            f"{log_text}\n\nERROR: {s['error']}",
            elapsed, True, False,
            prev_wave_v, prev_track_v, *counts, no_update,
        )
    if s['running']:
        return (
            log_text, elapsed, False, True,
            prev_wave_v, prev_track_v, no_update, no_update, no_update, no_update,
        )
    # Finished — push fresh versions.
    return (
        log_text, elapsed, True, False,
        s['wave_version'], s['track_version'], *counts, _ais_time_range_str(state),
    )


# ---------------------------------------------------------------------------
# Run example: load bundled example_data/ into the session.
# ---------------------------------------------------------------------------
app.clientside_callback(
    "function(session){ return !(session && session.session_id); }",
    Output('btn-run-example', 'disabled'),
    Input('_session', 'data'),
    prevent_initial_call=False,
)


app.clientside_callback(
    r"""
    async function(n, session) {
        const nu = window.dash_clientside.no_update;
        if (!n) return nu;
        const uploadStatus = document.getElementById('upload-status');
        const setStatus = (msg) => { if (uploadStatus) uploadStatus.textContent = msg; };
        const sessionId = session && session.session_id;
        if (!sessionId) {
            setStatus('Session is still initializing. Please wait a moment.');
            return nu;
        }
        setStatus('Loading example data...');
        try {
            const resp = await fetch(
                '/api/example?session_id=' + encodeURIComponent(sessionId),
                {method: 'POST', headers: {'X-Session-ID': sessionId}},
            );
            const j = await resp.json();
            if (!resp.ok || j.error) throw new Error(j.error || resp.statusText);
            // Populate the window mirror and the Dash store.
            window.__uploaded = {};
            for (const [role, info] of Object.entries(j.roles)) {
                window.__uploaded[role] = info;
                const button = document.getElementById(`native-upload-${role}-button`);
                if (button) {
                    button.textContent = info.filename;
                    button.title = (info.files || [info.filename]).join('\n');
                }
                const rowStatus = document.getElementById(`native-upload-${role}-status`);
                if (rowStatus) rowStatus.textContent = info.warning || '';
                const previewInput = document.getElementById(`native-preview-${role}`);
                if (previewInput) previewInput.checked = true;
            }
            if (window.dash_clientside?.set_props) {
                window.dash_clientside.set_props('_uploaded_files',
                    {data: Object.assign({}, window.__uploaded)});
                const ais = j.roles.ais;
                if (ais) {
                    window.dash_clientside.set_props('_ais_import',
                        {data: {path: ais.path, nonce: Date.now()}});
                }
                const previewStores = {
                    ais: '_pv_ais', coast: '_pv_coast',
                    land: '_pv_land', bathy: '_pv_bathy',
                };
                for (const [role, storeId] of Object.entries(previewStores)) {
                    const info = j.roles[role];
                    if (info) {
                        window.dash_clientside.set_props(storeId,
                            {data: {visible: true, path: info.path}});
                    }
                }
            }
            setStatus('Example loaded — click Calculate Waves to run the pipeline.');
        } catch (e) {
            setStatus('Error loading example: ' + e.message);
        }
        return nu;
    }
    """,
    Output('upload-status', 'children', allow_duplicate=True),
    Input('btn-run-example', 'n_clicks'),
    State('_session', 'data'),
    prevent_initial_call=True,
)


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
    State('_session', 'data'),
    prevent_initial_call=False,
)
def _populate_filter_options(_, session_data):
    mmsi_opts = []
    try:
        state = _get_session((session_data or {}).get('session_id'))
        if state.seg_meta:
            mmsis = sorted(set(s['mmsi'] for s in state.seg_meta))
            mmsi_opts = [{'label': str(m), 'value': m} for m in mmsis]
    except Exception:
        pass
    type_opts = [{'label': label, 'value': cat} for cat, label in _VESSEL_CATEGORIES]
    return mmsi_opts, type_opts


@app.callback(
    Output('fil-segs', 'options'),
    Output('fil-segs', 'value'),
    Input('fil-mmsi', 'value'),
    State('_session', 'data'),
    prevent_initial_call=False,
)
def _populate_seg_options(mmsi, session_data):
    if mmsi is None:
        return [], None
    try:
        state = _get_session((session_data or {}).get('session_id'))
    except Exception:
        return [], None
    segs = sorted(s['segment_id'] for s in state.seg_meta if s['mmsi'] == int(mmsi))
    return [{'label': f'seg {s}', 'value': s} for s in segs], None


@app.callback(
    Output('_filter_structural', 'data', allow_duplicate=True),
    Output('fil-type', 'value', allow_duplicate=True),
    Input('btn-fil-clear', 'n_clicks'),
    State('_filter_structural', 'data'),
    prevent_initial_call=True,
)
def _clear_track_filter(_, prev):
    nonce = ((prev or {}).get('nonce', 0) + 1)
    return {'mmsi': None, 'seg_ids': [], 'types': [], 'nonce': nonce, '_clear': True}, None


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
async function(n) {
    if (!n || window.__deck_initialized) return window.dash_clientside.no_update;
    const container = document.getElementById('deck-container');
    if (!container) return window.dash_clientside.no_update;
    if (typeof deck === 'undefined') {
        document.getElementById('status').textContent = 'waiting for deck.gl...';
        setTimeout(() => { window.__deck_initialized = false; }, 100);
        return window.dash_clientside.no_update;
    }
    window.__deck_initialized = true;
    if (!window.__sessionId) {
        const sessionResp = await fetch('/api/session', {method: 'POST'});
        const session = await sessionResp.json();
        if (!sessionResp.ok || session.error) {
            document.getElementById('status').textContent = 'ERROR: ' + (session.error || 'session failed');
            return window.dash_clientside.no_update;
        }
        window.__sessionId = session.session_id;
        if (window.dash_clientside?.set_props) {
            window.dash_clientside.set_props('_session', {data: session});
        }
        const nativeFetch = window.fetch.bind(window);
        window.fetch = function(input, init) {
            init = init || {};
            let url = (typeof input === 'string') ? input : input.url;
            if (url && url.startsWith('/api/') && !url.startsWith('/api/session')) {
                const sep = url.indexOf('?') >= 0 ? '&' : '?';
                url = url + sep + 'session_id=' + encodeURIComponent(window.__sessionId);
                const headers = new Headers(init.headers || {});
                headers.set('X-Session-ID', window.__sessionId);
                init = Object.assign({}, init, {headers});
                return nativeFetch(url, init);
            }
            return nativeFetch(input, init);
        };
    }
    const uploadStatus = document.getElementById('upload-status');
    const pulseElement = (el, className, duration = 1200) => {
        if (!el) return;
        if (el.__pulseTimer) window.clearTimeout(el.__pulseTimer);
        el.classList.remove(className);
        void el.offsetWidth;
        el.classList.add(className);
        el.__pulseTimer = window.setTimeout(() => {
            el.classList.remove(className);
            el.__pulseTimer = null;
        }, duration);
    };
    window.__highlightCtrlHint = () =>
        pulseElement(document.getElementById('ctrl-hint'), 'ctrl-highlight', 2800);
    for (const id of ['upload-status', 'export-status', 'fil-status']) {
        const el = document.getElementById(id);
        if (!el || el.dataset.highlightReady) continue;
        el.dataset.highlightReady = '1';
        new MutationObserver(() => {
            if ((el.textContent || '').trim()) pulseElement(el, 'status-highlight');
        }).observe(el, {childList: true, characterData: true, subtree: true});
    }
    // Role specs: must match ROLE_SPECS in dash_app.py (backend canonical source).
    const labels = {
        ais:   {host: 'upload-ais-host',   label: 'AIS CSV', accept: '.csv',
                previewStore: '_pv_ais'},
        coast: {host: 'upload-coast-host', label: 'Coastline Shapefile',
                accept: '.shp,.shx,.dbf,.prj,.xml,.sbn,.sbx,.cpg',
                multiple: true, previewStore: '_pv_coast'},
        land:  {host: 'upload-land-host',  label: 'Land Mask Shapefile',
                accept: '.shp,.shx,.dbf,.prj,.xml,.sbn,.sbx,.cpg',
                multiple: true, previewStore: '_pv_land'},
        bathy: {host: 'upload-bathy-host', label: 'Bathymetry',
                accept: '.mesh,.dfsu', previewStore: '_pv_bathy'},
        tide:  {host: 'upload-tide-host',  label: 'Tide DFS0 (optional)', accept: '.dfs0'},
    };
    // Client-side mirror of _uploaded_files store (keyed by role).
    window.__uploaded = window.__uploaded || {};
    const setStatus = (msg) => { if (uploadStatus) uploadStatus.textContent = msg; };
    const makeUploadRow = (role, spec) => {
        const host = document.getElementById(spec.host);
        if (!host || host.dataset.ready) return;
        host.dataset.ready = '1';
        const multiple = spec.multiple ? ' multiple' : '';
        const preview = spec.previewStore
            ? `<label class="upload-preview"><input id="native-preview-${role}" type="checkbox"> preview</label>`
            : '';
        host.innerHTML =
            `<div class="upload-control">` +
            `<label class="upload-control-label">${spec.label}</label>` +
            `<div class="upload-control-row">` +
            `<button id="native-upload-${role}-button" type="button" class="upload-select-btn">` +
            `Choose ${spec.label}</button>${preview}</div>` +
            `<input id="native-upload-${role}" type="file" accept="${spec.accept}"${multiple} hidden>` +
            `<div id="native-upload-${role}-status" class="upload-row-status"></div>` +
            `</div>`;
        const input = document.getElementById(`native-upload-${role}`);
        const button = document.getElementById(`native-upload-${role}-button`);
        const previewInput = document.getElementById(`native-preview-${role}`);
        const rowStatus = document.getElementById(`native-upload-${role}-status`);
        button.addEventListener('click', () => input.click());
        if (previewInput) {
            previewInput.addEventListener('change', () => {
                const entry = window.__uploaded[role];
                if (!entry || !entry.path) {
                    previewInput.checked = false;
                    return;
                }
                if (window.dash_clientside?.set_props) {
                    window.dash_clientside.set_props(spec.previewStore, {
                        data: {visible: previewInput.checked, path: entry.path},
                    });
                }
            });
        }
        input.addEventListener('change', async () => {
            const files = Array.from(input.files || []);
            if (files.length === 0) return;
            button.textContent = 'Uploading...';
            button.title = '';
            rowStatus.textContent = '';
            setStatus('');
            const fd = new FormData();
            if (spec.multiple) {
                files.forEach(file => fd.append('files', file, file.name));
            } else {
                fd.append('file', files[0], files[0].name);
            }
            try {
                const resp = await fetch(`/api/upload/${role}`, {method: 'POST', body: fd});
                const j = await resp.json();
                if (!resp.ok || j.error) throw new Error(j.error || resp.statusText);
                button.textContent = j.filename;
                button.title = (j.files || [j.filename]).join('\n');
                rowStatus.textContent = j.warning || '';
                // Merge into the uploaded-files mirror and push to Dash store.
                const entry = {path: j.path, filename: j.filename};
                window.__uploaded[role] = entry;
                if (window.dash_clientside?.set_props) {
                    window.dash_clientside.set_props('_uploaded_files',
                        {data: Object.assign({}, window.__uploaded)});
                    if (spec.previewStore) {
                        previewInput.checked = true;
                        window.dash_clientside.set_props(spec.previewStore,
                            {data: {visible: true, path: j.path}});
                    }
                    if (role === 'ais') {
                        window.dash_clientside.set_props('_ais_import',
                            {data: {path: j.path, nonce: Date.now()}});
                    }
                }
            } catch (e) {
                button.textContent = `Choose ${spec.label}`;
                rowStatus.textContent = 'ERROR: ' + e.message;
                setStatus('Upload failed: ' + e.message);
            }
        });
    };
    Object.keys(labels).forEach(k => makeUploadRow(k, labels[k]));
    const exportButton = document.getElementById('btn-fil-export');
    const exportStatus = document.getElementById('export-status');
    window.__setExportBusy = (busy, message) => {
        if (exportButton) {
            exportButton.classList.toggle('export-busy', !!busy);
            exportButton.setAttribute('aria-busy', busy ? 'true' : 'false');
        }
        if (exportStatus && message != null) exportStatus.textContent = message;
    };
    if (exportButton && !exportButton.dataset.savePickerReady) {
        exportButton.dataset.savePickerReady = '1';
        exportButton.addEventListener('click', () => {
            const canPick = window.isSecureContext
                && typeof window.showSaveFilePicker === 'function';
            window.__setExportBusy(
                true,
                canPick
                    ? 'Choose a save location, then the export will be prepared...'
                    : 'Preparing export... The browser will use its download settings.',
            );
            if (!canPick) {
                window.__exportFileHandlePromise = null;
                return;
            }
            window.__exportFileHandlePromise = window.showSaveFilePicker({
                suggestedName: 'aiswakepy_export.zip',
                types: [{
                    description: 'ZIP archive',
                    accept: {'application/zip': ['.zip']},
                }],
            }).then(handle => ({handle})).catch(error => ({error}));
        });
    }
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
        let wSourceIdx = new Int32Array(0);
        let rMMSI = new BigInt64Array(0), rSegId = new Int32Array(0);
        let rSourcePos = new Float32Array(0), rEndPos = new Float32Array(0);
        let rSourceTime = new BigInt64Array(0), rDistance = new Float32Array(0);
        let rReached = new Uint8Array(0), rSide = (_i) => '';
        let rWakeDir = new Float32Array(0), rTheta = new Float32Array(0);
        let rSogMs = new Float32Array(0), rPhaseSpeed = new Float32Array(0);
        let rGroupSpeed = new Float32Array(0), rCuspAngle = new Float32Array(0);
        let rTransverseSpeed = new Float32Array(0);
        let rCuspDir = new Float32Array(0), rCuspEndPos = new Float32Array(0);
        let rCuspDistance = new Float32Array(0), rCuspReached = new Uint8Array(0);
        let raysBySegKey = new Map();
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
            wSourceIdx = Int32Array.from({length: wLon.length}, (_, i) => i);
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
                wSourceIdx = reorder(wSourceIdx);
                const _origSide = wSide, _origTime = wTime;
                wSide = (i) => _origSide(perm[i]);
                wTime = (i) => _origTime(perm[i]);
            }
            buildWaveSegMapping();
        }
        function rebuildAnimationRayArrays(rT) {
            const n = rT.numRows;
            const get = name => rT.getChild(name);
            rMMSI = get('MMSI') ? get('MMSI').toArray() : new BigInt64Array(0);
            rSegId = get('segment_id') ? get('segment_id').toArray() : new Int32Array(n);
            rSourceTime = get('SourceTime') ? get('SourceTime').toArray() : new BigInt64Array(n);
            rDistance = get('Distance_m') ? get('Distance_m').toArray() : new Float32Array(n);
            rReached = get('ReachedShore') ? get('ReachedShore').toArray() : new Uint8Array(n);
            rWakeDir = get('WakeDirection_deg') ? get('WakeDirection_deg').toArray() : new Float32Array(n);
            rTheta = get('Theta_deg') ? get('Theta_deg').toArray() : new Float32Array(n);
            rSogMs = get('SOGms') ? get('SOGms').toArray() : new Float32Array(n);
            rPhaseSpeed = get('PhaseSpeed_mps') ? get('PhaseSpeed_mps').toArray() : new Float32Array(n);
            rGroupSpeed = get('GroupSpeed_mps') ? get('GroupSpeed_mps').toArray() : new Float32Array(n);
            rCuspAngle = get('CuspAngle_deg') ? get('CuspAngle_deg').toArray() : new Float32Array(n);
            rTransverseSpeed = get('TransverseSpeed_mps') ? get('TransverseSpeed_mps').toArray() : new Float32Array(n);
            rCuspDir = get('CuspDirection_deg') ? get('CuspDirection_deg').toArray() : new Float32Array(n);
            rCuspDistance = get('CuspDistance_m') ? get('CuspDistance_m').toArray() : new Float32Array(n);
            rCuspReached = get('CuspReachedShore') ? get('CuspReachedShore').toArray() : new Uint8Array(n);
            const sideCol = get('Side');
            rSide = i => sideCol ? sideCol.get(i) : '';
            const srcLon = get('SourceLongitude')?.toArray() || new Float32Array(n);
            const srcLat = get('SourceLatitude')?.toArray() || new Float32Array(n);
            const endLon = get('EndLongitude')?.toArray() || new Float32Array(n);
            const endLat = get('EndLatitude')?.toArray() || new Float32Array(n);
            const cuspEndLon = get('CuspEndLongitude')?.toArray() || new Float32Array(n);
            const cuspEndLat = get('CuspEndLatitude')?.toArray() || new Float32Array(n);
            rSourcePos = new Float32Array(n * 2);
            rEndPos = new Float32Array(n * 2);
            rCuspEndPos = new Float32Array(n * 2);
            raysBySegKey = new Map();
            for (let i = 0; i < n; i++) {
                rSourcePos[i*2] = srcLon[i]; rSourcePos[i*2+1] = srcLat[i];
                rEndPos[i*2] = endLon[i]; rEndPos[i*2+1] = endLat[i];
                rCuspEndPos[i*2] = cuspEndLon[i]; rCuspEndPos[i*2+1] = cuspEndLat[i];
                const key = `${Number(rMMSI[i])}|${Number(rSegId[i])}`;
                if (!raysBySegKey.has(key)) raysBySegKey.set(key, []);
                raysBySegKey.get(key).push(i);
            }
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

        const animationButton = document.getElementById('btn-animation');
        const animation = new window.VesselWaveAnimationController({
            realTimeScale: 50,
            onChange: state => {
                if (animationButton) {
                    animationButton.style.display = state.selection ? '' : 'none';
                    animationButton.textContent = state.playing ? '⏸' : '▶';
                    animationButton.classList.toggle('playing', state.playing);
                }
                if (window.deckInstance && typeof window.__rebuild === 'function') {
                    window.__rebuild();
                }
            },
        });
        window.__animationController = animation;
        if (animationButton) animationButton.addEventListener('click', () => animation.toggle());
        const rayPhaseSpeed = ri => {
            const stored = Number(rPhaseSpeed[ri]);
            if (Number.isFinite(stored) && stored > 0) return stored;
            const sog = Number(rSogMs[ri]);
            const theta = Number(rTheta[ri]);
            if (Number.isFinite(sog) && sog > 0 && Number.isFinite(theta)) {
                const computed = sog * Math.cos(theta * Math.PI / 180);
                if (computed > 0) return computed;
            }
            const group = Number(rGroupSpeed[ri]);
            if (Number.isFinite(group) && group > 0) return group * 2;
            return 0;
        };
        const rayGroupSpeed = ri => {
            const stored = Number(rGroupSpeed[ri]);
            if (Number.isFinite(stored) && stored > 0) return stored;
            return 0.5 * rayPhaseSpeed(ri);
        };
        const rayCog = ri => {
            const wakeDir = Number(rWakeDir[ri]);
            const theta = Number(rTheta[ri]);
            if (!Number.isFinite(wakeDir) || !Number.isFinite(theta)) return wakeDir || 0;
            return rSide(ri) === 'port' ? wakeDir + theta : wakeDir - theta;
        };
        const rayCuspDirection = ri => {
            const stored = Number(rCuspDir[ri]);
            if (Number.isFinite(stored) && Math.abs(stored) > 1e-9) return stored;
            const cog = rayCog(ri);
            const angle = Number(rCuspAngle[ri]);
            if (Number.isFinite(cog) && Number.isFinite(angle)) {
                return rSide(ri) === 'port' ? cog + angle : cog - angle;
            }
            return cog;
        };
        // Direction the cusp front travels: the divergent-wave / wake direction
        // (COG +/- theta), which WakeDirection_deg already encodes.
        const rayMoveDirection = ri => {
            const stored = Number(rWakeDir[ri]);
            if (Number.isFinite(stored)) return stored;
            const cog = rayCog(ri);
            const theta = Number(rTheta[ri]);
            if (Number.isFinite(cog) && Number.isFinite(theta)) {
                return rSide(ri) === 'port' ? cog - theta : cog + theta;
            }
            return cog;
        };
        const selectAnimationSegment = (segIdx, waveIdx = null) => {
            if (segIdx == null || segIdx < 0 || segIdx >= tMMSI.length) {
                animation.clear();
                return;
            }
            const start = startIndices[segIdx], end = startIndices[segIdx + 1];
            const firstNs = Number(pointTime[start] || 0);
            const lastNs = Number(pointTime[Math.max(start, end - 1)] || firstNs);
            const trackDurationS = Math.max(1, (lastNs - firstNs) / 1e9);
            let loopDurationS = trackDurationS;
            const rayIdxs = raysBySegKey.get(`${Number(tMMSI[segIdx])}|${Number(tSeg[segIdx])}`) || [];
            for (const ri of rayIdxs) {
                const sourceOffsetS = Math.max(0, (Number(rSourceTime[ri]) - firstNs) / 1e9);
                const speed = rayPhaseSpeed(ri);
                const travelS = speed > 0 ? (Number(rDistance[ri]) || 0) / speed : 0;
                loopDurationS = Math.max(loopDurationS, sourceOffsetS + travelS);
                const cuspSpeed = rayGroupSpeed(ri);
                const cuspDist = Number(rCuspDistance[ri]) || Number(rDistance[ri]) || 0;
                const cuspTravelS = cuspSpeed > 0 ? cuspDist / cuspSpeed : 0;
                loopDurationS = Math.max(loopDurationS, sourceOffsetS + cuspTravelS);
                // Let the loop run until the crest front (group speed along Distance_m)
                // reaches the coastline / max distance, so crests finish propagating.
                const crestTravelS = cuspSpeed > 0 ? (Number(rDistance[ri]) || 0) / cuspSpeed : 0;
                loopDurationS = Math.max(loopDurationS, sourceOffsetS + crestTravelS);
            }
            animation.select({
                segIdx,
                mmsi: Number(tMMSI[segIdx]),
                segmentId: Number(tSeg[segIdx]),
                waveIdx,
                firstNs,
                trackDurationS,
                loopDurationS: Math.max(1, loopDurationS),
            });
        };

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
        window.__filterState = {
            mmsi: null, seg_ids: [], types: [], freehand: null,
            similar: null, waveBox: null, inverted: false,
        };
        window.__visibleSegIdxs = null; // null = show all; Set<segIdx> = filtered
        window.__baseVisibleSegIdxs = null; // combined filters before optional inversion
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
            const segVis = window.__baseVisibleSegIdxs;
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
            if (noBox && noTrackFilter && !fs.inverted) {
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
                let passes = !boxSet || boxSet.has(i);
                if (!noTrackFilter) {
                    const sIdx = waveToSegIdx ? waveToSegIdx[i] : -1;
                    passes = passes && sIdx >= 0 && segVis.has(sIdx);
                }
                if (fs.inverted) passes = !passes;
                if (!passes) continue;
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
                            fs.waveBox == null && !fs.inverted;
            if (N === 0 || allNull) {
                window.__visibleSegIdxs = null;
                window.__baseVisibleSegIdxs = null;
                rebuildFilteredArrays(null);
                rebuildFilteredWaveArrays();
                window.__rebuild();
                const stat = document.getElementById('fil-status');
                if (stat) stat.textContent = N > 0 ? `All ${N.toLocaleString()} tracks visible` : '';
                if (window.dash_clientside?.set_props) {
                    window.dash_clientside.set_props('_any_filter_active', {data: false});
                }
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
                window.__baseVisibleSegIdxs = fs.inverted
                    ? new Set(Array.from({length: N}, (_, i) => i))
                    : null;
            } else {
                sets.sort((a, b) => a.size - b.size);
                let result = sets[0];
                for (let k = 1; k < sets.length; k++) {
                    const next = new Set();
                    for (const v of result) { if (sets[k].has(v)) next.add(v); }
                    result = next;
                }
                window.__baseVisibleSegIdxs = result;
            }
            if (fs.inverted) {
                const result = new Set();
                for (let i = 0; i < N; i++) {
                    if (window.__baseVisibleSegIdxs === null ||
                            !window.__baseVisibleSegIdxs.has(i)) result.add(i);
                }
                window.__visibleSegIdxs = result;
            } else {
                window.__visibleSegIdxs = window.__baseVisibleSegIdxs;
            }
            rebuildFilteredArrays(
                window.__visibleSegIdxs === null ? null : Array.from(window.__visibleSegIdxs));
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
                const isActive = sets.length > 0 || fs.inverted;
                window.dash_clientside.set_props('_any_filter_active', {data: isActive});
            }
        };

        window.__invertFilters = () => {
            window.__filterState.inverted = !window.__filterState.inverted;
            window.__recomputeVisibility();
            return window.__filterState.inverted
                ? 'Filter selection inverted'
                : 'Filter inversion removed';
        };

        window.__applyStructuralFilter = (structural) => {
            if (!structural) return '';
            if (structural._clear) {
                // Full reset from the Reset button
                window.__filterState.mmsi    = null;
                window.__filterState.seg_ids = [];
                window.__filterState.types   = [];
                window.__filterState.freehand = null;
                window.__filterState.similar  = null;
                window.__filterState.waveBox  = null;
                window.__filterState.inverted = false;
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
                if (window.__polygonController) window.__polygonController.cancel();
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
            if (typeof window.__highlightCtrlHint === 'function') window.__highlightCtrlHint();
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

        // ---- Wave-arrival-area polygon mode ----
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
        const pointInPolygon = (x, y, points) => {
            let inside = false;
            for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
                const xi = points[i][0], yi = points[i][1];
                const xj = points[j][0], yj = points[j][1];
                const crosses = ((yi > y) !== (yj > y)) &&
                    (x < (xj - xi) * (y - yi) / ((yj - yi) || Number.EPSILON) + xi);
                if (crosses) inside = !inside;
            }
            return inside;
        };
        const applyWavePolygon = (polygon) => {
            const polygonWaveIdxs = [];
            const polygonSegIdxs = new Set();
            for (let i = 0; i < wMMSI.length; i++) {
                const lon = wPos[i*2], lat = wPos[i*2+1];
                if (pointInPolygon(lon, lat, polygon)) {
                    polygonWaveIdxs.push(i);
                    const si = waveToSegIdx ? waveToSegIdx[i] : -1;
                    if (si >= 0) polygonSegIdxs.add(si);
                }
            }
            if (polygonWaveIdxs.length === 0) return;
            window.__filterState.waveBox = {
                waveIdxs: new Set(polygonWaveIdxs),
                segIdxs: polygonSegIdxs,
            };
            window.__recomputeVisibility();
        };
        const polygonController = new window.AiswakePolygonController({
            container,
            canvas: getOrCreateWaveBoxCanvas(),
            button: document.getElementById('btn-wavebox'),
            getViewport: () => window.deckInstance.getViewports()[0],
            onComplete: applyWavePolygon,
            onStateChange: () => {
                if (window._hoveredWave !== null) {
                    window._hoveredWave = null;
                    hideTip();
                    if (window._pinnedWave === null &&
                            typeof window.__rebuild === 'function') {
                        window.__rebuild();
                    }
                }
                if (typeof window.__updateDeckCursor === 'function') {
                    window.__updateDeckCursor();
                }
            },
        });
        window.__polygonController = polygonController;
        window.__enterWaveBoxMode = () => {
            if (!window.__hasWaves || wMMSI.length === 0) return;
            polygonController.setCtrlHeld(window.__ctrlHeld);
            const armed = polygonController.arm();
            if (armed && typeof window.__highlightCtrlHint === 'function') {
                window.__highlightCtrlHint();
            }
        };
        window.__activateWaveBoxDraw = () => polygonController.activate();
        window.__cancelWaveBoxDraw = () => polygonController.cancel();
        window.__redrawWavePolygon = () => polygonController.redraw();
        if (window.__pendingWaveBoxClick) {
            window.__pendingWaveBoxClick = false;
            window.__enterWaveBoxMode();
        }

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
            return arr ? Array.from(arr, i => Number(wSourceIdx[i])) : null;
        };

        // ---- Reset all filters (used when waves are recalculated) ----
        window.__resetAllFilters = () => {
            window.__filterState.mmsi     = null;
            window.__filterState.seg_ids  = [];
            window.__filterState.types    = [];
            window.__filterState.freehand = null;
            window.__filterState.similar  = null;
            window.__filterState.waveBox  = null;
            window.__filterState.inverted = false;
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
            if (window.__polygonController) window.__polygonController.cancel();
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
            if (typeof window.__highlightCtrlHint === 'function') window.__highlightCtrlHint();
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

        const pointRowToSegIdx = row => {
            let lo = 0, hi = startIndices.length - 1;
            while (lo < hi - 1) {
                const mid = (lo + hi) >> 1;
                if (startIndices[mid] <= row) lo = mid;
                else hi = mid;
            }
            return lo;
        };
        const offsetMeters = (pos, eastM, northM) => {
            const latRad = (pos[1] || 0) * Math.PI / 180;
            const dLat = northM / 111111.0;
            const dLon = eastM / Math.max(1e-6, 111111.0 * Math.cos(latRad));
            return [pos[0] + dLon, pos[1] + dLat];
        };
        const toMeters = (pos, origin) => {
            const latRad = (origin[1] || 0) * Math.PI / 180;
            return [
                (pos[0] - origin[0]) * Math.max(1e-6, 111111.0 * Math.cos(latRad)),
                (pos[1] - origin[1]) * 111111.0,
            ];
        };
        const fromMeters = (xy, origin) => offsetMeters(origin, xy[0], xy[1]);
        const bearingVector = bearingDeg => {
            const rad = Number(bearingDeg) * Math.PI / 180;
            return [Math.sin(rad), Math.cos(rad)];
        };
        const add2 = (a, b) => [a[0] + b[0], a[1] + b[1]];
        const mul2 = (a, s) => [a[0] * s, a[1] * s];
        // Intersect infinite line p+t*d with q+s*e (2D); null if (near-)parallel.
        const lineIntersect = (p, d, q, e) => {
            const denom = d[0] * e[1] - d[1] * e[0];
            if (Math.abs(denom) < 1e-6) return null;
            const t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / denom;
            return [p[0] + d[0] * t, p[1] + d[1] * t];
        };
        // Compass bearing (deg, 0=N, CW) from lon/lat a to lon/lat b.
        const bearingDeg = (a, b) => {
            const latRad = (a[1] || 0) * Math.PI / 180;
            const east = (b[0] - a[0]) * Math.cos(latRad);
            const north = (b[1] - a[1]);
            return (Math.atan2(east, north) * 180 / Math.PI + 360) % 360;
        };
        const selectedAnimationGeometry = state => {
            const selection = state.selection;
            if (!selection) return null;
            const si = selection.segIdx;
            const start = startIndices[si], end = startIndices[si + 1];
            if (start == null || end == null || end <= start) return null;
            const firstNs = selection.firstNs != null ? Number(selection.firstNs) : Number(pointTime[start]);
            const lastNs = Number(pointTime[end - 1]);
            const spanNs = Math.max(1, lastNs - firstNs);
            const targetNs = firstNs + state.trackProgress * spanNs;
            let row = start;
            while (row < end - 2 && Number(pointTime[row + 1]) < targetNs) row++;
            const t0 = Number(pointTime[row]), t1 = Number(pointTime[Math.min(row + 1, end - 1)]);
            const f = t1 > t0 ? Math.max(0, Math.min(1, (targetNs - t0) / (t1 - t0))) : 0;
            const vessel = [
                cPos[row*2] + (cPos[(row+1)*2] - cPos[row*2]) * f,
                cPos[row*2+1] + (cPos[(row+1)*2+1] - cPos[row*2+1]) * f,
            ];
            const rayIdxs = raysBySegKey.get(
                `${selection.mmsi}|${selection.segmentId}`
            ) || [];
            const transverseCircles = [];
            const cuspSegments = [];
            const propagationRays = [];  // debug: source → current front point per crest
            const cuspJoints = [];   // intersection points, drawn in debug mode
            // Every cusp source collected per side (one per AIS track point, ungated)
            // so each crest's length can be derived from its track spacing.
            const cuspBySide = {port: [], stbd: []};
            const emittedTransverse = new Set();
            // Front positions (abs lon/lat) by ray, kept across frames for the live
            // movement bearing shown in debug mode.
            const cuspPrevFront = window.__cuspPrevFront instanceof Map
                ? window.__cuspPrevFront : new Map();
            const cuspNextFront = new Map();
            const stride = Math.max(1, Math.ceil(rayIdxs.length / 220));
            // The vessel sits at the origin in local metres (the live generation point
            // through which the newest crest passes).
            const vessel0 = [0, 0];
            // ---- Collect every source (no time gate) + transverse circles ----
            for (const ri of rayIdxs) {
                const source = [rSourcePos[ri*2], rSourcePos[ri*2+1]];
                const side = rSide(ri);
                cuspBySide[side].push({
                    ri,
                    srcM: toMeters(source, vessel),
                    srcTime: Number(rSourceTime[ri]),
                    moveDir: bearingVector(rayMoveDirection(ri)),
                    lineDir: bearingVector(rayCuspDirection(ri)),
                    lineOriDeg: Number(rayCuspDirection(ri)),
                    ca: Number(rCuspAngle[ri]),
                    groupSpeed: rayGroupSpeed(ri),
                    // Distance the front travels along its propagation ray (COG+/-theta)
                    // before hitting the coastline or the max calculation distance.
                    propDist: Number(rDistance[ri]) || 0,
                });
                if (ri % stride === 0) {
                    const key = `${Number(rSourceTime[ri])}|${source[0].toFixed(6)}|${source[1].toFixed(6)}`;
                    if (!emittedTransverse.has(key)) {
                        emittedTransverse.add(key);
                        const sourceOffsetS = Math.max(0, (Number(rSourceTime[ri]) - firstNs) / 1e9);
                        const phaseProgress = animation.frontProgress(
                            sourceOffsetS, rayPhaseSpeed(ri), rDistance[ri]);
                        // Expand circles at the same group speed (0.5*cos(theta)*V).
                        const radius = animation.transverseRadius(sourceOffsetS, rayGroupSpeed(ri));
                        if (radius > 0) {
                            transverseCircles.push({
                                position: source,
                                radius: Math.min(radius, rDistance[ri]),
                                age: phaseProgress,
                            });
                        }
                    }
                }
            }
            // ---- Build each side's crest arm with determined lengths ----
            const dist2 = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
            const dot2 = (a, b) => a[0]*b[0] + a[1]*b[1];
            const sub2 = (a, b) => [a[0] - b[0], a[1] - b[1]];
            for (const side of ['port', 'stbd']) {
                const all = cuspBySide[side];
                all.sort((a, b) => a.srcTime - b.srcTime);
                const N = all.length;
                if (!N) continue;
                // Newest source index k: the last source whose midpoint-from-previous
                // the vessel has passed (appears half a step before its source time).
                let k = 0;
                for (let i = 1; i < N; i++) {
                    if (targetNs >= (all[i-1].srcTime + all[i].srcTime) / 2) k = i; else break;
                }
                // Determined per-source crest geometry for active indices [0..k].
                const cr = [];
                for (let i = 0; i <= k; i++) {
                    const s = all[i];
                    // Propagation time keeps advancing after the vessel reaches the
                    // track end (simElapsedS runs on to loopDurationS, while targetNs
                    // freezes), so crests keep moving until they hit shore / the limit.
                    const elapsedS = Math.max(0, state.simElapsedS - (s.srcTime - firstNs) / 1e9);
                    const frontDist = s.groupSpeed * elapsedS;
                    // The crest keeps propagating while playing, and disappears only
                    // once its front reaches the coastline / max calculation distance.
                    if (s.propDist > 0 && frontDist >= s.propDist) continue;
                    const C = add2(s.srcM, mul2(s.moveDir, frontDist));   // front/division point
                    const dPrev = i > 0 ? dist2(s.srcM, all[i-1].srcM)
                        : (i+1 < N ? dist2(s.srcM, all[i+1].srcM) : 0);
                    const dNext = i+1 < N ? dist2(s.srcM, all[i+1].srcM)
                        : (i > 0 ? dist2(s.srcM, all[i-1].srcM) : 0);
                    const c = Math.cos((s.ca || 0) * Math.PI / 180);
                    const Lp = 0.5 * c * dPrev;   // prev-half length
                    const Ln = 0.5 * c * dNext;   // next-half length
                    // Orientation signed so +u points toward the next (newer) source.
                    const ref = i+1 < N ? sub2(all[i+1].srcM, s.srcM)
                        : sub2(s.srcM, all[Math.max(0, i-1)].srcM);
                    const u = mul2(s.lineDir, dot2(s.lineDir, ref) >= 0 ? 1 : -1);
                    cr.push({s, C, u, Lp, Ln,
                        back: add2(C, mul2(u, -Lp)),   // prev/outward end
                        fwd:  add2(C, mul2(u,  Ln))});  // next/vessel-ward end
                }
                const m = cr.length;
                if (!m) continue;
                // While the vessel is still moving, the newest crest is being
                // generated: its FRONT point is the live generation point at the
                // vessel, growing by the length generated through this point's
                // ownership interval (the line passes through the vessel even before
                // the front reaches the AIS source point). Once the vessel reaches the
                // track end it stops generating and propagates like the others.
                if (state.trackProgress < 1) {
                    const nw = cr[m-1];
                    const sk = all[k];
                    const tBefore = k > 0 ? (all[k-1].srcTime + sk.srcTime) / 2 : sk.srcTime;
                    const tAfter = k+1 < N ? (sk.srcTime + all[k+1].srcTime) / 2
                        : sk.srcTime + (sk.srcTime - tBefore);
                    let g;
                    if (targetNs < sk.srcTime) {
                        // Prev-half: vessel between midpoint-before and the AIS point.
                        const denom = sk.srcTime - tBefore;
                        const p1 = denom > 0 ? (targetNs - tBefore) / denom : 1;
                        g = Math.max(0, Math.min(1, p1)) * nw.Lp;
                    } else {
                        // Next-half: vessel past the AIS point toward midpoint-after.
                        const denom = tAfter - sk.srcTime;
                        const p2 = denom > 0 ? (targetNs - sk.srcTime) / denom : 1;
                        g = nw.Lp + Math.max(0, Math.min(1, p2)) * nw.Ln;
                    }
                    nw.C = vessel0;                          // front point = vessel
                    nw.fwd = vessel0;
                    nw.back = add2(vessel0, mul2(nw.u, -g)); // grown length behind it
                }
                // Is X inside a crest's own finite extent (along its direction)?
                const within = (c0, X) => {
                    const tB = dot2(sub2(c0.back, c0.C), c0.u);
                    const tF = dot2(sub2(c0.fwd,  c0.C), c0.u);
                    const tX = dot2(sub2(X,        c0.C), c0.u);
                    return tX >= Math.min(tB, tF) - 1e-6 && tX <= Math.max(tB, tF) + 1e-6;
                };
                // Neighbour crossings; only trim when inside BOTH finite crests.
                const join = new Array(Math.max(0, m - 1)).fill(null);
                for (let i = 0; i < m - 1; i++) {
                    const X = lineIntersect(cr[i].C, cr[i].u, cr[i+1].C, cr[i+1].u);
                    const ok = !!X && within(cr[i], X) && within(cr[i+1], X);
                    join[i] = ok ? X : null;
                    cuspJoints.push({position: fromMeters(X || cr[i].fwd, vessel),
                        side, kind: ok ? 'neighbour' : 'rejected'});
                }
                cuspJoints.push({position: fromMeters(cr[m-1].fwd, vessel), side, kind: 'vessel'});
                // Emit segments: shared joints connect neighbours, else full length.
                for (let i = 0; i < m; i++) {
                    const c0 = cr[i];
                    const back = (i > 0 && join[i-1]) ? join[i-1] : c0.back;
                    const fwd  = (i < m-1 && join[i]) ? join[i] : c0.fwd;
                    const frontAbs = fromMeters(c0.C, vessel);
                    const prevFront = cuspPrevFront.get(c0.s.ri);
                    const movDeg = prevFront
                        && (Math.abs(frontAbs[0] - prevFront[0]) > 1e-9
                            || Math.abs(frontAbs[1] - prevFront[1]) > 1e-9)
                        ? bearingDeg(prevFront, frontAbs) : null;
                    cuspNextFront.set(c0.s.ri, frontAbs);
                    cuspSegments.push({
                        source: fromMeters(back, vessel),
                        target: fromMeters(fwd, vessel),
                        labelPos: frontAbs,
                        side,
                        direction: c0.s.lineOriDeg,
                        speedKn: c0.s.groupSpeed * 1.943844,
                        oriDeg: (c0.s.lineOriDeg % 360 + 360) % 360,
                        movDeg,
                    });
                    propagationRays.push({
                        source: fromMeters(c0.s.srcM, vessel),
                        target: frontAbs,
                        side,
                    });
                }
            }
            window.__cuspPrevFront = cuspNextFront;
            const geometry = {vessel, cuspSegments, propagationRays, transverseCircles, cuspJoints};
            window.__animationLastGeometry = {
                state,
                rayCount: rayIdxs.length,
                cuspSegmentCount: cuspSegments.length,
                transverseCount: transverseCircles.length,
            };
            return geometry;
        };

        const buildLayers = (zoom, hoveredIdx) => {
            const animationState = animation.getState();
            const selectedSegIdx = animationState.selection?.segIdx ?? null;
            const useRaster = zoom < """ + str(ZOOM_RASTER_THRESHOLD) + r"""
                && selectedSegIdx == null;
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
                        getColor: (_, {index}) => trackColor(
                            tType[index],
                            selectedSegIdx == null ? 80 : (index === selectedSegIdx ? 240 : 18),
                        ),
                        getWidth: 1.5, widthUnits: 'pixels', widthMinPixels: 1.5,
                        updateTriggers: { getColor: [tType, selectedSegIdx] },
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
                        getColor: (_, {index}) => {
                            const si = filteredSegIdxs[index];
                            return trackColor(
                                tType[si],
                                selectedSegIdx == null ? 160 : (si === selectedSegIdx ? 240 : 18),
                            );
                        },
                        getWidth: 2.5, widthUnits: 'pixels', widthMinPixels: 2,
                        updateTriggers: { getColor: [filteredSegIdxs, selectedSegIdx] },
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
                                return trackColor(
                                    tType[si],
                                    selectedSegIdx == null ? 160 : (si === selectedSegIdx ? 240 : 12),
                                );
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
                if (vis === null && cPos.length / 2 <= MAX_FILTERED_POINTS) {
                    layers.push(new deck.ScatterplotLayer({
                        id: 'tracks-pts',
                        data: { length: cPos.length / 2,
                                attributes: { getPosition: { value: cPos, size: 2 } } },
                        getRadius: 4, radiusUnits: 'pixels', radiusMinPixels: 2,
                        getFillColor: [255, 255, 255, 0],
                        pickable: true,
                    }));
                }
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
                        const polygonState = window.__polygonController?.getState();
                        if (polygonState?.armed || polygonState?.drawing ||
                                !window.__ctrlHeld || index < 0) {
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

            const animationGeometry = selectedAnimationGeometry(animationState);
            if (animationGeometry) {
                const animationLayerIds = [];
                if (animationGeometry.transverseCircles.length > 0) {
                    animationLayerIds.push('animation-transverse');
                    layers.push(new deck.ScatterplotLayer({
                        id: 'animation-transverse',
                        data: animationGeometry.transverseCircles,
                        getPosition: d => d.position,
                        getRadius: d => d.radius,
                        radiusUnits: 'meters',
                        stroked: true,
                        filled: false,
                        getLineColor: d => [255, 255, 255, Math.max(35, 150 - d.age * 80)],
                        lineWidthMinPixels: 1,
                        pickable: false,
                    }));
                }
                if (animationGeometry.cuspSegments.length > 0) {
                    animationLayerIds.push('animation-cusp-lines');
                    layers.push(new deck.LineLayer({
                        id: 'animation-cusp-lines',
                        data: animationGeometry.cuspSegments,
                        getSourcePosition: d => d.source,
                        getTargetPosition: d => d.target,
                        getColor: d => d.side === 'port'
                            ? [60, 190, 255, 245]
                            : [255, 135, 70, 245],
                        getWidth: 3.5,
                        widthMinPixels: 3,
                        pickable: false,
                    }));
                    if (window.__cuspDebug) {
                        // Tag a sampled subset so the canvas stays readable.
                        const segs = animationGeometry.cuspSegments;
                        const labelStride = Math.max(1, Math.ceil(segs.length / 28));
                        const labels = segs.filter((d, i) => i % labelStride === 0);
                        animationLayerIds.push('animation-cusp-debug');
                        layers.push(new deck.TextLayer({
                            id: 'animation-cusp-debug',
                            data: labels,
                            getPosition: d => d.labelPos,
                            getText: d => `${d.speedKn.toFixed(1)} kn`
                                + `\nori ${Math.round(d.oriDeg)}°`
                                + `\nmov ${d.movDeg == null ? '--' : Math.round(d.movDeg) + '°'}`,
                            getColor: d => d.side === 'port'
                                ? [10, 70, 130, 255] : [150, 55, 10, 255],
                            getSize: 11,
                            sizeUnits: 'pixels',
                            getTextAnchor: 'start',
                            getAlignmentBaseline: 'center',
                            background: true,
                            getBackgroundColor: [255, 255, 255, 205],
                            backgroundPadding: [3, 2],
                            fontFamily: 'monospace',
                            pickable: false,
                        }));
                    }
                }
                if (window.__cuspDebug && animationGeometry.propagationRays
                    && animationGeometry.propagationRays.length > 0) {
                    animationLayerIds.push('animation-cusp-rays');
                    layers.push(new deck.LineLayer({
                        id: 'animation-cusp-rays',
                        data: animationGeometry.propagationRays,
                        getSourcePosition: d => d.source,
                        getTargetPosition: d => d.target,
                        getColor: d => d.side === 'port'
                            ? [60, 190, 255, 120]
                            : [255, 135, 70, 120],
                        getWidth: 1.5,
                        widthMinPixels: 1,
                        pickable: false,
                    }));
                }
                if (window.__cuspDebug && animationGeometry.cuspJoints
                    && animationGeometry.cuspJoints.length > 0) {
                    // Draw every intersection point for verification: magenta = an
                    // accepted neighbour trim, lime = newest-crest vessel-ray end,
                    // grey = a crossing rejected for falling outside the crest lengths.
                    animationLayerIds.push('animation-cusp-joints');
                    layers.push(new deck.ScatterplotLayer({
                        id: 'animation-cusp-joints',
                        data: animationGeometry.cuspJoints,
                        getPosition: d => d.position,
                        getRadius: d => d.kind === 'rejected' ? 3 : 4, radiusUnits: 'pixels',
                        radiusMinPixels: 3, radiusMaxPixels: 6,
                        getFillColor: d => d.kind === 'vessel'
                            ? [60, 230, 60, 255]
                            : (d.kind === 'rejected' ? [150, 150, 150, 180] : [240, 40, 200, 255]),
                        stroked: true, getLineColor: [20, 20, 20, 220],
                        lineWidthMinPixels: 1, pickable: false,
                    }));
                }
                animationLayerIds.push('animation-vessel');
                layers.push(new deck.ScatterplotLayer({
                    id: 'animation-vessel',
                    data: [{position: animationGeometry.vessel}],
                    getPosition: d => d.position,
                    getRadius: 9, radiusUnits: 'pixels', radiusMinPixels: 7,
                    getFillColor: [255, 255, 255, 255],
                    stroked: true, getLineColor: [0, 110, 210, 255],
                    lineWidthMinPixels: 3, pickable: false,
                }));
                window.__animationLastLayerIds = animationLayerIds;
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
            const polygonState = window.__polygonController?.getState();
            if (window.__freehandMode) return CURSOR_PENCIL;
            if (polygonState?.drawing && window.__ctrlHeld) return 'crosshair';
            if (polygonState?.drawing) return isDragging ? 'grabbing' : 'grab';
            if (window.__ctrlHeld) {
                if (window.__freehandArmed) return CURSOR_PENCIL;
                if (polygonState?.armed) return 'crosshair';
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
                    animation.clear();
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
                    const si = waveToSegIdx ? waveToSegIdx[origIdx] : -1;
                    selectAnimationSegment(si, origIdx);
                    window.deckInstance.setProps({ layers: buildLayers(window._currentZoom, window._hoveredWave) });
                } else if (layer.id === 'tracks') {
                    const si = (window.__visibleSegIdxs !== null && filteredSegIdxs.length > 0)
                        ? filteredSegIdxs[index] : index;
                    copyMmsi = tMMSI[si];
                    msg = `track MMSI=${tMMSI[si]} seg=${tSeg[si]}`;
                    window._pinnedWave = null;
                    selectAnimationSegment(si);
                } else if (layer.id === 'tracks-pts') {
                    const row = window.__visibleSegIdxs !== null
                        ? filteredPointRow[index] : index;
                    const si = window.__visibleSegIdxs !== null
                        ? filteredPointSeg[index] : pointRowToSegIdx(row);
                    copyMmsi = tMMSI[si];
                    msg = `track point MMSI=${tMMSI[si]} seg=${tSeg[si]}`;
                    window._pinnedWave = null;
                    selectAnimationSegment(si);
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
                if (typeof window.__redrawWavePolygon === 'function') {
                    requestAnimationFrame(window.__redrawWavePolygon);
                }
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
            if (window.__polygonController) window.__polygonController.setCtrlHeld(true);
        });
        window.addEventListener('keyup', (e) => {
            if (e.key !== 'Control' || !window.__ctrlHeld) return;
            window.__ctrlHeld = false;
            hideTip();
            if (window.__freehandMode && typeof window.__cancelFreehandDraw === 'function') window.__cancelFreehandDraw();
            if (window.__polygonController) window.__polygonController.setCtrlHeld(false);
            window.__updateDeckCursor();
        });
        // Clear inspect mode if window loses focus while Ctrl is held (otherwise
        // ctrl-tabbing away leaves the flag stuck on with no key event to clear it)
        window.addEventListener('blur', () => {
            if (!window.__ctrlHeld) return;
            window.__ctrlHeld = false;
            hideTip();
            if (window.__freehandMode && typeof window.__cancelFreehandDraw === 'function') window.__cancelFreehandDraw();
            if (window.__polygonController) window.__polygonController.setCtrlHeld(false);
            window.__updateDeckCursor();
        });
        container.addEventListener('mousedown', (e) => {
            // Sync Ctrl state from the real event so first-interaction Ctrl+click works
            if (e.ctrlKey && !window.__ctrlHeld) {
                window.__ctrlHeld = true;
                window.__updateDeckCursor();
                // Activate armed modes in case keydown didn't fire before Ctrl was held
                if (window.__freehandArmed && !window.__freehandMode) window.__activateFreehandDraw();
                if (window.__polygonController) window.__polygonController.setCtrlHeld(true);
            }
            if (!window.__freehandMode && !window.__polygonController?.getState().drawing) {
                window.__updateDeckCursor(true);
            }
        });
        window.addEventListener('mouseup', () => {
            if (!window.__freehandMode && !window.__polygonController?.getState().drawing) {
                window.__updateDeckCursor(false);
            }
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

        // Post-pipeline refresh hooks: show the same progress overlay as before, then rebuild layers,
        // then show a "Rendering..." pill until deck.gl has painted at least one frame.
        window.__refreshWaveLayer = async (version) => {
            animation.clear();
            const [buf, rayBuf] = await fetchAssetsWithProgress([
                { key: 'waves', url: `/api/waves.arrow?v=${version}`, label: 'wave impacts' },
                { key: 'wave_animation', url: `/api/wave_animation.arrow?v=${version}`,
                  label: 'wave animation rays' },
            ], 'Loading wave impacts');
            setRenderStatus('Rendering waves...', false);
            rebuildWaveArrays(window.tableFromIPC(buf));
            rebuildAnimationRayArrays(window.tableFromIPC(rayBuf));
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
            animation.clear();
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
            animation.clear();
            const tv = result.track_version || 0;
            const wv = result.wave_version || 0;
            const hasWaves = (result.n_waves || 0) > 0;
            const assets = [
                { key: 'track_coords',  url: `/api/track_coords.arrow?v=${tv}`,  label: 'track coords' },
                { key: 'track_meta',    url: `/api/track_meta.arrow?v=${tv}`,    label: 'track metadata' },
                { key: 'track_offsets', url: `/api/track_offsets.arrow?v=${tv}`, label: 'track offsets' },
            ];
            if (hasWaves) assets.push({ key: 'waves', url: `/api/waves.arrow?v=${wv}`, label: 'wave impacts' });
            assets.push({ key: 'wave_animation', url: `/api/wave_animation.arrow?v=${wv}`,
                          label: 'wave animation rays' });
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
                    rebuildAnimationRayArrays(window.tableFromIPC(buffers[4]));
                } else {
                    // Empty wave caches so cross-filter logic stays consistent
                    window.__hasWaves = false;
                    window.__waveCount = 0;
                    rebuildAnimationRayArrays(window.tableFromIPC(buffers[3]));
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
                    ? `time: ${window.__previews.ais.timeMin}  →  ${window.__previews.ais.timeMax}`
                    : '';
                return range;
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
                    ? `time: ${window.__previews.ais.timeMin}  →  ${window.__previews.ais.timeMax}`
                    : '';
                return range;
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
            // Server filters mesh elements to AIS bbox expanded to 2× range (0.5× padding each side).
            let bboxParam = '';
            if (window.__aisBbox) {
                const [bw, bs, be, bn] = window.__aisBbox;
                const dLon = (be - bw) * 0.5, dLat = (bn - bs) * 0.5;
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
    const previewStores = {
        ais: '_pv_ais', coast: '_pv_coast',
        land: '_pv_land', bathy: '_pv_bathy',
    };
    for (const [role, storeId] of Object.entries(previewStores)) {
        const checkbox = document.getElementById(`native-preview-${role}`);
        if (checkbox) checkbox.checked = false;
        const uploaded = window.__uploaded?.[role];
        if (window.dash_clientside?.set_props) {
            window.dash_clientside.set_props(storeId, {
                data: {visible: false, path: uploaded?.path || null},
            });
        }
    }
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

# Invert the final intersection produced by all active filters.
app.clientside_callback(
    r"""
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        if (typeof window.__invertFilters !== 'function') return 'Filter controls not ready';
        return window.__invertFilters();
    }
    """,
    Output('fil-status', 'children', allow_duplicate=True),
    Input('btn-fil-invert', 'n_clicks'),
    prevent_initial_call=True,
)

# Wave-arrival-area button click → enter polygon-draw mode.
app.clientside_callback(
    r"""
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        if (typeof window.__enterWaveBoxMode === 'function') {
            window.__enterWaveBoxMode();
        } else {
            window.__pendingWaveBoxClick = true;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('fil-status', 'children', allow_duplicate=True),
    Input('btn-wavebox', 'n_clicks'),
    prevent_initial_call=True,
)

# Enable/disable the track-filter section based on whether waves are loaded.
app.clientside_callback(
    "function(n) { return (n && n > 0) ? {} : {pointerEvents: 'none', opacity: '0.5'}; }",
    Output('filter-section-wrap', 'style'),
    Input('_wave_n', 'data'),
    prevent_initial_call=False,
)


# Export button click → use a native save picker where available, then prepare the ZIP.
app.clientside_callback(
    r"""
    async function(n, uploaded_files, filter_active) {
        const nu = window.dash_clientside.no_update;
        if (!n) return [nu, nu];
        let fileHandle = null;
        const pickerPromise = window.__exportFileHandlePromise;
        window.__exportFileHandlePromise = null;
        if (pickerPromise) {
            const pickerResult = await pickerPromise;
            if (pickerResult.error) {
                if (typeof window.__setExportBusy === 'function') {
                    window.__setExportBusy(false,
                        pickerResult.error.name === 'AbortError'
                            ? 'Export cancelled'
                            : `Error opening save dialog: ${pickerResult.error.message}`);
                }
                return [
                    pickerResult.error.name === 'AbortError'
                        ? 'Export cancelled'
                        : `Error opening save dialog: ${pickerResult.error.message}`,
                    nu,
                ];
            }
            fileHandle = pickerResult.handle;
        }
        if (typeof window.__setExportBusy === 'function') {
            window.__setExportBusy(true, 'Preparing export archive...');
        }
        const seg_keys = filter_active && typeof window.__getFilteredSegKeys === 'function'
            ? window.__getFilteredSegKeys() : [];
        const wave_idxs = filter_active && typeof window.__getFilteredWaveIdxs === 'function'
            ? window.__getFilteredWaveIdxs() : null;
        const sel_ais = (uploaded_files && uploaded_files.ais) ? uploaded_files.ais.path : '';
        const body = JSON.stringify({
            filtered: !!filter_active, seg_keys, wave_idxs, sel_ais,
        });
        try {
            const resp = await fetch('/api/export/filtered',
                { method: 'POST', headers: {'Content-Type': 'application/json'}, body });
            if (!resp.ok) {
                let msg = resp.statusText;
                try {
                    const j = await resp.json();
                    msg = j.error || msg;
                } catch (_) {}
                if (typeof window.__setExportBusy === 'function') {
                    window.__setExportBusy(false, `Error: ${msg}`);
                }
                return [`Error: ${msg}`, nu];
            }
            const result = await resp.json();
            if (!result.download_url) {
                if (typeof window.__setExportBusy === 'function') {
                    window.__setExportBusy(false,
                        'Error: download URL missing from server response');
                }
                return ['Error: download URL missing from server response', nu];
            }
            const scope = filter_active ? 'filtered results' : 'full inputs and results';
            if (fileHandle) {
                if (typeof window.__setExportBusy === 'function') {
                    window.__setExportBusy(true, 'Downloading and saving export archive...');
                }
                const download = await fetch(result.download_url);
                if (!download.ok) throw new Error(`download failed: ${download.statusText}`);
                const writable = await fileHandle.createWritable();
                await writable.write(await download.blob());
                await writable.close();
                if (typeof window.__setExportBusy === 'function') {
                    window.__setExportBusy(false, `Saved ${scope}`);
                }
                return [`Saved ${scope}`, Date.now()];
            }
            if (typeof window.__setExportBusy === 'function') {
                window.__setExportBusy(false, `Downloading ${scope}`);
            }
            window.location.assign(result.download_url);
            return [`Downloading ${scope}`, Date.now()];
        } catch (e) {
            if (typeof window.__setExportBusy === 'function') {
                window.__setExportBusy(false, `Error: ${e.message}`);
            }
            return [`Error: ${e.message}`, nu];
        }
    }
    """,
    Output('export-status', 'children'),
    Output('_rescan_count', 'data', allow_duplicate=True),
    Input('btn-fil-export', 'n_clicks'),
    State('_uploaded_files', 'data'),
    State('_any_filter_active', 'data'),
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
