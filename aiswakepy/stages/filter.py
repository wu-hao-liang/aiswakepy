"""Stage 1 — AIS filtering and interpolation.

Steps:
1. Load raw AIS CSV, parse timestamps, retain required columns.
2. Segment trajectories (time-gap-based).
3. Validate speed (conservative: min of reported SOG and computed speed).
4. Interpolate gaps > trigger_m to spacing_m resolution.
5. Mask land points using coastline polygon.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

from aiswakepy.geo.geodesy import forward_point, geodetic_distance

_KNOTS_TO_MS = 0.5144444

_REQUIRED_COLS = [
    "mmsi", "width", "length", "draught",
    "obstime", "longitude", "latitude",
    "sog", "cog", "typecargo",
]


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load_ais(csv_path: str | Path) -> pd.DataFrame:
    """Read raw AIS CSV.  Parses obstime to datetime; retains required columns.

    Extra columns beyond the required set are silently dropped.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalise column names to lower-case
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"AIS CSV is missing required columns: {missing}")

    df = df[_REQUIRED_COLS].copy()
    df["obstime"] = pd.to_datetime(df["obstime"], utc=False, errors="coerce")
    df = df.dropna(subset=["obstime"])
    return df


# ---------------------------------------------------------------------------
# 2. Segment trajectories
# ---------------------------------------------------------------------------

def segment_trajectories(df: pd.DataFrame, gap_s: float = 600.0) -> pd.DataFrame:
    """Sort by mmsi + obstime and assign integer segment_id.

    A new segment starts when the time gap to the previous fix of the same
    vessel exceeds ``gap_s`` seconds.
    """
    df = df.sort_values(["mmsi", "obstime"]).copy()

    dt = df.groupby("mmsi")["obstime"].diff().dt.total_seconds().fillna(gap_s + 1)
    new_seg = (dt > gap_s) | (df["mmsi"] != df["mmsi"].shift(1))
    df["segment_id"] = new_seg.cumsum().astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Speed validation
# ---------------------------------------------------------------------------

def validate_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Compute geodetic speed between consecutive fixes; cap SOG at computed value.

    Within each segment, compute distance and time between consecutive points.
    The conservative speed used downstream is ``min(sog_reported, v_calc)``.
    First point of each segment keeps its reported SOG unchanged.
    """
    df = df.copy()

    lons = df["longitude"].to_numpy()
    lats = df["latitude"].to_numpy()
    times = df["obstime"].to_numpy()
    segs = df["segment_id"].to_numpy()

    v_calc = np.full(len(df), np.nan)
    dist_m = np.full(len(df), np.nan)

    same_seg = segs[1:] == segs[:-1]
    d = geodetic_distance(lons[:-1], lats[:-1], lons[1:], lats[1:])
    dt = (times[1:] - times[:-1]) / np.timedelta64(1, "s")
    dist_m[1:] = np.where(same_seg, d, np.nan)
    v_calc[1:] = np.where(same_seg & (dt > 0), d / dt / _KNOTS_TO_MS, np.nan)

    # First point of each segment: keep reported SOG
    sog = df["sog"].to_numpy(dtype=float).copy()
    valid = ~np.isnan(v_calc)
    sog[valid] = np.minimum(sog[valid], v_calc[valid])

    df["sog"] = sog
    df["_dist_to_prev_m"] = dist_m
    return df


# ---------------------------------------------------------------------------
# 4. Interpolation
# ---------------------------------------------------------------------------

def interpolate_trajectories(
    df: pd.DataFrame,
    spacing_m: float = 20.0,
    trigger_m: float = 100.0,
) -> pd.DataFrame:
    """Insert linearly interpolated fixes for gaps > trigger_m.

    Only numeric columns are interpolated; obstime is linearly interpolated
    as well (converted to float seconds and back).  The segment_id of
    interpolated rows matches the parent segment.
    """
    numeric_cols = [
        c for c in df.columns
        if c not in ("obstime", "segment_id", "_dist_to_prev_m")
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    out_chunks: list[pd.DataFrame] = []

    for _, seg_df in df.groupby("segment_id", sort=False):
        seg_df = seg_df.reset_index(drop=True)
        n = len(seg_df)

        # One numpy buffer per column; avoids O(N²) DataFrame concat within segment
        col_bufs: dict[str, list[np.ndarray]] = {c: [] for c in numeric_cols}
        time_buf: list[np.ndarray] = []
        segid_buf: list[np.ndarray] = []

        def _push_single(idx: int) -> None:
            for col in numeric_cols:
                col_bufs[col].append(np.array([seg_df.at[idx, col]], dtype=float))
            time_buf.append(np.array([seg_df.at[idx, "obstime"].timestamp()]))
            segid_buf.append(np.array([seg_df.at[idx, "segment_id"]], dtype=int))

        _push_single(0)

        for i in range(1, n):
            dist = seg_df.at[i, "_dist_to_prev_m"]
            if pd.isna(dist) or dist <= trigger_m:
                _push_single(i)
            else:
                n_pts = max(2, int(np.ceil(dist / spacing_m)) + 1)
                ts = np.linspace(0.0, 1.0, n_pts)[1:]

                t0 = seg_df.at[i - 1, "obstime"].timestamp()
                t1 = seg_df.at[i, "obstime"].timestamp()
                time_buf.append(t0 + (t1 - t0) * ts)
                segid_buf.append(
                    np.full(len(ts), seg_df.at[i, "segment_id"], dtype=int)
                )

                for col in numeric_cols:
                    v0 = float(seg_df.at[i - 1, col])
                    v1 = float(seg_df.at[i, col])
                    col_bufs[col].append(v0 + (v1 - v0) * ts)

        data = {col: np.concatenate(col_bufs[col]) for col in numeric_cols}
        data["obstime"] = pd.to_datetime(
            np.concatenate(time_buf), unit="s", utc=False
        )
        data["segment_id"] = np.concatenate(segid_buf)
        out_chunks.append(pd.DataFrame(data))

    result = pd.concat(out_chunks, ignore_index=True)
    if "_dist_to_prev_m" in result.columns:
        result = result.drop(columns=["_dist_to_prev_m"])
    return result


# ---------------------------------------------------------------------------
# 5. Land masking
# ---------------------------------------------------------------------------

def mask_land(df: pd.DataFrame, coastline_shp: str | Path) -> pd.DataFrame:
    """Remove AIS points that fall inside the coastline polygon."""
    coast = gpd.read_file(coastline_shp)
    land = unary_union(coast.geometry)

    points = gpd.GeoSeries(
        gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    in_land = points.within(land)
    return df[~in_land].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Orchestrator
# ---------------------------------------------------------------------------

def filter_ais(
    csv_path: str | Path,
    coastline_shp: str | Path,
    gap_s: float = 600.0,
    spacing_m: float = 20.0,
    trigger_m: float = 100.0,
) -> pd.DataFrame:
    """Run the full AIS filtering pipeline and return a cleaned DataFrame."""
    df = load_ais(csv_path)
    df = segment_trajectories(df, gap_s=gap_s)
    df = validate_speed(df)
    df = interpolate_trajectories(df, spacing_m=spacing_m, trigger_m=trigger_m)
    df = mask_land(df, coastline_shp)
    return df
