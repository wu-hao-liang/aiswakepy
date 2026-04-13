"""Stage 1 — AIS filtering and interpolation.

Pipeline:
 1. load_ais                — read CSV, parse timestamps, retain required columns
 2. deduplicate             — drop exact (mmsi, obstime) duplicates
 3. uniformize_vessel_info  — set width/length/typecargo to mode per MMSI
 4. remove_zero_dimensions  — drop rows where width/length/draught <= 0
 5. segment_trajectories    — time-gap-based segmentation (default 180 s)
 6. clean_error_coords      — Kinematic Consistency Check: remove GPS spikes
 7. clean_error_speed       — Acceleration Check: replace erroneous SOG/COG
 8. validate_speed          — secondary cap: SOG = min(reported, geodetic-derived)
 9. interpolate_trajectories — Cubic Hermite Spline at fixed time intervals
10. filter_study_area       — optional: keep only points inside a polygon
11. mask_land               — remove points inside coastline polygon
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline
from shapely.ops import unary_union

from aiswakepy.geo.geodesy import forward_point, geodetic_distance

_KNOTS_TO_MS = 0.5144444   # 1 knot in m/s
_DEG_TO_M = 111111.0       # flat-Earth: degrees → metres (good for ratio tests)

_REQUIRED_COLS = [
    "mmsi", "width", "length", "draught",
    "obstime", "longitude", "latitude",
    "sog", "cog", "typecargo",
]


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load_ais(csv_path: str | Path) -> pd.DataFrame:
    """Read raw AIS CSV. Parses obstime; retains required columns; drops NaN coords."""
    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="load_ais")

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"AIS CSV is missing required columns: {missing}")

    df = df[_REQUIRED_COLS].copy()
    df["obstime"] = pd.to_datetime(df["obstime"], utc=False, errors="coerce")
    df = df.dropna(subset=["obstime", "longitude", "latitude"])
    spinner.done(rows=len(df))
    return df


# ---------------------------------------------------------------------------
# 2. Deduplicate
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with duplicate (mmsi, obstime), keeping the first occurrence.

    Duplicates arise when multiple AIS receivers capture the same broadcast.
    Without this step, dt=0 between consecutive records causes division by zero
    in speed calculations.
    """
    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="deduplicate")
    result = df.drop_duplicates(subset=["mmsi", "obstime"], keep="first").reset_index(drop=True)
    spinner.done(rows=len(result))
    return result


# ---------------------------------------------------------------------------
# 3. Vessel info uniformization
# ---------------------------------------------------------------------------

def uniformize_vessel_info(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Set vessel attribute columns to the mode (most frequent value) per MMSI.

    AIS feeds sometimes report slightly different width/length/typecargo values
    across records for the same vessel. This step ensures a single consistent
    value per vessel.
    """
    if columns is None:
        columns = ["width", "length", "typecargo"]

    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="uniformize_vessel_info")
    df = df.copy()
    for col in columns:
        if col in df.columns:
            # agg once per MMSI → map back: O(n_unique) not O(n_rows)
            mode_map = df.groupby("mmsi")[col].agg(lambda s: s.mode().iloc[0])
            df[col] = df["mmsi"].map(mode_map)
    spinner.done(rows=len(df))
    return df


# ---------------------------------------------------------------------------
# 4. Zero-dimension removal
# ---------------------------------------------------------------------------

def remove_zero_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where width, length, or draught is <= 0 or NaN.

    These vessel dimensions are required for the Kriebel wake model; records
    with zero or missing dimensions cannot produce valid wake calculations.
    """
    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="remove_zero_dimensions")
    mask = (
        (df["width"] > 0) & df["width"].notna()
        & (df["length"] > 0) & df["length"].notna()
        & (df["draught"] > 0) & df["draught"].notna()
    )
    result = df[mask].reset_index(drop=True)
    spinner.done(rows=len(result))
    return result


# ---------------------------------------------------------------------------
# 5. Segment trajectories
# ---------------------------------------------------------------------------

def segment_trajectories(df: pd.DataFrame, gap_s: float = 180.0) -> pd.DataFrame:
    """Sort by mmsi + obstime and assign integer segment_id.

    A new segment starts when the time gap to the previous fix of the same
    vessel exceeds ``gap_s`` seconds (default 180 s).
    """
    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="segment_trajectories")
    df = df.sort_values(["mmsi", "obstime"]).copy()
    dt = df.groupby("mmsi")["obstime"].diff().dt.total_seconds().fillna(gap_s + 1)
    new_seg = (dt > gap_s) | (df["mmsi"] != df["mmsi"].shift(1))
    df["segment_id"] = new_seg.cumsum().astype(int)
    result = df.reset_index(drop=True)
    spinner.done(rows=len(result))
    return result


# ---------------------------------------------------------------------------
# 6. Error coordinate cleaning — Kinematic Consistency Check
# ---------------------------------------------------------------------------

def clean_error_coords(
    df: pd.DataFrame,
    max_velocity_knots: float = 12.0,
) -> pd.DataFrame:
    """Remove GPS spike points using a Kinematic Consistency Check.

    Algorithm
    ---------
    For each consecutive pair (i, i+1) within a segment, compute average speed.
    If average speed > ``max_velocity_knots``, flag both endpoints as suspicious
    (+1 flag each). After all pairs are checked, resolve by flag count:

    - Flag 2  : point is the GPS spike (both adjacent segments are too fast) → remove.
    - Flag 1, neighbour has flag 2 : point is clean (neighbour was the spike) → keep.
    - Flag 1, neighbour has flag 1 : ambiguous drifted sequence. Search outward
      from each end for the nearest other flag-1 point; remove everything between.
      If search reaches the trajectory boundary, remove the shorter-in-time half.
    - Flag 0  : clean → keep.

    Uses flat-Earth distances (deg × 111111) for speed computation — acceptable
    because the test is a dimensionless threshold comparison, not a precision
    measurement.
    """
    from aiswakepy._progress import Spinner
    max_velocity_ms = max_velocity_knots * _KNOTS_TO_MS
    keep_mask = np.ones(len(df), dtype=bool)
    idx_arr = df.index.to_numpy()

    spinner = Spinner(desc="clean_error_coords")
    for _si, (seg_id, seg_df) in enumerate(df.groupby("segment_id", sort=False)):
        spinner.update(_si + 1)
        if len(seg_df) < 2:
            continue

        seg_idx = seg_df.index.to_numpy()
        lons = seg_df["longitude"].to_numpy()
        lats = seg_df["latitude"].to_numpy()
        t_ns = seg_df["obstime"].to_numpy().astype("datetime64[ns]").astype(np.int64)

        x = lons * _DEG_TO_M
        y = lats * _DEG_TO_M
        t_s = (t_ns - t_ns[0]) / 1e9

        n = len(seg_idx)
        flags = np.zeros(n, dtype=int)

        # Step 1: flag endpoints of too-fast segments
        dx = np.diff(x)
        dy = np.diff(y)
        dl = np.sqrt(dx**2 + dy**2)
        dt = np.diff(t_s)
        dt_safe = np.where(dt > 0, dt, np.finfo(float).eps)
        avg_speed = dl / dt_safe  # m/s

        fast = avg_speed > max_velocity_ms
        flags[:-1] += fast.astype(int)
        flags[1:] += fast.astype(int)

        if not fast.any():
            continue  # all segments normal — skip resolution

        # Step 2: resolve flags
        remove = np.zeros(n, dtype=bool)
        processed = np.zeros(n, dtype=bool)

        for i in range(n):
            if processed[i]:
                continue
            if flags[i] == 0:
                processed[i] = True
                continue
            if flags[i] == 2:
                remove[i] = True
                processed[i] = True
                continue
            # flags[i] == 1: find the other endpoint of each flagged segment
            # that involves point i
            # Segments involving point i: segment (i-1, i) and (i, i+1)
            # Find which fast-segment(s) involve this point
            if flags[i] == 1:
                # Determine what's on each side
                # Left side: segment i-1 → i
                left_flag2 = (i > 0) and fast[i - 1] and (flags[i - 1] == 2)
                right_flag2 = (i < n - 1) and fast[i] and (flags[i + 1] == 2)

                if left_flag2 or right_flag2:
                    # Neighbour is the spike; this point is clean
                    processed[i] = True
                    continue

                # Case C: both sides have flag 1 → find boundary by searching outward.
                # Skip the direct neighbour that shares the same fast segment as i,
                # because we need a flag-1 point from a DIFFERENT suspicious segment.
                bwd_boundary = None
                for j in range(i - 1, -1, -1):
                    # Skip direct left neighbour if fast segment j→i caused i's flag
                    if j == i - 1 and i > 0 and fast[i - 1]:
                        continue
                    if flags[j] == 1 and not processed[j]:
                        bwd_boundary = j
                        break

                fwd_boundary = None
                for j in range(i + 1, n):
                    # Skip direct right neighbour if fast segment i→j caused i's flag
                    if j == i + 1 and i < len(fast) and fast[i]:
                        continue
                    if flags[j] == 1 and not processed[j]:
                        fwd_boundary = j
                        break

                if bwd_boundary is not None and fwd_boundary is not None:
                    # Remove everything from bwd_boundary to fwd_boundary (inclusive)
                    remove[bwd_boundary:fwd_boundary + 1] = True
                    processed[bwd_boundary:fwd_boundary + 1] = True
                elif bwd_boundary is not None:
                    # No forward boundary — remove backward half if shorter in time
                    t_bwd = t_s[bwd_boundary]
                    t_mid = t_s[i]
                    t_end = t_s[-1]
                    if (t_mid - t_s[0]) <= (t_end - t_mid):
                        remove[:i + 1] = True
                        processed[:i + 1] = True
                    else:
                        remove[bwd_boundary:] = True
                        processed[bwd_boundary:] = True
                elif fwd_boundary is not None:
                    # No backward boundary
                    t_mid = t_s[i]
                    if (t_mid - t_s[0]) <= (t_s[-1] - t_mid):
                        remove[:fwd_boundary + 1] = True
                        processed[:fwd_boundary + 1] = True
                    else:
                        remove[i:] = True
                        processed[i:] = True
                else:
                    # No other flag-1 point found in either direction
                    # Remove shorter-in-time half
                    t_mid = t_s[i]
                    t_start = t_s[0]
                    t_end = t_s[-1]
                    if (t_mid - t_start) <= (t_end - t_mid):
                        remove[: i + 1] = True
                    else:
                        remove[i:] = True
                    processed[i] = True

        keep_mask[seg_idx[remove]] = False

    result = df[keep_mask].reset_index(drop=True)
    spinner.done(rows=len(result))
    return result


# ---------------------------------------------------------------------------
# 7. Error speed cleaning — Acceleration Check
# ---------------------------------------------------------------------------

def clean_error_speed(
    df: pd.DataFrame,
    max_acceleration_ms2: float = 0.2,
) -> pd.DataFrame:
    """Replace erroneous SOG/COG using an Acceleration Check.

    Must be called after ``clean_error_coords`` (GPS spikes removed first).

    For each point, the AIS-reported velocity (from SOG/COG) is compared to the
    velocity implied by adjacent point positions.  The acceleration required to
    transition from the AIS velocity to the segment-average velocity within half
    the time interval is computed.  If |acceleration| exceeds
    ``max_acceleration_ms2`` in either the x or y direction, the point's SOG/COG
    is replaced with the distance-weighted average of the adjacent finite-difference
    velocities.
    """
    from aiswakepy._progress import Spinner
    df = df.copy()
    sog_arr = df["sog"].to_numpy(dtype=float).copy()
    cog_arr = df["cog"].to_numpy(dtype=float).copy()
    _eps = np.finfo(float).eps

    spinner = Spinner(desc="clean_error_speed")
    for _si, (seg_id, seg_df) in enumerate(df.groupby("segment_id", sort=False)):
        spinner.update(_si + 1)
        n = len(seg_df)
        if n < 2:
            continue

        seg_pos = seg_df.index.to_numpy()
        lons = seg_df["longitude"].to_numpy()
        lats = seg_df["latitude"].to_numpy()
        t_ns = seg_df["obstime"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        t_s = (t_ns - t_ns[0]) / 1e9

        # AIS-reported velocity vectors (m/s)
        sog_ms = sog_arr[seg_pos] * _KNOTS_TO_MS
        cog_rad = np.radians(cog_arr[seg_pos])
        vx_ais = sog_ms * np.sin(cog_rad)
        vy_ais = sog_ms * np.cos(cog_rad)

        # Segment finite-difference velocities
        dx = np.diff(lons) * _DEG_TO_M
        dy = np.diff(lats) * _DEG_TO_M
        dt = np.diff(t_s)
        dt_safe = np.where(dt > 0, dt, _eps)
        vx_seg = dx / dt_safe   # velocity of segment i→i+1
        vy_seg = dy / dt_safe

        # ---- Vectorised acceleration checks ----
        # Forward check: point i (0..n-2) vs segment i→i+1
        half_fwd = dt_safe / 2.0
        bad_fwd = (
            (np.abs(vx_ais[:-1] - vx_seg) / half_fwd > max_acceleration_ms2) |
            (np.abs(vy_ais[:-1] - vy_seg) / half_fwd > max_acceleration_ms2)
        )
        # Backward check: point i (1..n-1) vs segment i-1→i
        half_bwd = dt_safe / 2.0
        bad_bwd = (
            (np.abs(vx_ais[1:] - vx_seg) / half_bwd > max_acceleration_ms2) |
            (np.abs(vy_ais[1:] - vy_seg) / half_bwd > max_acceleration_ms2)
        )
        bad = np.zeros(n, dtype=bool)
        bad[:-1] |= bad_fwd
        bad[1:]  |= bad_bwd

        if not bad.any():
            continue

        # Pre-compute distance-weighted replacements for interior points (0-alloc)
        dl = np.sqrt(dx**2 + dy**2)
        dl_pre  = np.maximum(dl[:-1], _eps)   # length n-2
        dl_post = np.maximum(dl[1:],  _eps)   # length n-2
        w_sum   = 1.0 / dl_pre + 1.0 / dl_post
        repl_vx = (vx_seg[:-1] / dl_pre + vx_seg[1:] / dl_post) / w_sum  # length n-2
        repl_vy = (vy_seg[:-1] / dl_pre + vy_seg[1:] / dl_post) / w_sum

        # Apply only to flagged points
        for i in np.where(bad)[0]:
            if i == 0:
                new_vx, new_vy = vx_seg[0], vy_seg[0]
            elif i == n - 1:
                new_vx, new_vy = vx_seg[-1], vy_seg[-1]
            else:
                new_vx, new_vy = repl_vx[i - 1], repl_vy[i - 1]

            new_speed = np.sqrt(new_vx**2 + new_vy**2)
            sog_arr[seg_pos[i]] = new_speed / _KNOTS_TO_MS
            cog_arr[seg_pos[i]] = (np.degrees(np.arctan2(new_vx, new_vy)) + 360.0) % 360.0

    df["sog"] = sog_arr
    df["cog"] = cog_arr
    spinner.done(rows=len(df))
    return df


# ---------------------------------------------------------------------------
# 8. Speed validation (secondary cap)
# ---------------------------------------------------------------------------

def validate_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Secondary cap: SOG = min(reported, geodetic-derived) within each segment.

    After ``clean_error_speed`` the reported SOG/COG should already be consistent
    with the trajectory. This step acts as a final safety net by capping the
    reported SOG at the geodetically-computed speed between consecutive fixes.
    """
    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="validate_speed")
    df = df.copy()

    lons = df["longitude"].to_numpy()
    lats = df["latitude"].to_numpy()
    times = df["obstime"].to_numpy()
    segs = df["segment_id"].to_numpy()

    n = len(df)
    v_calc = np.full(n, np.nan)
    dist_m = np.full(n, np.nan)

    same_seg = np.zeros(n, dtype=bool)
    same_seg[1:] = segs[1:] == segs[:-1]
    idx = np.where(same_seg)[0]
    if len(idx):
        d = geodetic_distance(lons[idx - 1], lats[idx - 1], lons[idx], lats[idx])
        dt = (times[idx] - times[idx - 1]) / np.timedelta64(1, "s")
        dist_m[idx] = d
        valid_dt = dt > 0
        v_calc[idx[valid_dt]] = d[valid_dt] / dt[valid_dt] / _KNOTS_TO_MS

    sog = df["sog"].to_numpy(dtype=float).copy()
    valid = ~np.isnan(v_calc)
    sog[valid] = np.minimum(sog[valid], v_calc[valid])

    df["sog"] = sog
    df["_dist_to_prev_m"] = dist_m
    spinner.done(rows=len(df))
    return df


# ---------------------------------------------------------------------------
# 9. Interpolation — Cubic Hermite Spline (time-based)
# ---------------------------------------------------------------------------

def interpolate_trajectories(
    df: pd.DataFrame,
    interval_s: float = 30.0,
) -> pd.DataFrame:
    """Interpolate each trajectory segment using Cubic Hermite Splines.

    For each segment with >= 2 points:
    - Convert lon/lat to flat-Earth x/y for spline fitting.
    - Build two splines: x(t) and y(t) using SOG/COG as velocity constraints.
    - Evaluate at uniform ``interval_s`` time intervals.
    - Derive interpolated SOG and COG from spline first derivatives.
    - Draught is nearest-neighbour in time (not interpolated).

    Segments with only 1 point pass through unchanged.
    """
    if df.empty:
        return df

    # Drop internal column not needed downstream
    if "_dist_to_prev_m" in df.columns:
        df = df.drop(columns=["_dist_to_prev_m"])

    # Sort by segment_id so rows are contiguous per segment
    df = df.sort_values("segment_id").reset_index(drop=True)

    # Extract all columns as numpy arrays once — avoids per-segment pandas overhead
    seg_ids   = df["segment_id"].to_numpy()
    lons_all  = df["longitude"].to_numpy(dtype=float)
    lats_all  = df["latitude"].to_numpy(dtype=float)
    t_ns_all  = df["obstime"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    sog_all   = df["sog"].to_numpy(dtype=float)
    cog_all   = df["cog"].to_numpy(dtype=float)
    draught_all   = df["draught"].to_numpy(dtype=float)
    mmsi_all      = df["mmsi"].to_numpy()
    width_all     = df["width"].to_numpy(dtype=float)
    length_all    = df["length"].to_numpy(dtype=float)
    typecargo_all = df["typecargo"].to_numpy()

    # Find segment boundaries without groupby — O(n) numpy only
    boundaries = np.where(np.diff(seg_ids) != 0)[0] + 1
    seg_starts = np.concatenate([[0], boundaries])
    seg_ends   = np.concatenate([boundaries, [len(df)]])

    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="interpolate_trajectories")

    # Accumulate results as lists of numpy arrays — no per-segment DataFrame
    out: dict[str, list] = {k: [] for k in (
        "mmsi", "width", "length", "draught", "obstime_ns",
        "longitude", "latitude", "sog", "cog", "typecargo", "segment_id",
    )}

    for seg_i, (start, end) in enumerate(zip(seg_starts, seg_ends)):
        spinner.update(seg_i + 1)
        n   = int(end - start)
        sid = seg_ids[start]

        if n < 2:
            out["mmsi"].append(mmsi_all[start:end])
            out["width"].append(width_all[start:end])
            out["length"].append(length_all[start:end])
            out["draught"].append(draught_all[start:end])
            out["obstime_ns"].append(t_ns_all[start:end])
            out["longitude"].append(lons_all[start:end])
            out["latitude"].append(lats_all[start:end])
            out["sog"].append(sog_all[start:end])
            out["cog"].append(cog_all[start:end])
            out["typecargo"].append(typecargo_all[start:end])
            out["segment_id"].append(np.full(n, sid))
            continue

        t_ns = t_ns_all[start:end]
        t_s  = (t_ns - t_ns[0]) / 1e9

        x = lons_all[start:end] * _DEG_TO_M
        y = lats_all[start:end] * _DEG_TO_M

        speed_ms = sog_all[start:end] * _KNOTS_TO_MS
        cog_rad  = np.radians(cog_all[start:end])
        dxdt = speed_ms * np.sin(cog_rad)
        dydt = speed_ms * np.cos(cog_rad)

        xspline = CubicHermiteSpline(t_s, x, dxdt)
        yspline = CubicHermiteSpline(t_s, y, dydt)

        t_interp = np.arange(t_s[0], t_s[-1], interval_s, dtype=float)
        if t_interp[-1] < t_s[-1]:
            t_interp = np.append(t_interp, t_s[-1])

        x_interp    = xspline(t_interp, nu=0)
        dxdt_interp = xspline(t_interp, nu=1)
        y_interp    = yspline(t_interp, nu=0)
        dydt_interp = yspline(t_interp, nu=1)

        speed_interp = np.sqrt(dxdt_interp**2 + dydt_interp**2)
        sog_interp   = speed_interp / _KNOTS_TO_MS
        cog_interp   = (np.degrees(np.arctan2(dxdt_interp, dydt_interp)) + 360.0) % 360.0

        lon_interp = x_interp / _DEG_TO_M
        lat_interp = y_interp / _DEG_TO_M

        # Draught: nearest-neighbour via searchsorted — O(m log n) not O(m×n)
        idx = np.searchsorted(t_s, t_interp).clip(1, n - 1)
        left_dist  = np.abs(t_interp - t_s[idx - 1])
        right_dist = np.abs(t_interp - t_s[idx])
        nearest    = np.where(left_dist <= right_dist, idx - 1, idx)
        draught_interp = draught_all[start:end][nearest]

        ni = len(t_interp)
        out["mmsi"].append(np.full(ni, mmsi_all[start]))
        out["width"].append(np.full(ni, width_all[start]))
        out["length"].append(np.full(ni, length_all[start]))
        out["draught"].append(draught_interp)
        out["obstime_ns"].append(t_ns[0] + (t_interp * 1e9).astype(np.int64))
        out["longitude"].append(lon_interp)
        out["latitude"].append(lat_interp)
        out["sog"].append(sog_interp)
        out["cog"].append(cog_interp)
        out["typecargo"].append(np.full(ni, typecargo_all[start]))
        out["segment_id"].append(np.full(ni, sid))

    # Single DataFrame construction from concatenated numpy arrays
    result = pd.DataFrame({
        "mmsi":       np.concatenate(out["mmsi"]),
        "width":      np.concatenate(out["width"]),
        "length":     np.concatenate(out["length"]),
        "draught":    np.concatenate(out["draught"]),
        "obstime":    pd.to_datetime(np.concatenate(out["obstime_ns"]), unit="ns"),
        "longitude":  np.concatenate(out["longitude"]),
        "latitude":   np.concatenate(out["latitude"]),
        "sog":        np.concatenate(out["sog"]),
        "cog":        np.concatenate(out["cog"]),
        "typecargo":  np.concatenate(out["typecargo"]),
        "segment_id": np.concatenate(out["segment_id"]),
    })
    spinner.done(rows=len(result))
    return result


# ---------------------------------------------------------------------------
# 10. Study-area polygon filter (optional)
# ---------------------------------------------------------------------------

def filter_study_area(
    df: pd.DataFrame,
    polygon_shp: str | Path | None,
) -> pd.DataFrame:
    """Keep only AIS points that fall inside a study-area polygon.

    If ``polygon_shp`` is None, the DataFrame is returned unchanged.
    """
    if polygon_shp is None:
        return df

    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="filter_study_area")
    study_area = gpd.read_file(str(polygon_shp))
    region = unary_union(study_area.geometry)
    points = gpd.GeoSeries(
        gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    inside = points.within(region)
    result = df[inside].reset_index(drop=True)
    spinner.done(rows=len(result))
    return result


# ---------------------------------------------------------------------------
# 11. Land masking
# ---------------------------------------------------------------------------

def mask_land(df: pd.DataFrame, coastline_shp: str | Path) -> pd.DataFrame:
    """Remove AIS points that fall inside the coastline polygon."""
    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="mask_land")
    coast = gpd.read_file(coastline_shp)
    land = unary_union(coast.geometry)
    points = gpd.GeoSeries(
        gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    in_land = points.within(land)
    result = df[~in_land].reset_index(drop=True)
    spinner.done(rows=len(result))
    return result


# ---------------------------------------------------------------------------
# 12. Orchestrator
# ---------------------------------------------------------------------------

def filter_ais(
    csv_path: str | Path,
    coastline_shp: str | Path,
    gap_s: float = 180.0,
    max_velocity_knots: float = 12.0,
    max_acceleration_ms2: float = 0.2,
    interval_s: float = 30.0,
    study_area_shp: str | Path | None = None,
) -> pd.DataFrame:
    """Run the full AIS filtering pipeline and return a cleaned DataFrame."""
    df = load_ais(csv_path)
    df = deduplicate(df)
    df = uniformize_vessel_info(df)
    df = remove_zero_dimensions(df)
    df = segment_trajectories(df, gap_s=gap_s)
    df = clean_error_coords(df, max_velocity_knots=max_velocity_knots)
    df = clean_error_speed(df, max_acceleration_ms2=max_acceleration_ms2)
    df = validate_speed(df)
    df = interpolate_trajectories(df, interval_s=interval_s)
    df = filter_study_area(df, polygon_shp=study_area_shp)
    df = mask_land(df, coastline_shp)
    return df
