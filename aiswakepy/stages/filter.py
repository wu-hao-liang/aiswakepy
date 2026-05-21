"""Stage 1 — AIS filtering and interpolation.

Pipeline (execution order):
 1. load_ais                     — read CSV, parse timestamps, retain required columns
 2. deduplicate                  — drop exact (mmsi, obstime) duplicates
 3. uniformize_vessel_info       — set width/length/typecargo to mode per MMSI
 4. remove_zero_dimensions       — drop rows where width/length/draught <= 0
 5. remove_invalid_draught       — drop rows where draught > width (implausible)
 6. mask_land ×2                 — remove raw points on land (before segmentation)
 7. segment_trajectories         — time-gap-based segmentation of surviving points
 8. clean_error_coords           — Kinematic Consistency Check (per segment)
 9. clean_error_speed            — Acceleration + speed-consistency check (per segment)
10. filter_low_speed             — strip rows with SOG < min_speed_knots
11. segment_trajectories (re-run) — re-segment after low-speed point removal
12. interpolate_trajectories     — straight-line / Hermite / mixed to uniform grid
13a. mask_land (land)            — drop post-interpolation land points
13b. mask_land (coastline)       — drop points inside coastline
13c. segment_trajectories        — re-segment after land point removal
14. filter_study_area            — optional: keep only points inside a polygon
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline
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
    if df["obstime"].dt.tz is not None:
        df["obstime"] = df["obstime"].dt.tz_localize(None)
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
# 5. Invalid-draught removal
# ---------------------------------------------------------------------------

def remove_invalid_draught(
    df: pd.DataFrame,
    max_draught_to_width: float = 1.0,
) -> pd.DataFrame:
    """Drop rows where draught exceeds ``max_draught_to_width`` × width.

    Real vessels have draught well below their beam (typical T/B ≲ 0.5).
    Records with draught greater than the beam are almost always AIS
    misreports (e.g. unit confusion or stuck-default values) and would
    produce nonsensical wake predictions.
    """
    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="remove_invalid_draught")
    mask = df["draught"] <= max_draught_to_width * df["width"]
    result = df[mask].reset_index(drop=True)
    spinner.done(rows=len(result))
    return result


# ---------------------------------------------------------------------------
# 10. Low-speed filter
# ---------------------------------------------------------------------------

def filter_low_speed(
    df: pd.DataFrame,
    min_speed_knots: float = 0.0,
) -> pd.DataFrame:
    """Remove rows where SOG is below ``min_speed_knots``.

    Vessels that are berthed, anchored, or drifting produce negligible wake.
    This step runs early (before segmentation) to strip those rows cheaply.
    """
    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="filter_low_speed")
    mask = df["sog"] >= min_speed_knots
    result = df[mask].reset_index(drop=True)
    spinner.done(rows=len(result))
    return result


# ---------------------------------------------------------------------------
# 7. Segment trajectories
# ---------------------------------------------------------------------------

def segment_trajectories(df: pd.DataFrame, gap_s: float = 180.0,
                         use_force_break: bool = False) -> pd.DataFrame:
    """Sort by mmsi + obstime and assign integer segment_id.

    A new segment starts when the time gap to the previous fix of the same
    vessel exceeds ``gap_s`` seconds (default 180 s), or when crossing to
    a different MMSI.

    If ``use_force_break`` is True and the DataFrame has a ``_force_break``
    column, any row with ``_force_break=True`` also starts a new segment
    (regardless of time gap).  ``mask_land`` sets these flags when it
    removes land points, so the trajectory is split at land crossings.

    Safe to call multiple times — completely re-assigns segment_ids from
    scratch based on the current row order.
    """
    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="segment_trajectories")
    df = df.sort_values(["mmsi", "obstime"]).copy()
    dt = df.groupby("mmsi")["obstime"].diff().dt.total_seconds().fillna(gap_s + 1)
    new_seg = (dt > gap_s) | (df["mmsi"] != df["mmsi"].shift(1))
    if use_force_break and "_force_break" in df.columns:
        new_seg = new_seg | df["_force_break"].to_numpy(dtype=bool)
    df["segment_id"] = new_seg.cumsum().astype(int)
    result = df.reset_index(drop=True)
    spinner.done(rows=len(result))
    return result


# ---------------------------------------------------------------------------
# 8. Error coordinate cleaning — Kinematic Consistency Check
# ---------------------------------------------------------------------------

def clean_error_coords(
    df: pd.DataFrame,
    max_velocity_knots: float = 36.0,
    low_sog_threshold_ms: float = 1.0,
    velocity_ratio_threshold: float = 2.0,
) -> pd.DataFrame:
    """Remove GPS spike points using a Kinematic Consistency Check.

    Algorithm
    ---------
    For each consecutive pair (i, i+1) within a segment, compute average speed.
    A pair is flagged when:

    * average speed > ``max_velocity_knots`` (hard speed cap), OR
    * both endpoints report SOG < ``low_sog_threshold_ms`` (nominally stationary)
      but the computed displacement speed exceeds 2× that threshold — this
      catches GPS position spikes that would otherwise pass the speed cap.
    * the displacement speed exceeds ``velocity_ratio_threshold`` × the
      average of the two reported SOG values — the positions moved far
      faster than the transponder claims, indicating a coordinate spike.

    After all pairs are checked, resolve by flag count:

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
    spike_speed_ms = 2.0 * low_sog_threshold_ms  # e.g. 2 m/s when threshold is 1 m/s
    keep_mask = np.ones(len(df), dtype=bool)

    # Group by MMSI — this runs before segment_trajectories, so segment_id may
    # not exist yet. MMSI-level grouping gives the same kinematic isolation.
    grouping = df.groupby("mmsi", sort=False) if "segment_id" not in df.columns \
               else df.groupby("segment_id", sort=False)

    spinner = Spinner(desc="clean_error_coords")
    for _si, (_, seg_df) in enumerate(grouping):
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

        # Primary check: raw speed cap
        fast = avg_speed > max_velocity_ms

        # Secondary check: low-SOG GPS spikes.
        # Both endpoints report near-zero speed, but the positions moved
        # significantly — the GPS coordinate is a spike, not real motion.
        sog_arr = seg_df["sog"].to_numpy(dtype=float)
        sog_arr_ms = sog_arr * _KNOTS_TO_MS
        low_sog_pair = (sog_arr_ms[:-1] < low_sog_threshold_ms) & \
                       (sog_arr_ms[1:]  < low_sog_threshold_ms)
        fast = fast | (low_sog_pair & (avg_speed > spike_speed_ms))

        # Tertiary check: velocity-consistency ratio.
        # If the positions moved more than velocity_ratio_threshold × faster
        # than the average of the two reported SOG values, the coordinate is
        # likely a GPS spike that the speed cap and low-SOG checks missed.
        sog_avg_ms = 0.5 * (sog_arr_ms[:-1] + sog_arr_ms[1:])
        sog_avg_safe = np.maximum(sog_avg_ms, 1e-3)  # avoid /0
        fast = fast | ((avg_speed / sog_avg_safe) > velocity_ratio_threshold)

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
# 9. Error speed cleaning — Acceleration + speed-consistency check
# ---------------------------------------------------------------------------

def clean_error_speed(
    df: pd.DataFrame,
    max_acceleration_ms2: float = 10.0,
    speed_consistency_ratio: float = 0.5,
) -> pd.DataFrame:
    """Replace erroneous SOG/COG using an Acceleration Check and a speed-
    consistency check.

    Must be called after ``clean_error_coords`` (GPS spikes removed first).

    Two independent checks flag a point for SOG/COG replacement:

    1. **Acceleration check**: if the AIS-reported velocity (SOG/COG) requires
       acceleration exceeding ``max_acceleration_ms2`` to match the segment-
       average velocity within half the time interval, the point is flagged.

    2. **Speed-consistency check**: for each consecutive pair (i, i+1), if the
       position-derived speed (dl/dt) is less than ``speed_consistency_ratio`` ×
       the magnitude of the vector-averaged AIS velocity, both endpoints are
       flagged.  This catches cases where AIS reports movement (SOG ≫ 0) but
       the GPS positions show near-stationary — the SOG/COG is unreliable.

    Flagged points get their SOG/COG replaced with the distance-weighted
    average of adjacent segment velocities.
    """
    from aiswakepy._progress import Spinner
    df = df.copy()
    sog_arr = df["sog"].to_numpy(dtype=float).copy()
    cog_arr = df["cog"].to_numpy(dtype=float).copy()
    _eps = np.finfo(float).eps

    grouping = df.groupby("mmsi", sort=False) if "segment_id" not in df.columns \
               else df.groupby("segment_id", sort=False)

    spinner = Spinner(desc="clean_error_speed")
    for _si, (_, seg_df) in enumerate(grouping):
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

        # ---- Speed-consistency check ----
        # For each pair (i, i+1): if dl/dt ≪ |v_i + v_{i+1}| / 2,
        # the positions say the vessel is near-stationary but the AIS
        # transponder claims it is moving — the SOG/COG is suspect.
        dl_pair = np.sqrt(dx**2 + dy**2)        # distance per pair (m)
        dist_speed = dl_pair / dt_safe          # speed from positions (m/s)
        vx_avg = 0.5 * (vx_ais[:-1] + vx_ais[1:])
        vy_avg = 0.5 * (vy_ais[:-1] + vy_ais[1:])
        vec_avg_mag = np.sqrt(vx_avg**2 + vy_avg**2)  # |(v_i+v_{i+1})/2|
        inconsistent = dist_speed < (speed_consistency_ratio * vec_avg_mag)
        bad[:-1] |= inconsistent
        bad[1:]  |= inconsistent

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
# 12. Interpolation — linear / Hermite / mixed (time-based)
# ---------------------------------------------------------------------------

def interpolate_trajectories(
    df: pd.DataFrame,
    interval_s: float = 30.0,
    method: str = "linear",
    low_sog_threshold_ms: float = 1.0,
) -> pd.DataFrame:
    """Interpolate each trajectory segment to a uniform time grid.

    Parameters
    ----------
    df : input vessel positions with columns: mmsi, segment_id, obstime,
        longitude, latitude, sog, cog, width, length, draught, typecargo.
    interval_s : output time spacing in seconds.
    method : ``"linear"``, ``"hermite"``, or ``"mixed"``.
        Position interpolation (lon, lat) follows the chosen method.  SOG/COG
        are always linearly interpolated as vectors between the two raw
        endpoints (SOG via np.interp; COG via sin/cos circular mean) in all
        three methods — this avoids segment-velocity collapse and spline
        overshoot contaminating the speed columns.
        - ``"linear"``: straight-line between consecutive raw points.
        - ``"hermite"``: CubicHermiteSpline with SOG/COG as velocity constraints.
        - ``"mixed"``: per-bracket hybrid — linear when BOTH endpoints have
          SOG < ``low_sog_threshold_ms`` (both stationary), else hermite.
    low_sog_threshold_ms : speed threshold (m/s) for ``"mixed"`` method.
        Default 1.0 m/s (≈ 2 knots).

    For each segment with >= 2 points, interpolation is done per-bracket
    (between consecutive raw points) using ``np.linspace`` so that every
    surviving raw point is reproduced exactly.  Intermediate points are
    evenly distributed across each bracket.  Draught is nearest-neighbour
    in time (not interpolated).  Segments with 1 point pass through unchanged.
    """
    method = (method or "linear").lower()
    if method not in ("linear", "hermite", "mixed"):
        raise ValueError(f"interpolate_trajectories: unknown method {method!r}; "
                         "expected 'linear', 'hermite', or 'mixed'")
    if df.empty:
        return df

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

    # ---- Helper: bracket-based interpolation for one segment ----
    def _bracket_times(t_s, interval_s):
        """Return list of evenly-spaced time arrays per bracket.

        Every raw point time t_s[i] appears exactly once in the concatenated
        result.  Intermediate points are distributed evenly across each bracket
        so the spacing is as close to ``interval_s`` as the bracket length allows.
        """
        parts = []
        for i in range(len(t_s) - 1):
            ta, tb = t_s[i], t_s[i + 1]
            n_step = max(1, int((tb - ta) / interval_s))
            ti = np.linspace(ta, tb, n_step + 1)          # includes both endpoints
            if i > 0:
                ti = ti[1:]                                # drop duplicate raw point
            parts.append(ti)
        return parts

    def _build_output(t_interp, x_interp, y_interp, sog_interp, cog_interp,
                      t_ns0, sid, draught_arr, t_s):
        """Append a per-segment output block to ``out``."""
        ni = len(t_interp)
        lon_i = x_interp / _DEG_TO_M
        lat_i = y_interp / _DEG_TO_M
        dr_idx = np.searchsorted(t_s, t_interp).clip(1, len(t_s) - 1)
        left_dist  = np.abs(t_interp - t_s[dr_idx - 1])
        right_dist = np.abs(t_interp - t_s[dr_idx])
        nearest    = np.where(left_dist <= right_dist, dr_idx - 1, dr_idx)
        draught_i = draught_arr[nearest]
        out["mmsi"].append(np.full(ni, mmsi_all[start]))
        out["width"].append(np.full(ni, width_all[start]))
        out["length"].append(np.full(ni, length_all[start]))
        out["draught"].append(draught_i)
        out["obstime_ns"].append(t_ns0 + (t_interp * 1e9).astype(np.int64))
        out["longitude"].append(lon_i)
        out["latitude"].append(lat_i)
        out["sog"].append(sog_interp)
        out["cog"].append(cog_interp)
        out["typecargo"].append(np.full(ni, typecargo_all[start]))
        out["segment_id"].append(np.full(ni, sid))

    # ---- Per-segment loop ----
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

        t_ns_seg = t_ns_all[start:end]
        t_s  = (t_ns_seg - t_ns_seg[0]) / 1e9
        x = lons_all[start:end] * _DEG_TO_M
        y = lats_all[start:end] * _DEG_TO_M
        speed_ms   = sog_all[start:end] * _KNOTS_TO_MS
        cog_rad_seg = np.radians(cog_all[start:end])

        # Build per-bracket time arrays — every raw point guaranteed.
        t_parts = _bracket_times(t_s, interval_s)
        t_interp = np.concatenate(t_parts)

        # SOG: linear scalar interpolation between raw endpoints.
        sog_interp = np.interp(t_interp, t_s, sog_all[start:end])
        # COG: sin/cos circular-mean interpolation.
        cos_interp = np.interp(t_interp, t_s, np.cos(cog_rad_seg))
        sin_interp = np.interp(t_interp, t_s, np.sin(cog_rad_seg))
        cog_interp = (np.degrees(np.arctan2(sin_interp, cos_interp)) + 360.0) % 360.0

        if method == "mixed":
            # Per-bracket: hermite if both endpoint SOG ≥ threshold, else linear.
            vx_ais = speed_ms * np.sin(cog_rad_seg)
            vy_ais = speed_ms * np.cos(cog_rad_seg)
            x_parts, y_parts = [], []
            for i in range(n - 1):
                ta, tb = t_s[i], t_s[i + 1]
                ti = t_parts[i]
                use_linear = (speed_ms[i] < low_sog_threshold_ms and
                              speed_ms[i + 1] < low_sog_threshold_ms)
                if use_linear:
                    xi = np.interp(ti, t_s[i : i + 2], x[i : i + 2])
                    yi = np.interp(ti, t_s[i : i + 2], y[i : i + 2])
                else:
                    xs = CubicHermiteSpline(t_s[i : i + 2], x[i : i + 2],
                                            vx_ais[i : i + 2])
                    ys = CubicHermiteSpline(t_s[i : i + 2], y[i : i + 2],
                                            vy_ais[i : i + 2])
                    xi = xs(ti, nu=0); yi = ys(ti, nu=0)
                x_parts.append(xi); y_parts.append(yi)
            x_interp = np.concatenate(x_parts) if x_parts else np.array([], dtype=float)
            y_interp = np.concatenate(y_parts) if y_parts else np.array([], dtype=float)
        elif method == "hermite":
            vx = speed_ms * np.sin(cog_rad_seg)
            vy = speed_ms * np.cos(cog_rad_seg)
            xspline = CubicHermiteSpline(t_s, x, vx)
            yspline = CubicHermiteSpline(t_s, y, vy)
            x_interp = xspline(t_interp, nu=0)
            y_interp = yspline(t_interp, nu=0)
        else:  # linear
            x_interp = np.interp(t_interp, t_s, x)
            y_interp = np.interp(t_interp, t_s, y)

        _build_output(t_interp, x_interp, y_interp, sog_interp, cog_interp,
                      t_ns_seg[0], sid, draught_all[start:end], t_s)

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
# 14. Study-area polygon filter (optional — runs last)
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
    pts_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts_gdf, study_area, predicate="within", how="left")
    result = df[joined["index_right"].notna().to_numpy()].reset_index(drop=True)
    spinner.done(rows=len(result))
    return result


def _set_force_breaks(df: pd.DataFrame, remove_mask: np.ndarray) -> None:
    """Flag the surviving point after each removed point as a forced segment break.

    Modifies ``df`` in place, adding or updating the ``_force_break`` column.
    Only considers same-MMSI adjacency (cross-vessel boundaries are ignored —
    ``segment_trajectories`` already splits on MMSI change).
    """
    if not remove_mask.any():
        return
    force = np.zeros(len(df), dtype=bool)
    force[1:] = remove_mask[:-1]
    mmsi_arr = df["mmsi"].to_numpy()
    same_mmsi = np.zeros(len(df), dtype=bool)
    same_mmsi[1:] = mmsi_arr[1:] == mmsi_arr[:-1]
    force = force & same_mmsi
    if "_force_break" in df.columns:
        force = force | df["_force_break"].to_numpy(dtype=bool)
    df["_force_break"] = force


# ---------------------------------------------------------------------------
# 6. Land masking (pre-segmentation)
# ---------------------------------------------------------------------------

def mask_land(df: pd.DataFrame, land_shp: str | Path,
              track_breaks: bool = False) -> pd.DataFrame:
    """Remove AIS points that fall inside the land polygon.

    ``land_shp`` is a separate shapefile from the coastline used for
    wave-impact shore intersection — this one defines the land area
    to exclude AIS points from (e.g. Singapore main island, Jurong).

    If ``track_breaks`` is True, a ``_force_break`` column is added:
    the first surviving point after each removed land point is flagged
    so that ``segment_trajectories`` can split the trajectory at land
    crossings regardless of the time gap.

    Uses ``gpd.sjoin`` with an R-tree spatial index (O(N log M)).
    """
    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="mask_land")
    land_gdf = gpd.read_file(land_shp)
    if land_gdf.crs is None:
        raise ValueError(
            f"Land shapefile {land_shp!r} has no CRS.  Include a .prj file "
            "defining the coordinate reference system as EPSG:4326 (WGS 84)."
        )
    if land_gdf.crs.to_epsg() != 4326:
        land_gdf = land_gdf.to_crs("EPSG:4326")
    # Sort so that adjacency after removal is meaningful.
    df = df.sort_values(["mmsi", "obstime"]).reset_index(drop=True)
    pts_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts_gdf, land_gdf, predicate="within", how="left")
    in_land = joined["index_right"].notna().to_numpy()

    if track_breaks:
        _set_force_breaks(df, in_land)

    result = df[~in_land].reset_index(drop=True)
    spinner.done(rows=len(result))
    return result


# ---------------------------------------------------------------------------
# 14. Orchestrator
# ---------------------------------------------------------------------------

def filter_ais(
    csv_path: str | Path,
    land_shp: str | Path,
    coastline_shp: str | Path,
    gap_s: float = 180.0,
    max_velocity_knots: float = 36.0,
    max_acceleration_ms2: float = 10.0,
    interval_s: float = 30.0,
    max_draught_to_width: float = 1.0,
    min_speed_knots: float = 0.0,
    study_area_shp: str | Path | None = None,
    interp_method: str = "linear",
    low_sog_threshold_ms: float = 1.0,
    velocity_ratio_threshold: float = 2.0,
    speed_consistency_ratio: float = 0.5,
    # --- Depth-clearance check, applied AFTER interpolation/land masking ---
    bathy_path: str | Path | None = None,
    tide_dfs0_path: str | Path | None = None,
    tide_item: str | None = None,
    underkeel_margin_m: float = 1.0,
) -> pd.DataFrame:
    """Run the full AIS filtering pipeline and return a cleaned DataFrame.

    ``interp_method`` is ``"linear"``, ``"hermite"``, or ``"mixed"``.

    ``low_sog_threshold_ms`` feeds the kinematic error-coord check
    and the mixed interpolation method.

    ``velocity_ratio_threshold`` controls the velocity-consistency check in
    error-coord detection.

    When ``bathy_path`` is provided, the under-keel clearance check is folded
    into the post-interpolation tail of the pipeline (between the final
    ``mask_land`` pair and the last ``segment_trajectories``). A point with a
    bad ``draught`` value but a valid lon/lat in water is therefore still
    available as a position constraint for the interpolation step; only the
    final, interpolated frame is checked against bathymetry + tide.

    Pipeline order (see module docstring for full list):
    1-5.    Load, dedupe, uniformize, drop zero dims, drop invalid draught.
    6.      ``mask_land`` ×2 — remove raw land points before segmentation.
    7.      ``segment_trajectories`` — assign IDs to surviving points.
    8-9.    Kinematic cleaning (per segment): coord errors, speed errors.
    10.     ``filter_low_speed`` — strip SOG < min_speed_knots.
    11.     ``segment_trajectories`` — re-segment after low-speed removal.
    12.     ``interpolate_trajectories`` — uniform time-grid.
    13a.    ``mask_land`` (land) — drop post-interpolation land points.
    13b.    ``mask_land`` (coastline) — drop points inside coastline.
    13c.    ``assign_depth`` — add WaterDepth, drop under-keel violations
            (skipped when ``bathy_path is None``).
    13d.    ``segment_trajectories`` — re-segment after land + depth removals.
    14.     ``filter_study_area`` — optional polygon filter.
    """
    df = load_ais(csv_path)
    df = deduplicate(df)
    df = uniformize_vessel_info(df)
    df = remove_zero_dimensions(df)
    df = remove_invalid_draught(df, max_draught_to_width=max_draught_to_width)
    # 6. Remove raw land points — before segmentation.
    #    track_breaks=True → surviving points after a land gap get _force_break.
    df = mask_land(df, land_shp, track_breaks=True)
    df = mask_land(df, coastline_shp, track_breaks=True)
    # 7. Segment with force-break flags so land crossings start new segments.
    df = segment_trajectories(df, gap_s=gap_s, use_force_break=True)
    # 8-9. Per-segment kinematic cleaning.
    df = clean_error_coords(df, max_velocity_knots=max_velocity_knots,
                            low_sog_threshold_ms=low_sog_threshold_ms,
                            velocity_ratio_threshold=velocity_ratio_threshold)
    df = clean_error_speed(df, max_acceleration_ms2=max_acceleration_ms2,
                           speed_consistency_ratio=speed_consistency_ratio)
    # 10. Strip rows whose final SOG is below min_speed_knots
    # (clean_error_speed may have replaced SOG values).
    df = filter_low_speed(df, min_speed_knots=min_speed_knots)
    # 11. Re-segment after low-speed point removal — time-gap only (no
    #     force-break: we want interpolation to work within each segment).
    df = segment_trajectories(df, gap_s=gap_s)
    # 12. Interpolation to uniform time grid.
    df = interpolate_trajectories(df, interval_s=interval_s, method=interp_method,
                                  low_sog_threshold_ms=low_sog_threshold_ms)
    # 13. Post-interpolation cleanup. Land + depth checks all flag _force_break
    # on surviving points so the single segment_trajectories at the end picks
    # up every break in one pass.
    # 13a. Drop points on land and flag breaks.
    df = mask_land(df, land_shp, track_breaks=True)
    # 13b. Drop points inside coastline and flag breaks.
    df = mask_land(df, coastline_shp, track_breaks=True)
    # 13c. Add WaterDepth and drop points failing the under-keel clearance
    # check (skipped when no bathymetry is provided — e.g. unit tests).
    if bathy_path is not None:
        from aiswakepy.stages.depth import assign_depth
        df = assign_depth(
            df,
            bathy_path=bathy_path,
            tide_dfs0_path=tide_dfs0_path,
            tide_item=tide_item,
            underkeel_margin_m=underkeel_margin_m,
        )
    # 13d. Re-segment with force-break flags so land + depth gaps split tracks.
    df = segment_trajectories(df, gap_s=gap_s, use_force_break=True)
    # 14. Study-area filter — optional, last.
    df = filter_study_area(df, polygon_shp=study_area_shp)
    return df
