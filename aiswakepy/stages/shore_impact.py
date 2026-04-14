"""Stage 4 — Shore impact calculation.

For each wake event (port and starboard), cast a ray in the wake propagation
direction, find the intersection with the coastline, and compute the decayed
wave height at the shore using the Kriebel distance-decay formula.

Also provides ``compute_point_impact`` which finds wake arrivals at a fixed
measurement point (e.g. an OSSI wave gauge) by solving for the exact position
along each trajectory segment where the wake direction points at the gauge.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from aiswakepy.geo.coastline import (
    build_coastline_index,
    find_shore_intersection_indexed,
    load_coastline,
)
from aiswakepy.geo.geodesy import forward_point, geodetic_bearing, geodetic_distance


# ---------------------------------------------------------------------------
# Helpers for point-impact geometry
# ---------------------------------------------------------------------------

def _normalize_angle(deg: float | np.ndarray) -> float | np.ndarray:
    """Normalize angle(s) to the range (-180, 180]."""
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def _angular_lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate between two compass bearings handling wrap-around.

    The interpolation takes the shortest angular path from *a* to *b*.
    """
    diff = _normalize_angle(b - a)
    return float(_normalize_angle(a + t * diff))


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)



def compute_point_impact(
    df_wave: pd.DataFrame,
    point_lon: float,
    point_lat: float,
    g: float = 9.78,
    bisect_tol_deg: float = 1e-3,
) -> pd.DataFrame:
    """Find all wake arrivals at a fixed measurement point (e.g. an OSSI gauge).

    For each pair of consecutive AIS positions within a trajectory segment,
    solve for the exact position along the segment where the vessel's wake
    direction points directly at the measurement point.  At that position the
    perpendicular propagation distance from the sailing line is computed and
    the Kriebel wave-height decay formula is applied.

    Parameters
    ----------
    df_wave:         Wave-parameters DataFrame (output of ``compute_wave_params``).
                     Must contain columns: longitude, latitude, obstime,
                     WakeDirPort, WakeDirStarboard, Theta, Tc, BF, LengthWL,
                     SOGms, cog, segment_id, mmsi, width, length, FroudeM, sog.
    point_lon:       Longitude of the measurement point (decimal degrees).
    point_lat:       Latitude of the measurement point (decimal degrees).
    g:               Gravitational acceleration (m/s²). Default 9.78.
    bisect_tol_deg:  Angular convergence tolerance for bisection (degrees).

    Returns
    -------
    DataFrame with one row per wake-arrival event, columns:
        MMSI, PointLongitude, PointLatitude, WaveHeight, WavePeriod,
        PropDist_m, DistPerp_m, DateTime, ArrivalTime,
        FroudeM, VesselWidth, VesselLength, SOG, Side, segment_id.
    """
    _OUT_COLS = [
        "MMSI", "PointLongitude", "PointLatitude", "WaveHeight", "WavePeriod",
        "PropDist_m", "DistPerp_m", "DateTime", "ArrivalTime",
        "FroudeM", "VesselWidth", "VesselLength", "SOG", "Side", "segment_id",
        # Additional columns required by empirical models in compare_empirical.py
        "SOGms", "WaterDepth", "length", "width", "draught",
        "bow_entry_m", "displacement_m3",
    ]

    if df_wave.empty:
        return pd.DataFrame(columns=_OUT_COLS)

    records: list[dict] = []

    def _gcol(row, col: str) -> float:
        """Safely read an optional column from an itertuples row."""
        val = getattr(row, col, float("nan"))
        return float(val) if val is not None else float("nan")

    for seg_id, grp in df_wave.groupby("segment_id", sort=False):
        grp = grp.sort_values("obstime")
        rows = list(grp.itertuples(index=False))
        if len(rows) < 2:
            continue

        for i in range(len(rows) - 1):
            r0 = rows[i]
            r1 = rows[i + 1]

            lon0, lat0 = float(r0.longitude), float(r0.latitude)
            lon1, lat1 = float(r1.longitude), float(r1.latitude)
            dt_span = (r1.obstime - r0.obstime).total_seconds()

            # Check both port and starboard independently.
            # The sign-change test is the rigorous criterion — do not pre-filter
            # by side.  A vessel that is unberthing or turning may have a wake on
            # the "unexpected" side reaching the gauge; checking both sides handles
            # this correctly.  The wrong side will simply have no sign change and
            # will be skipped at zero cost.
            for side, wd0, wd1 in [
                ("port", float(r0.WakeDirPort),       float(r1.WakeDirPort)),
                ("stbd", float(r0.WakeDirStarboard),  float(r1.WakeDirStarboard)),
            ]:
                # Angular mismatch function f(t) = wake_dir(t) - bearing_to_point(t).
                # A root (f=0) means the wake ray points exactly at the gauge.
                def f(t: float, _wd0: float = wd0, _wd1: float = wd1) -> float:
                    lon_t = _lerp(lon0, lon1, t)
                    lat_t = _lerp(lat0, lat1, t)
                    wd_t = _angular_lerp(_wd0, _wd1, t)
                    brg_t = float(geodetic_bearing(lon_t, lat_t, point_lon, point_lat))
                    return float(_normalize_angle(wd_t - brg_t))

                f0 = f(0.0)
                f1 = f(1.0)

                # Skip if no sign change — no root in this segment for this side
                if f0 * f1 >= 0.0:
                    continue

                # Bisection to find t* where f(t*) ≈ 0
                lo, hi = 0.0, 1.0
                _f0 = f0
                for _ in range(60):  # up to ~1e-18 precision; exits early on tol
                    mid = 0.5 * (lo + hi)
                    fm = f(mid)
                    if abs(fm) < bisect_tol_deg:
                        break
                    if _f0 * fm < 0.0:
                        hi = mid
                    else:
                        lo = mid
                        _f0 = fm
                t_star = 0.5 * (lo + hi)

                # Interpolate all vessel parameters at t*
                lon_s = _lerp(lon0, lon1, t_star)
                lat_s = _lerp(lat0, lat1, t_star)
                theta_s = _lerp(float(r0.Theta), float(r1.Theta), t_star)
                bf_s = _lerp(float(r0.BF), float(r1.BF), t_star)
                lwl_s = _lerp(float(r0.LengthWL), float(r1.LengthWL), t_star)
                sogms_s = _lerp(float(r0.SOGms), float(r1.SOGms), t_star)
                tc_s = _lerp(float(r0.Tc), float(r1.Tc), t_star)
                frm_s = _lerp(float(r0.FroudeM), float(r1.FroudeM), t_star)
                sog_s = _lerp(float(r0.sog), float(r1.sog), t_star)
                obs_s = r0.obstime + pd.Timedelta(seconds=t_star * dt_span)

                # Propagation distance from intersection point to gauge
                prop_dist = float(geodetic_distance(lon_s, lat_s, point_lon, point_lat))

                # Lateral (perpendicular) distance from sailing line
                theta_rad = math.radians(theta_s)
                dist_perp = prop_dist * math.sin(theta_rad)

                if lwl_s <= 0.0 or dist_perp <= 0.0:
                    continue

                # Kriebel wave height at the gauge
                h_point = (
                    bf_s
                    * (dist_perp / lwl_s) ** (-1.0 / 3.0)
                    / g
                    * sogms_s ** 2
                )

                # Deep-water group velocity: c_g = g*Tc/(4π)
                c_g = g * tc_s / (4.0 * math.pi)
                if c_g < 0.01:
                    c_g = 0.01
                travel_s = prop_dist / c_g
                arrival_time = obs_s + pd.Timedelta(seconds=travel_s)

                records.append({
                    "MMSI": int(r0.mmsi),
                    "PointLongitude": point_lon,
                    "PointLatitude": point_lat,
                    "WaveHeight": h_point,
                    "WavePeriod": tc_s,
                    "PropDist_m": prop_dist,
                    "DistPerp_m": dist_perp,
                    "DateTime": obs_s,
                    "ArrivalTime": arrival_time,
                    "FroudeM": frm_s,
                    "VesselWidth": float(r0.width),
                    "VesselLength": float(r0.length),
                    "SOG": sog_s,
                    "Side": side,
                    "segment_id": seg_id,
                    # Model input columns (vessel-fixed; use endpoint values)
                    "SOGms": sogms_s,
                    "WaterDepth": _lerp(_gcol(r0, "WaterDepth"), _gcol(r1, "WaterDepth"), t_star),
                    "length": float(r0.length),
                    "width": float(r0.width),
                    "draught": _gcol(r0, "draught"),
                    "bow_entry_m": _gcol(r0, "bow_entry_m"),
                    "displacement_m3": _gcol(r0, "displacement_m3"),
                })

    if not records:
        return pd.DataFrame(columns=_OUT_COLS)

    return pd.DataFrame(records).reset_index(drop=True)


def compute_shore_impact(
    df_wave: pd.DataFrame,
    coastline_shp: str | Path,
    max_propagation_m: float = 2000.0,
    wake_cutoff_m: float = 0.01,
    g: float = 9.78,
) -> pd.DataFrame:
    """Compute wave height at shoreline for each wake event.

    Parameters
    ----------
    df_wave:           Wave parameters DataFrame (output of compute_wave_params).
    coastline_shp:     Path to coastline shapefile.
    max_propagation_m: Maximum ray length (m).
    wake_cutoff_m:     Minimum H_shore threshold (m); events below this are discarded.
    g:                 Gravitational acceleration (m/s²).

    Returns
    -------
    DataFrame with one row per shoreline intersection (port and/or starboard),
    columns: MMSI, ShLongitude, ShLatitude, WaveHeight, WavePeriod,
             DistLoc_km, DateTime, FroudeM, VesselWidth, VesselLength,
             SOG, Side.
    """
    if df_wave.empty:
        return pd.DataFrame(columns=[
            "MMSI", "ShLongitude", "ShLatitude", "WaveHeight", "WavePeriod",
            "DistLoc_km", "DateTime", "FroudeM", "VesselWidth", "VesselLength",
            "SOG", "Side",
        ])

    coastline = load_coastline(coastline_shp)
    strtree, segments = build_coastline_index(coastline)

    # Vectorize ray endpoint computation for both sides at once
    lons = df_wave["longitude"].to_numpy()
    lats = df_wave["latitude"].to_numpy()
    dist = np.full(len(lons), max_propagation_m)
    port_lon2, port_lat2 = forward_point(
        lons, lats, df_wave["WakeDirPort"].to_numpy(), dist
    )
    stbd_lon2, stbd_lat2 = forward_point(
        lons, lats, df_wave["WakeDirStarboard"].to_numpy(), dist
    )

    from aiswakepy._progress import Spinner
    total = len(df_wave)
    spinner = Spinner(total=total, desc="Shore impact")

    records = []
    for i, row in enumerate(df_wave.itertuples(index=False)):
        spinner.update(i + 1)
        for side, end_lon, end_lat in [
            ("port", float(port_lon2[i]), float(port_lat2[i])),
            ("stbd", float(stbd_lon2[i]), float(stbd_lat2[i])),
        ]:
            ray = LineString([(row.longitude, row.latitude), (end_lon, end_lat)])
            hit = find_shore_intersection_indexed(ray, strtree, segments)
            if hit is None:
                continue

            sh_lon, sh_lat, dist_m = hit

            # Lateral (perpendicular) distance from the sailing track to the
            # shore intersection point.  The wake ray travels at angle Theta
            # off the vessel heading, so the transverse component is:
            #   y = dist_ray * sin(Theta)
            # This is the distance variable in the Kriebel decay formula.
            # DistLoc_km in the output retains the full ray distance.
            theta_rad = np.radians(row.Theta)
            dist_perp = dist_m * np.sin(theta_rad)

            # Kriebel distance-decay: H = BF * (y / L_WL)^(-1/3) * V^2 / g
            l_wl = row.LengthWL
            if l_wl <= 0 or dist_perp <= 0:
                continue

            h_shore = (
                row.BF
                * (dist_perp / l_wl) ** (-1.0 / 3.0)
                / g
                * row.SOGms ** 2
            )

            if h_shore < wake_cutoff_m:
                continue

            records.append({
                "MMSI": int(row.mmsi),
                "ShLongitude": sh_lon,
                "ShLatitude": sh_lat,
                "WaveHeight": h_shore,
                "WavePeriod": row.Tc,
                "DistLoc_km": dist_perp / 1000.0,
                "DateTime": row.obstime,
                "FroudeM": row.FroudeM,
                "VesselWidth": row.width,
                "VesselLength": row.length,
                "SOG": row.sog,
                "Side": side,
            })

    spinner.done(total)

    if not records:
        return pd.DataFrame(columns=[
            "MMSI", "ShLongitude", "ShLatitude", "WaveHeight", "WavePeriod",
            "DistLoc_km", "DateTime", "FroudeM", "VesselWidth", "VesselLength",
            "SOG", "Side",
        ])

    return pd.DataFrame(records).reset_index(drop=True)
