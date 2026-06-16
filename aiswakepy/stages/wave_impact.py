"""Stage 4 — Wave impact calculation.

For each wake event (port and starboard), cast a ray in the wake propagation
direction, find the intersection with the coastline, add the perpendicular
distance ``dist_perp`` as a column, then call the selected empirical formula
to compute wave height at the impact point.  Each formula provides its own
distance-decay law.

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
from aiswakepy.models.bhowmik import compute_bhowmik
from aiswakepy.models.blaauw import compute_blaauw
from aiswakepy.models.gates import compute_gates
from aiswakepy.models.kriebel import compute_kriebel
from aiswakepy.models.maynord import compute_maynord
from aiswakepy.models.pianc import compute_pianc
from aiswakepy.models.sorensen import compute_sorensen

# Maps formula name → (compute function, H-column name produced)
_FORMULA_REGISTRY: dict[str, tuple] = {
    "bhowmik":  (compute_bhowmik,  "H_Bhowmik"),
    "blaauw":   (compute_blaauw,   "H_Blaauw"),
    "gates":    (compute_gates,    "H_Gates"),
    "kriebel":  (compute_kriebel,  "H_Kriebel"),
    "maynord":  (compute_maynord,  "H_Maynord"),
    "pianc":    (compute_pianc,    "H_PIANC"),
    "sorensen": (compute_sorensen, "H_Sorensen"),
}


# ---------------------------------------------------------------------------
# Helpers for point-impact geometry
# ---------------------------------------------------------------------------

def _normalize_angle(deg: float | np.ndarray) -> float | np.ndarray:
    """Normalize angle(s) to the range (-180, 180]."""
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def _angular_lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate between two compass bearings handling wrap-around."""
    diff = _normalize_angle(b - a)
    return float(_normalize_angle(a + t * diff))


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def compute_point_impact(
    df_vessel: pd.DataFrame,
    point_lon: float,
    point_lat: float,
    g: float = 9.78,
    max_prop_m: float | None = None,
    bisect_tol_deg: float = 1e-3,
) -> pd.DataFrame:
    """Find all geometric wake arrivals at a fixed measurement point.

    For each pair of consecutive AIS positions within a trajectory segment,
    solve for the exact position along the segment where the vessel's wake
    direction points directly at the measurement point.  Returns the geometric
    record (lateral distance, propagation distance, arrival time) plus the
    vessel columns each empirical formula needs, so the caller can apply any
    subset of formulae to every geometric hit.  The wake-height formula itself
    is **not** applied here — that's the caller's job.

    Parameters
    ----------
    df_vessel:       Vessel-parameters DataFrame (output of
                     ``compute_vessel_params``).  Must contain columns:
                     longitude, latitude, obstime, WakeDirPort,
                     WakeDirStarboard, Theta, Tc, SOGms,
                     cog, segment_id, mmsi, width, length, Froude_D.
    point_lon:       Longitude of the measurement point (decimal degrees).
    point_lat:       Latitude of the measurement point (decimal degrees).
    g:               Gravitational acceleration (m/s²). Default 9.78.
    max_prop_m:      Maximum propagation distance (m). Trajectory segment pairs
                     where both endpoints are farther than this from the point
                     are skipped before the bisection search.  Good approximation
                     when the segment heading is roughly aligned with the track.
                     ``None`` disables the filter (default).
    bisect_tol_deg:  Angular convergence tolerance for bisection (degrees).

    Returns
    -------
    DataFrame with one row per geometric wake-arrival event.  Includes
    ``dist_perp`` so empirical formula functions can be called directly on it.
    """
    _OUT_COLS = [
        "MMSI", "PointLongitude", "PointLatitude", "WavePeriod",
        "PropDist_m", "DistPerp_m", "dist_perp", "DateTime", "ArrivalTime",
        "Froude_D", "VesselWidth", "VesselLength", "SOG", "Side", "segment_id",
        "SOGms", "WaterDepth", "length", "width", "draught",
        "block_coeff", "bow_entry_m", "displacement_m3",
    ]

    if df_vessel.empty:
        return pd.DataFrame(columns=_OUT_COLS)

    def _gcol(row, col: str) -> float:
        val = getattr(row, col, float("nan"))
        return float(val) if val is not None else float("nan")

    # Collect all intersection records before batch-computing wave heights
    hit_records: list[dict] = []
    hit_vessel_data: list[dict] = []   # vessel columns at t* for formula

    for seg_id, grp in df_vessel.groupby("segment_id", sort=False):
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

            # Pre-filter: skip pairs where both endpoints exceed max_prop_m.
            # Approximation — valid when the segment is short relative to the
            # propagation distance (vessel heading ≈ segment heading).
            if max_prop_m is not None:
                d0 = float(geodetic_distance(lon0, lat0, point_lon, point_lat))
                d1 = float(geodetic_distance(lon1, lat1, point_lon, point_lat))
                if min(d0, d1) > max_prop_m:
                    continue

            for side, wd0, wd1 in [
                ("port", float(r0.WakeDirPort),       float(r1.WakeDirPort)),
                ("stbd", float(r0.WakeDirStarboard),  float(r1.WakeDirStarboard)),
            ]:
                def f(t: float, _wd0: float = wd0, _wd1: float = wd1) -> float:
                    lon_t = _lerp(lon0, lon1, t)
                    lat_t = _lerp(lat0, lat1, t)
                    wd_t = _angular_lerp(_wd0, _wd1, t)
                    brg_t = float(geodetic_bearing(lon_t, lat_t, point_lon, point_lat))
                    return float(_normalize_angle(wd_t - brg_t))

                f0 = f(0.0)
                f1 = f(1.0)

                if f0 * f1 >= 0.0:
                    continue

                lo, hi = 0.0, 1.0
                _f0 = f0
                for _ in range(60):
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

                lon_s = _lerp(lon0, lon1, t_star)
                lat_s = _lerp(lat0, lat1, t_star)
                theta_s = _lerp(float(r0.Theta), float(r1.Theta), t_star)
                sogms_s = _lerp(float(r0.SOGms), float(r1.SOGms), t_star)
                tc_s    = _lerp(float(r0.Tc), float(r1.Tc), t_star)
                frd_s   = _lerp(float(r0.Froude_D), float(r1.Froude_D), t_star)
                sog_s   = _lerp(float(r0.sog), float(r1.sog), t_star)
                obs_s   = r0.obstime + pd.Timedelta(seconds=t_star * dt_span)

                prop_dist = float(geodetic_distance(lon_s, lat_s, point_lon, point_lat))

                theta_rad = math.radians(theta_s)
                dist_perp = prop_dist * math.sin(theta_rad)

                if dist_perp <= 0.0:
                    continue

                hit_records.append({
                    "MMSI": int(r0.mmsi),
                    "PointLongitude": point_lon,
                    "PointLatitude": point_lat,
                    "WavePeriod": tc_s,
                    "PropDist_m": prop_dist,
                    "DistPerp_m": dist_perp,
                    "DateTime": obs_s,
                    "Froude_D": frd_s,
                    "VesselWidth": float(r0.width),
                    "VesselLength": float(r0.length),
                    "SOG": sog_s,
                    "Side": side,
                    "segment_id": seg_id,
                })
                # Collect vessel columns needed by formula
                hit_vessel_data.append({
                    "SOGms":          sogms_s,
                    "Froude_D":        frd_s,
                    "WaterDepth":     _lerp(_gcol(r0, "WaterDepth"), _gcol(r1, "WaterDepth"), t_star),
                    "length":         float(r0.length),
                    "width":          float(r0.width),
                    "draught":        _gcol(r0, "draught"),
                    "block_coeff":    _gcol(r0, "block_coeff"),
                    "bow_entry_m":    _gcol(r0, "bow_entry_m"),
                    "displacement_m3": _gcol(r0, "displacement_m3"),
                    "dist_perp":      dist_perp,
                })

    if not hit_records:
        return pd.DataFrame(columns=_OUT_COLS)

    # Deep-water group velocity for travel-time computation
    tc_arr = np.array([r["WavePeriod"] for r in hit_records])
    # Floor of 0.01 m/s guards against division-by-zero when Tc is 0 or near-zero.
    c_g = np.maximum(g * tc_arr / (4.0 * math.pi), 0.01)
    prop_dist_arr = np.array([r["PropDist_m"] for r in hit_records])
    travel_s = prop_dist_arr / c_g

    records_out: list[dict] = []
    for idx, (rec, trav) in enumerate(zip(hit_records, travel_s)):
        obs_s = rec["DateTime"]
        arrival = obs_s + pd.Timedelta(seconds=float(trav))
        vd = hit_vessel_data[idx]
        records_out.append({
            **rec,
            "ArrivalTime": arrival,
            "SOGms":           vd["SOGms"],
            "WaterDepth":      vd["WaterDepth"],
            "length":          vd["length"],
            "width":           vd["width"],
            "draught":         vd["draught"],
            "block_coeff":     vd["block_coeff"],
            "bow_entry_m":     vd["bow_entry_m"],
            "displacement_m3": vd["displacement_m3"],
            "dist_perp":       vd["dist_perp"],
        })

    if not records_out:
        return pd.DataFrame(columns=_OUT_COLS)

    return pd.DataFrame(records_out).reset_index(drop=True)


_ANIMATION_RAY_COLS = [
    "MMSI", "segment_id", "SourceLongitude", "SourceLatitude",
    "EndLongitude", "EndLatitude", "SourceTime", "Side",
    "Distance_m", "ReachedShore", "WakeDirection_deg", "Theta_deg",
    "SOGms", "PhaseSpeed_mps", "GroupSpeed_mps",
    "CuspAngle_deg", "TransverseSpeed_mps",
]


def kelvin_cusp_angle(theta_deg: float | np.ndarray) -> float | np.ndarray:
    """Return the Kelvin cusp/envelope angle for a divergent-wave angle.

    ``theta_deg`` is the divergent-wave propagation angle relative to vessel
    heading. In the deep-water Kelvin case theta=asin(1/sqrt(3)), this returns
    atan(1/(2*sqrt(2))) ≈ 19.47 degrees.
    """
    theta = np.radians(theta_deg)
    numerator = 0.5 * np.sin(theta) * np.cos(theta)
    denominator = 1.0 - 0.5 * np.cos(theta) ** 2
    return np.degrees(np.arctan2(numerator, denominator))


def compute_wave_impact_with_rays(
    df_vessel: pd.DataFrame,
    coastline_shp: str | Path,
    formula: str = "kriebel",
    max_propagation_m: float = 2000.0,
    wake_cutoff_m: float = 0.01,
    g: float = 9.78,
    rho: float = 1026.0,
    **formula_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute shoreline impacts and the exact rays used by the calculation.

    The ray frame contains both coastline hits and misses. Misses terminate at
    ``max_propagation_m`` so the frontend can animate the same geometry that
    was evaluated by this stage.
    """
    if formula not in _FORMULA_REGISTRY:
        raise ValueError(
            f"Unknown formula {formula!r}. Supported: {list(_FORMULA_REGISTRY)}"
        )
    compute_fn, h_col = _FORMULA_REGISTRY[formula]

    _OUT_COLS = [
        "MMSI", "ShLongitude", "ShLatitude", "WaveHeight", "WavePeriod",
        "E_max", "E_tot", "DistLoc_km", "DateTime", "Froude_D",
        "VesselLongitude", "VesselLatitude", "VesselCOG", "VesselDraught",
        "VesselWidth", "VesselLength", "SOG", "Side", "segment_id", "typecargo",
    ]

    if df_vessel.empty:
        return (
            pd.DataFrame(columns=_OUT_COLS),
            pd.DataFrame(columns=_ANIMATION_RAY_COLS),
        )

    coastline = load_coastline(coastline_shp)
    strtree, segments = build_coastline_index(coastline)

    lons = df_vessel["longitude"].to_numpy()
    lats = df_vessel["latitude"].to_numpy()
    dist = np.full(len(lons), max_propagation_m)
    port_lon2, port_lat2 = forward_point(
        lons, lats, df_vessel["WakeDirPort"].to_numpy(), dist
    )
    stbd_lon2, stbd_lat2 = forward_point(
        lons, lats, df_vessel["WakeDirStarboard"].to_numpy(), dist
    )

    from aiswakepy._progress import Spinner
    total = len(df_vessel)
    spinner = Spinner(total=total, desc="Wave impact")

    hit_records: list[dict] = []
    hit_vessel_rows: list[int] = []
    hit_dist_perp: list[float] = []
    ray_records: list[dict] = []

    for i, row in enumerate(df_vessel.itertuples(index=False)):
        spinner.update(i + 1)
        theta_deg = float(row.Theta)
        phase_speed = float(row.SOGms) * math.cos(math.radians(theta_deg))
        group_speed = 0.5 * phase_speed
        cusp_angle_deg = float(kelvin_cusp_angle(theta_deg))
        transverse_speed = float(row.SOGms) * math.sin(math.radians(cusp_angle_deg))
        for side, wake_dir, limit_lon, limit_lat in [
            ("port", float(row.WakeDirPort), float(port_lon2[i]), float(port_lat2[i])),
            ("stbd", float(row.WakeDirStarboard), float(stbd_lon2[i]), float(stbd_lat2[i])),
        ]:
            ray = LineString([
                (row.longitude, row.latitude),
                (limit_lon, limit_lat),
            ])
            hit = find_shore_intersection_indexed(ray, strtree, segments)
            reached_shore = hit is not None
            if reached_shore:
                end_lon, end_lat, distance_m = hit
            else:
                end_lon, end_lat, distance_m = (
                    limit_lon, limit_lat, float(max_propagation_m)
                )
            ray_records.append({
                "MMSI": int(row.mmsi),
                "segment_id": int(row.segment_id),
                "SourceLongitude": float(row.longitude),
                "SourceLatitude": float(row.latitude),
                "EndLongitude": float(end_lon),
                "EndLatitude": float(end_lat),
                "SourceTime": row.obstime,
                "Side": side,
                "Distance_m": float(distance_m),
                "ReachedShore": bool(reached_shore),
                "WakeDirection_deg": wake_dir,
                "Theta_deg": theta_deg,
                "SOGms": float(row.SOGms),
                "PhaseSpeed_mps": phase_speed,
                "GroupSpeed_mps": group_speed,
                "CuspAngle_deg": cusp_angle_deg,
                "TransverseSpeed_mps": transverse_speed,
            })
            if not reached_shore:
                continue

            sh_lon, sh_lat = end_lon, end_lat
            theta_rad = np.radians(row.Theta)
            dist_perp = distance_m * np.sin(theta_rad)
            if dist_perp <= 0:
                continue

            hit_records.append({
                "MMSI":             int(row.mmsi),
                "ShLongitude":      sh_lon,
                "ShLatitude":       sh_lat,
                "WavePeriod":       row.Tc,
                "DistLoc_km":       dist_perp / 1000.0,
                "DateTime":         row.obstime,
                "Froude_D":         row.Froude_D,
                "VesselLongitude":  row.longitude,
                "VesselLatitude":   row.latitude,
                "VesselCOG":        row.cog,
                "VesselDraught":    row.draught,
                "VesselWidth":      row.width,
                "VesselLength":     row.length,
                "SOG":              row.sog,
                "Side":             side,
                "segment_id":       int(row.segment_id),
                "typecargo":        int(getattr(row, "typecargo", 0)),
            })
            hit_vessel_rows.append(i)
            hit_dist_perp.append(dist_perp)

    spinner.done(total)
    rays = pd.DataFrame(ray_records, columns=_ANIMATION_RAY_COLS)
    if not hit_records:
        return pd.DataFrame(columns=_OUT_COLS), rays

    hit_df = df_vessel.iloc[hit_vessel_rows].copy().reset_index(drop=True)
    hit_df["dist_perp"] = hit_dist_perp
    h_series = compute_fn(hit_df, g=g, **formula_kwargs)

    records_out: list[dict] = []
    for rec, h_val in zip(hit_records, h_series):
        if np.isnan(h_val) or h_val < wake_cutoff_m:
            continue
        T_val = rec["WavePeriod"]
        E_max = rho * g * g * h_val * h_val * T_val * T_val / (16.0 * math.pi)
        E_tot = 10.8 * E_max ** 0.82
        records_out.append({
            **rec,
            "WaveHeight": h_val,
            "E_max": E_max,
            "E_tot": E_tot,
        })

    impacts = (
        pd.DataFrame(records_out).reset_index(drop=True)
        if records_out else pd.DataFrame(columns=_OUT_COLS)
    )
    return impacts, rays


def compute_wave_impact(
    df_vessel: pd.DataFrame,
    coastline_shp: str | Path,
    formula: str = "kriebel",
    max_propagation_m: float = 2000.0,
    wake_cutoff_m: float = 0.01,
    g: float = 9.78,
    rho: float = 1026.0,
    **formula_kwargs,
) -> pd.DataFrame:
    """Compute wave height at shoreline for each wake event.

    The stage proceeds in three steps:
    1. **Geometry** — cast a ray from each vessel position along the wake
       direction (port and starboard) and find the coastline intersection.
       Adds ``dist_perp`` (perpendicular distance from sailing track) to the
       working DataFrame.
    2. **Formula** — calls the selected empirical formula (which reads
       ``dist_perp`` from the DataFrame) to compute wave height at the shore.
       Each formula includes its own distance-decay law.
    3. **Filtering** — discards events below ``wake_cutoff_m``.

    Parameters
    ----------
    df_vessel:         Vessel-parameters DataFrame (output of
                       ``compute_vessel_params``).
    coastline_shp:     Path to coastline shapefile.
    formula:           Empirical wake model to use (default ``"kriebel"``).
    max_propagation_m: Maximum ray length (m).
    wake_cutoff_m:     Minimum H threshold (m); events below are discarded.
    g:                 Gravitational acceleration (m/s²).
    rho:               Water density (kg/m³). Default 1026.
    **formula_kwargs:  Extra keyword arguments forwarded to the formula function.

    Returns
    -------
    DataFrame with one row per shoreline intersection (port and/or starboard).
    Columns: MMSI, ShLongitude, ShLatitude, WaveHeight, WavePeriod,
             E_max, E_tot, DistLoc_km, DateTime, Froude_D, VesselWidth,
             VesselLength, SOG, Side.

    Wave energy columns use deep-water linear theory:
        E_max = ρ g² H² T² / (16π)         [J/m]
        E_tot = 10.8 · E_max^0.82          [J/m]  (empirical scaling)
    """
    impacts, _ = compute_wave_impact_with_rays(
        df_vessel=df_vessel,
        coastline_shp=coastline_shp,
        formula=formula,
        max_propagation_m=max_propagation_m,
        wake_cutoff_m=wake_cutoff_m,
        g=g,
        rho=rho,
        **formula_kwargs,
    )
    return impacts
