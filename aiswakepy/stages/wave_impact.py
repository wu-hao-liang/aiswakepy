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
    formula: str = "kriebel",
    g: float = 9.78,
    bisect_tol_deg: float = 1e-3,
    **formula_kwargs,
) -> pd.DataFrame:
    """Find all wake arrivals at a fixed measurement point (e.g. an OSSI gauge).

    For each pair of consecutive AIS positions within a trajectory segment,
    solve for the exact position along the segment where the vessel's wake
    direction points directly at the measurement point.  At that position the
    perpendicular propagation distance from the sailing line is added as
    ``dist_perp`` and the selected empirical formula computes the wave height.

    Parameters
    ----------
    df_vessel:       Vessel-parameters DataFrame (output of
                     ``compute_vessel_params``).  Must contain columns:
                     longitude, latitude, obstime, WakeDirPort,
                     WakeDirStarboard, Theta, Tc, SOGms, LengthWL,
                     cog, segment_id, mmsi, width, length, FroudeD.
    point_lon:       Longitude of the measurement point (decimal degrees).
    point_lat:       Latitude of the measurement point (decimal degrees).
    formula:         Empirical wake model to use (default ``"kriebel"``).
    g:               Gravitational acceleration (m/s²). Default 9.78.
    bisect_tol_deg:  Angular convergence tolerance for bisection (degrees).
    **formula_kwargs: Extra keyword arguments forwarded to the formula function.

    Returns
    -------
    DataFrame with one row per wake-arrival event.
    """
    if formula not in _FORMULA_REGISTRY:
        raise ValueError(
            f"Unknown formula {formula!r}. Supported: {list(_FORMULA_REGISTRY)}"
        )
    compute_fn, h_col = _FORMULA_REGISTRY[formula]

    _OUT_COLS = [
        "MMSI", "PointLongitude", "PointLatitude", "WaveHeight", "WavePeriod",
        "PropDist_m", "DistPerp_m", "DateTime", "ArrivalTime",
        "FroudeD", "VesselWidth", "VesselLength", "SOG", "Side", "segment_id",
        "SOGms", "WaterDepth", "length", "width", "draught",
        "bow_entry_m", "displacement_m3",
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
                lwl_s   = _lerp(float(r0.LengthWL), float(r1.LengthWL), t_star)
                tc_s    = _lerp(float(r0.Tc), float(r1.Tc), t_star)
                frd_s   = _lerp(float(r0.FroudeD), float(r1.FroudeD), t_star)
                sog_s   = _lerp(float(r0.sog), float(r1.sog), t_star)
                obs_s   = r0.obstime + pd.Timedelta(seconds=t_star * dt_span)

                prop_dist = float(geodetic_distance(lon_s, lat_s, point_lon, point_lat))

                theta_rad = math.radians(theta_s)
                dist_perp = prop_dist * math.sin(theta_rad)

                if lwl_s <= 0.0 or dist_perp <= 0.0:
                    continue

                hit_records.append({
                    "MMSI": int(r0.mmsi),
                    "PointLongitude": point_lon,
                    "PointLatitude": point_lat,
                    "WavePeriod": tc_s,
                    "PropDist_m": prop_dist,
                    "DistPerp_m": dist_perp,
                    "DateTime": obs_s,
                    "FroudeD": frd_s,
                    "VesselWidth": float(r0.width),
                    "VesselLength": float(r0.length),
                    "SOG": sog_s,
                    "Side": side,
                    "segment_id": seg_id,
                })
                # Collect vessel columns needed by formula
                hit_vessel_data.append({
                    "SOGms":          sogms_s,
                    "LengthWL":       lwl_s,
                    "FroudeD":        frd_s,
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

    # Batch compute wave heights with selected formula
    hit_df = pd.DataFrame(hit_vessel_data)
    h_series = compute_fn(hit_df, g=g, **formula_kwargs)

    # Deep-water group velocity for travel-time computation
    tc_arr = np.array([r["WavePeriod"] for r in hit_records])
    c_g = np.maximum(g * tc_arr / (4.0 * math.pi), 0.01)
    prop_dist_arr = np.array([r["PropDist_m"] for r in hit_records])
    travel_s = prop_dist_arr / c_g

    records_out: list[dict] = []
    for idx, (rec, h_val, trav) in enumerate(zip(hit_records, h_series, travel_s)):
        if np.isnan(h_val) or h_val <= 0:
            continue
        obs_s = rec["DateTime"]
        arrival = obs_s + pd.Timedelta(seconds=float(trav))
        vd = hit_vessel_data[idx]
        records_out.append({
            **rec,
            "WaveHeight": h_val,
            "ArrivalTime": arrival,
            "SOGms":           vd["SOGms"],
            "WaterDepth":      vd["WaterDepth"],
            "length":          vd["length"],
            "width":           vd["width"],
            "draught":         vd["draught"],
            "bow_entry_m":     vd["bow_entry_m"],
            "displacement_m3": vd["displacement_m3"],
        })

    if not records_out:
        return pd.DataFrame(columns=_OUT_COLS)

    return pd.DataFrame(records_out).reset_index(drop=True)


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
             DistLoc_km, DateTime, FroudeD, VesselWidth, VesselLength,
             SOG, Side.
    """
    if formula not in _FORMULA_REGISTRY:
        raise ValueError(
            f"Unknown formula {formula!r}. Supported: {list(_FORMULA_REGISTRY)}"
        )
    compute_fn, h_col = _FORMULA_REGISTRY[formula]

    _OUT_COLS = [
        "MMSI", "ShLongitude", "ShLatitude", "WaveHeight", "WavePeriod",
        "DistLoc_km", "DateTime", "FroudeD", "VesselWidth", "VesselLength",
        "SOG", "Side",
    ]

    if df_vessel.empty:
        return pd.DataFrame(columns=_OUT_COLS)

    coastline = load_coastline(coastline_shp)
    strtree, segments = build_coastline_index(coastline)

    # Vectorize ray endpoint computation for both sides
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

    # Step 1: Geometry — collect all ray-coastline hits
    hit_records: list[dict] = []
    hit_vessel_rows: list[int] = []
    hit_dist_perp: list[float] = []

    for i, row in enumerate(df_vessel.itertuples(index=False)):
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

            # Perpendicular distance from the sailing track to the intersection:
            #   dist_perp = dist_ray * sin(Theta)
            # This is the lateral distance variable in all empirical formulae.
            theta_rad = np.radians(row.Theta)
            dist_perp = dist_m * np.sin(theta_rad)

            if dist_perp <= 0:
                continue

            hit_records.append({
                "MMSI":          int(row.mmsi),
                "ShLongitude":   sh_lon,
                "ShLatitude":    sh_lat,
                "WavePeriod":    row.Tc,
                "DistLoc_km":    dist_perp / 1000.0,
                "DateTime":      row.obstime,
                "FroudeD":       row.FroudeD,
                "VesselWidth":   row.width,
                "VesselLength":  row.length,
                "SOG":           row.sog,
                "Side":          side,
            })
            hit_vessel_rows.append(i)
            hit_dist_perp.append(dist_perp)

    spinner.done(total)

    if not hit_records:
        return pd.DataFrame(columns=_OUT_COLS)

    # Step 2: Formula — batch compute wave heights
    # Build a slice of df_vessel for all hits (preserving all vessel columns)
    hit_df = df_vessel.iloc[hit_vessel_rows].copy().reset_index(drop=True)
    hit_df["dist_perp"] = hit_dist_perp

    h_series = compute_fn(hit_df, g=g, **formula_kwargs)

    # Step 3: Filtering and output
    records_out: list[dict] = []
    for rec, h_val in zip(hit_records, h_series):
        if np.isnan(h_val) or h_val < wake_cutoff_m:
            continue
        records_out.append({**rec, "WaveHeight": h_val})

    if not records_out:
        return pd.DataFrame(columns=_OUT_COLS)

    return pd.DataFrame(records_out).reset_index(drop=True)
