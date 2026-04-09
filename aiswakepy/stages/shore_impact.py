"""Stage 4 — Shore impact calculation.

For each wake event (port and starboard), cast a ray in the wake propagation
direction, find the intersection with the coastline, and compute the decayed
wave height at the shore using the Kriebel distance-decay formula.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import track
from shapely.geometry import LineString

from aiswakepy.geo.coastline import (
    build_coastline_index,
    find_shore_intersection_indexed,
    load_coastline,
)
from aiswakepy.geo.geodesy import forward_point


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
    port_lon2, port_lat2 = forward_point(
        lons, lats, df_wave["WakeDirPort"].to_numpy(), max_propagation_m
    )
    stbd_lon2, stbd_lat2 = forward_point(
        lons, lats, df_wave["WakeDirStarboard"].to_numpy(), max_propagation_m
    )

    records = []
    for i, row in track(
        enumerate(df_wave.itertuples(index=False)),
        total=len(df_wave),
        description="Shore impact",
        transient=True,
    ):
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

    if not records:
        return pd.DataFrame(columns=[
            "MMSI", "ShLongitude", "ShLatitude", "WaveHeight", "WavePeriod",
            "DistLoc_km", "DateTime", "FroudeM", "VesselWidth", "VesselLength",
            "SOG", "Side",
        ])

    return pd.DataFrame(records).reset_index(drop=True)
