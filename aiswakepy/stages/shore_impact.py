"""Stage 4 — Shore impact calculation.

For each wake event (port and starboard), cast a ray in the wake propagation
direction, find the intersection with the coastline, and compute the decayed
wave height at the shore using the Kriebel distance-decay formula.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aiswakepy.geo.coastline import build_ray, find_shore_intersection, load_coastline
from aiswakepy.geo.geodesy import geodetic_distance


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
    coastline = load_coastline(coastline_shp)

    records = []
    for _, row in df_wave.iterrows():
        for side, bearing_col in [("port", "WakeDirPort"), ("stbd", "WakeDirStarboard")]:
            bearing = row[bearing_col]
            ray = build_ray(
                row["longitude"], row["latitude"],
                bearing, max_propagation_m,
            )
            hit = find_shore_intersection(ray, coastline)
            if hit is None:
                continue

            sh_lon, sh_lat, dist_m = hit

            # Lateral (perpendicular) distance from the sailing track to the
            # shore intersection point.  The wake ray travels at angle Theta
            # off the vessel heading, so the transverse component is:
            #   y = dist_ray * sin(Theta)
            # This is the distance variable in the Kriebel decay formula.
            # DistLoc_km in the output retains the full ray distance.
            theta_rad = np.radians(row["Theta"])
            dist_perp = dist_m * np.sin(theta_rad)

            # Kriebel distance-decay: H = BF * (y / L_WL)^(-1/3) * V^2 / g
            l_wl = row["LengthWL"]
            if l_wl <= 0 or dist_perp <= 0:
                continue

            h_shore = (
                row["BF"]
                * (dist_perp / l_wl) ** (-1.0 / 3.0)
                / g
                * row["SOGms"] ** 2
            )

            if h_shore < wake_cutoff_m:
                continue

            records.append({
                "MMSI": int(row["mmsi"]),
                "ShLongitude": sh_lon,
                "ShLatitude": sh_lat,
                "WaveHeight": h_shore,
                "WavePeriod": row["Tc"],
                "DistLoc_km": dist_perp / 1000.0,
                "DateTime": row["obstime"],
                "FroudeM": row["FroudeM"],
                "VesselWidth": row["width"],
                "VesselLength": row["length"],
                "SOG": row["sog"],
                "Side": side,
            })

    if not records:
        return pd.DataFrame(columns=[
            "MMSI", "ShLongitude", "ShLatitude", "WaveHeight", "WavePeriod",
            "DistLoc_km", "DateTime", "FroudeM", "VesselWidth", "VesselLength",
            "SOG", "Side",
        ])

    return pd.DataFrame(records).reset_index(drop=True)
