"""Geodetic utility functions using WGS84 ellipsoid via pyproj."""

from __future__ import annotations

import numpy as np
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")


def geodetic_distance(
    lon1: float | np.ndarray,
    lat1: float | np.ndarray,
    lon2: float | np.ndarray,
    lat2: float | np.ndarray,
) -> float | np.ndarray:
    """Geodetic distance in metres between point(s) (lon1, lat1) and (lon2, lat2).

    Accepts scalars or numpy arrays (vectorised).
    """
    _, _, dist = _GEOD.inv(lon1, lat1, lon2, lat2)
    return dist


def geodetic_bearing(
    lon1: float | np.ndarray,
    lat1: float | np.ndarray,
    lon2: float | np.ndarray,
    lat2: float | np.ndarray,
) -> float | np.ndarray:
    """Forward azimuth in degrees (0 = north, 90 = east) from point 1 to point 2.

    Accepts scalars or numpy arrays (vectorised).
    """
    fwd_az, _, _ = _GEOD.inv(lon1, lat1, lon2, lat2)
    return fwd_az


def forward_point(
    lon: float | np.ndarray,
    lat: float | np.ndarray,
    bearing_deg: float | np.ndarray,
    distance_m: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return (lon2, lat2) reached by travelling distance_m along bearing_deg from (lon, lat).

    Accepts scalars or numpy arrays (vectorised).
    Returns (lon2, lat2).
    """
    lon2, lat2, _ = _GEOD.fwd(lon, lat, bearing_deg, distance_m)
    return lon2, lat2
