"""Coastline operations: load shapefile, build rays, find shoreline intersections."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from aiswakepy.geo.geodesy import forward_point, geodetic_distance


def load_coastline(shp_path: str | Path) -> MultiPolygon | Polygon:
    """Load a coastline shapefile and return a unified (Multi)Polygon."""
    gdf = gpd.read_file(str(shp_path))
    return unary_union(gdf.geometry)


def build_ray(
    lon: float,
    lat: float,
    bearing_deg: float,
    distance_m: float,
) -> LineString:
    """Build a LineString ray from (lon, lat) in direction bearing_deg for distance_m.

    Uses Vincenty geodesy to compute the endpoint.
    Returns a 2-point LineString in WGS84 (lon, lat) coordinates.
    """
    lon2, lat2 = forward_point(lon, lat, bearing_deg, distance_m)
    return LineString([(lon, lat), (lon2, lat2)])


def find_shore_intersection(
    ray: LineString,
    coastline,
) -> tuple[float, float, float] | None:
    """Find the closest intersection of a ray with a coastline polygon.

    Parameters
    ----------
    ray:        Ray LineString from vessel to max propagation point.
    coastline:  Coastline (Multi)Polygon loaded by load_coastline().

    Returns
    -------
    (lon, lat, distance_m) of the closest intersection to the ray start,
    or None if no intersection exists.
    """
    origin = ray.coords[0]   # (lon, lat) of vessel

    intersection = ray.intersection(coastline.boundary)

    if intersection.is_empty:
        return None

    # Collect all candidate points
    geom_type = intersection.geom_type
    if geom_type == "Point":
        candidates = [intersection]
    elif geom_type == "MultiPoint":
        candidates = list(intersection.geoms)
    elif geom_type in ("LineString", "MultiLineString", "GeometryCollection"):
        # Extract all points from any geometry in the collection
        candidates = []
        geoms = (
            list(intersection.geoms)
            if hasattr(intersection, "geoms")
            else [intersection]
        )
        for g in geoms:
            if g.geom_type == "Point":
                candidates.append(g)
            elif hasattr(g, "coords"):
                candidates.extend(Point(c) for c in g.coords)
    else:
        candidates = [intersection]

    if not candidates:
        return None

    # Select the candidate closest to the ray origin
    best_pt = None
    best_dist = np.inf
    for pt in candidates:
        d = geodetic_distance(origin[0], origin[1], pt.x, pt.y)
        if d < best_dist:
            best_dist = d
            best_pt = pt

    if best_pt is None:
        return None

    return best_pt.x, best_pt.y, best_dist
