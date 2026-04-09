"""Coastline operations: load shapefile, build rays, find shoreline intersections."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from aiswakepy.geo.geodesy import forward_point, geodetic_distance


def _extract_boundary_segments(geom) -> list[LineString]:
    """Recursively decompose a geometry's boundary into individual 2-point LineStrings."""
    segments: list[LineString] = []
    if geom.geom_type in ("LinearRing", "LineString"):
        coords = list(geom.coords)
        for i in range(len(coords) - 1):
            segments.append(LineString([coords[i], coords[i + 1]]))
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            segments.extend(_extract_boundary_segments(g))
    return segments


def build_coastline_index(
    coastline: MultiPolygon | Polygon,
) -> tuple[STRtree, list[LineString]]:
    """Build a Shapely STRtree from individual boundary segments of the coastline.

    Returns ``(strtree, segments)`` for use with
    :func:`find_shore_intersection_indexed`.  Building the tree once and reusing
    it across many rays reduces the shore-impact stage from O(N × C) to
    O(N × (log C + k)).
    """
    segments = _extract_boundary_segments(coastline.boundary)
    return STRtree(segments), segments


def find_shore_intersection_indexed(
    ray: LineString,
    strtree: STRtree,
    segments: list[LineString],
) -> tuple[float, float, float] | None:
    """Find the closest intersection using a pre-built STRtree.

    Equivalent to :func:`find_shore_intersection` but uses the spatial index to
    test only candidate segments whose bounding box overlaps the ray.
    """
    origin = ray.coords[0]
    candidate_indices = strtree.query(ray)

    best_pt: Point | None = None
    best_dist = np.inf

    for idx in candidate_indices:
        isect = ray.intersection(segments[idx])
        if isect.is_empty:
            continue
        pts = (
            list(isect.geoms)
            if hasattr(isect, "geoms")
            else [isect]
        )
        for pt in pts:
            if pt.geom_type != "Point":
                continue
            d = geodetic_distance(origin[0], origin[1], pt.x, pt.y)
            if d < best_dist:
                best_dist = d
                best_pt = pt

    if best_pt is None:
        return None
    return best_pt.x, best_pt.y, best_dist


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
