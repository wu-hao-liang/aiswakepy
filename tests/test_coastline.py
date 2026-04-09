"""Tests for shipwake.geo.coastline — Step 7."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from aiswakepy.geo.coastline import (
    build_ray,
    find_shore_intersection,
    load_coastline,
)

SG_LON, SG_LAT = 103.85, 1.29


# ---------------------------------------------------------------------------
# load_coastline
# ---------------------------------------------------------------------------

def _write_shp(tmp_path: Path, polygon: Polygon) -> Path:
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    p = tmp_path / "coast.shp"
    gdf.to_file(p)
    return p


def test_load_coastline_valid(tmp_path):
    poly = Polygon([(103.84, 1.28), (103.86, 1.28), (103.86, 1.30), (103.84, 1.30)])
    shp = _write_shp(tmp_path, poly)
    coast = load_coastline(shp)
    assert coast.area > 0


def test_load_real_coastline():
    shp = Path(__file__).parent.parent / "examples/coastline/Coast_P1.shp"
    if not shp.exists():
        pytest.skip("Real coastline shapefile not available")
    coast = load_coastline(shp)
    assert coast.area > 0


# ---------------------------------------------------------------------------
# build_ray
# ---------------------------------------------------------------------------

def test_build_ray_length():
    """Ray endpoint should be ~1000 m from origin."""
    from aiswakepy.geo.geodesy import geodetic_distance
    ray = build_ray(SG_LON, SG_LAT, 90.0, 1000.0)
    start = ray.coords[0]
    end = ray.coords[-1]
    dist = geodetic_distance(start[0], start[1], end[0], end[1])
    assert abs(dist - 1000.0) < 1.0


def test_build_ray_is_linestring():
    ray = build_ray(SG_LON, SG_LAT, 45.0, 500.0)
    assert isinstance(ray, LineString)
    assert len(ray.coords) == 2


# ---------------------------------------------------------------------------
# find_shore_intersection
# ---------------------------------------------------------------------------

def test_intersection_ray_hits_polygon(tmp_path):
    """Ray from open sea toward land should find an intersection."""
    # Polygon: land to the east, vessel to the west, ray heading east
    poly = Polygon([(103.87, 1.28), (103.90, 1.28), (103.90, 1.30), (103.87, 1.30)])
    ray = build_ray(103.85, 1.29, 90.0, 3000.0)   # heading east
    coast = poly
    result = find_shore_intersection(ray, coast)
    assert result is not None
    sh_lon, sh_lat, dist_m = result
    assert sh_lon > 103.85   # intersection east of vessel
    assert dist_m > 0


def test_intersection_no_hit():
    """Ray heading away from polygon → None."""
    poly = Polygon([(103.87, 1.28), (103.90, 1.28), (103.90, 1.30), (103.87, 1.30)])
    ray = build_ray(103.85, 1.29, 270.0, 1000.0)  # heading west, away from polygon
    result = find_shore_intersection(ray, poly)
    assert result is None


def test_intersection_distance_reasonable():
    """Distance to intersection should be < max_propagation_m."""
    # Polygon close enough to be within a 2000 m ray
    # 0.01° lon ≈ ~1100 m at lat 1.29°
    poly = Polygon([(103.86, 1.28), (103.90, 1.28), (103.90, 1.30), (103.86, 1.30)])
    ray = build_ray(103.85, 1.29, 90.0, 2000.0)
    result = find_shore_intersection(ray, poly)
    assert result is not None
    _, _, dist_m = result
    assert dist_m < 2000.0
