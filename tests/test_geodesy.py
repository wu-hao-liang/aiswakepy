"""Tests for shipwake.geo.geodesy — Step 2."""

import numpy as np
import pytest

from aiswakepy.geo.geodesy import forward_point, geodetic_bearing, geodetic_distance

# Singapore approx centre
SG_LON, SG_LAT = 103.85, 1.29


def test_distance_known_east():
    """1 km due east of Singapore: longitude increases by ~0.009°."""
    lon2, lat2 = forward_point(SG_LON, SG_LAT, 90.0, 1000.0)
    dist = geodetic_distance(SG_LON, SG_LAT, lon2, lat2)
    assert abs(dist - 1000.0) < 0.5   # within 0.5 m


def test_distance_known_north():
    lon2, lat2 = forward_point(SG_LON, SG_LAT, 0.0, 500.0)
    dist = geodetic_distance(SG_LON, SG_LAT, lon2, lat2)
    assert abs(dist - 500.0) < 0.5


def test_bearing_due_north():
    lon2, lat2 = forward_point(SG_LON, SG_LAT, 0.0, 1000.0)
    b = geodetic_bearing(SG_LON, SG_LAT, lon2, lat2)
    assert abs(b - 0.0) < 0.01


def test_bearing_due_east():
    lon2, lat2 = forward_point(SG_LON, SG_LAT, 90.0, 1000.0)
    b = geodetic_bearing(SG_LON, SG_LAT, lon2, lat2)
    assert abs(b - 90.0) < 0.01


def test_bearing_due_south():
    lon2, lat2 = forward_point(SG_LON, SG_LAT, 180.0, 1000.0)
    b = geodetic_bearing(SG_LON, SG_LAT, lon2, lat2)
    assert abs(abs(b) - 180.0) < 0.01


def test_round_trip():
    """forward_point then geodetic_distance should recover the original distance."""
    for bearing in [0, 45, 90, 135, 180, 225, 270, 315]:
        lon2, lat2 = forward_point(SG_LON, SG_LAT, bearing, 1234.5)
        dist = geodetic_distance(SG_LON, SG_LAT, lon2, lat2)
        assert abs(dist - 1234.5) < 1.0, f"bearing={bearing}"


def test_vectorised_distance():
    n = 100
    rng = np.random.default_rng(42)
    lons1 = rng.uniform(103.6, 104.0, n)
    lats1 = rng.uniform(1.1, 1.5, n)
    lons2 = lons1 + rng.uniform(-0.01, 0.01, n)
    lats2 = lats1 + rng.uniform(-0.01, 0.01, n)

    vec = geodetic_distance(lons1, lats1, lons2, lats2)
    scalar = np.array([
        geodetic_distance(lons1[i], lats1[i], lons2[i], lats2[i])
        for i in range(n)
    ])
    np.testing.assert_allclose(vec, scalar, rtol=1e-10)


def test_vectorised_forward_point():
    n = 50
    rng = np.random.default_rng(7)
    lons = rng.uniform(103.6, 104.0, n)
    lats = rng.uniform(1.1, 1.5, n)
    bearings = rng.uniform(0, 360, n)
    dists = rng.uniform(100, 2000, n)

    lon2v, lat2v = forward_point(lons, lats, bearings, dists)
    for i in range(n):
        l2s, a2s = forward_point(lons[i], lats[i], bearings[i], dists[i])
        assert abs(lon2v[i] - l2s) < 1e-9
        assert abs(lat2v[i] - a2s) < 1e-9
