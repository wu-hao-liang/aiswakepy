"""Tests for shipwake.viz — Step 9."""

import os
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd
import pytest
from shapely.geometry import Polygon

matplotlib.use("Agg")

from aiswakepy.viz.vessel_diagram import plot_vessel_wake
from aiswakepy.viz.wave_map import plot_wave_height_map, plot_wave_period_map


def _write_shp(tmp_path: Path, polygon: Polygon) -> Path:
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    p = tmp_path / "coast.shp"
    gdf.to_file(p)
    return p


def _make_impact_df():
    return pd.DataFrame({
        "MMSI": [1, 1, 2],
        "ShLongitude": [103.87, 103.88, 103.86],
        "ShLatitude": [1.29, 1.28, 1.30],
        "WaveHeight": [0.05, 0.12, 0.08],
        "WavePeriod": [2.2, 2.5, 1.8],
        "DistLoc_km": [1.1, 1.3, 0.9],
        "Side": ["port", "stbd", "port"],
    })


def _make_wave_df():
    return pd.DataFrame({
        "mmsi": [1, 1, 1],
        "longitude": [103.85, 103.851, 103.852],
        "latitude": [1.29, 1.291, 1.292],
        "WakeDirPort": [55.0, 55.0, 55.0],
        "WakeDirStarboard": [125.0, 125.0, 125.0],
        "WaveHeight": [0.05, 0.07, 0.09],
    })


# ---------------------------------------------------------------------------
# Wave height map
# ---------------------------------------------------------------------------

def test_wave_height_map_creates_file(tmp_path):
    poly = Polygon([(103.84, 1.27), (103.92, 1.27), (103.92, 1.32), (103.84, 1.32)])
    shp = _write_shp(tmp_path, poly)
    out = tmp_path / "wave_height.png"
    plot_wave_height_map(_make_impact_df(), shp, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_wave_period_map_creates_file(tmp_path):
    poly = Polygon([(103.84, 1.27), (103.92, 1.27), (103.92, 1.32), (103.84, 1.32)])
    shp = _write_shp(tmp_path, poly)
    out = tmp_path / "wave_period.png"
    plot_wave_period_map(_make_impact_df(), shp, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_wave_height_map_empty_df(tmp_path):
    """Empty DataFrame → still produces a PNG with a warning."""
    poly = Polygon([(103.84, 1.27), (103.92, 1.27), (103.92, 1.32), (103.84, 1.32)])
    shp = _write_shp(tmp_path, poly)
    out = tmp_path / "wave_height_empty.png"
    with pytest.warns(UserWarning):
        plot_wave_height_map(pd.DataFrame(), shp, out)
    assert out.exists()


# ---------------------------------------------------------------------------
# Vessel diagram
# ---------------------------------------------------------------------------

def test_vessel_diagram_creates_file(tmp_path):
    poly = Polygon([(103.84, 1.27), (103.92, 1.27), (103.92, 1.32), (103.84, 1.32)])
    shp = _write_shp(tmp_path, poly)
    out = tmp_path / "vessel_1.png"
    plot_vessel_wake(1, _make_wave_df(), _make_impact_df(), shp, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_vessel_diagram_unknown_mmsi(tmp_path):
    """Unknown MMSI → blank figure, still saved."""
    poly = Polygon([(103.84, 1.27), (103.92, 1.27), (103.92, 1.32), (103.84, 1.32)])
    shp = _write_shp(tmp_path, poly)
    out = tmp_path / "vessel_99.png"
    plot_vessel_wake(99, _make_wave_df(), _make_impact_df(), shp, out)
    assert out.exists()


def test_output_dir_created(tmp_path):
    """Output parent directory is created if it does not exist."""
    poly = Polygon([(103.84, 1.27), (103.92, 1.27), (103.92, 1.32), (103.84, 1.32)])
    shp = _write_shp(tmp_path, poly)
    out = tmp_path / "subdir" / "deep" / "wave_height.png"
    plot_wave_height_map(_make_impact_df(), shp, out)
    assert out.exists()
