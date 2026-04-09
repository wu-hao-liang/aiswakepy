"""Tests for shipwake.stages.shore_impact — Step 8."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from aiswakepy.stages.shore_impact import compute_shore_impact

_G = 9.78


def _write_shp(tmp_path: Path, polygon: Polygon) -> Path:
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    p = tmp_path / "coast.shp"
    gdf.to_file(p)
    return p


def _make_wave_row(**kwargs) -> pd.DataFrame:
    """Minimal wave params row that passes through the shore impact stage."""
    defaults = dict(
        mmsi=123456789,
        longitude=103.85,
        latitude=1.29,
        obstime=pd.Timestamp("2024-01-01"),
        sog=8.0,
        cog=90.0,
        SOGms=8.0 * 0.5144444,
        LengthWL=160.0,
        Beta=7.0,
        FroudeM=0.13,
        BF=0.15,
        Tc=2.2,
        Theta=35.0,          # cog=90 ± 35 → WakeDir 55/125
        WakeDirPort=55.0,
        WakeDirStarboard=125.0,
        width=30.0,
        length=200.0,
        WaterDepth=15.0,
    )
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# Basic shore impact
# ---------------------------------------------------------------------------

def test_shore_impact_hits_land(tmp_path):
    """Vessel heading east with land polygon to the east → intersection found."""
    poly = Polygon([(103.86, 1.28), (103.90, 1.28), (103.90, 1.30), (103.86, 1.30)])
    shp = _write_shp(tmp_path, poly)

    # WakeDirPort=90 and WakeDirStarboard=90 both east → 2 hits
    df = _make_wave_row(WakeDirPort=90.0, WakeDirStarboard=90.0)
    result = compute_shore_impact(df, shp, max_propagation_m=5000.0)
    assert len(result) >= 1


def test_shore_impact_no_hit(tmp_path):
    """Rays pointing west away from land polygon to the east → no hit."""
    poly = Polygon([(103.86, 1.28), (103.90, 1.28), (103.90, 1.30), (103.86, 1.30)])
    shp = _write_shp(tmp_path, poly)

    df = _make_wave_row(WakeDirPort=270.0, WakeDirStarboard=270.0)
    result = compute_shore_impact(df, shp, max_propagation_m=5000.0)
    assert len(result) == 0


def test_shore_impact_columns(tmp_path):
    """Output has correct columns."""
    poly = Polygon([(103.86, 1.28), (103.90, 1.28), (103.90, 1.30), (103.86, 1.30)])
    shp = _write_shp(tmp_path, poly)
    df = _make_wave_row(WakeDirPort=90.0, WakeDirStarboard=90.0)
    result = compute_shore_impact(df, shp, max_propagation_m=5000.0)
    for col in ["MMSI", "ShLongitude", "ShLatitude", "WaveHeight",
                "WavePeriod", "DistLoc_km", "Side"]:
        assert col in result.columns


def test_shore_impact_port_and_starboard(tmp_path):
    """Port and starboard rays both hit land → 2 output rows."""
    # Polygon to the east
    poly = Polygon([(103.86, 1.27), (103.92, 1.27), (103.92, 1.32), (103.86, 1.32)])
    shp = _write_shp(tmp_path, poly)

    # Port and starboard both directed eastward at slightly different angles
    df = _make_wave_row(WakeDirPort=80.0, WakeDirStarboard=100.0)
    result = compute_shore_impact(df, shp, max_propagation_m=5000.0)
    sides = set(result["Side"].tolist())
    assert "port" in sides
    assert "stbd" in sides


def test_shore_impact_below_cutoff(tmp_path):
    """Wave height below cutoff → filtered out."""
    poly = Polygon([(103.86, 1.28), (103.90, 1.28), (103.90, 1.30), (103.86, 1.30)])
    shp = _write_shp(tmp_path, poly)
    # Set BF very small so H_shore < cutoff
    df = _make_wave_row(WakeDirPort=90.0, WakeDirStarboard=90.0, BF=1e-10)
    result = compute_shore_impact(df, shp, max_propagation_m=5000.0, wake_cutoff_m=0.01)
    assert len(result) == 0


def test_empty_df_returns_empty(tmp_path):
    """Empty input → empty output with correct columns."""
    poly = Polygon([(103.86, 1.28), (103.90, 1.28), (103.90, 1.30), (103.86, 1.30)])
    shp = _write_shp(tmp_path, poly)
    df = pd.DataFrame(columns=_make_wave_row().columns)
    result = compute_shore_impact(df, shp)
    assert len(result) == 0
    assert "WaveHeight" in result.columns


def test_dist_loc_positive(tmp_path):
    """DistLoc_km should be positive."""
    poly = Polygon([(103.86, 1.28), (103.90, 1.28), (103.90, 1.30), (103.86, 1.30)])
    shp = _write_shp(tmp_path, poly)
    df = _make_wave_row(WakeDirPort=90.0, WakeDirStarboard=90.0)
    result = compute_shore_impact(df, shp, max_propagation_m=5000.0)
    if len(result) > 0:
        assert (result["DistLoc_km"] > 0).all()
