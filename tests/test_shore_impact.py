"""Tests for aiswakepy.stages.wave_impact — compute_wave_impact."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from aiswakepy.stages.wave_impact import compute_wave_impact

_G = 9.78


def _write_shp(tmp_path: Path, polygon: Polygon) -> Path:
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    p = tmp_path / "coast.shp"
    gdf.to_file(p)
    return p


def _make_vessel_row(**kwargs) -> pd.DataFrame:
    """Minimal vessel-params row that passes through the wave impact stage.

    Includes all columns produced by compute_vessel_params that formulae need.
    dist_perp is NOT included here — it is computed inside compute_wave_impact
    from the ray-coastline intersection geometry.
    """
    defaults = dict(
        mmsi=123456789,
        longitude=103.85,
        latitude=1.29,
        obstime=pd.Timestamp("2024-01-01"),
        sog=8.0,
        cog=90.0,
        SOGms=8.0 * 0.5144444,
        FroudeD=0.3397,
        # Kriebel intermediates — computed by compute_kriebel internally
        # (not required in vessel stage output, but kept here for completeness)
        block_coeff=0.86,
        bow_entry_m=200.0 / 7,
        displacement_m3=30.0 * 10.0 * 200.0 * 0.95 * 0.86,
        Tc=2.2,
        Theta=35.0,          # cog=90 ± 35 → WakeDir 55/125
        WakeDirPort=55.0,
        WakeDirStarboard=125.0,
        width=30.0,
        length=200.0,
        draught=10.0,
        WaterDepth=15.0,
    )
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# Basic wave impact
# ---------------------------------------------------------------------------

def test_wave_impact_hits_land(tmp_path):
    """Vessel heading east with land polygon to the east → intersection found."""
    poly = Polygon([(103.86, 1.28), (103.90, 1.28), (103.90, 1.30), (103.86, 1.30)])
    shp = _write_shp(tmp_path, poly)

    # wake_cutoff_m=0.0 so any positive H passes — this test checks geometry only
    df = _make_vessel_row(WakeDirPort=90.0, WakeDirStarboard=90.0)
    result = compute_wave_impact(df, shp, max_propagation_m=5000.0, wake_cutoff_m=0.0)
    assert len(result) >= 1


def test_wave_impact_no_hit(tmp_path):
    """Rays pointing west away from land polygon to the east → no hit."""
    poly = Polygon([(103.86, 1.28), (103.90, 1.28), (103.90, 1.30), (103.86, 1.30)])
    shp = _write_shp(tmp_path, poly)

    df = _make_vessel_row(WakeDirPort=270.0, WakeDirStarboard=270.0)
    result = compute_wave_impact(df, shp, max_propagation_m=5000.0, wake_cutoff_m=0.0)
    assert len(result) == 0


def test_wave_impact_columns(tmp_path):
    """Output has correct columns."""
    poly = Polygon([(103.86, 1.28), (103.90, 1.28), (103.90, 1.30), (103.86, 1.30)])
    shp = _write_shp(tmp_path, poly)
    df = _make_vessel_row(WakeDirPort=90.0, WakeDirStarboard=90.0)
    result = compute_wave_impact(df, shp, max_propagation_m=5000.0, wake_cutoff_m=0.0)
    for col in ["MMSI", "ShLongitude", "ShLatitude", "WaveHeight",
                "WavePeriod", "DistLoc_km", "Side"]:
        assert col in result.columns


def test_wave_impact_port_and_starboard(tmp_path):
    """Port and starboard rays both hit land → 2 output rows."""
    poly = Polygon([(103.86, 1.27), (103.92, 1.27), (103.92, 1.32), (103.86, 1.32)])
    shp = _write_shp(tmp_path, poly)

    df = _make_vessel_row(WakeDirPort=80.0, WakeDirStarboard=100.0)
    result = compute_wave_impact(df, shp, max_propagation_m=5000.0, wake_cutoff_m=0.0)
    sides = set(result["Side"].tolist())
    assert "port" in sides
    assert "stbd" in sides


def test_wave_impact_below_cutoff(tmp_path):
    """Wave height below cutoff → filtered out.

    SOGms=0.5 gives FroudeM < 0.1 (Kriebel lower limit) → H_Kriebel = NaN
    → row is excluded regardless of wake_cutoff_m.
    """
    poly = Polygon([(103.86, 1.28), (103.90, 1.28), (103.90, 1.30), (103.86, 1.30)])
    shp = _write_shp(tmp_path, poly)
    df = _make_vessel_row(WakeDirPort=90.0, WakeDirStarboard=90.0, SOGms=0.5)
    result = compute_wave_impact(df, shp, max_propagation_m=5000.0, wake_cutoff_m=0.01)
    assert len(result) == 0


def test_empty_df_returns_empty(tmp_path):
    """Empty input → empty output with correct columns."""
    poly = Polygon([(103.86, 1.28), (103.90, 1.28), (103.90, 1.30), (103.86, 1.30)])
    shp = _write_shp(tmp_path, poly)
    df = pd.DataFrame(columns=_make_vessel_row().columns)
    result = compute_wave_impact(df, shp)
    assert len(result) == 0
    assert "WaveHeight" in result.columns


def test_dist_loc_positive(tmp_path):
    """DistLoc_km should be positive."""
    poly = Polygon([(103.86, 1.28), (103.90, 1.28), (103.90, 1.30), (103.86, 1.30)])
    shp = _write_shp(tmp_path, poly)
    df = _make_vessel_row(WakeDirPort=90.0, WakeDirStarboard=90.0)
    result = compute_wave_impact(df, shp, max_propagation_m=5000.0, wake_cutoff_m=0.0)
    if len(result) > 0:
        assert (result["DistLoc_km"] > 0).all()

