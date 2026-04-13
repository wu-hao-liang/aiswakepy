"""Tests for shipwake.geo.bathymetry and shipwake.stages.depth — Step 5."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from aiswakepy.geo.bathymetry import snap_to_tide
from aiswakepy.stages.depth import assign_depth

# ---------------------------------------------------------------------------
# Helpers: synthetic BathymetryMesh stub
# ---------------------------------------------------------------------------

def _make_bathy_stub(depth_value: float = 10.0, nan_outside: bool = False):
    """Return a mock BathymetryMesh that always returns a constant depth."""
    stub = MagicMock()
    def get_depth(lons, lats):
        depths = np.full(len(lons), depth_value, dtype=float)
        if nan_outside:
            # Simulate points far outside returning NaN
            depths[np.asarray(lons) > 200.0] = np.nan
        return depths
    stub.get_depth.side_effect = get_depth
    return stub


def _make_df(n=3, draught=3.0, lon=103.85, lat=1.29):
    return pd.DataFrame({
        "mmsi": [1] * n,
        "width": [10] * n,
        "length": [50] * n,
        "draught": [draught] * n,
        "obstime": pd.date_range("2024-01-01", periods=n, freq="10min"),
        "longitude": [lon] * n,
        "latitude": [lat] * n,
        "sog": [5.0] * n,
        "cog": [90.0] * n,
        "typecargo": [70] * n,
    })


# ---------------------------------------------------------------------------
# snap_to_tide
# ---------------------------------------------------------------------------

def _make_tide_series(start="2024-01-01", periods=10, freq="6min", level=0.5):
    idx = pd.date_range(start, periods=periods, freq=freq)
    return pd.Series(level, index=idx)


def test_snap_to_tide_basic():
    tide = _make_tide_series(level=1.0)
    # Timestamp exactly on a tide index
    obstimes = pd.Series(pd.date_range("2024-01-01 00:00:00", periods=1, freq="6min"))
    levels = snap_to_tide(obstimes, tide)
    assert levels[0] == pytest.approx(1.0)


def test_snap_to_tide_rounds_to_nearest():
    tide = _make_tide_series(level=2.0, freq="6min")
    # Timestamp at 00:04 → nearest is 00:06 (2 min away vs 4 min to 00:00)
    obstimes = pd.Series([pd.Timestamp("2024-01-01 00:04:00")])
    levels = snap_to_tide(obstimes, tide)
    assert levels[0] == pytest.approx(2.0)


def test_snap_outside_range_returns_nan():
    tide = _make_tide_series(periods=5)
    obstimes = pd.Series([pd.Timestamp("2030-01-01 00:00:00")])
    levels = snap_to_tide(obstimes, tide)
    assert np.isnan(levels[0])


# ---------------------------------------------------------------------------
# assign_depth — using stub bathymetry
# ---------------------------------------------------------------------------

def test_assign_depth_adds_column():
    df = _make_df(n=3, draught=3.0)
    stub = _make_bathy_stub(depth_value=15.0)
    result = assign_depth(df, bathy_path="dummy.mesh", _bathy=stub)
    assert "WaterDepth" in result.columns
    np.testing.assert_allclose(result["WaterDepth"].to_numpy(), 15.0)


def test_assign_depth_filters_nan():
    """Points outside the mesh (NaN depth) should be removed."""
    df = _make_df(n=3)
    # All points return NaN
    stub = _make_bathy_stub(depth_value=np.nan)
    result = assign_depth(df, bathy_path="dummy.mesh", _bathy=stub)
    assert len(result) == 0


def test_assign_depth_underkeel_filter():
    """depth=4.0, draught=3.0, margin=1.0 → WaterDepth >= 4.0 required; 4.0 exactly passes."""
    df = _make_df(n=3, draught=3.0)
    stub = _make_bathy_stub(depth_value=4.0)
    result = assign_depth(df, bathy_path="dummy.mesh", underkeel_margin_m=1.0, _bathy=stub)
    assert len(result) == 3


def test_assign_depth_underkeel_filter_removes_shallow():
    """depth=3.5, draught=3.0, margin=1.0 → WaterDepth=3.5 < 4.0 required → all removed."""
    df = _make_df(n=3, draught=3.0)
    stub = _make_bathy_stub(depth_value=3.5)
    result = assign_depth(df, bathy_path="dummy.mesh", underkeel_margin_m=1.0, _bathy=stub)
    assert len(result) == 0


def test_assign_depth_with_tide():
    """bathy=10 m, tide=+1 m → WaterDepth=11 m."""
    df = _make_df(n=3, draught=3.0)
    stub = _make_bathy_stub(depth_value=10.0)

    # Build a tide series spanning the obstime
    tide = pd.Series(
        1.0,
        index=pd.date_range("2024-01-01", periods=100, freq="1min"),
    )

    # Patch load_tide to return our synthetic series
    import aiswakepy.stages.depth as depth_mod
    original = depth_mod.load_tide
    depth_mod.load_tide = lambda _, item=None: tide
    try:
        result = assign_depth(df, bathy_path="dummy.mesh",
                              tide_dfs0_path="dummy.dfs0", _bathy=stub)
    finally:
        depth_mod.load_tide = original

    np.testing.assert_allclose(result["WaterDepth"].to_numpy(), 11.0)


def test_assign_depth_tide_outside_range_drops_rows():
    """Timestamps outside tide range → NaN → rows dropped."""
    df = _make_df(n=3)
    stub = _make_bathy_stub(depth_value=10.0)

    # Tide series ends before df.obstime
    tide = pd.Series(
        0.5,
        index=pd.date_range("2020-01-01", periods=5, freq="6min"),
    )

    import aiswakepy.stages.depth as depth_mod
    original = depth_mod.load_tide
    depth_mod.load_tide = lambda _, item=None: tide
    try:
        result = assign_depth(df, bathy_path="dummy.mesh",
                              tide_dfs0_path="dummy.dfs0", _bathy=stub)
    finally:
        depth_mod.load_tide = original

    assert len(result) == 0


# ---------------------------------------------------------------------------
# Integration test using real mesh file (skipped if file missing)
# ---------------------------------------------------------------------------

MESH_PATH = Path(__file__).parent.parent / (
    "examples/bathymetry/61803960_WestCoast_HD_25m_mCD_Prod_v20260220.mesh"
)


@pytest.mark.skipif(not MESH_PATH.exists(), reason="Real mesh file not available")
def test_load_real_mesh_depth():
    from aiswakepy.geo.bathymetry import load_bathymetry
    bathy = load_bathymetry(MESH_PATH)
    # Query a point roughly in Singapore Strait (~1.26°N, 103.82°E)
    depths = bathy.get_depth(np.array([103.82]), np.array([1.26]))
    assert depths[0] > 0, "Expected positive depth in Singapore Strait"
    assert depths[0] < 200, "Depth seems unrealistically large"
