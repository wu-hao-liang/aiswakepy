"""Tests for aiswakepy.stages.vessel — compute_vessel_params."""

import numpy as np
import pandas as pd
import pytest

from aiswakepy.stages.vessel import compute_vessel_params, export_gis

_G = 9.78
_KNOTS_TO_MS = 0.5144444


def _make_row(**kwargs) -> pd.DataFrame:
    """Build a single-row DataFrame with all required columns."""
    defaults = dict(
        mmsi=123456789,
        width=30.0,
        length=200.0,
        draught=10.0,
        obstime=pd.Timestamp("2024-01-01 00:00:00"),
        longitude=103.85,
        latitude=1.29,
        sog=8.0,
        cog=90.0,
        typecargo=80,      # tanker
        WaterDepth=15.0,
        segment_id=1,
    )
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# Hand-calculated reference values for a tanker at 8 kts, depth 15 m
# L=200, B=30, d=10, SOG=8kts, depth=15m, type=tanker (L_Le method)
# Cb=0.86, Le=200/7≈28.571
# SOGms = 8*0.5144444 = 4.1155...
# Froude_D = 4.1155 / sqrt(9.78*15) = 4.1155 / 12.116 ≈ 0.3397
# Theta = 35.27*(1-exp(12*(0.3397-1))) ≈ 35.257
# WakeDirPort = 90 - 35.257 ≈ 54.74
# WakeDirStarboard = 90 + 35.257 ≈ 125.26
# ---------------------------------------------------------------------------

def test_hand_calc_single_row():
    df = _make_row(sog=8.0, cog=90.0, WaterDepth=15.0)
    result = compute_vessel_params(df, cb_method="L_Le", g=_G)
    assert len(result) == 1, "Row should pass all filters"

    r = result.iloc[0]
    assert r["SOGms"]      == pytest.approx(8.0 * _KNOTS_TO_MS, rel=1e-4)
    assert r["block_coeff"] == pytest.approx(0.86, rel=1e-4)
    assert r["bow_entry_m"] == pytest.approx(200.0 / 7, rel=1e-4)
    assert r["Froude_D"]    == pytest.approx(0.3397, rel=0.01)
    assert r["Theta"]      == pytest.approx(35.26, abs=0.5)


def test_wake_dir_uses_theta_not_90():
    """WakeDirPort = COG - θ, NOT COG - 90."""
    df = _make_row(sog=8.0, cog=90.0, WaterDepth=15.0)
    result = compute_vessel_params(df, cb_method="L_Le", g=_G)
    assert len(result) == 1
    r = result.iloc[0]
    theta = r["Theta"]
    assert r["WakeDirPort"]      == pytest.approx(90.0 - theta, rel=1e-6)
    assert r["WakeDirStarboard"] == pytest.approx(90.0 + theta, rel=1e-6)
    assert abs(r["WakeDirPort"] - 0.0) > 5.0


def test_sog_limit_filtered():
    df = _make_row(sog=15.0, WaterDepth=15.0)
    result = compute_vessel_params(df, max_sog_knots=12.0)
    assert len(result) == 0


def test_bl_ratio_filtered():
    """Beam/Length = 30/50 = 0.6 > 0.3 → filtered."""
    df = _make_row(length=50.0, width=30.0, WaterDepth=15.0, sog=5.0)
    result = compute_vessel_params(df, max_bl_ratio=0.3)
    assert len(result) == 0


def test_zero_depth_filtered():
    df = _make_row(WaterDepth=0.0)
    result = compute_vessel_params(df)
    assert len(result) == 0


def test_multiple_rows_mixed_filter():
    """SOG and BL filters applied; formula-validity filter is NOT in vessel stage."""
    rows = [
        _make_row(sog=8.0,  WaterDepth=15.0).iloc[0],   # valid
        _make_row(sog=15.0, WaterDepth=15.0).iloc[0],   # SOG too high → filtered
    ]
    df = pd.DataFrame(rows).reset_index(drop=True)
    result = compute_vessel_params(df)
    assert len(result) == 1


def test_gis_export_columns():
    df = _make_row(sog=8.0, WaterDepth=15.0)
    vessel = compute_vessel_params(df)
    gis = export_gis(vessel)
    assert "WaterDepth" in gis.columns
    assert "WakeDirPort" in gis.columns
    assert "WakeDirStarboard" in gis.columns
    assert "SOGms" in gis.columns
    assert "Froude_D" in gis.columns
    assert len(gis.columns) == 14


def test_does_not_mutate_input():
    df = _make_row(sog=8.0, WaterDepth=15.0)
    original_cols = set(df.columns)
    _ = compute_vessel_params(df)
    assert set(df.columns) == original_cols
