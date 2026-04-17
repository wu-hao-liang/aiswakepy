"""Tests for aiswakepy.stages.wave_impact — compute_point_impact.

Tests verify that compute_point_impact preserves vessel parameters
(especially block_coeff, bow_entry_m, displacement_m3) in output.
"""

import pandas as pd
import pytest

from aiswakepy.stages.wave_impact import compute_point_impact


_G = 9.78


def _make_vessel_segment(**kwargs) -> pd.DataFrame:
    """Create a 2-point vessel trajectory segment.

    The second point is positioned such that the wake direction rays
    intersect a gauge point at (103.73, 1.27).
    """
    defaults = dict(
        mmsi=123456789,
        segment_id=1,
        typecargo=70,
        # Start point
        longitude=103.75,
        latitude=1.27,
        obstime=pd.Timestamp("2024-01-01 12:00:00"),
        sog=8.0,
        cog=90.0,
        SOGms=8.0 * 0.5144444,
        Froude_D=0.3397,
        block_coeff=0.86,
        bow_entry_m=200.0 / 7,
        displacement_m3=30.0 * 10.0 * 200.0 * 0.95 * 0.86,
        Tc=2.2,
        Theta=35.0,
        WakeDirPort=55.0,
        WakeDirStarboard=125.0,
        width=30.0,
        length=200.0,
        draught=10.0,
        WaterDepth=15.0,
    )
    defaults.update(kwargs)

    row1 = defaults.copy()
    row1_df = pd.DataFrame([row1])

    # Second point 30 seconds later
    row2 = defaults.copy()
    row2["longitude"] = 103.73
    row2["latitude"] = 1.27
    row2["obstime"] = pd.Timestamp("2024-01-01 12:00:30")
    row2_df = pd.DataFrame([row2])

    return pd.concat([row1_df, row2_df], ignore_index=True)


def test_point_impact_output_columns():
    """compute_point_impact output includes all expected columns."""
    df_vessel = _make_vessel_segment()
    result = compute_point_impact(df_vessel, 103.73, 1.27, formula="kriebel", g=_G)

    expected_cols = [
        "MMSI", "PointLongitude", "PointLatitude", "WaveHeight", "WavePeriod",
        "PropDist_m", "DistPerp_m", "DateTime", "ArrivalTime",
        "Froude_D", "VesselWidth", "VesselLength", "SOG", "Side", "segment_id",
        "SOGms", "WaterDepth", "length", "width", "draught",
        "block_coeff", "bow_entry_m", "displacement_m3",
    ]

    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_point_impact_preserves_block_coeff():
    """block_coeff from input is preserved in output."""
    df_vessel = _make_vessel_segment(block_coeff=0.75)
    result = compute_point_impact(df_vessel, 103.73, 1.27, formula="kriebel", g=_G)

    if len(result) > 0:
        assert "block_coeff" in result.columns
        assert (result["block_coeff"] == 0.75).all()


def test_point_impact_preserves_bow_entry():
    """bow_entry_m from input is preserved in output."""
    bow_entry = 35.0
    df_vessel = _make_vessel_segment(bow_entry_m=bow_entry)
    result = compute_point_impact(df_vessel, 103.73, 1.27, formula="kriebel", g=_G)

    if len(result) > 0:
        assert "bow_entry_m" in result.columns
        assert (result["bow_entry_m"] == bow_entry).all()


def test_point_impact_preserves_displacement():
    """displacement_m3 from input is preserved in output."""
    disp = 5000.0
    df_vessel = _make_vessel_segment(displacement_m3=disp)
    result = compute_point_impact(df_vessel, 103.73, 1.27, formula="kriebel", g=_G)

    if len(result) > 0:
        assert "displacement_m3" in result.columns
        assert (result["displacement_m3"] == disp).all()


def test_point_impact_empty_input():
    """Empty input returns empty DataFrame with all expected columns."""
    df_vessel = _make_vessel_segment().iloc[:0]  # Empty
    result = compute_point_impact(df_vessel, 103.73, 1.27, formula="kriebel", g=_G)

    assert len(result) == 0
    expected_cols = [
        "block_coeff", "bow_entry_m", "displacement_m3",
        "MMSI", "PointLongitude", "PointLatitude", "WaveHeight",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column in empty output: {col}"


def test_point_impact_finds_arrival():
    """compute_point_impact finds wake arrival at gauge point."""
    df_vessel = _make_vessel_segment()
    result = compute_point_impact(df_vessel, 103.73, 1.27, formula="kriebel", g=_G)

    # Should find at least one event at the gauge
    assert len(result) > 0, "No wake arrivals found at gauge point"
    assert "ArrivalTime" in result.columns
    assert result["ArrivalTime"].notna().any()


def test_point_impact_segment_id_preserved():
    """segment_id from input is preserved in output."""
    df_vessel = _make_vessel_segment(segment_id=42)
    result = compute_point_impact(df_vessel, 103.73, 1.27, formula="kriebel", g=_G)

    if len(result) > 0:
        assert "segment_id" in result.columns
        assert (result["segment_id"] == 42).all()


def test_point_impact_formulas_have_required_columns():
    """All formulae can access required columns in compute_point_impact output."""
    df_vessel = _make_vessel_segment()

    for formula_name in ["kriebel", "pianc", "bhowmik", "gates", "blaauw", "sorensen", "maynord"]:
        result = compute_point_impact(df_vessel, 103.73, 1.27, formula=formula_name, g=_G)

        # Should have output without errors; formula functions need block_coeff, bow_entry_m, etc.
        # If any column was missing, the formula would raise KeyError
        assert "WaveHeight" in result.columns or len(result) == 0
