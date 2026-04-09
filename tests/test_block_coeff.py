"""Tests for shipwake.vessel.block_coeff — Step 3."""

import numpy as np
import pandas as pd
import pytest

from aiswakepy.vessel.block_coeff import get_vessel_params, get_vessel_params_df


# ---------------------------------------------------------------------------
# Method L_Le
# ---------------------------------------------------------------------------

def test_L_Le_tanker():
    p = get_vessel_params(200.0, 30.0, ship_type=80, method="L_Le")
    assert p["block_coeff"] == pytest.approx(0.86)
    assert p["bow_entry_m"] == pytest.approx(200.0 / 7)


def test_L_Le_tanker_upper_bound():
    p = get_vessel_params(300.0, 50.0, ship_type=89, method="L_Le")
    assert p["block_coeff"] == pytest.approx(0.86)


def test_L_Le_cargo():
    p = get_vessel_params(120.0, 18.0, ship_type=75, method="L_Le")
    assert p["block_coeff"] == pytest.approx(0.80)
    assert p["bow_entry_m"] == pytest.approx(120.0 / 5)


def test_L_Le_dredger():
    p = get_vessel_params(80.0, 14.0, ship_type=33, method="L_Le")
    assert p["block_coeff"] == pytest.approx(0.80)
    assert p["bow_entry_m"] == pytest.approx(80.0 / 5)


def test_L_Le_other_fishing():
    p = get_vessel_params(20.0, 5.0, ship_type=30, method="L_Le")
    assert p["block_coeff"] == pytest.approx(0.67)
    assert p["bow_entry_m"] == pytest.approx(20.0 / 3)


def test_L_Le_unknown_type():
    # Unknown type code → "other" category
    p = get_vessel_params(50.0, 10.0, ship_type=99, method="L_Le")
    assert p["block_coeff"] == pytest.approx(0.67)
    assert p["bow_entry_m"] == pytest.approx(50.0 / 3)


# ---------------------------------------------------------------------------
# Method B_Le
# ---------------------------------------------------------------------------

def test_B_Le_tanker():
    p = get_vessel_params(200.0, 30.0, ship_type=82, method="B_Le")
    assert p["block_coeff"] == pytest.approx(0.80)
    assert p["bow_entry_m"] == pytest.approx(30.0 / 1.0)


def test_B_Le_cargo():
    p = get_vessel_params(120.0, 20.0, ship_type=70, method="B_Le")
    assert p["block_coeff"] == pytest.approx(0.70)
    assert p["bow_entry_m"] == pytest.approx(20.0 / 0.7)


def test_B_Le_other():
    p = get_vessel_params(30.0, 6.0, ship_type=60, method="B_Le")
    assert p["block_coeff"] == pytest.approx(0.60)
    assert p["bow_entry_m"] == pytest.approx(6.0 / 0.4)


# ---------------------------------------------------------------------------
# Method table
# ---------------------------------------------------------------------------

def test_table_tanker_large():
    """Large tanker: L=350, B=63 should match a row in the tanker range."""
    p = get_vessel_params(350.0, 63.0, ship_type=80, method="table")
    assert 0.80 <= p["block_coeff"] <= 0.90
    assert p["bow_entry_m"] > 0


def test_table_returns_positive():
    p = get_vessel_params(120.0, 20.0, ship_type=75, method="table")
    assert p["block_coeff"] > 0
    assert p["bow_entry_m"] > 0


# ---------------------------------------------------------------------------
# Bad method
# ---------------------------------------------------------------------------

def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        get_vessel_params(100.0, 15.0, ship_type=80, method="magic")


# ---------------------------------------------------------------------------
# Vectorised DataFrame interface
# ---------------------------------------------------------------------------

def _make_df():
    return pd.DataFrame({
        "length": [200.0, 120.0, 20.0],
        "width":  [30.0,  18.0,  5.0],
        "typecargo": [80, 75, 30],
    })


def test_df_L_Le_shape():
    df = get_vessel_params_df(_make_df(), method="L_Le")
    assert "block_coeff" in df.columns
    assert "bow_entry_m" in df.columns
    assert len(df) == 3


def test_df_L_Le_values():
    df = get_vessel_params_df(_make_df(), method="L_Le")
    assert df.iloc[0]["block_coeff"] == pytest.approx(0.86)   # tanker
    assert df.iloc[1]["block_coeff"] == pytest.approx(0.80)   # cargo
    assert df.iloc[2]["block_coeff"] == pytest.approx(0.67)   # fishing


def test_df_B_Le_values():
    df = get_vessel_params_df(_make_df(), method="B_Le")
    assert df.iloc[0]["bow_entry_m"] == pytest.approx(30.0 / 1.0)
    assert df.iloc[1]["bow_entry_m"] == pytest.approx(18.0 / 0.7)
    assert df.iloc[2]["bow_entry_m"] == pytest.approx(5.0 / 0.4)


def test_df_does_not_modify_original():
    original = _make_df()
    _ = get_vessel_params_df(original, method="L_Le")
    assert "block_coeff" not in original.columns
