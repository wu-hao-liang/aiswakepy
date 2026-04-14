"""Tests for the 6 additional empirical ship-wake models.

Hand-calculated reference values derived from the MATLAB reference
scripts WUHL_01_EmpiricalFormulations_AISdata_B_Le.m.

Shared test fixture:
    tanker, L=200m, B=30m, d=10m, SOG=8kts, WaterDepth=15m
    SOGms = 8 * 0.5144444 = 4.1155 m/s
    block_coeff = 0.80 (B_Le tanker), bow_entry_m = 30/1.0 = 30.0
    displacement_m3 = 30 * 10 * 200 * 0.95 * 0.80 = 45600 m³
    FroudeD = 4.1155 / sqrt(9.78*15) = 0.3397
    Fr_length = 4.1155 / sqrt(9.78*200) = 0.09296
"""

import numpy as np
import pandas as pd
import pytest

from aiswakepy.models.pianc import compute_pianc
from aiswakepy.models.bhowmik import compute_bhowmik
from aiswakepy.models.gates import compute_gates
from aiswakepy.models.blaauw import compute_blaauw, A_LOADED, A_MODERATE, A_LIGHT
from aiswakepy.models.sorensen import compute_sorensen
from aiswakepy.models.maynord import compute_maynord
from aiswakepy.vessel.block_coeff import get_vessel_params_df

_G = 9.78
_KNOTS_TO_MS = 0.5144444
_SOG_KTS = 8.0
_SOGMS = _SOG_KTS * _KNOTS_TO_MS   # ≈ 4.1155
_DIST_M = 500.0


def _make_df(**overrides) -> pd.DataFrame:
    """Single-row DataFrame with standard tanker vessel parameters."""
    defaults = dict(
        mmsi=123456789,
        width=30.0,
        length=200.0,
        draught=10.0,
        typecargo=80,       # tanker
        WaterDepth=15.0,
        SOGms=_SOGMS,
        sog=_SOG_KTS,
        FroudeD=_SOGMS / np.sqrt(_G * 15.0),
        # B_Le tanker: Cb=0.80, Le=30/1.0=30
        block_coeff=0.80,
        bow_entry_m=30.0,
        displacement_m3=30.0 * 10.0 * 200.0 * 0.95 * 0.80,  # 45600
    )
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# block_coeff.get_vessel_params_df — displacement_m3 added
# ---------------------------------------------------------------------------

def test_displacement_computed():
    df = pd.DataFrame([{
        "width": 30.0, "length": 200.0, "draught": 10.0, "typecargo": 80,
    }])
    out = get_vessel_params_df(df, method="B_Le")
    # B_Le tanker: Cb=0.80, W = 30*10*200*0.95*0.80 = 45600
    assert "displacement_m3" in out.columns
    assert out["displacement_m3"].iloc[0] == pytest.approx(45600.0, rel=1e-6)


def test_displacement_zero_draught():
    df = pd.DataFrame([{
        "width": 30.0, "length": 200.0, "draught": 0.0, "typecargo": 80,
    }])
    out = get_vessel_params_df(df, method="B_Le")
    assert out["displacement_m3"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PIANC (1987)
# ---------------------------------------------------------------------------
# Fd = 4.1155/sqrt(9.78*15) ≈ 0.3397
# Hmax = 1.0 * 15 * (500/15)^(-1/3) * 0.3397^4
#      = 15 * 33.33^(-1/3) * 0.01333
#      = 15 * 0.3217 * 0.01333 ≈ 0.0644 m

def test_pianc_known_value():
    df = _make_df()
    h = compute_pianc(df, dist_m=_DIST_M, g=_G)
    fd = _SOGMS / np.sqrt(_G * 15.0)
    expected = 1.0 * 15.0 * (_DIST_M / 15.0) ** (-1.0 / 3.0) * fd ** 4
    assert h.iloc[0] == pytest.approx(expected, rel=1e-4)


def test_pianc_filter_fr_high():
    """Fr >= 0.7 → NaN."""
    df = _make_df(SOGms=30.0, length=10.0)   # huge Fr
    h = compute_pianc(df, dist_m=_DIST_M, g=_G)
    assert np.isnan(h.iloc[0])


def test_pianc_filter_fd_high():
    """Fd >= 0.7 → NaN.
    SOGms=5, depth=5: Fd=5/sqrt(9.78*5)=5/6.99≈0.715 ≥ 0.7
    Fr=5/sqrt(9.78*1000)=0.016 < 0.7 (Fr filter does not trigger).
    """
    df = _make_df(SOGms=5.0, WaterDepth=5.0, length=1000.0)
    h = compute_pianc(df, dist_m=_DIST_M, g=_G)
    assert np.isnan(h.iloc[0])


def test_pianc_zero_depth_nan():
    df = _make_df(WaterDepth=0.0)
    h = compute_pianc(df, dist_m=_DIST_M, g=_G)
    assert np.isnan(h.iloc[0])


# ---------------------------------------------------------------------------
# Bhowmik et al. (1982)
# ---------------------------------------------------------------------------
# Fr_d = 4.1155 / sqrt(9.78*10) = 4.1155/9.8895 ≈ 0.4162
# Hmax = 0.133 * 0.4162 * 10 ≈ 0.5535 m

def test_bhowmik_known_value():
    df = _make_df()
    h = compute_bhowmik(df, g=_G)
    fr_d = _SOGMS / np.sqrt(_G * 10.0)
    expected = 0.133 * fr_d * 10.0
    assert h.iloc[0] == pytest.approx(expected, rel=1e-4)


def test_bhowmik_zero_draught_nan():
    df = _make_df(draught=0.0)
    h = compute_bhowmik(df, g=_G)
    assert np.isnan(h.iloc[0])


def test_bhowmik_no_distance_dependency():
    """Bhowmik takes no dist_m — two rows with same speed/draught → same H."""
    rows = [_make_df().iloc[0], _make_df().iloc[0]]
    df = pd.DataFrame(rows).reset_index(drop=True)
    h = compute_bhowmik(df, g=_G)
    assert h.iloc[0] == pytest.approx(h.iloc[1])


# ---------------------------------------------------------------------------
# Gates & Herbich (1977)
# ---------------------------------------------------------------------------
# Standard tanker: SOGms=4.1155, L=200m, B=30m, Le=30m, dist_m=500m
# Fr = 4.1155/sqrt(9.78*200) = 0.09306  <  FR_BREAK(≈0.273)
# Kw = _KW_SLOPE*Fr + _KW_INTERCEPT
# N  = (y_ft*3*g_ft*sqrt(3)/(2*V_ft²*π) - 1.5) / 2
# H_ft = (1.5/(2N+1.5))^(1/3) * Kw*(B/Le) * V_ft²/(2*g_ft)
# H_m  = H_ft * 0.3048

from aiswakepy.models.gates import _KW_SLOPE, _KW_INTERCEPT, _FR_BREAK, _FT


def test_gates_known_value():
    df = _make_df()
    h = compute_gates(df, dist_m=_DIST_M, g=_G)

    # Kw (local g)
    fr = _SOGMS / np.sqrt(_G * 200.0)
    kw = _KW_SLOPE * fr + _KW_INTERCEPT   # fr < FR_BREAK

    # Imperial conversions
    v_ft = _SOGMS * _FT
    g_ft = _G * _FT
    y_ft = _DIST_M * _FT

    # Cusp number N
    term = y_ft * 3.0 * g_ft * np.sqrt(3.0) / (2.0 * v_ft ** 2 * np.pi)
    n = max((term - 1.5) / 2.0, 0.0)

    # Wave height
    h_ft = (1.5 / (2.0 * n + 1.5)) ** (1.0 / 3.0) * (kw * 30.0 / 30.0) * v_ft ** 2 / (2.0 * g_ft)
    expected = h_ft * 0.3048
    assert h.iloc[0] == pytest.approx(expected, rel=1e-4)


def test_gates_filter_fr_high():
    """Fr >= 0.7 → NaN."""
    df = _make_df(SOGms=25.0, length=10.0)
    h = compute_gates(df, dist_m=_DIST_M, g=_G)
    assert np.isnan(h.iloc[0])


def test_gates_fn_above_break():
    """Fr >= FR_BREAK → Kw = 1.133 (constant branch)."""
    # Need Fr = V/sqrt(g*L) >= FR_BREAK ≈ 0.273
    # With L=10m, g=9.78: need V >= 0.273 * sqrt(9.78*10) = 2.703 m/s
    # Use SOGms=3.5, L=10m → Fr=3.5/sqrt(97.8)=0.354 > FR_BREAK
    df = _make_df(SOGms=3.5, length=10.0, width=3.0, bow_entry_m=3.0)
    h = compute_gates(df, dist_m=_DIST_M, g=_G)
    # Fr = 3.5/sqrt(9.78*10) ≈ 0.354 < 0.7 → not filtered
    assert not np.isnan(h.iloc[0])
    assert h.iloc[0] > 0


def test_gates_zero_distance_nan():
    """dist_m <= 0 → NaN."""
    df = _make_df()
    assert np.isnan(compute_gates(df, dist_m=0.0, g=_G).iloc[0])
    assert np.isnan(compute_gates(df, dist_m=-100.0, g=_G).iloc[0])


def test_gates_decay_with_distance():
    """Larger dist_m → smaller H (N grows → (1.5/(2N+1.5))^(1/3) shrinks)."""
    df = _make_df()
    h_near = compute_gates(df, dist_m=100.0, g=_G).iloc[0]
    h_far  = compute_gates(df, dist_m=1000.0, g=_G).iloc[0]
    assert h_near > h_far


# ---------------------------------------------------------------------------
# Blaauw et al. (1985)
# ---------------------------------------------------------------------------
# Fd = 0.3397
# A=0.80: Hmax = 0.80*15*(500/15)^(-1/3)*0.3397^2.67
#   0.3397^2.67: exp(2.67*ln(0.3397)) = exp(2.67*(-1.079)) = exp(-2.881) ≈ 0.0561
#   Hmax ≈ 0.80*15*0.3217*0.0561 ≈ 0.2171 m

def test_blaauw_loaded_known_value():
    df = _make_df()
    h = compute_blaauw(df, dist_m=_DIST_M, g=_G, A=A_LOADED)
    fd = _SOGMS / np.sqrt(_G * 15.0)
    expected = A_LOADED * 15.0 * (_DIST_M / 15.0) ** (-1.0 / 3.0) * fd ** 2.67
    assert h.iloc[0] == pytest.approx(expected, rel=1e-4)


def test_blaauw_three_variants_ordered():
    """A_LOADED > A_MODERATE > A_LIGHT → Hmax decreases in same order."""
    df = _make_df()
    h1 = compute_blaauw(df, dist_m=_DIST_M, g=_G, A=A_LOADED).iloc[0]
    h2 = compute_blaauw(df, dist_m=_DIST_M, g=_G, A=A_MODERATE).iloc[0]
    h3 = compute_blaauw(df, dist_m=_DIST_M, g=_G, A=A_LIGHT).iloc[0]
    assert h1 > h2 > h3 > 0


def test_blaauw_filter_fd_high():
    """Fd >= 0.7 → NaN.
    SOGms=5, depth=3: Fd=5/sqrt(9.78*3)=5/5.42≈0.923 ≥ 0.7.
    """
    df = _make_df(SOGms=5.0, WaterDepth=3.0)
    h = compute_blaauw(df, dist_m=_DIST_M, g=_G)
    assert np.isnan(h.iloc[0])


# ---------------------------------------------------------------------------
# Sorensen (1984)
# ---------------------------------------------------------------------------

def test_sorensen_returns_finite():
    """Valid inputs → finite positive result."""
    df = _make_df()
    h = compute_sorensen(df, dist_m=_DIST_M, g=_G)
    assert np.isfinite(h.iloc[0])
    assert h.iloc[0] > 0


def test_sorensen_zero_displacement_nan():
    df = _make_df(displacement_m3=0.0)
    h = compute_sorensen(df, dist_m=_DIST_M, g=_G)
    assert np.isnan(h.iloc[0])


def test_sorensen_zero_depth_nan():
    df = _make_df(WaterDepth=0.0)
    h = compute_sorensen(df, dist_m=_DIST_M, g=_G)
    assert np.isnan(h.iloc[0])


def test_sorensen_increases_with_dist_decreasing():
    """n < 0 for typical conditions → larger dist → smaller Hmax."""
    df_near = _make_df()
    df_far = _make_df()
    h_near = compute_sorensen(df_near, dist_m=200.0, g=_G).iloc[0]
    h_far = compute_sorensen(df_far, dist_m=1000.0, g=_G).iloc[0]
    # With negative exponent n, Hmax decreases as dist increases
    assert h_near > h_far


# ---------------------------------------------------------------------------
# Maynord (2005)
# ---------------------------------------------------------------------------

def test_maynord_large_ship_nan():
    """Typical large slow ship → all three applicability conditions fail → NaN."""
    df = _make_df()
    # Fr_dis = 4.1155/sqrt(9.78*45600^(1/3)) ≈ very small
    h = compute_maynord(df, dist_m=_DIST_M, g=_G)
    assert np.isnan(h.iloc[0])


def test_maynord_fast_small_craft():
    """Small fast craft with Fr_dis >= 1.5 → valid result."""
    # W = 3*1*10*0.95*0.67 ≈ 19.1 m³, W^(1/3) ≈ 2.67
    # Fr_dis = V/sqrt(9.78*2.67) ≈ V/5.11; need V >= 7.67 m/s
    df = _make_df(
        width=3.0, length=10.0, draught=1.0,
        block_coeff=0.67, bow_entry_m=3.0,
        displacement_m3=3.0 * 1.0 * 10.0 * 0.95 * 0.67,  # ≈ 19.1
        SOGms=8.0,
        WaterDepth=5.0,
    )
    h = compute_maynord(df, dist_m=50.0, g=_G)
    # Should NOT be NaN — Fr_dis > 1.5
    assert not np.isnan(h.iloc[0])
    assert h.iloc[0] > 0


def test_maynord_zero_displacement_nan():
    df = _make_df(displacement_m3=0.0)
    h = compute_maynord(df, dist_m=_DIST_M, g=_G)
    assert np.isnan(h.iloc[0])


# ---------------------------------------------------------------------------
# Series index alignment
# ---------------------------------------------------------------------------

def test_index_preserved():
    """Output Series index matches input DataFrame index."""
    df = _make_df()
    df.index = [42]
    for fn, kwargs in [
        (compute_pianc,    {"dist_m": _DIST_M, "g": _G}),
        (compute_bhowmik,  {"g": _G}),
        (compute_gates,    {"dist_m": _DIST_M, "g": _G}),
        (compute_blaauw,   {"dist_m": _DIST_M, "g": _G}),
        (compute_sorensen, {"dist_m": _DIST_M, "g": _G}),
        (compute_maynord,  {"dist_m": _DIST_M, "g": _G}),
    ]:
        result = fn(df, **kwargs)
        assert list(result.index) == [42], f"{fn.__name__} did not preserve index"
