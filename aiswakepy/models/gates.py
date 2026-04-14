"""Gates & Herbich (1977) empirical ship-wake model.

Reference
---------
Gates, E.T. and Herbich, J.B. (1977). Mathematical Model to Predict the
Behaviour of Deep-Draft Vessels in Restricted Waterways.
Texas A&M University, Sea Grant Publication TAMU-SG-77-206.

(Based on NPL Teddington experiments; Figure 52.F from the original book.)

Description
-----------
Imperial-unit formulation. Distance-dependent wave height via cusp number N.

Kw coefficient (piecewise linear in Froude number Fn = V/sqrt(g_NPL * L)):

    Fn_break = 0.9191 / sqrt(g_NPL_ft)            g_NPL = 9.81 m/s²
    Kw = 1.133                                      if Fn >= Fn_break
    Kw = _KW_SLOPE * Fn + _KW_INTERCEPT            otherwise
         (fitted to Fig 52.F digitised data)

Cusp-line distance: the perpendicular distance y from the sailing line to
cusp N is

    y_ft = 2 * V_ft² * (2N+1.5) * π / (3 * g_ft * sqrt(3))

Solving for N given the observation lateral distance dist_m:

    term   = y_ft * 3 * g_ft * sqrt(3) / (2 * V_ft² * π)
    N      = (term - 1.5) / 2,  clamped to [0, ∞)

Wave height (Imperial, then converted to metres):

    H_ft = (1.5 / (2N+1.5))^(1/3) * Kw * (B / Le) * V_ft² / (2 * g_ft)
    H_m  = H_ft * 0.3048

where:
    B    — vessel beam (m)
    Le   — bow entry length (m)
    V    — vessel speed (m/s)
    g    — gravitational acceleration (m/s²)
    dist_m — lateral (perpendicular) distance from sailing line (m)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Kw regression constants ───────────────────────────────────────────────────
# g used in original NPL experiments for Froude-number scaling
_G_NPL = 9.81          # m/s²
_FT = 3.28084          # m → ft conversion

# Fn break point: Tq = V_ft/sqrt(L_ft) = 0.9191 (Gates threshold)
# Fn = Tq / sqrt(g_ft),  g_ft = 9.81 * 3.28084 = 32.185 ft/s²
_FN_BREAK = 0.9191 / np.sqrt(_G_NPL * _FT)  # ≈ 0.16201

# Linear fit: Kw = _KW_SLOPE * Fn + _KW_INTERCEPT  for Fn < _FN_BREAK
# Anchored at (_FN_BREAK, 1.133); least-squares fit to Fig 52.F data:
#   Tq:  0.62   0.68   0.78   0.82   0.88   (0.9191 → 1.133)
#   Kw:  2.85   2.55   2.20   1.60   1.85
_KW_SLOPE = -34.395
_KW_INTERCEPT = 6.705


def compute_gates(
    df: pd.DataFrame,
    dist_m: "np.ndarray | float",
    g: float = 9.78,
) -> pd.Series:
    """Apply the Gates & Herbich (1977) formula to each AIS fix.

    Parameters
    ----------
    df:     DataFrame with ``SOGms``, ``length``, ``width``, ``bow_entry_m``.
    dist_m: Lateral (perpendicular) distance from sailing line to the
            observation point (m). Scalar or 1-D array aligned with df rows.
    g:      Gravitational acceleration (m/s²). Default 9.78 (Singapore).

    Returns
    -------
    pd.Series of H (m).  NaN where applicability filter fails:
        - length Froude Fr = V/sqrt(g*L) >= 0.7
        - Le <= 0
        - dist_m <= 0
        - V <= 0
    """
    v   = df["SOGms"].to_numpy(dtype=float)
    l   = df["length"].to_numpy(dtype=float)
    b   = df["width"].to_numpy(dtype=float)
    le  = df["bow_entry_m"].to_numpy(dtype=float)
    y   = np.asarray(dist_m, dtype=float) * np.ones(len(df))

    # ── Kw via Froude number (g_NPL = 9.81 m/s², NPL Teddington standard) ──
    fn = v / np.sqrt(_G_NPL * l)
    kw = np.where(
        fn >= _FN_BREAK,
        1.133,
        _KW_SLOPE * fn + _KW_INTERCEPT,
    )

    # ── Imperial units for wave height and cusp formula ──────────────────────
    v_ft = v   * _FT
    g_ft = g   * _FT          # project g (9.78 m/s²) → ft/s²
    y_ft = y   * _FT

    # ── Cusp number N from lateral observation distance ───────────────────────
    # y_ft = 2 * V_ft² * (2N+1.5) * π / (3 * g_ft * sqrt(3))
    # => (2N+1.5) = y_ft * 3 * g_ft * sqrt(3) / (2 * V_ft² * π)
    term = y_ft * 3.0 * g_ft * np.sqrt(3.0) / (2.0 * v_ft ** 2 * np.pi)
    n = np.maximum((term - 1.5) / 2.0, 0.0)

    # ── Wave height (imperial → metres) ──────────────────────────────────────
    h_ft = (
        (1.5 / (2.0 * n + 1.5)) ** (1.0 / 3.0)
        * (kw * b / le)
        * v_ft ** 2 / (2.0 * g_ft)
    )
    h = h_ft * 0.3048

    # ── Applicability filter ──────────────────────────────────────────────────
    fr = v / np.sqrt(g * l)
    invalid = (fr >= 0.7) | (le <= 0) | (y <= 0) | (v <= 0)
    h[invalid] = np.nan

    return pd.Series(h, index=df.index, name="H_Gates")
