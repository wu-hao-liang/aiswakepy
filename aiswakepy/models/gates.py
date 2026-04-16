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

Kw coefficient (piecewise linear in Froude number Fr = V/sqrt(g * L)):

    Fr_break = 0.27344   (dimensionless; derived from Taylor quotient
                          threshold Tq = 0.9191 kt/√ft at NPL Teddington,
                          where velocity is in knots per Saunders (1957),
                          Hydrodynamics in Ship Design, vol. 2, p. 244 §52.5,
                          g_Teddington = 9.81 m/s²)
    Kw = 1.133                                      if Fr >= Fr_break
    Kw = _KW_SLOPE * Fr + _KW_INTERCEPT            otherwise
         (fitted to Fig 52.F digitised data)

Cusp-line distance: the perpendicular distance y from the sailing line to
cusp N is

    y_ft = 2 * V_ft² * (2N+1.5) * π / (3 * g_ft * sqrt(3))

Solving for N given the observation lateral distance dist_perp:

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
    dist_perp — lateral (perpendicular) distance from sailing line (m)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Kw regression constants ───────────────────────────────────────────────────
_FT = 3.28084          # m → ft conversion

# Fr break point: dimensionless Froude number threshold derived from the
# Taylor quotient Tq = V_kt/sqrt(L_ft) = 0.9191 measured at NPL Teddington,
# where the x-axis of Fig 52.F uses velocity in KNOTS (Saunders 1957,
# Hydrodynamics in Ship Design, vol. 2, p. 244 §52.5; g_Teddington = 9.81 m/s²).
# Conversion: Fr = Tq_kt × 0.514444 / sqrt(0.3048 × g_Teddington) = Tq_kt × 0.29751
# Precomputed as a dimensionless constant; local g is used at runtime.
_FR_BREAK = 0.27344    # = 0.9191 × 0.29751

# Linear fit: Kw = _KW_SLOPE * Fr + _KW_INTERCEPT  for Fr < _FR_BREAK
# Anchored at (_FR_BREAK, 1.133); anchored least-squares fit to Fig 52.F data
# (Tq in kt/√ft converted to Fr via factor 0.29751):
#   Tq (kt/√ft):  0.6843  0.7626  0.7797  0.8443  0.8740  (0.9191 → 1.133)
#   Fr:           0.2036  0.2269  0.2320  0.2512  0.2600
#   Kw:           2.8839  2.3980  2.1438  1.7313  1.4473
_KW_SLOPE = -25.49
_KW_INTERCEPT = 8.104


def compute_gates(
    df: pd.DataFrame,
    g: float = 9.78,
    max_Froude_L: float = 0.7,
) -> pd.Series:
    """Apply the Gates & Herbich (1977) formula to each AIS fix.

    Parameters
    ----------
    df:           DataFrame with ``SOGms``, ``length``, ``width``, ``bow_entry_m``,
                  ``dist_perp`` columns.
    g:            Gravitational acceleration (m/s²). Default 9.78 (Singapore).
    max_Froude_L: Maximum length Froude number (default 0.7). Formula valid for Froude_L < max_Froude_L.

    Returns
    -------
    pd.Series of H (m).  NaN where Froude_L >= max_Froude_L.
    """
    v   = df["SOGms"].to_numpy(dtype=float)
    l   = df["length"].to_numpy(dtype=float)
    b   = df["width"].to_numpy(dtype=float)
    le  = df["bow_entry_m"].to_numpy(dtype=float)
    y   = df["dist_perp"].to_numpy(dtype=float)

    # ── Froude number (local g) — used for both Kw lookup and validity filter ─
    Froude_L = v / np.sqrt(g * l)
    kw = np.where(
        Froude_L >= _FR_BREAK,
        1.133,
        _KW_SLOPE * Froude_L + _KW_INTERCEPT,
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
    invalid = (Froude_L >= max_Froude_L)
    h[invalid] = np.nan

    return pd.Series(h, index=df.index, name="H_Gates")
