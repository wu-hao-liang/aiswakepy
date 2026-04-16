"""Kriebel & Seelig (2005) empirical ship-wake model.

Reference
---------
Kriebel, D.L. and Seelig, W.N. (2005). An Empirical Model for Ship-Generated
Waves. Proceedings of the Fifth International Symposium on Ocean Wave
Measurement and Analysis.

Description
-----------
Computes H_Kriebel at a given lateral distance from the sailing line.

Reads ``SOGms``, ``length``, ``Froude_D`` from the DataFrame (computed
upstream by ``compute_vessel_params``). The ``dist_perp`` column must be
present in the DataFrame and is used as the lateral distance.

Formula-specific intermediates (``Alpha``, ``Beta``, ``Froude_M``, ``BF``,
``GHV2``) are computed internally and are not added to the DataFrame.

    Alpha  = 2.35 * (1 - Cb)
    Beta   = 1 + 8 * tanh(0.45 * (L/Le - 2))^3
    Froude_M = (V / sqrt(g * L)) * exp(Alpha * d / h)
    BF       = Beta * (Froude_M - 0.1)^2
    GHV2   = BF * (y / L)^(-1/3)
    H      = GHV2 / g * V^2
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_kriebel(
    df: pd.DataFrame,
    g: float = 9.78,
    min_Froude_M: float = 0.1,
    max_Froude_M: float = 0.5,
    max_bf: float = 0.4,
    max_Froude_D: float = 1.0,
) -> pd.Series:
    """Apply the Kriebel & Seelig (2005) empirical formula to each AIS fix.

    Parameters
    ----------
    df:            DataFrame with ``SOGms``, ``length``, ``Froude_D``,
                   ``block_coeff``, ``bow_entry_m``, ``draught``,
                   ``WaterDepth``, ``dist_perp`` columns (all produced by
                   ``compute_vessel_params`` or ``compute_wave_impact``).
    g:             Local gravitational acceleration (m/s²). Default 9.78.
    min_Froude_M:  Lower bound for modified Froude number (default 0.1).
    max_Froude_M:  Upper bound for modified Froude number (default 0.5).
    max_bf:        Upper bound for BF shape factor (default 0.4).
    max_Froude_D:  Maximum depth Froude number (default 1.0).

    Returns
    -------
    pd.Series of H_Kriebel values (m).  NaN where applicability limits are
    exceeded (Froude_M out of range, BF > max_bf, or Froude_D >= max_Froude_D).
    """
    v       = df["SOGms"].to_numpy(dtype=float)
    l       = df["length"].to_numpy(dtype=float)
    cb      = df["block_coeff"].to_numpy(dtype=float)
    le      = df["bow_entry_m"].to_numpy(dtype=float)
    draught = df["draught"].to_numpy(dtype=float)
    depth    = df["WaterDepth"].to_numpy(dtype=float)
    Froude_D = df["Froude_D"].to_numpy(dtype=float)

    # --- Alpha = 2.35 * (1 - Cb): finite-depth adjustment exponent ---
    # Scales the shallow-water amplification of the modified Froude number
    # via the hull block coefficient. — Kriebel & Seelig (2005)
    alpha = 2.35 * (1.0 - cb)

    # --- Beta = 1 + 8 * tanh(0.45 * (L/Le - 2))^3: hull shape factor ---
    # Derived from bow entry length Le (half-angle of entrance).
    # β→1 for fine bows (L/Le≈2), β→9 for blunt bows. — Kriebel & Seelig (2005)
    ratio = l / le
    beta = 1.0 + 8.0 * np.tanh(0.45 * (ratio - 2.0)) ** 3

    # --- Froude_M = (V/√(g·L)) * exp(α·d/h): modified Froude number Fr* ---
    # Adjusts the length-based Froude number for finite water depth via the
    # draught/depth ratio. — Kriebel & Seelig (2005)
    Froude_M = (v / np.sqrt(g * l)) * np.exp(alpha * draught / depth)

    # --- BF = β * (Fr* − 0.1)²: dimensionless wave height at y/L = 1 ---
    # Represents the normalised wave height at one vessel-length lateral distance.
    # Filter BF > 0.4: no data in Kriebel's 16-source dataset exceeds this limit.
    # — Kriebel & Seelig (2005)
    bf = beta * (Froude_M - 0.1) ** 2

    # --- Lateral distance: dist_perp from df ---
    y = df["dist_perp"].to_numpy(dtype=float)

    # --- H = BF * (y / L)^(-1/3) * V² / g ---
    # GHV2 = g·H/V² = BF * (y/L)^(-1/3); H recovered as GHV2 * V²/g.
    # — Kriebel & Seelig (2005)
    h = bf * (y / l) ** (-1.0 / 3.0) / g * v ** 2

    # --- Applicability filter: NaN where outside Kriebel (2005) valid range ---
    invalid = (
        (Froude_M < min_Froude_M) |
        (Froude_M > max_Froude_M) |
        (bf > max_bf) |
        (Froude_D >= max_Froude_D)
    )
    h[invalid] = np.nan

    return pd.Series(h, index=df.index, name="H_Kriebel")
