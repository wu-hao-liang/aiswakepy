"""Kriebel & Seelig (2005) empirical ship-wake model.

Reference
---------
Kriebel, D.L. and Seelig, W.N. (2005). An Empirical Model for Ship-Generated
Waves. Proceedings of the Fifth International Symposium on Ocean Wave
Measurement and Analysis.

Description
-----------
Computes H_Kriebel at a given lateral distance from the sailing line.

Reads ``SOGms``, ``LengthWL``, ``FroudeD`` from the DataFrame (computed
upstream by ``compute_vessel_params``).  If ``dist_perp`` is present in the
DataFrame, it is used as the lateral distance; otherwise the function falls
back to the hull-origin reference distance y = B/2 (i.e. y/L = B/(2·L)).

Formula-specific intermediates (``Alpha``, ``Beta``, ``FroudeM``, ``BF``,
``GHV2``) are computed internally and are not added to the DataFrame.

    Alpha  = 2.35 * (1 - Cb)
    Beta   = 1 + 8 * tanh(0.45 * (L_WL/Le - 2))^3
    FroudeM = (V / sqrt(g * L_WL)) * exp(Alpha * d / h)
    BF     = Beta * (FroudeM - 0.1)^2
    GHV2   = BF * (y / L_WL)^(-1/3)
    H      = GHV2 / g * V^2
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_kriebel(
    df: pd.DataFrame,
    g: float = 9.78,
    min_froude_m: float = 0.1,
    max_froude_m: float = 0.5,
    max_bf: float = 0.4,
) -> pd.Series:
    """Apply the Kriebel & Seelig (2005) empirical formula to each AIS fix.

    Parameters
    ----------
    df:            DataFrame with ``SOGms``, ``LengthWL``, ``FroudeD``,
                   ``block_coeff``, ``bow_entry_m``, ``draught``,
                   ``WaterDepth``, ``width``, ``length`` columns (all
                   produced by ``compute_vessel_params``).
                   If ``dist_perp`` is present it is used as the lateral
                   distance; otherwise the hull-origin reference B/2 is used.
    g:             Local gravitational acceleration (m/s²). Default 9.78.
    min_froude_m:  Lower bound for modified Froude number (default 0.1).
    max_froude_m:  Upper bound for modified Froude number (default 0.5).
    max_bf:        Upper bound for BF shape factor (default 0.4).

    Returns
    -------
    pd.Series of H_Kriebel values (m).  NaN where applicability limits are
    exceeded (FroudeM out of range or BF > max_bf).
    """
    sogms   = df["SOGms"].to_numpy(dtype=float)
    lwl     = df["LengthWL"].to_numpy(dtype=float)
    cb      = df["block_coeff"].to_numpy(dtype=float)
    le      = df["bow_entry_m"].to_numpy(dtype=float)
    draught = df["draught"].to_numpy(dtype=float)
    h_depth = df["WaterDepth"].to_numpy(dtype=float)
    width   = df["width"].to_numpy(dtype=float)

    # --- Alpha = 2.35 * (1 - Cb): finite-depth adjustment exponent ---
    # Scales the shallow-water amplification of the modified Froude number
    # via the hull block coefficient. — Kriebel & Seelig (2005)
    alpha = 2.35 * (1.0 - cb)

    # --- Beta = 1 + 8 * tanh(0.45 * (L/Le - 2))^3: hull shape factor ---
    # Derived from bow entry length Le (half-angle of entrance).
    # β→1 for fine bows (L/Le≈2), β→9 for blunt bows. — Kriebel & Seelig (2005)
    ratio = lwl / le
    beta = 1.0 + 8.0 * np.tanh(0.45 * (ratio - 2.0)) ** 3

    # --- FroudeM = (V/√(g·L)) * exp(α·d/h): modified Froude number Fr* ---
    # Adjusts the length-based Froude number for finite water depth via the
    # draught/depth ratio. — Kriebel & Seelig (2005)
    froude_m = (sogms / np.sqrt(g * lwl)) * np.exp(alpha * draught / h_depth)

    # --- BF = β * (Fr* − 0.1)²: dimensionless wave height at y/L = 1 ---
    # Represents the normalised wave height at one vessel-length lateral distance.
    # Filter BF > 0.4: no data in Kriebel's 16-source dataset exceeds this limit.
    # — Kriebel & Seelig (2005)
    bf = beta * (froude_m - 0.1) ** 2

    # --- Lateral distance: dist_perp from df, or hull-origin reference B/2 ---
    if "dist_perp" in df.columns:
        y = df["dist_perp"].to_numpy(dtype=float)
    else:
        y = width / 2.0   # hull-origin: y = B/2

    # --- H = BF * (y / L_WL)^(-1/3) * V² / g ---
    # GHV2 = g·H/V² = BF * (y/L)^(-1/3); H recovered as GHV2 * V²/g.
    # — Kriebel & Seelig (2005)
    h = bf * (y / lwl) ** (-1.0 / 3.0) / g * sogms ** 2

    # --- Applicability filter: NaN where outside Kriebel (2005) valid range ---
    invalid = (
        (froude_m < min_froude_m) |
        (froude_m > max_froude_m) |
        (bf > max_bf)
    )
    h = h.copy()
    h[invalid] = np.nan

    return pd.Series(h, index=df.index, name="H_Kriebel")
