"""Kriebel & Seelig (2005) empirical ship-wake model.

Reference
---------
Kriebel, D.L. and Seelig, W.N. (2005). An Empirical Model for Ship-Generated
Waves. Proceedings of the Fifth International Symposium on Ocean Wave
Measurement and Analysis.

Description
-----------
Computes the Kriebel wave height at the vessel origin (``H_Kriebel``) and
the intermediate quantities specific to this formula:
``Beta``, ``Alpha``, ``FroudeM``, ``FroudeD``, ``BF``, ``GHV2``.

General wave propagation parameters (wave period, energy, spreading angle,
wake directions) are computed in the calling stage
(``aiswakepy.stages.wave_params``) and are not part of this model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aiswakepy.vessel.block_coeff import get_vessel_params_df

_KNOTS_TO_MS = 0.5144444


def compute_kriebel(
    df: pd.DataFrame,
    cb_method: str = "L_Le",
    g: float = 9.78,
) -> pd.DataFrame:
    """Apply the Kriebel & Seelig (2005) empirical formula to each AIS fix.

    Adds the following columns to a copy of *df*:

    =========  =============================================================
    SOGms      Speed over ground (m/s)
    LengthWL   Waterline length = 0.8 * LOA (m)
    Alpha      Depth-adjustment coefficient = 2.35 * (1 - Cb)
    Beta       Shape factor = 1 + 8 * tanh(0.45 * (L/Le - 2))^3
    FroudeM    Modified Froude number Fm = (V / sqrt(g*L)) * exp(alpha*d/h)
    FroudeD    Depth Froude number Fd = V / sqrt(g*h)
    BF         Beta * (FroudeM - 0.1)^2
    GHV2       BF * (B / 2L)^(-1/3)
    H_Kriebel  Wave height at vessel origin (m) = GHV2 / g * V^2
    =========  =============================================================

    Parameters
    ----------
    df:        AIS DataFrame with ``WaterDepth`` column (output of assign_depth).
    cb_method: Block coefficient method: ``"L_Le"``, ``"B_Le"``, or ``"table"``.
    g:         Local gravitational acceleration (m/s^2). Default 9.78 (Singapore).

    Returns
    -------
    Copy of *df* with the columns above added.
    """
    df = df.copy()

    # --- Vessel params (block_coeff + bow_entry_m) ---
    df = get_vessel_params_df(df, method=cb_method)

    # --- Speed and length conversions ---
    df["SOGms"] = df["sog"] * _KNOTS_TO_MS
    df["LengthWL"] = df["length"] * 0.8

    # --- Alpha = 2.35 * (1 - Cb): finite-depth adjustment exponent ---
    # Scales the shallow-water amplification of the modified Froude number
    # via the hull block coefficient. — Kriebel & Seelig (2005)
    df["Alpha"] = 2.35 * (1.0 - df["block_coeff"])

    # --- Beta = 1 + 8 * tanh(0.45 * (L/Le - 2))^3: hull shape factor ---
    # Derived from bow entry length Le (half-angle of entrance).
    # β→1 for fine bows (L/Le≈2), β→9 for blunt bows. — Kriebel & Seelig (2005)
    ratio = df["LengthWL"] / df["bow_entry_m"]
    df["Beta"] = 1.0 + 8.0 * np.tanh(0.45 * (ratio - 2.0)) ** 3

    # --- FroudeM = (V/√(g·L)) * exp(α·d/h): modified Froude number Fr* ---
    # Adjusts the length-based Froude number for finite water depth via the
    # draught/depth ratio. — Kriebel & Seelig (2005)
    df["FroudeM"] = (
        df["SOGms"] / np.sqrt(g * df["LengthWL"])
    ) * np.exp(df["Alpha"] * df["draught"] / df["WaterDepth"])

    # --- FroudeD = V/√(g·h): depth Froude number ---
    df["FroudeD"] = df["SOGms"] / np.sqrt(g * df["WaterDepth"])

    # --- BF = β * (Fr* − 0.1)²: dimensionless wave height at y/L = 1 ---
    # Represents the normalised wave height at one vessel-length lateral distance.
    # Filter BF > 0.4: no data in Kriebel's 16-source dataset exceeds this limit.
    # — Kriebel & Seelig (2005)
    df["BF"] = df["Beta"] * (df["FroudeM"] - 0.1) ** 2

    # --- GHV2 = g·H/V² = BF * (y/L)^(-1/3): dimensionless wave height ---
    # Core Kriebel output. Evaluated at normalised lateral distance y/L = B/(2L)
    # from the sailing line (y = B/2 at the hull, L = vessel length).
    # — Kriebel & Seelig (2005)
    df["GHV2"] = df["BF"] * (df["width"] / (2.0 * df["length"])) ** (-1.0 / 3.0)

    # --- H_Kriebel = GHV2 * V²/g: dimensional maximum wave height at the hull ---
    # Recovers H from the dimensionless ratio: H = (g·H/V²) × V²/g.
    df["H_Kriebel"] = df["GHV2"] / g * df["SOGms"] ** 2

    return df
