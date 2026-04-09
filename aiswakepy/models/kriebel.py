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

    # --- Depth-adjustment coefficient Alpha = 2.35 * (1 - Cb) ---
    df["Alpha"] = 2.35 * (1.0 - df["block_coeff"])

    # --- Shape factor Beta = 1 + 8 * tanh(0.45 * (L/Le - 2))^3 ---
    ratio = df["LengthWL"] / df["bow_entry_m"]
    df["Beta"] = 1.0 + 8.0 * np.tanh(0.45 * (ratio - 2.0)) ** 3

    # --- Modified Froude number Fm = (V / sqrt(g*L)) * exp(alpha * d / h) ---
    df["FroudeM"] = (
        df["SOGms"] / np.sqrt(g * df["LengthWL"])
    ) * np.exp(df["Alpha"] * df["draught"] / df["WaterDepth"])

    # --- Depth Froude number Fd = V / sqrt(g * h) ---
    df["FroudeD"] = df["SOGms"] / np.sqrt(g * df["WaterDepth"])

    # --- BF = Beta * (Fm - 0.1)^2 ---
    df["BF"] = df["Beta"] * (df["FroudeM"] - 0.1) ** 2

    # --- GH/V^2 = BF * (B / 2L)^(-1/3) ---
    df["GHV2"] = df["BF"] * (df["width"] / (2.0 * df["length"])) ** (-1.0 / 3.0)

    # --- Wave height at origin: H = GHV2 / g * V^2 ---
    df["H_Kriebel"] = df["GHV2"] / g * df["SOGms"] ** 2

    return df
