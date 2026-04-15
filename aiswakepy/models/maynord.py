"""Maynord (2005) empirical ship-wake model.

Reference
---------
Maynord, S.T. (2005). Wave Height from Planing and Semi-Planing Small Boats.
River Research and Applications, 21(1), 1–17.

Description
-----------
Developed for semi-planing and planing small craft (displacement Froude ≥ 1.5).

    Fr_dis = V / sqrt(g * W^(1/3))      (displacement Froude number)
    Hmax   = C * Fr_dis^(-0.58) * (y / W^(1/3))^(-0.42) * W^(1/3)

where:
    W  — volumetric displacement (m³) = width * draught * length * 0.95 * Cb
    y  — lateral distance from sailing line to point of interest (m)
    C  — empirical coefficient (default 0.82)

Applicability: the formula is considered valid when at least one of:
    - Fr_dis >= 1.5
    - Fr (length Froude) >= 0.6
    - depth/L >= 0.35
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_maynord(
    df: pd.DataFrame,
    g: float = 9.78,
    C: float = 0.82,
    min_fr_dis: float = 1.5,
    min_fr: float = 0.6,
    min_depth_ratio: float = 0.35,
) -> pd.Series:
    """Apply the Maynord (2005) formula to each AIS fix.

    Parameters
    ----------
    df:     DataFrame with ``SOGms``, ``length``, ``WaterDepth``,
            ``displacement_m3``, ``dist_perp`` columns.
    g:      Gravitational acceleration (m/s²). Default 9.78 (Singapore).
    C:      Empirical coefficient. Default 0.82.
    min_fr_dis: Minimum displacement Froude number (default 1.5).
    min_fr:     Minimum length Froude number (default 0.6).
    min_depth_ratio: Minimum depth/length ratio (default 0.35).
                     Formula valid if ANY of: Fr_dis >= min_fr_dis OR Fr >= min_fr OR depth/L >= min_depth_ratio.

    Returns
    -------
    pd.Series of Hmax values (m).  NaN where ALL applicability conditions fail:
    Fr_dis < min_fr_dis AND Fr < min_fr AND depth/L < min_depth_ratio.
    """
    v = df["SOGms"].to_numpy(dtype=float)
    l = df["length"].to_numpy(dtype=float)
    h = df["WaterDepth"].to_numpy(dtype=float)
    w = df["displacement_m3"].to_numpy(dtype=float)
    y = df["dist_perp"].to_numpy(dtype=float)

    w_cbrt = w ** (1.0 / 3.0)
    fr_dis = v / np.sqrt(g * w_cbrt)
    fr = v / np.sqrt(g * l)
    depth_l = h / l

    hmax = C * fr_dis ** (-0.58) * (y / w_cbrt) ** (-0.42) * w_cbrt

    # Remove where ALL three applicability conditions fail simultaneously
    # (formula developed for high-speed craft; not valid for slow large ships)
    not_applicable = (fr_dis < min_fr_dis) & (fr < min_fr) & (depth_l < min_depth_ratio)
    hmax[not_applicable] = np.nan

    return pd.Series(hmax, index=df.index, name="H_Maynord")
