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
    dist_m: np.ndarray | float,
    g: float = 9.78,
    C: float = 0.82,
) -> pd.Series:
    """Apply the Maynord (2005) formula to each AIS fix.

    Parameters
    ----------
    df:     DataFrame with ``SOGms``, ``length``, ``WaterDepth``,
            ``displacement_m3`` columns.
    dist_m: Lateral distance from sailing line to point of interest (m).
            Scalar or 1-D array aligned with df rows.
    g:      Gravitational acceleration (m/s²). Default 9.78 (Singapore).
    C:      Empirical coefficient. Default 0.82.

    Returns
    -------
    pd.Series of Hmax values (m).  NaN where applicability filter fails
    (Fr_dis < 1.5 AND Fr < 0.6 AND depth/L < 0.35) or displacement <= 0.
    """
    v = df["SOGms"].to_numpy(dtype=float)
    l = df["length"].to_numpy(dtype=float)
    h = df["WaterDepth"].to_numpy(dtype=float)
    w = df["displacement_m3"].to_numpy(dtype=float)
    y = np.asarray(dist_m, dtype=float) * np.ones(len(df))

    w_cbrt = w ** (1.0 / 3.0)
    fr_dis = v / np.sqrt(g * w_cbrt)
    fr = v / np.sqrt(g * l)
    depth_l = h / l

    hmax = C * fr_dis ** (-0.58) * (y / w_cbrt) ** (-0.42) * w_cbrt

    # Remove where ALL three applicability conditions fail simultaneously
    # (formula developed for high-speed craft; not valid for slow large ships)
    not_applicable = (fr_dis < 1.5) & (fr < 0.6) & (depth_l < 0.35)
    invalid = not_applicable | (w <= 0) | (y <= 0)
    hmax[invalid] = np.nan

    return pd.Series(hmax, index=df.index, name="H_Maynord")
