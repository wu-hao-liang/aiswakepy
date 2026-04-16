"""Blaauw et al. (1985) empirical ship-wake model.

Reference
---------
Blaauw, H.G., de Groot, M.T., Knaap, F.C.M. and Pilarczyk, K.W. (1985).
Design of Bank Protection of Inland Waterways. Proceedings of the Conference
on Flexible Armoured Revetments incorporating Geotextiles, London.

Description
-----------
Developed for large vessels in deep water. Similar structure to PIANC but
uses a different Froude exponent and hull-type coefficient A.

    Hmax = A * h * (y/h)^(-1/3) * Froude_D^2.67

where:
    h  — water depth (m)
    y  — lateral distance from sailing line to point of interest (m)
    Froude_D — depth Froude number = V / sqrt(g * h)
    A  — hull-type coefficient:
            0.80  loaded vessel
            0.35  vessel with moderate load
            0.25  lightly loaded vessel
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Hull-type coefficient for loaded vessel (only variant used)
A_LOADED = 0.80


def compute_blaauw(
    df: pd.DataFrame,
    g: float = 9.78,
    A: float = A_LOADED,
    max_Froude_D: float = 0.7,
) -> pd.Series:
    """Apply the Blaauw et al. (1985) formula to each AIS fix.

    Parameters
    ----------
    df:           DataFrame with ``SOGms``, ``WaterDepth``, ``dist_perp``, ``width`` columns.
    g:            Gravitational acceleration (m/s²). Default 9.78 (Singapore).
    A:            Hull-type coefficient. Default: ``A_LOADED`` (0.80).
    max_Froude_D: Maximum depth Froude number (default 0.7). Formula valid for Froude_D < max_Froude_D.

    Returns
    -------
    pd.Series of Hmax values (m).  NaN where Froude_D >= max_Froude_D.
    """
    v = df["SOGms"].to_numpy(dtype=float)
    depth = df["WaterDepth"].to_numpy(dtype=float)
    y = df["dist_perp"].to_numpy(dtype=float)
    b = df["width"].to_numpy(dtype=float)

    Froude_D = v / np.sqrt(g * depth)
    hmax = A * depth * ((y - b / 2) / depth) ** (-1.0 / 3.0) * Froude_D ** 2.67

    invalid = (Froude_D >= max_Froude_D)
    hmax[invalid] = np.nan

    return pd.Series(hmax, index=df.index, name="H_Blaauw")
