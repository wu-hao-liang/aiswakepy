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

    Hmax = A * h * (y/h)^(-1/3) * Fd^2.67

where:
    h  — water depth (m)
    y  — lateral distance from sailing line to point of interest (m)
    Fd — depth Froude number = V / sqrt(g * h)
    A  — hull-type coefficient:
            0.80  loaded vessel
            0.35  vessel with moderate load
            0.25  lightly loaded vessel
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Standard A coefficients by loading condition
A_LOADED   = 0.80
A_MODERATE = 0.35
A_LIGHT    = 0.25


def compute_blaauw(
    df: pd.DataFrame,
    g: float = 9.78,
    A: float = A_LOADED,
    max_fd: float = 0.7,
) -> pd.Series:
    """Apply the Blaauw et al. (1985) formula to each AIS fix.

    Parameters
    ----------
    df:     DataFrame with ``SOGms``, ``WaterDepth``, ``dist_perp`` columns.
    g:      Gravitational acceleration (m/s²). Default 9.78 (Singapore).
    A:      Hull-type coefficient. Use ``A_LOADED`` (0.80), ``A_MODERATE``
            (0.35), or ``A_LIGHT`` (0.25). Default: ``A_LOADED``.
    max_fd: Maximum depth Froude number (default 0.7). Formula valid for Fd < max_fd.

    Returns
    -------
    pd.Series of Hmax values (m).  NaN where depth Froude Fd >= max_fd.
    """
    v = df["SOGms"].to_numpy(dtype=float)
    h = df["WaterDepth"].to_numpy(dtype=float)
    y = df["dist_perp"].to_numpy(dtype=float)

    fd = v / np.sqrt(g * h)
    hmax = A * h * (y / h) ** (-1.0 / 3.0) * fd ** 2.67

    invalid = (fd >= max_fd)
    hmax[invalid] = np.nan

    return pd.Series(hmax, index=df.index, name="H_Blaauw")
