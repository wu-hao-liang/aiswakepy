"""PIANC (1987) empirical ship-wake model.

Reference
---------
PIANC (1987). Guidelines for the Design and Construction of Flexible Revetments
Incorporating Geotextiles for Inland Waterways. Report of Working Group 4,
Permanent International Association of Navigation Congresses.

Description
-----------
Developed for vessels in inland waterways. Relates Hmax to water depth,
lateral distance, and the depth Froude number.

    Hmax = A * h * (y/h)^(-1/3) * Fd^4

where:
    h  — water depth (m)
    y  — lateral distance from sailing line to point of interest (m)
    Fd — depth Froude number = V / sqrt(g * h)
    A  — hull-type coefficient (1.0 for tugs, patrol boats, loaded inland boats)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_pianc(
    df: pd.DataFrame,
    dist_m: np.ndarray | float,
    g: float = 9.78,
    A: float = 1.0,
) -> pd.Series:
    """Apply the PIANC (1987) formula to each AIS fix.

    Parameters
    ----------
    df:     DataFrame with ``SOGms``, ``WaterDepth``, ``length`` columns.
    dist_m: Lateral distance from sailing line to point of interest (m).
            Scalar or 1-D array aligned with df rows.
    g:      Gravitational acceleration (m/s²). Default 9.78 (Singapore).
    A:      Hull-type coefficient. Default 1.0.

    Returns
    -------
    pd.Series of Hmax values (m).  NaN where applicability filters fail:
        - length Froude Fr = V/sqrt(g*L) >= 0.7
        - depth  Froude Fd = V/sqrt(g*h) >= 0.7
    """
    v = df["SOGms"].to_numpy(dtype=float)
    h = df["WaterDepth"].to_numpy(dtype=float)
    y = np.asarray(dist_m, dtype=float) * np.ones(len(df))
    l = df["length"].to_numpy(dtype=float)

    fd = v / np.sqrt(g * h)
    fr = v / np.sqrt(g * l)

    hmax = A * h * (y / h) ** (-1.0 / 3.0) * fd ** 4

    # Applicability filters — set to NaN outside valid range
    invalid = (fr >= 0.7) | (fd >= 0.7) | (h <= 0) | (y <= 0)
    hmax[invalid] = np.nan

    return pd.Series(hmax, index=df.index, name="H_PIANC")
