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
    g: float = 9.78,
    A: float = 1.0,
    max_fr: float = 0.7,
    max_fd: float = 0.7,
) -> pd.Series:
    """Apply the PIANC (1987) formula to each AIS fix.

    Parameters
    ----------
    df:     DataFrame with ``SOGms``, ``WaterDepth``, ``length``,
            ``dist_perp`` columns.
    g:      Gravitational acceleration (m/s²). Default 9.78 (Singapore).
    A:      Hull-type coefficient. Default 1.0.
    max_fr: Maximum length Froude number (default 0.7). Formula valid for Fr < max_fr.
    max_fd: Maximum depth Froude number (default 0.7). Formula valid for Fd < max_fd.

    Returns
    -------
    pd.Series of Hmax values (m).  NaN where Fr >= max_fr or Fd >= max_fd.
    """
    v = df["SOGms"].to_numpy(dtype=float)
    h = df["WaterDepth"].to_numpy(dtype=float)
    y = df["dist_perp"].to_numpy(dtype=float)
    l = df["length"].to_numpy(dtype=float)

    fd = v / np.sqrt(g * h)
    fr = v / np.sqrt(g * l)

    hmax = A * h * (y / h) ** (-1.0 / 3.0) * fd ** 4

    # Applicability filters — set to NaN outside valid range
    invalid = (fr >= max_fr) | (fd >= max_fd)
    hmax[invalid] = np.nan

    return pd.Series(hmax, index=df.index, name="H_PIANC")
