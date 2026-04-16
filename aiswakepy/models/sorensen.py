"""Sorensen (1984) empirical ship-wake model.

Reference
---------
Sorensen, R.M. (1984). Investigation of Ship-Generated Waves.
Journal of Waterway, Port, Coastal and Ocean Engineering, 110(4), 471–487.

Description
-----------
Developed for displacement vessels.
Uses piecewise coefficients dependent on depth Froude number Fr = V/sqrt(g*h).

    If Fr < 0.55:
        β = -0.225 * Fr^(-0.699)
        δ = -0.118 * Fr^(-0.356)
    Else:
        β = -0.342
        δ = -0.146

    a = -0.6 / Fr
    b = 0.75 * Fr^(-1.125)
    c = 2.653 * Fr - 1.95

    depth_adim = depth / W^(1/3)          (dimensionless water depth)
    n          = β * depth_adim^δ
    log_α      = a + b*ln(depth_adim) + c*ln(depth_adim)²
    α          = exp(log_α)

    H_adim     = α * (dist_perp / W^(1/3))^n   (dimensionless wave height)
    Hmax       = H_adim * W^(1/3)               (wave height in metres)

where:
    depth     — water depth (m)
    dist_perp — perpendicular distance from sailing line to point of interest (m)
    W         — volumetric displacement (m³) = width * draught * length * 0.95 * Cb
    L         — vessel length (m)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sorensen(
    df: pd.DataFrame,
    g: float = 9.78,
    min_Froude_D: float = 0.2,
    max_Froude_D: float = 0.8,
) -> pd.Series:
    """Apply the Sorensen (1984) formula to each AIS fix.

    Parameters
    ----------
    df:           DataFrame with ``SOGms``, ``WaterDepth``,
                  ``displacement_m3``, ``dist_perp`` columns.
    g:            Gravitational acceleration (m/s²). Default 9.78 (Singapore).
    min_Froude_D: Minimum depth Froude number (default 0.2). Formula valid for Froude_D >= min_Froude_D.
    max_Froude_D: Maximum depth Froude number (default 0.8). Formula valid for Froude_D <= max_Froude_D.

    Returns
    -------
    pd.Series of Hmax values (m).  NaN where Froude_D < min_Froude_D or Froude_D > max_Froude_D.
    """
    v = df["SOGms"].to_numpy(dtype=float)
    depth = df["WaterDepth"].to_numpy(dtype=float)
    w = df["displacement_m3"].to_numpy(dtype=float)
    y = df["dist_perp"].to_numpy(dtype=float)

    Froude_D = v / np.sqrt(g * depth)

    # Piecewise β and δ coefficients
    low_Froude_D = Froude_D < 0.55
    beta = np.where(low_Froude_D, -0.225 * Froude_D ** (-0.699), -0.342)
    delta = np.where(low_Froude_D, -0.118 * Froude_D ** (-0.356), -0.146)

    a = -0.6 / Froude_D
    b = 0.75 * Froude_D ** (-1.125)
    c = 2.653 * Froude_D - 1.95

    w_cbrt = w ** (1.0 / 3.0)
    depth_adim = depth / w_cbrt
    ln_depth_adim = np.log(depth_adim)

    n = beta * depth_adim ** delta
    log_alpha = a + b * ln_depth_adim + c * ln_depth_adim ** 2
    alpha = np.exp(log_alpha)

    dist_adim = y / w_cbrt
    h_adim = alpha * dist_adim ** n
    hmax = h_adim * w_cbrt

    # Applicability filter — valid for depth Froude number in (min_Froude_D, max_Froude_D)
    invalid = (Froude_D <= min_Froude_D) | (Froude_D >= max_Froude_D)
    hmax[invalid] = np.nan

    return pd.Series(hmax, index=df.index, name="H_Sorensen")
