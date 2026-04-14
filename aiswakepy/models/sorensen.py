"""Sorensen (1984) empirical ship-wake model.

Reference
---------
Sorensen, R.M. (1984). Investigation of Ship-Generated Waves.
Journal of Waterway, Port, Coastal and Ocean Engineering, 110(4), 471–487.

Description
-----------
Developed for displacement vessels with length Froude numbers 0.2–0.8.
Uses piecewise coefficients dependent on Fr = V/sqrt(g*L).

    If Fr < 0.55:
        β = -0.225 * Fr^(-0.699)
        δ = -0.118 * Fr^(-0.356)
    Else:
        β = -0.342
        δ = -0.146

    a = -0.6 / Fr
    b = 0.75 * Fr^(-1.125)
    c = 2.653 * Fr - 1.95

    h_adim = h / W^(1/3)          (dimensionless water depth)
    n      = β * h_adim^δ
    log_α  = a + b*ln(h_adim) + c*ln(h_adim)²
    α      = exp(log_α)

    Hmax   = α * (y / W^(1/3))^n

where:
    h  — water depth (m)
    y  — lateral distance from sailing line to point of interest (m)
    W  — volumetric displacement (m³) = width * draught * length * 0.95 * Cb
    L  — vessel length (m)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sorensen(
    df: pd.DataFrame,
    dist_m: np.ndarray | float,
    g: float = 9.78,
) -> pd.Series:
    """Apply the Sorensen (1984) formula to each AIS fix.

    Parameters
    ----------
    df:     DataFrame with ``SOGms``, ``length``, ``WaterDepth``,
            ``displacement_m3`` columns.
    dist_m: Lateral distance from sailing line to point of interest (m).
            Scalar or 1-D array aligned with df rows.
    g:      Gravitational acceleration (m/s²). Default 9.78 (Singapore).

    Returns
    -------
    pd.Series of Hmax values (m).
    """
    v = df["SOGms"].to_numpy(dtype=float)
    l = df["length"].to_numpy(dtype=float)
    h = df["WaterDepth"].to_numpy(dtype=float)
    w = df["displacement_m3"].to_numpy(dtype=float)
    y = np.asarray(dist_m, dtype=float) * np.ones(len(df))

    fr = v / np.sqrt(g * l)

    # Piecewise β and δ coefficients
    low_fr = fr < 0.55
    beta = np.where(low_fr, -0.225 * fr ** (-0.699), -0.342)
    delta = np.where(low_fr, -0.118 * fr ** (-0.356), -0.146)

    a = -0.6 / fr
    b = 0.75 * fr ** (-1.125)
    c = 2.653 * fr - 1.95

    w_cbrt = w ** (1.0 / 3.0)
    h_adim = h / w_cbrt

    # Guard against log of non-positive values
    h_adim_safe = np.where(h_adim > 0, h_adim, np.nan)
    ln_h = np.log(h_adim_safe)

    n = beta * h_adim_safe ** delta
    log_alpha = a + b * ln_h + c * ln_h ** 2
    alpha = np.exp(log_alpha)

    dist_adim = y / w_cbrt
    dist_adim_safe = np.where(dist_adim > 0, dist_adim, np.nan)

    hmax = alpha * dist_adim_safe ** n

    return pd.Series(hmax, index=df.index, name="H_Sorensen")
