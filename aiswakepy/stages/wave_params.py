"""Stage 3 — Ship-wake wave parameter calculation.

Calls the configured empirical model (default: Kriebel & Seelig 2005) to
obtain the origin wave height, then computes general propagation parameters
(wave period, energy, spreading angle, wake directions) and applies filters.

To use a different empirical model, swap the import and the call to
``compute_kriebel`` below with the new model function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aiswakepy.models.kriebel import compute_kriebel

_TWO_PI = 2.0 * np.pi

# GIS output column selection (15 columns)
_GIS_COLS = [
    "mmsi", "longitude", "latitude", "Etot",
    "WakeDirPort", "WakeDirStarboard", "H_Kriebel",
    "obstime", "Beta", "FroudeM", "SOGms",
    "width", "length", "Tc", "WaterDepth",
]


def compute_wave_params(
    df: pd.DataFrame,
    cb_method: str = "L_Le",
    g: float = 9.78,
    rho: float = 1026.0,
    min_froude_m: float = 0.1,
    max_froude_m: float = 0.5,
    max_bf: float = 0.4,
    max_sog_knots: float = 12.0,
    max_bl_ratio: float = 0.3,
) -> pd.DataFrame:
    """Compute all wave parameters for each AIS fix.

    Parameters
    ----------
    df:             AIS DataFrame with WaterDepth column (output of assign_depth).
    cb_method:      Block coefficient method: ``"L_Le"``, ``"B_Le"``, or ``"table"``.
    g:              Local gravitational acceleration (m/s^2). Default 9.78 (Singapore).
    rho:            Water density (kg/m^3). Default 1026.
    min_froude_m:   Lower bound for modified Froude filter.
    max_froude_m:   Upper bound for modified Froude filter.
    max_bf:         Maximum BF filter.
    max_sog_knots:  Maximum vessel speed filter (knots).
    max_bl_ratio:   Maximum beam/length ratio filter.

    Returns
    -------
    DataFrame with all computed columns, filtered rows removed.
    """
    # --- Kriebel origin wave height and Froude numbers ---
    df = compute_kriebel(df, cb_method=cb_method, g=g)

    # --- Wave period (empirical, s) ---
    df["T"] = 0.27 * df["sog"]

    # --- Wave energy ---
    df["Emax"] = (rho * g ** 2 * df["H_Kriebel"] ** 2 * df["T"] ** 2) / (16.0 * np.pi)
    df["Etot"] = 10.8 * df["Emax"] ** 0.82

    # --- Wake spreading angle theta (deg) ---
    # Approaches arcsin(1/sqrt(3)) ~= 35.26 deg in deep water (Fd -> 0)
    df["Theta"] = 35.27 * (1.0 - np.exp(12.0 * (df["FroudeD"] - 1.0)))

    # --- Wave celerity component and characteristic period ---
    df["Cel"] = df["SOGms"] * np.cos(np.radians(df["Theta"]))
    df["Tc"] = (_TWO_PI * df["Cel"]) / g

    # --- Wake propagation directions: COG +/- theta ---
    df["WakeDirPort"] = df["cog"] - df["Theta"]
    df["WakeDirStarboard"] = df["cog"] + df["Theta"]

    # --- Beam / Length ratio (used for filtering) ---
    df["BLratio"] = df["width"] / df["length"]

    # --- Row filters ---
    mask = (
        (df["FroudeM"] >= min_froude_m) &
        (df["FroudeM"] <= max_froude_m) &
        (df["BF"] <= max_bf) &
        (df["sog"] <= max_sog_knots) &
        (df["BLratio"] <= max_bl_ratio) &
        (df["WaterDepth"] > 0)
    )
    return df[mask].reset_index(drop=True)


def export_gis(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 15-column GIS-ready subset."""
    cols = [c for c in _GIS_COLS if c in df.columns]
    return df[cols].copy()
