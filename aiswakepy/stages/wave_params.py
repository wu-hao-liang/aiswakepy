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
    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="compute_wave_params")

    # --- Kriebel origin wave height and Froude numbers ---
    df = compute_kriebel(df, cb_method=cb_method, g=g)

    # --- T = 0.27 * SOG (knots→s): divergent wave period ---
    # From deep-water Kelvin wake dispersion: T = 2π·V·cos(θ)/g, where
    # θ = arcsin(1/√3) ≈ 35.26°, cos(θ) ≈ 0.8165, and 1 knot = 0.5144 m/s
    # gives coefficient 2π × 0.8165 × 0.5144 / 9.81 ≈ 0.27. — Kirkegaard et al. (1998)
    df["T"] = 0.27 * df["sog"]

    # --- Emax = ρg²H²T²/(16π): maximum wave energy per unit crest width (J/m) ---
    # Energy of one wave crest = E_density × L₀, where E_density = ρgH²/8
    # (energy per unit surface area) and L₀ = gT²/(2π) (deep-water wavelength).
    # Using H_Kriebel approximates the energy of the maximum crest. — linear wave theory
    df["Emax"] = (rho * g ** 2 * df["H_Kriebel"] ** 2 * df["T"] ** 2) / (16.0 * np.pi)

    # --- Etot = 10.8 * Emax^0.82: total wake group energy (J/m) ---
    # Empirical power-law fit relating total wake energy to the maximum crest
    # energy; accounts for all waves in the wake group. — Sorensen (1997)
    df["Etot"] = 10.8 * df["Emax"] ** 0.82

    # --- Theta: angle of diverging waves relative to vessel heading (deg) ---
    # Deep-water Kelvin limit → arcsin(1/√3) ≈ 35.26°; shrinks to 0° as Fd→1
    # (critical depth speed). Empirical curve fit to Havelock (1908) finite-depth
    # theory. — Kriebel & Seelig (2005); Havelock (1908)
    _theta_deep = np.degrees(np.arcsin(1.0 / np.sqrt(3.0)))
    df["Theta"] = _theta_deep * (1.0 - np.exp(12.0 * (df["FroudeD"] - 1.0)))

    # --- Cel = V·cos(θ): divergent wave celerity (m/s) ---
    # Vessel speed component in the wave propagation direction; the Kelvin
    # stationarity condition fixes the wave pattern relative to the ship.
    df["Cel"] = df["SOGms"] * np.cos(np.radians(df["Theta"]))

    # --- Tc = 2π·Cel/g: depth-adjusted divergent wave period (s) ---
    # From deep-water dispersion T = 2πc/g; generalises T = 0.27×SOG
    # to finite depth where θ (and therefore Cel) varies with Fd.
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
    result = df[mask].reset_index(drop=True)
    spinner.done(rows=len(result))
    return result


def export_gis(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 15-column GIS-ready subset."""
    cols = [c for c in _GIS_COLS if c in df.columns]
    return df[cols].copy()
