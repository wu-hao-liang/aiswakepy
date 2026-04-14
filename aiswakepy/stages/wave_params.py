"""Stage 3 — Ship-wake wave parameter calculation.

Calls the configured empirical model (default: Kriebel & Seelig 2005) via the
``formula`` argument to obtain the origin wave height, then computes general
propagation parameters (wave period, energy, spreading angle, wake directions)
and applies filters.

Supported formulas
------------------
``"kriebel"`` (default) — Kriebel & Seelig (2005).  Formula-specific validity
limits (FroudeM range, BF ceiling) are applied inside the formula function;
rows outside those limits have their H column set to NaN and are dropped here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aiswakepy.models.kriebel import compute_kriebel
from aiswakepy.models.bhowmik import compute_bhowmik
from aiswakepy.models.blaauw import compute_blaauw
from aiswakepy.models.gates import compute_gates
from aiswakepy.models.maynord import compute_maynord
from aiswakepy.models.pianc import compute_pianc
from aiswakepy.models.sorensen import compute_sorensen
from aiswakepy.vessel.block_coeff import get_vessel_params_df

_TWO_PI = 2.0 * np.pi

# Maps formula name → (compute function, H-column name produced by that function)
_FORMULA_REGISTRY: dict[str, tuple] = {
    "bhowmik": (compute_bhowmik, "H_Bhowmik"),
    "blaauw": (compute_blaauw, "H_Blaauw"),
    "gates": (compute_gates, "H_Gates"),
    "kriebel": (compute_kriebel, "H_Kriebel"),
    "maynord": (compute_maynord, "H_Maynord"),
    "pianc": (compute_pianc, "H_PIANC"),
    "sorensen": (compute_sorensen, "H_Sorensen"),
}

# GIS output column selection (15 columns)
_GIS_COLS = [
    "mmsi", "longitude", "latitude", "Etot",
    "WakeDirPort", "WakeDirStarboard", "H_Kriebel",
    "obstime", "Beta", "FroudeM", "SOGms",
    "width", "length", "Tc", "WaterDepth",
]


def compute_wave_params(
    df: pd.DataFrame,
    formula: str = "kriebel",
    cb_method: str = "L_Le",
    g: float = 9.78,
    rho: float = 1026.0,
    max_sog_knots: float = 12.0,
    max_bl_ratio: float = 0.3,
    **formula_kwargs,
) -> pd.DataFrame:
    """Compute all wave parameters for each AIS fix.

    Parameters
    ----------
    df:             AIS DataFrame with WaterDepth column (output of assign_depth).
    formula:        Empirical wake model to use. Options: ``"bhowmik"``, ``"blaauw"``,
                    ``"gates"``, ``"kriebel"`` (default), ``"maynord"``, ``"pianc"``,
                    ``"sorensen"``.
    cb_method:      Block coefficient method: ``"L_Le"``, ``"B_Le"``, or ``"table"``.
                    Determines how ``block_coeff`` and ``bow_entry_m`` are derived;
                    applied before calling any formula.
    g:              Local gravitational acceleration (m/s^2). Default 9.78 (Singapore).
    rho:            Water density (kg/m^3). Default 1026.
    max_sog_knots:  Maximum vessel speed filter (knots).
    max_bl_ratio:   Maximum beam/length ratio filter.
    **formula_kwargs:
                    Extra keyword arguments forwarded to the formula function:
                    - kriebel: ``min_froude_m``, ``max_froude_m``, ``max_bf``
                    - blaauw: ``max_fd`` (max depth Froude)
                    - gates: ``max_fr`` (max length Froude)
                    - maynord: ``min_fr_dis``, ``min_fr``, ``min_depth_ratio``
                    - pianc: ``max_fr``, ``max_fd``
                    - bhowmik, sorensen: no formula-specific parameters

    Returns
    -------
    DataFrame with all computed columns, filtered rows removed.
    """
    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="compute_wave_params")

    if formula not in _FORMULA_REGISTRY:
        raise ValueError(
            f"Unknown formula {formula!r}. Supported: {list(_FORMULA_REGISTRY)}"
        )
    compute_fn, h_col = _FORMULA_REGISTRY[formula]

    # --- Vessel params (block_coeff, bow_entry_m, displacement_m3) ---
    df = get_vessel_params_df(df, method=cb_method)

    # --- Origin wave height and Froude numbers (formula-specific) ---
    df = compute_fn(df, g=g, **formula_kwargs)

    # --- T = 0.27 * SOG (knots→s): divergent wave period ---
    # From deep-water Kelvin wake dispersion: T = 2π·V·cos(θ)/g, where
    # θ = arcsin(1/√3) ≈ 35.26°, cos(θ) ≈ 0.8165, and 1 knot = 0.5144 m/s
    # gives coefficient 2π × 0.8165 × 0.5144 / 9.81 ≈ 0.27. — Kirkegaard et al. (1998)
    df["T"] = 0.27 * df["sog"]

    # --- Emax = ρg²H²T²/(16π): maximum wave energy per unit crest width (J/m) ---
    # Energy of one wave crest = E_density × L₀, where E_density = ρgH²/8
    # (energy per unit surface area) and L₀ = gT²/(2π) (deep-water wavelength).
    # Uses the formula's origin wave height H to approximate the maximum crest
    # energy. — linear wave theory
    df["Emax"] = (rho * g ** 2 * df[h_col] ** 2 * df["T"] ** 2) / (16.0 * np.pi)

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
    # Formula-specific limits (e.g. FroudeM, BF for Kriebel) are already applied
    # inside the formula function as NaN on the H column.
    mask = (
        df[h_col].notna() &
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
