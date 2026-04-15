"""Stage 3 — Vessel parameter computation and wake propagation geometry.

Computes vessel-specific parameters (block coefficient, displacement, speed)
and wake propagation quantities (Kelvin angle, wake directions, wave period)
for each AIS fix.  Does NOT compute wave height — that is done in stage 4
(``compute_wave_impact``) once shore/point distances are known.

Supported formulas, selected and applied in the wave_impact stage
-----------------------------------------------------------------
``"kriebel"`` (default) — Kriebel & Seelig (2005)
``"sorensen"``          — Sorensen (1984)
``"gates"``             — Gates & Herbich (1977)
``"blaauw"``            — Blaauw et al. (1985)
``"maynord"``           — Maynord (2005)
``"pianc"``             — PIANC (1987)
``"bhowmik"``           — Bhowmik et al. (1982)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aiswakepy.vessel.block_coeff import get_vessel_params_df

_KNOTS_TO_MS = 0.5144444
_TWO_PI = 2.0 * np.pi

# GIS output column selection — vessel propagation parameters
_GIS_COLS = [
    "mmsi", "longitude", "latitude",
    "WakeDirPort", "WakeDirStarboard",
    "obstime", "FroudeD", "SOGms",
    "width", "length", "Tc", "WaterDepth",
    "LengthWL", "Theta", "BLratio",
]


def compute_vessel_params(
    df: pd.DataFrame,
    cb_method: str = "L_Le",
    g: float = 9.78,
    max_sog_knots: float = 12.0,
    max_bl_ratio: float = 0.3,
) -> pd.DataFrame:
    """Compute vessel and wake propagation parameters for each AIS fix.

    Parameters
    ----------
    df:             AIS DataFrame with ``WaterDepth`` column (output of
                    ``assign_depth``).
    cb_method:      Block coefficient method: ``"L_Le"``, ``"B_Le"``, or
                    ``"table"``.  Determines how ``block_coeff`` and
                    ``bow_entry_m`` are derived.
    g:              Local gravitational acceleration (m/s²). Default 9.78.
    max_sog_knots:  Maximum vessel speed filter (knots).
    max_bl_ratio:   Maximum beam/length ratio filter.

    Returns
    -------
    DataFrame with vessel and propagation columns, filtered rows removed.
    Wave height is NOT computed here — formula-specific intermediates (e.g.
    Kriebel's Alpha, Beta, FroudeM, BF) are computed inside each formula
    function in the wave_impact stage.

    Columns added
    -------------
    block_coeff, bow_entry_m, displacement_m3   — vessel params
    SOGms    — speed (m/s)
    LengthWL — waterline length = 0.8 * LOA (m)
    FroudeD  — depth Froude number V / sqrt(g * h)
    T        — divergent wave period 0.27 * SOG_knots (s)
    Theta    — Kelvin wake half-angle (deg)
    Cel      — divergent wave celerity V * cos(Theta) (m/s)
    Tc       — depth-adjusted wave period 2π * Cel / g (s)
    WakeDirPort, WakeDirStarboard — COG ± Theta (deg)
    BLratio  — beam/length ratio
    """
    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="compute_vessel_params")

    # --- Vessel params (block_coeff, bow_entry_m, displacement_m3) ---
    df = get_vessel_params_df(df, method=cb_method)

    # --- Universal intermediates ---

    # SOGms: speed over ground in m/s
    df["SOGms"] = df["sog"] * _KNOTS_TO_MS

    # LengthWL: waterline length = 0.8 * LOA
    df["LengthWL"] = df["length"] * 0.8

    # FroudeD = V / sqrt(g * h): depth Froude number
    df["FroudeD"] = df["SOGms"] / np.sqrt(g * df["WaterDepth"])

    # --- T = 0.27 * SOG (knots→s): divergent wave period ---
    # From deep-water Kelvin wake dispersion: T = 2π·V·cos(θ)/g, where
    # θ = arcsin(1/√3) ≈ 35.26°, cos(θ) ≈ 0.8165, and 1 knot = 0.5144 m/s
    # gives coefficient 2π × 0.8165 × 0.5144 / 9.81 ≈ 0.27. — Kirkegaard et al. (1998)
    df["T"] = 0.27 * df["sog"]

    # --- Theta: angle of diverging waves relative to vessel heading (deg) ---
    # Deep-water Kelvin limit → arcsin(1/√3) ≈ 35.26°; shrinks to 0° as Fd→1
    # (critical depth speed). Empirical curve fit to Havelock (1908) finite-depth
    # theory. — Kriebel & Seelig (2005); Havelock (1908)
    _theta_deep = np.degrees(np.arcsin(1.0 / np.sqrt(3.0)))
    df["Theta"] = _theta_deep * (1.0 - np.exp(12.0 * (df["FroudeD"] - 1.0)))

    # --- Cel = V·cos(θ): divergent wave celerity (m/s) ---
    df["Cel"] = df["SOGms"] * np.cos(np.radians(df["Theta"]))

    # --- Tc = 2π·Cel/g: depth-adjusted divergent wave period (s) ---
    df["Tc"] = (_TWO_PI * df["Cel"]) / g

    # --- Wake propagation directions: COG ± Theta ---
    df["WakeDirPort"] = df["cog"] - df["Theta"]
    df["WakeDirStarboard"] = df["cog"] + df["Theta"]

    # --- Beam / Length ratio (used for filtering) ---
    df["BLratio"] = df["width"] / df["length"]

    # --- Row filters ---
    # Formula-specific validity filters (e.g. FroudeM range for Kriebel) are
    # applied inside each formula function in the wave_impact stage.
    mask = (
        (df["sog"] <= max_sog_knots) &
        (df["BLratio"] <= max_bl_ratio) &
        (df["WaterDepth"] > 0)
    )
    result = df[mask].reset_index(drop=True)
    spinner.done(rows=len(result))
    return result


def export_gis(df: pd.DataFrame) -> pd.DataFrame:
    """Return the GIS-ready vessel-parameter subset."""
    cols = [c for c in _GIS_COLS if c in df.columns]
    return df[cols].copy()
