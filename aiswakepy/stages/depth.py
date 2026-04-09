"""Stage 2 — Water depth and tidal level assignment.

Combines static bathymetric depth (from .mesh/.dfsu) with predicted tidal
water level (from .dfs0), then filters records with insufficient under-keel
clearance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aiswakepy.geo.bathymetry import BathymetryMesh, load_bathymetry, load_tide, snap_to_tide


def assign_depth(
    df: pd.DataFrame,
    bathy_path: str | Path,
    tide_dfs0_path: str | Path | None = None,
    underkeel_margin_m: float = 1.0,
    _bathy: BathymetryMesh | None = None,
) -> pd.DataFrame:
    """Assign total water depth to each AIS fix; filter under-keel violations.

    Parameters
    ----------
    df:               Filtered AIS DataFrame (output of filter_ais).
    bathy_path:       Path to .mesh or .dfsu bathymetry file.
    tide_dfs0_path:   Optional path to .dfs0 tidal prediction file.
    underkeel_margin_m: Minimum required clearance above draught (m).
    _bathy:           Pre-loaded BathymetryMesh (for testing / caching).

    Returns
    -------
    DataFrame with added ``WaterDepth`` column; rows with NaN depth or
    insufficient under-keel clearance are removed.
    """
    df = df.copy()

    bathy = _bathy if _bathy is not None else load_bathymetry(bathy_path)

    lons = df["longitude"].to_numpy()
    lats = df["latitude"].to_numpy()
    depths = bathy.get_depth(lons, lats)

    # Add tidal level if provided
    if tide_dfs0_path is not None:
        tide_series = load_tide(tide_dfs0_path)
        tide_levels = snap_to_tide(df["obstime"], tide_series)
        # Only add tide where both depth and tide are valid
        valid_tide = ~np.isnan(tide_levels)
        depths[valid_tide] += tide_levels[valid_tide]
        # Outside tide range → NaN
        depths[~valid_tide] = np.nan

    df["WaterDepth"] = depths

    # Drop NaN depths (outside mesh or outside tide range)
    df = df.dropna(subset=["WaterDepth"])

    # Under-keel clearance filter
    df = df[df["WaterDepth"] >= df["draught"] + underkeel_margin_m]

    return df.reset_index(drop=True)
