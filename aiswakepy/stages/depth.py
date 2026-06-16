"""Stage 2 — Water depth and tidal level assignment.

Combines static bathymetric depth (from .mesh/.dfsu) with predicted tidal
water level (from .dfs0), then filters records with insufficient under-keel
clearance.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from aiswakepy.geo.bathymetry import BathymetryMesh, load_bathymetry, load_tide, snap_to_tide


def assign_depth(
    df: pd.DataFrame,
    bathy_path: str | Path | None = None,
    constant_depth_m: float = 15.0,
    tide_dfs0_path: str | Path | None = None,
    tide_item: str | None = None,
    underkeel_margin_m: float = 1.0,
    _bathy: BathymetryMesh | None = None,
) -> pd.DataFrame:
    """Assign total water depth to each AIS fix; filter under-keel violations.

    Parameters
    ----------
    df:               Filtered AIS DataFrame (output of filter_ais).
    bathy_path:       Optional path to .mesh or .dfsu bathymetry file.
    constant_depth_m: Depth used when no bathymetry file is supplied.
    tide_dfs0_path:   Optional path to .dfs0 tidal prediction file.
    tide_item:        Item name to read from the .dfs0 file.
    underkeel_margin_m: Minimum required clearance above draught (m).
    _bathy:           Pre-loaded BathymetryMesh (for testing / caching).

    Returns
    -------
    DataFrame with added ``WaterDepth`` column; rows with NaN depth or
    insufficient under-keel clearance are removed.
    """
    from aiswakepy._progress import Spinner
    spinner = Spinner(desc="assign depth and filter by draught")

    df = df.copy()

    lons = df["longitude"].to_numpy()
    lats = df["latitude"].to_numpy()
    if _bathy is not None:
        depths = _bathy.get_depth(lons, lats)
    elif bathy_path is not None:
        depths = load_bathymetry(bathy_path).get_depth(lons, lats)
    else:
        if constant_depth_m <= 0:
            raise ValueError("constant_depth_m must be positive")
        depths = np.full(len(df), float(constant_depth_m), dtype=float)

    # Add tidal level if provided
    if tide_dfs0_path is not None:
        tide_series = load_tide(tide_dfs0_path, item=tide_item)

        # Verify AIS time range is covered by the tide series
        ais_t_min = pd.Timestamp(df["obstime"].min())
        ais_t_max = pd.Timestamp(df["obstime"].max())
        tide_t_min = tide_series.index.min()
        tide_t_max = tide_series.index.max()
        if ais_t_min < tide_t_min or ais_t_max > tide_t_max:
            warnings.warn(
                f"AIS time range [{ais_t_min:%Y-%m-%d %H:%M} – {ais_t_max:%Y-%m-%d %H:%M}] "
                f"extends beyond tide series [{tide_t_min:%Y-%m-%d %H:%M} – {tide_t_max:%Y-%m-%d %H:%M}]. "
                f"Records outside this range will be dropped."
            )

        tide_levels = snap_to_tide(df["obstime"], tide_series)
        # Only add tide where both depth and tide are valid
        valid_tide = ~np.isnan(tide_levels)
        depths[valid_tide] += tide_levels[valid_tide]
        # Outside tide range → NaN
        depths[~valid_tide] = np.nan

    df["WaterDepth"] = depths

    # Sort for meaningful adjacency when setting force-break flags.
    df = df.sort_values(["mmsi", "obstime"]).reset_index(drop=True)

    # Combined removal mask: NaN depth OR insufficient clearance.
    remove = df["WaterDepth"].isna().to_numpy().copy()
    remove |= (df["WaterDepth"] < df["draught"] + underkeel_margin_m).to_numpy()

    from aiswakepy.stages.filter import _set_force_breaks
    _set_force_breaks(df, remove)

    df = df[~remove]
    result = df.reset_index(drop=True)
    spinner.done(rows=len(result))
    return result
