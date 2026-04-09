"""Bathymetry and tidal water level utilities.

Supports DHI .mesh and .dfsu files via mikeio.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mikeio
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bathymetry loading
# ---------------------------------------------------------------------------

class BathymetryMesh:
    """Wrapper around a loaded mikeio mesh/dfsu geometry for depth lookups."""

    def __init__(self, path: str | Path):
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".mesh":
            msh = mikeio.Mesh(str(path))
            self._geometry = msh.geometry
        elif suffix in (".dfsu", ".dfs2"):
            ds = mikeio.open(str(path))
            self._geometry = ds.geometry
        else:
            raise ValueError(f"Unsupported bathymetry file type: {suffix!r}")

        # Extract node or element coordinates and depths
        # For unstructured meshes, geometry has node_coordinates
        # z is positive-down depth from chart datum (negated bed level)
        geom = self._geometry
        if hasattr(geom, "element_coordinates"):
            coords = geom.element_coordinates   # shape (n, 3): x, y, z
        elif hasattr(geom, "node_coordinates"):
            coords = geom.node_coordinates      # shape (n, 3): x, y, z
        else:
            raise AttributeError("Cannot extract coordinates from geometry")

        self._xy = coords[:, :2]          # (n, 2) lon/lat
        # z stored as negative bed level (below datum) → depth = -z
        self._depth = -coords[:, 2]       # positive depth values

        # Build a simple index for fast nearest-node lookup
        from scipy.spatial import KDTree
        self._tree = KDTree(self._xy)

    def get_depth(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """Return water depth (m) at the nearest mesh node/element.

        Points outside the mesh extent return NaN (distance > 0.5 degree).
        """
        pts = np.column_stack([np.asarray(lons), np.asarray(lats)])
        dist, idx = self._tree.query(pts, workers=-1)
        depths = self._depth[idx].copy().astype(float)
        # Mark points that are too far from any mesh node as NaN
        depths[dist > 0.5] = np.nan
        return depths


def load_bathymetry(path: str | Path) -> BathymetryMesh:
    """Load a DHI .mesh or .dfsu bathymetry file."""
    return BathymetryMesh(path)


# ---------------------------------------------------------------------------
# Tidal water level
# ---------------------------------------------------------------------------

def load_tide(dfs0_path: str | Path) -> pd.Series:
    """Read a .dfs0 tidal prediction file and return a time-indexed Series.

    Values are water level in metres relative to Chart Datum (CD).
    """
    da = mikeio.read(str(dfs0_path), items=0)
    series = da.to_pandas().iloc[:, 0]
    series.index = pd.to_datetime(series.index, utc=False)
    return series


def snap_to_tide(
    obstimes: pd.Series,
    tide_series: pd.Series,
) -> np.ndarray:
    """Snap AIS timestamps to the nearest tide series interval.

    Returns an array of tidal water levels (m) aligned to obstimes.
    Timestamps outside the tide series range return NaN.
    """
    t_obs = pd.to_datetime(obstimes, utc=False)
    tide_idx = tide_series.index

    levels = np.full(len(t_obs), np.nan)
    t_min, t_max = tide_idx[0], tide_idx[-1]

    in_range = (t_obs >= t_min) & (t_obs <= t_max)
    if not in_range.any():
        return levels

    # Use searchsorted to find nearest index
    t_arr = t_obs[in_range]
    idx_right = np.searchsorted(tide_idx, t_arr)
    idx_right = np.clip(idx_right, 1, len(tide_idx) - 1)
    idx_left = idx_right - 1

    t_left = tide_idx[idx_left]
    t_right = tide_idx[idx_right]

    diff_left = (t_arr - t_left).dt.total_seconds().to_numpy()
    diff_right = (t_right - t_arr).dt.total_seconds().to_numpy()

    nearest_idx = np.where(diff_left <= diff_right, idx_left, idx_right)
    levels[in_range.to_numpy()] = tide_series.iloc[nearest_idx].to_numpy()
    return levels
