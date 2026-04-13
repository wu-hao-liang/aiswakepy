"""Wave height and wave period map plots."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.ops import linemerge, unary_union


def _downsample(
    df_impact: pd.DataFrame,
    coastline_shp: str | Path,
    max_points: int,
) -> pd.DataFrame:
    """Downsample to at most *max_points* by keeping the top-N highest-WaveHeight
    points per 1-metre coastline bin.

    Only activates when ``len(df_impact) > max_points``.  N per bin is computed
    as ``ceil(max_points / n_occupied_bins)`` so the actual output is ≤ max_points.

    Points are returned sorted ascending by WaveHeight so the highest values
    render on top (last-drawn = highest z-order in matplotlib).
    """
    if df_impact.empty or len(df_impact) <= max_points:
        return df_impact.sort_values("WaveHeight", ascending=True).reset_index(drop=True)

    coast = gpd.read_file(str(coastline_shp))

    # Reproject to a metre-based CRS so project() returns distances in metres
    metric_crs = coast.estimate_utm_crs()
    coast_m = coast.to_crs(metric_crs)
    boundary = unary_union(coast_m.geometry).boundary
    coastline_line_m = (
        linemerge(boundary)
        if boundary.geom_type == "MultiLineString"
        else boundary
    )

    # Reproject impact points to the same metric CRS
    points_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df_impact["ShLongitude"], df_impact["ShLatitude"]),
        crs="EPSG:4326",
    ).to_crs(metric_crs)

    dist_along = np.array([
        coastline_line_m.project(geom)
        for geom in points_gdf.geometry
    ])

    df = df_impact.copy()
    df["_bin"] = dist_along.astype(int)   # 1-metre bins in metric CRS

    n_occupied_bins = df["_bin"].nunique()
    top_n = math.ceil(max_points / n_occupied_bins)

    df = (
        df.sort_values("WaveHeight", ascending=False)
        .groupby("_bin", sort=False)
        .head(top_n)
        .drop(columns=["_bin"])
        .sort_values("WaveHeight", ascending=True)
        .reset_index(drop=True)
    )
    return df


def _plot_impact_map(
    df_impact: pd.DataFrame,
    coastline_shp: str | Path,
    value_col: str,
    label: str,
    cmap: str,
    output_path: str | Path,
    title: str = "",
    max_points: int = 100_000,
) -> None:
    """Internal helper for a colour-coded scatter map over coastline."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Coastline
    try:
        coast = gpd.read_file(str(coastline_shp))
        coast.plot(ax=ax, color="lightgray", edgecolor="black", linewidth=0.5)
    except Exception:
        pass

    if df_impact.empty:
        warnings.warn(f"No data to plot for {label}")
        ax.set_title(f"{title} (no data)")
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    df_plot = _downsample(df_impact, coastline_shp, max_points)

    sc = ax.scatter(
        df_plot["ShLongitude"],
        df_plot["ShLatitude"],
        c=df_plot[value_col],
        cmap=cmap,
        s=5,
        alpha=0.8,
        zorder=3,
    )
    plt.colorbar(sc, ax=ax, label=label)
    ax.set_title(title or label)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wave_height_map(
    df_impact: pd.DataFrame,
    coastline_shp: str | Path,
    output_path: str | Path,
    max_points: int = 100_000,
) -> None:
    """Scatter map of shore impact points colour-coded by WaveHeight (m)."""
    _plot_impact_map(
        df_impact, coastline_shp,
        value_col="WaveHeight",
        label="Wave Height (m)",
        cmap="YlOrRd",
        output_path=output_path,
        title="Ship-wake Shore Impact — Wave Height",
        max_points=max_points,
    )


def plot_wave_period_map(
    df_impact: pd.DataFrame,
    coastline_shp: str | Path,
    output_path: str | Path,
    max_points: int = 100_000,
) -> None:
    """Scatter map of shore impact points colour-coded by WavePeriod (s)."""
    _plot_impact_map(
        df_impact, coastline_shp,
        value_col="WavePeriod",
        label="Wave Period (s)",
        cmap="Blues",
        output_path=output_path,
        title="Ship-wake Shore Impact — Wave Period",
        max_points=max_points,
    )
