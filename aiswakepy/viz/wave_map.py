"""Wave height and wave period map plots."""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import linemerge, unary_union


def _bin_top_n(
    df_impact: pd.DataFrame,
    coastline_shp: str | Path,
    top_n: int,
) -> pd.DataFrame:
    """Keep at most *top_n* highest-WaveHeight points per 1-metre coastline bin.

    Points are sorted ascending by WaveHeight so the highest values render on
    top (last-drawn = highest z-order in matplotlib).
    """
    if df_impact.empty:
        return df_impact

    coast = gpd.read_file(str(coastline_shp))
    boundary = unary_union(coast.geometry).boundary
    coastline_line = (
        linemerge(boundary)
        if boundary.geom_type == "MultiLineString"
        else boundary
    )

    dist_along = np.array([
        coastline_line.project(Point(lon, lat))
        for lon, lat in zip(df_impact["ShLongitude"], df_impact["ShLatitude"])
    ])

    df = df_impact.copy()
    df["_bin"] = dist_along.astype(int)

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
    top_n_per_bin: int | None = None,
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

    df_plot = (
        _bin_top_n(df_impact, coastline_shp, top_n_per_bin)
        if top_n_per_bin is not None
        else df_impact
    )

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
    top_n_per_bin: int | None = None,
) -> None:
    """Scatter map of shore impact points colour-coded by WaveHeight (m)."""
    _plot_impact_map(
        df_impact, coastline_shp,
        value_col="WaveHeight",
        label="Wave Height (m)",
        cmap="YlOrRd",
        output_path=output_path,
        title="Ship-wake Shore Impact — Wave Height",
        top_n_per_bin=top_n_per_bin,
    )


def plot_wave_period_map(
    df_impact: pd.DataFrame,
    coastline_shp: str | Path,
    output_path: str | Path,
    top_n_per_bin: int | None = None,
) -> None:
    """Scatter map of shore impact points colour-coded by WavePeriod (s)."""
    _plot_impact_map(
        df_impact, coastline_shp,
        value_col="WavePeriod",
        label="Wave Period (s)",
        cmap="Blues",
        output_path=output_path,
        title="Ship-wake Shore Impact — Wave Period",
        top_n_per_bin=top_n_per_bin,
    )
