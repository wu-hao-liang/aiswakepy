"""Wave height and wave period map plots."""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.ops import unary_union


def _plot_impact_map(
    df_impact: pd.DataFrame,
    coastline_shp: str | Path,
    value_col: str,
    label: str,
    cmap: str,
    output_path: str | Path,
    title: str = "",
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

    sc = ax.scatter(
        df_impact["ShLongitude"],
        df_impact["ShLatitude"],
        c=df_impact[value_col],
        cmap=cmap,
        s=10,
        alpha=0.7,
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
) -> None:
    """Scatter map of shore impact points colour-coded by WaveHeight (m)."""
    _plot_impact_map(
        df_impact, coastline_shp,
        value_col="WaveHeight",
        label="Wave Height (m)",
        cmap="YlOrRd",
        output_path=output_path,
        title="Ship-wake Shore Impact — Wave Height",
    )


def plot_wave_period_map(
    df_impact: pd.DataFrame,
    coastline_shp: str | Path,
    output_path: str | Path,
) -> None:
    """Scatter map of shore impact points colour-coded by WavePeriod (s)."""
    _plot_impact_map(
        df_impact, coastline_shp,
        value_col="WavePeriod",
        label="Wave Period (s)",
        cmap="Blues",
        output_path=output_path,
        title="Ship-wake Shore Impact — Wave Period",
    )
