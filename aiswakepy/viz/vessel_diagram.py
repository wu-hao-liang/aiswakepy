"""Per-vessel wake diagram: track + wake rays + shore intersections."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from aiswakepy.geo.geodesy import forward_point


def plot_vessel_wake(
    vessel_mmsi: int,
    df_wave: pd.DataFrame,
    df_impact: pd.DataFrame,
    coastline_shp: str | Path,
    output_path: str | Path,
    max_propagation_m: float = 2000.0,
) -> None:
    """Plot a single vessel's track, wake rays, and shore intersection points.

    Parameters
    ----------
    vessel_mmsi:       MMSI of the vessel to plot.
    df_wave:           Wave parameters DataFrame (all vessels).
    df_impact:         Shore impact DataFrame (all vessels).
    coastline_shp:     Path to coastline shapefile.
    output_path:       PNG save path.
    max_propagation_m: Ray length for wake direction arrows.
    """
    vessel_wave = df_wave[df_wave["mmsi"] == vessel_mmsi]
    vessel_impact = df_impact[df_impact["MMSI"] == vessel_mmsi]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Coastline
    try:
        coast = gpd.read_file(str(coastline_shp))
        coast.plot(ax=ax, color="lightgray", edgecolor="black", linewidth=0.5)
    except Exception:
        pass

    if vessel_wave.empty:
        ax.set_title(f"MMSI {vessel_mmsi} — no data")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    # Vessel track
    ax.plot(
        vessel_wave["longitude"], vessel_wave["latitude"],
        "b-", linewidth=1, alpha=0.6, label="Vessel track",
    )

    # Sample wake rays (every 5th point to avoid clutter)
    for _, row in vessel_wave.iloc[::5].iterrows():
        for col in ["WakeDirPort", "WakeDirStarboard"]:
            if col not in row:
                continue
            lon2, lat2 = forward_point(
                row["longitude"], row["latitude"], row[col], max_propagation_m
            )
            ax.plot(
                [row["longitude"], lon2], [row["latitude"], lat2],
                "g-", alpha=0.15, linewidth=0.5,
            )

    # Shore impact points
    if not vessel_impact.empty:
        ax.scatter(
            vessel_impact["ShLongitude"], vessel_impact["ShLatitude"],
            c=vessel_impact["WaveHeight"], cmap="YlOrRd",
            s=40, zorder=5, label="Shore impact",
        )

    ax.set_title(f"MMSI {vessel_mmsi} — Wake diagram")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(fontsize=8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
