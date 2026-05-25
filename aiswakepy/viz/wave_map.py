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

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    vmax: float | None = None,
    lon0: float | None = None,
    lon1: float | None = None,
    lat0: float | None = None,
    lat1: float | None = None,
    zoom: int | str = "auto",
    show: bool = False,
) -> None:
    """Internal helper for a colour-coded scatter map over coastline with optional basemap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if df_impact.empty:
        warnings.warn(f"No data to plot for {label}")
        ax.set_title(f"{title} (no data)")
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return

    # Determine axis extent — user-supplied bounds take priority
    x0 = lon0 if lon0 is not None else df_impact["ShLongitude"].min() - 0.01
    x1 = lon1 if lon1 is not None else df_impact["ShLongitude"].max() + 0.01
    y0 = lat0 if lat0 is not None else df_impact["ShLatitude"].min() - 0.01
    y1 = lat1 if lat1 is not None else df_impact["ShLatitude"].max() + 0.01
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Add satellite basemap if contextily is available
    if HAS_CONTEXTILY:
        try:
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=zoom, crs="EPSG:4326", attribution="")
        except Exception as e:
            warnings.warn(f"Basemap unavailable: {e}")

    # Coastline overlay
    try:
        coast = gpd.read_file(str(coastline_shp))
        coast.plot(ax=ax, color="yellow", edgecolor="orange", linewidth=1.0, alpha=0.25, zorder=2)
    except Exception:
        pass

    df_plot = _downsample(df_impact, coastline_shp, max_points)

    vmin = 0.0
    if vmax is None:
        vmax = math.ceil(df_plot[value_col].max())

    sc = ax.scatter(
        df_plot["ShLongitude"],
        df_plot["ShLatitude"],
        c=df_plot[value_col],
        cmap=cmap,
        s=5,
        alpha=0.8,
        vmin=vmin,
        vmax=vmax,
        zorder=3,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    fig.colorbar(sc, cax=cax, label=label)
    ax.set_title(title or label)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_wave_height_map(
    df_impact: pd.DataFrame,
    coastline_shp: str | Path,
    output_path: str | Path,
    max_points: int = 100_000,
    lon0: float | None = None,
    lon1: float | None = None,
    lat0: float | None = None,
    lat1: float | None = None,
    zoom: int | str = "auto",
    show: bool = False,
) -> None:
    """Scatter map of shore impact points colour-coded by WaveHeight (m).

    Uses Esri WorldImagery basemap (if contextily available) and rainbow colormap.
    Colorbar max is fixed to 0.5 m for consistent scale across runs.
    Pass lon0/lon1/lat0/lat1 to restrict the plot extent; zoom controls tile resolution.
    Set show=True to display inline in Jupyter.
    """
    _plot_impact_map(
        df_impact, coastline_shp,
        value_col="WaveHeight",
        label="Wave Height (m)",
        cmap="turbo",
        output_path=output_path,
        title="Ship Wake Impact — Maximum Wave Height (m)",
        max_points=max_points,
        lon0=lon0, lon1=lon1, lat0=lat0, lat1=lat1,
        zoom=zoom,
        show=show,
    )


def plot_wave_period_map(
    df_impact: pd.DataFrame,
    coastline_shp: str | Path,
    output_path: str | Path,
    max_points: int = 100_000,
    lon0: float | None = None,
    lon1: float | None = None,
    lat0: float | None = None,
    lat1: float | None = None,
    zoom: int | str = "auto",
    show: bool = False,
) -> None:
    """Scatter map of shore impact points colour-coded by WavePeriod (s).

    Uses Esri WorldImagery basemap (if contextily available) and rainbow colormap.
    Colorbar auto-scales to the data range.
    Pass lon0/lon1/lat0/lat1 to restrict the plot extent; zoom controls tile resolution.
    Set show=True to display inline in Jupyter.
    """
    _plot_impact_map(
        df_impact, coastline_shp,
        value_col="WavePeriod",
        label="Wave Period (s)",
        cmap="turbo",
        output_path=output_path,
        title="Ship Wake Impact — Wave Period (s)",
        max_points=max_points,
        lon0=lon0, lon1=lon1, lat0=lat0, lat1=lat1,
        zoom=zoom,
        show=show,
    )
