"""Report-quality plots and tables for the ship-wake pipeline."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from aiswakepy.viz.wave_map import plot_wave_height_map, plot_wave_period_map

# ---------------------------------------------------------------------------
# AIS ship type code → human-readable category
# Each entry: (code_min, code_max, label)
# ---------------------------------------------------------------------------
_TYPE_RANGES = [
    (80, 89, "Tanker"),
    (70, 79, "Cargo"),
    (60, 69, "Passenger"),
    (40, 49, "High Speed Craft"),
    (52, 52, "Tug"),
    (31, 31, "Towing"),
    (32, 32, "Towing (large)"),
    (30, 30, "Fishing"),
    (33, 33, "Dredger"),
    (34, 34, "Diving Ops"),
    (35, 35, "Military"),
    (36, 36, "Sailing"),
    (37, 37, "Pleasure Craft"),
    (50, 50, "Pilot"),
    (51, 51, "SAR"),
    (53, 53, "Port Tender"),
    (55, 55, "Law Enforcement"),
    (58, 58, "Medical"),
    (90, 99, "Other"),
]

_TYPE_COLORS: dict[str, str] = {
    "Tanker":           "#0055FF",
    "Cargo":            "#FF6600",
    "Passenger":        "#00BB00",
    "High Speed Craft": "#EE0000",
    "Tug":              "#9900EE",
    "Towing":           "#AA3300",
    "Towing (large)":   "#CC8800",
    "Fishing":          "#FF00BB",
    "Dredger":          "#555555",
    "Diving Ops":       "#BBCC00",
    "Military":         "#00CCEE",
    "Sailing":          "#55AAFF",
    "Pleasure Craft":   "#FFCC00",
    "Pilot":            "#00CC55",
    "SAR":              "#FF6644",
    "Port Tender":      "#BB55FF",
    "Law Enforcement":  "#FF55AA",
    "Medical":          "#EEDD00",
    "Other":            "#AAAAAA",
}


def _typecargo_to_label(code) -> str:
    """Map AIS typecargo integer code to a human-readable vessel type label."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "Other"
    for lo, hi, label in _TYPE_RANGES:
        if lo <= code <= hi:
            return label
    return "Other"


def _wave_extent(
    df: pd.DataFrame, margin_frac: float = 0.15
) -> tuple[float, float, float, float]:
    """Return lon0, lon1, lat0, lat1 from ShLongitude/ShLatitude with fractional margin."""
    lon_min = float(df["ShLongitude"].min())
    lon_max = float(df["ShLongitude"].max())
    lat_min = float(df["ShLatitude"].min())
    lat_max = float(df["ShLatitude"].max())
    m_lon = max(lon_max - lon_min, 0.001) * margin_frac
    m_lat = max(lat_max - lat_min, 0.001) * margin_frac
    return lon_min - m_lon, lon_max + m_lon, lat_min - m_lat, lat_max + m_lat


def plot_wave_height_report(
    df_wave_impact: pd.DataFrame,
    coastline_shp: str | Path,
    output_path: str | Path,
    margin_frac: float = 0.15,
    max_points: int = 100_000,
    zoom: int | str = "auto",
    show: bool = False,
) -> None:
    """Wave height map with extent auto-fitted to wave impact points."""
    lon0, lon1, lat0, lat1 = _wave_extent(df_wave_impact, margin_frac)
    plot_wave_height_map(
        df_wave_impact, coastline_shp, output_path,
        max_points=max_points,
        lon0=lon0, lon1=lon1, lat0=lat0, lat1=lat1,
        zoom=zoom, show=show,
    )


def plot_wave_period_report(
    df_wave_impact: pd.DataFrame,
    coastline_shp: str | Path,
    output_path: str | Path,
    margin_frac: float = 0.15,
    max_points: int = 100_000,
    zoom: int | str = "auto",
    show: bool = False,
) -> None:
    """Wave period map with extent auto-fitted to wave impact points."""
    lon0, lon1, lat0, lat1 = _wave_extent(df_wave_impact, margin_frac)
    plot_wave_period_map(
        df_wave_impact, coastline_shp, output_path,
        max_points=max_points,
        lon0=lon0, lon1=lon1, lat0=lat0, lat1=lat1,
        zoom=zoom, show=show,
    )


def top_vessels_table(
    df_wave_impact: pd.DataFrame,
    n: int = 10,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Top-n vessels by peak shore wave height.

    Returns DataFrame columns: MMSI, VesselType, MaxWaveHeight_m.
    Requires 'typecargo' in df_wave_impact (carried through from wave_impact stage).
    """
    if df_wave_impact.empty:
        return pd.DataFrame(columns=["MMSI", "VesselType", "MaxWaveHeight_m"])

    top = (
        df_wave_impact.groupby("MMSI")["WaveHeight"]
        .max()
        .sort_values(ascending=False)
        .head(n)
        .rename("MaxWaveHeight_m")
        .reset_index()
    )

    if "typecargo" in df_wave_impact.columns:
        type_per_mmsi = (
            df_wave_impact.groupby("MMSI")["typecargo"]
            .agg(lambda x: x.mode().iloc[0])
        )
        top["VesselType"] = top["MMSI"].map(type_per_mmsi).apply(_typecargo_to_label)
    else:
        top["VesselType"] = "Unknown"

    result = top[["MMSI", "VesselType", "MaxWaveHeight_m"]]
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


def plot_vessel_track_scatter(
    df_vessel: pd.DataFrame,
    df_wave_impact: pd.DataFrame,
    output_path: str | Path,
    show: bool = False,
) -> None:
    """Scatter of vessel speed vs length, coloured by vessel type.

    Only plots vessel track points whose segment_id appears in df_wave_impact —
    i.e. tracks that produced at least one shore impact after any applied filters.
    x-axis: SOG (knots), y-axis: vessel length (m).
    """
    if df_wave_impact.empty or df_vessel.empty:
        print("plot_vessel_track_scatter: no data — skipping")
        return

    active_segs = set(int(s) for s in df_wave_impact["segment_id"].unique())
    df_plot = df_vessel[df_vessel["segment_id"].isin(active_segs)].copy()
    if df_plot.empty:
        print("plot_vessel_track_scatter: no vessel points on active segments — skipping")
        return

    import matplotlib.lines as mlines

    df_plot["_VesselType"] = df_plot["typecargo"].apply(_typecargo_to_label)
    types_present = set(df_plot["_VesselType"].unique())

    fig, ax = plt.subplots(figsize=(10, 7))
    for vtype, color in _TYPE_COLORS.items():
        if vtype not in types_present:
            continue
        mask = df_plot["_VesselType"] == vtype
        ax.scatter(
            df_plot.loc[mask, "sog"],
            df_plot.loc[mask, "length"],
            c=color,
            s=10,
            alpha=0.7,
            linewidths=0,
            zorder=3,
        )

    legend_handles = [
        mlines.Line2D(
            [], [], color=c, marker="o", linestyle="None", markersize=6, label=vt,
            alpha=1.0 if vt in types_present else 0.3,
        )
        for vt, c in _TYPE_COLORS.items()
    ]

    ax.set_xlabel("Vessel Speed (knots)")
    ax.set_ylabel("Vessel Length (m)")
    ax.set_title("Vessel Track Statistics by Type")
    ax.legend(
        handles=legend_handles,
        title="Vessel Type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        markerscale=1,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Vessel track scatter saved to {Path(output_path).resolve()}")
