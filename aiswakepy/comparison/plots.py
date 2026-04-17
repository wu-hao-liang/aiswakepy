"""Plotting helpers for empirical formula comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


COLOURS: dict[str, str] = {
    "Kriebel":  "olive",
    "PIANC":    "blue",
    "Sorensen": "green",
    "Maynord":  "magenta",
    "Bhowmik":  "#006857",
    "Gates":    "#C8C8C8",
    "Blaauw":   "#D95319",
    "OSSI":     "black",
}


def timeseries_plot(
    df: pd.DataFrame,
    ossi: pd.DataFrame,
    pred_cols: dict[str, str],
    title: str,
    out_path: str | Path,
    time_col: str = "ArrivalTime",
) -> None:
    """Scatter time-series of all formulae + OSSI measurements.

    Parameters
    ----------
    df        : Events DataFrame (one row per wake arrival).
    ossi      : OSSI measurements DataFrame with columns 'time' and 'Hmax'.
    pred_cols : Mapping of legend label → DataFrame column name.
    title     : Plot title.
    out_path  : Path to save the PNG.
    time_col  : Column in *df* that holds the arrival timestamps.
    """
    fig, ax = plt.subplots(figsize=(16, 5))

    ax.scatter(ossi["time"], ossi["Hmax"], s=10, c=COLOURS["OSSI"],
               label="Measurements", zorder=5)

    for label, col in pred_cols.items():
        if col not in df.columns:
            continue
        valid = df[df[col].notna() & (df[col] > 0)]
        ax.scatter(valid[time_col], valid[col], s=10,
                   c=COLOURS.get(label), label=label)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.set_ylabel("$H_{max}$ (m)")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  saved {Path(out_path).name}")
    return fig


def scatter_plot(
    df: pd.DataFrame,
    pred_cols: dict[str, str],
    out_path: str | Path,
) -> None:
    """Predicted vs measured Hmax scatter with 1:1 line.

    Parameters
    ----------
    df        : Events DataFrame with a 'Hmax_measured' column.
    pred_cols : Mapping of legend label → DataFrame column name.
    out_path  : Path to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    has_data = False
    for label, col in pred_cols.items():
        if col not in df.columns:
            continue
        valid = df[["Hmax_measured", col]].dropna()
        if valid.empty:
            continue
        ax.scatter(valid["Hmax_measured"], valid[col], s=20,
                   c=COLOURS.get(label), label=label, alpha=0.7)
        has_data = True

    if has_data:
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], "--k", lw=1, label="1:1")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

    ax.set_xlabel("$H_{max}$ measured (m)")
    ax.set_ylabel("$H_{max}$ predicted (m)")
    ax.set_title("Empirical formulae vs measurements")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  saved {Path(out_path).name}")
    return fig
