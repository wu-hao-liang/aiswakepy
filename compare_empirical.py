"""compare_empirical.py — Compare 7 empirical ship-wake formulae against
OSSI wave gauge measurements.

Replicates the analysis from MATLAB scripts:
    ref/WUHL_01_EmpiricalFormulations_AISdata_B_Le.m
    ref/WUHL_02_Comparison_AIS_OSSI_B_Le.m

The comparison uses geometrically correct wake propagation: for each pair of
consecutive AIS positions within a trajectory segment, the exact point where
the vessel's wake direction aligns with the bearing to the OSSI gauge is found
by bisection.  The propagation distance and lateral distance are computed at
that point and fed to all empirical formulae.  Arrival times use deep-water
group velocity (c_g = g·T / 4π).

Usage
-----
    python compare_empirical.py \\
        --ais   output/AIS_wave_params.csv \\
        --ossi  data/ShipWake_peaks_event3minutes_minHeight5cm.xlsx \\
        --gauge-lon 103.733335 \\
        --gauge-lat 1.265771 \\
        --out   output/comparison/

Arguments
---------
--ais           : Path to pipeline wave-params output (CSV or Parquet).
                  Must contain columns: obstime, longitude, latitude,
                  SOGms, length, width, draught, typecargo, WaterDepth,
                  FroudeD, FroudeM, Beta, BF, LengthWL, Tc, Theta,
                  WakeDirPort, WakeDirStarboard, cog, segment_id,
                  block_coeff, bow_entry_m.
                  If displacement_m3 is absent it is computed from
                  width, draught, length, block_coeff.
--ossi          : Excel (.xlsx) file with pre-extracted ship-wake events.
                  Sheet "SHIPWAKE" with columns:
                    col B (index 1): timestamp
                    col C (index 2): Hmax (m)
                    col E (index 4): T (s)
                  Time can be Excel serial date (float) or datetime.
--gauge-lon     : Longitude of OSSI gauge (decimal degrees).
--gauge-lat     : Latitude of OSSI gauge (decimal degrees).
--out           : Output directory for plots and CSV (default: output/comparison/).
--event-window  : Matching window in minutes (default: 0.5).
--g             : Gravity m/s² (default: 9.78).
--cb-method     : Block coefficient method: L_Le / B_Le / table (default: B_Le).
--clean-draft   : Drop records with draught = 0 (default: True).

Outputs
-------
  timeseries_all.png        — All formulae + measurements vs time
  timeseries_matched.png    — Matched events vs time
  scatter_predicted_vs_measured.png — Hmax predicted vs measured (1:1 line)
  matched_events.csv        — Table of matched (AIS, OSSI) pairs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Model imports
# ---------------------------------------------------------------------------
from aiswakepy.models.pianc    import compute_pianc
from aiswakepy.models.bhowmik  import compute_bhowmik
from aiswakepy.models.gates    import compute_gates
from aiswakepy.models.blaauw   import compute_blaauw, A_LOADED, A_MODERATE, A_LIGHT
from aiswakepy.models.sorensen import compute_sorensen
from aiswakepy.models.maynord  import compute_maynord
from aiswakepy.vessel.block_coeff import get_vessel_params_df
from aiswakepy.stages.shore_impact import compute_point_impact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ais(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    df["obstime"] = pd.to_datetime(df["obstime"])
    return df


def _ensure_vessel_cols(df: pd.DataFrame, cb_method: str) -> pd.DataFrame:
    """Add block_coeff, bow_entry_m, displacement_m3 if missing."""
    need_params = "block_coeff" not in df.columns or "bow_entry_m" not in df.columns
    if need_params:
        df = get_vessel_params_df(df, method=cb_method)
    if "displacement_m3" not in df.columns:
        df["displacement_m3"] = (
            df["width"].to_numpy(dtype=float)
            * df["draught"].to_numpy(dtype=float)
            * df["length"].to_numpy(dtype=float)
            * 0.95
            * df["block_coeff"].to_numpy(dtype=float)
        )
    return df



def _excel_serial_to_datetime(serial: float) -> pd.Timestamp:
    """Convert MATLAB/Excel serial date (days since 1899-12-30) to Timestamp."""
    return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(serial))


def _load_ossi(path: str | Path) -> pd.DataFrame:
    """Load OSSI wave gauge events from Excel.

    Expected sheet: 'SHIPWAKE'
    Column B (index 1): time (Excel serial or datetime)
    Column C (index 2): Hmax (m)
    Column E (index 4): T (s)
    """
    path = Path(path)
    raw = pd.read_excel(path, sheet_name="SHIPWAKE", header=None)

    time_col = raw.iloc[:, 1]
    hmax_col = raw.iloc[:, 2].to_numpy(dtype=float)
    t_col    = raw.iloc[:, 4].to_numpy(dtype=float)

    # Parse time: could be datetime (pandas Timestamp) or Excel serial float
    if pd.api.types.is_float_dtype(time_col) or pd.api.types.is_integer_dtype(time_col):
        times = pd.to_datetime(
            [_excel_serial_to_datetime(v) for v in time_col.to_numpy(dtype=float)]
        )
    else:
        times = pd.to_datetime(time_col)

    ossi = pd.DataFrame({"time": times, "Hmax": hmax_col, "T": t_col})
    ossi = ossi.dropna(subset=["time", "Hmax"]).reset_index(drop=True)
    return ossi


def _match_events(
    ais_times: pd.Series,
    ossi: pd.DataFrame,
    window_min: float = 0.5,
) -> np.ndarray:
    """For each AIS timestamp find the unique OSSI event within ±window_min.

    Returns array of OSSI Hmax values aligned to ais_times.
    NaN where no unique match is found (0 or >1 OSSI events in window).
    """
    window_td = pd.Timedelta(minutes=window_min)
    ossi_times = ossi["time"].to_numpy()
    ossi_hmax  = ossi["Hmax"].to_numpy(dtype=float)

    matched = np.full(len(ais_times), np.nan)

    for i, t in enumerate(ais_times):
        in_win = (ossi["time"] >= t - window_td) & (ossi["time"] <= t + window_td)
        indices = np.where(in_win.to_numpy())[0]
        if len(indices) == 1:
            matched[i] = ossi_hmax[indices[0]]
        # 0 or >1 → NaN (no unique match)

    return matched


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COLOURS = {
    "Kriebel":   "olive",
    "PIANC":     "blue",
    "Sorensen":  "green",
    "Maynord":   "magenta",
    "Bhowmik":   "#006857",   # dark green
    "Gates":     "#C8C8C8",   # gray
    "Blaauw1":   "#D95319",   # orange
    "Blaauw2":   "#BF00BF",   # purple
    "Blaauw3":   "#BFBF00",   # olive-yellow
    "OSSI":      "black",
}


def _timeseries_plot(
    df: pd.DataFrame,
    ossi: pd.DataFrame,
    pred_cols: dict[str, str],
    title: str,
    out_path: Path,
) -> None:
    """Scatter time-series of all formulae + measurements."""
    fig, ax = plt.subplots(figsize=(16, 5))

    ax.scatter(ossi["time"], ossi["Hmax"], s=10, c=_COLOURS["OSSI"],
               label="Measurements", zorder=5)

    for label, col in pred_cols.items():
        if col in df.columns:
            ax.scatter(df["obstime_adj"], df[col], s=10,
                       c=_COLOURS.get(label, None), label=label)

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
    plt.close(fig)
    print(f"  saved {out_path.name}")


def _scatter_plot(
    df: pd.DataFrame,
    pred_cols: dict[str, str],
    out_path: Path,
) -> None:
    """Predicted vs measured Hmax scatter with 1:1 line."""
    fig, ax = plt.subplots(figsize=(7, 7))

    has_data = False
    for label, col in pred_cols.items():
        if col not in df.columns:
            continue
        valid = df[["Hmax_measured", col]].dropna()
        if valid.empty:
            continue
        ax.scatter(valid["Hmax_measured"], valid[col], s=20,
                   c=_COLOURS.get(label, None), label=label, alpha=0.7)
        has_data = True

    if has_data:
        lim = ax.get_xlim()[1]
        ax.plot([0, lim], [0, lim], "--k", lw=1, label="1:1")

    ax.set_xlabel("$H_{max}$ measurement (m)")
    ax.set_ylabel("$H_{max}$ empirical (m)")
    ax.set_title("Empirical formulae vs measurements")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(
    ais_path: str | Path,
    ossi_path: str | Path,
    gauge_lon: float,
    gauge_lat: float,
    out_dir: str | Path = "output/comparison",
    event_window_min: float = 0.5,
    g: float = 9.78,
    cb_method: str = "B_Le",
    clean_draft: bool = True,
) -> pd.DataFrame:
    """Run the full comparison and produce output plots + CSV.

    Wake arrivals at the gauge are found by solving for the exact position on
    each trajectory segment where the vessel's wake direction aligns with the
    bearing to the gauge (bisection on each consecutive AIS pair).  Arrival
    times use deep-water group velocity (c_g = g·Tc / 4π).

    Returns the events DataFrame (one row per wake arrival at the gauge).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load AIS ---
    print("Loading AIS data...")
    df = _load_ais(ais_path)
    print(f"  {len(df)} AIS records")

    # --- Ensure vessel params and displacement ---
    df = _ensure_vessel_cols(df, cb_method)

    # --- Optional: remove zero-draught records ---
    if clean_draft and "draught" in df.columns:
        n_before = len(df)
        df = df[df["draught"] > 0].reset_index(drop=True)
        print(f"  {n_before - len(df)} records removed (draught = 0)")

    # --- Find wake arrivals at the gauge via ray-segment intersection ---
    print(f"Finding wake arrivals at gauge ({gauge_lon:.6f}, {gauge_lat:.6f})...")
    events = compute_point_impact(df, gauge_lon, gauge_lat, g=g)
    print(f"  {len(events)} wake-arrival events found")

    if events.empty:
        print("No wake events reached the gauge — check that the AIS data contains "
              "segment_id, Theta, WakeDirPort, WakeDirStarboard, Tc, BF, LengthWL columns.")
        return events

    # --- Compute all empirical wave heights at the lateral distance ---
    print("Computing empirical wave heights...")
    dist_perp = events["DistPerp_m"].to_numpy(dtype=float)

    # Kriebel (already in WaveHeight column from compute_point_impact, kept for completeness)
    if all(c in events.columns for c in ["WaveHeight"]):
        events["H_Kriebel_dist"] = events["WaveHeight"]

    events["H_PIANC"]    = compute_pianc(events, dist_perp, g=g).values
    events["H_Bhowmik"]  = compute_bhowmik(events, g=g).values
    events["H_Gates"]    = compute_gates(events, dist_perp, g=g).values
    events["H_Blaauw1"]  = compute_blaauw(events, dist_perp, g=g, A=A_LOADED).values
    events["H_Blaauw2"]  = compute_blaauw(events, dist_perp, g=g, A=A_MODERATE).values
    events["H_Blaauw3"]  = compute_blaauw(events, dist_perp, g=g, A=A_LIGHT).values
    events["H_Sorensen"] = compute_sorensen(events, dist_perp, g=g).values
    events["H_Maynord"]  = compute_maynord(events, dist_perp, g=g).values

    # --- Load OSSI measurements ---
    print(f"Loading OSSI measurements from {Path(ossi_path).name}...")
    ossi = _load_ossi(ossi_path)
    print(f"  {len(ossi)} wave events")

    # --- Match wake arrivals ↔ OSSI measurements using group-velocity arrival time ---
    print(f"Matching events (±{event_window_min} min window)...")
    events["Hmax_measured"] = _match_events(events["ArrivalTime"], ossi, event_window_min)
    n_matched = events["Hmax_measured"].notna().sum()
    print(f"  {n_matched} matched wake↔OSSI pairs")

    # Column map: legend label → DataFrame column
    pred_cols = {
        "Kriebel":  "H_Kriebel_dist",
        "PIANC":    "H_PIANC",
        "Sorensen": "H_Sorensen",
        "Maynord":  "H_Maynord",
        "Bhowmik":  "H_Bhowmik",
        "Gates":    "H_Gates",
        "Blaauw1":  "H_Blaauw1",
        "Blaauw2":  "H_Blaauw2",
        "Blaauw3":  "H_Blaauw3",
    }

    # Rename ArrivalTime to obstime_adj for compatibility with existing plot functions
    events["obstime_adj"] = events["ArrivalTime"]

    # --- Plots ---
    print("Producing plots...")
    _timeseries_plot(
        events, ossi, pred_cols,
        title="All formulae and measurements vs time (arrival at gauge)",
        out_path=out_dir / "timeseries_all.png",
    )
    _timeseries_plot(
        events[events["Hmax_measured"].notna()].reset_index(drop=True),
        ossi, pred_cols,
        title=f"Matched events (±{event_window_min} min window) vs time",
        out_path=out_dir / "timeseries_matched.png",
    )
    _scatter_plot(events, pred_cols, out_dir / "scatter_predicted_vs_measured.png")

    # --- Save matched events CSV ---
    matched_cols = [
        "DateTime", "ArrivalTime", "MMSI", "PropDist_m", "DistPerp_m",
        "SOG", "VesselLength", "VesselWidth", "Side", "segment_id",
        "Hmax_measured",
    ] + list(pred_cols.values())
    out_cols = [c for c in matched_cols if c in events.columns]
    matched_df = events[out_cols][events["Hmax_measured"].notna()].copy()
    matched_csv = out_dir / "matched_events.csv"
    matched_df.to_csv(matched_csv, index=False)
    print(f"  saved {matched_csv.name} ({len(matched_df)} rows)")

    return events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare empirical ship-wake formulae against OSSI measurements."
    )
    parser.add_argument("--ais",        required=True, help="AIS wave-params CSV or Parquet")
    parser.add_argument("--ossi",       required=True, help="OSSI Excel file (.xlsx)")
    parser.add_argument("--gauge-lon",  required=True, type=float, help="Gauge longitude")
    parser.add_argument("--gauge-lat",  required=True, type=float, help="Gauge latitude")
    parser.add_argument("--out",        default="output/comparison", help="Output directory")
    parser.add_argument("--event-window", type=float, default=0.5,
                        help="Matching window in minutes (default 0.5)")
    parser.add_argument("--g",          type=float, default=9.78,
                        help="Gravity m/s² (default 9.78)")
    parser.add_argument("--cb-method",  default="B_Le",
                        choices=["L_Le", "B_Le", "table"],
                        help="Block coefficient method (default B_Le)")
    parser.add_argument("--no-clean-draft", action="store_true",
                        help="Keep records with draught = 0")
    args = parser.parse_args()

    run_comparison(
        ais_path=args.ais,
        ossi_path=args.ossi,
        gauge_lon=args.gauge_lon,
        gauge_lat=args.gauge_lat,
        out_dir=args.out,
        event_window_min=args.event_window,
        g=args.g,
        cb_method=args.cb_method,
        clean_draft=not args.no_clean_draft,
    )


if __name__ == "__main__":
    main()
