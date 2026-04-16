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
                    col B (index 1): timestamp (MATLAB datenum or datetime)
                    col C (index 2): Hmax (m)
                    col E (index 4): T (s)
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
from pathlib import Path

import pandas as pd

from aiswakepy.comparison       import load_ossi, match_events, timeseries_plot, scatter_plot
from aiswakepy.models.pianc     import compute_pianc
from aiswakepy.models.bhowmik   import compute_bhowmik
from aiswakepy.models.gates     import compute_gates
from aiswakepy.models.blaauw    import compute_blaauw
from aiswakepy.models.sorensen  import compute_sorensen
from aiswakepy.models.maynord   import compute_maynord
from aiswakepy.vessel.block_coeff import get_vessel_params_df
from aiswakepy.stages.wave_impact import compute_point_impact


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
    if "block_coeff" not in df.columns or "bow_entry_m" not in df.columns:
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

    df = _ensure_vessel_cols(df, cb_method)

    if clean_draft and "draught" in df.columns:
        n_before = len(df)
        df = df[df["draught"] > 0].reset_index(drop=True)
        print(f"  {n_before - len(df)} records removed (draught = 0)")

    # --- Find wake arrivals at the gauge ---
    print(f"Finding wake arrivals at gauge ({gauge_lon:.6f}, {gauge_lat:.6f})...")
    events = compute_point_impact(df, gauge_lon, gauge_lat, g=g)
    print(f"  {len(events)} wake-arrival events found")

    if events.empty:
        print("No wake events reached the gauge — check that the AIS data contains "
              "segment_id, Theta, WakeDirPort, WakeDirStarboard, Tc, SOGms, LengthWL columns.")
        return events

    # --- Compute all empirical wave heights ---
    print("Computing empirical wave heights...")
    events["dist_perp"] = events["DistPerp_m"]

    if "WaveHeight" in events.columns:
        events["H_Kriebel"] = events["WaveHeight"]

    events["H_PIANC"]    = compute_pianc(events, g=g).values
    events["H_Bhowmik"]  = compute_bhowmik(events, g=g).values
    events["H_Gates"]    = compute_gates(events, g=g).values
    events["H_Blaauw"]   = compute_blaauw(events, g=g).values
    events["H_Sorensen"] = compute_sorensen(events, g=g).values
    events["H_Maynord"]  = compute_maynord(events, g=g).values

    # --- Load OSSI measurements ---
    print(f"Loading OSSI measurements from {Path(ossi_path).name}...")
    ossi = load_ossi(ossi_path)
    print(f"  {len(ossi)} wave events")

    # --- Match wake arrivals ↔ OSSI ---
    print(f"Matching events (±{event_window_min} min window)...")
    events["Hmax_measured"] = match_events(events["ArrivalTime"], ossi, event_window_min)
    n_matched = events["Hmax_measured"].notna().sum()
    print(f"  {n_matched} matched wake↔OSSI pairs")

    pred_cols = {
        "Kriebel":  "H_Kriebel",
        "PIANC":    "H_PIANC",
        "Sorensen": "H_Sorensen",
        "Maynord":  "H_Maynord",
        "Bhowmik":  "H_Bhowmik",
        "Gates":    "H_Gates",
        "Blaauw":   "H_Blaauw",
    }

    # --- Plots ---
    print("Producing plots...")
    timeseries_plot(
        events, ossi, pred_cols,
        title="All formulae and measurements vs time (arrival at gauge)",
        out_path=out_dir / "timeseries_all.png",
    )
    timeseries_plot(
        events[events["Hmax_measured"].notna()].reset_index(drop=True),
        ossi, pred_cols,
        title=f"Matched events (±{event_window_min} min window) vs time",
        out_path=out_dir / "timeseries_matched.png",
    )
    scatter_plot(events, pred_cols, out_dir / "scatter_predicted_vs_measured.png")

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
