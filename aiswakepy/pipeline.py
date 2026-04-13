"""Pipeline orchestration: run all stages or a selected subset."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import pandas as pd

from aiswakepy.config import ShipwakeConfig, load_config

Stage = Literal["filter", "depth", "wave", "impact", "viz"]
ALL_STAGES: list[Stage] = ["filter", "depth", "wave", "impact", "viz"]


def run_pipeline(
    config: str | dict | Path | ShipwakeConfig,
    stages: list[Stage] | None = None,
) -> dict:
    """Run the aiswakepy pipeline.

    Parameters
    ----------
    config: ShipwakeConfig, or any source accepted by load_config().
    stages: Subset of stages to run (default: all).

    Returns
    -------
    dict with keys for each completed stage:
        ``df_filtered``, ``df_depth``, ``df_wave``, ``df_impact``.
    """
    if not isinstance(config, ShipwakeConfig):
        config = load_config(config)

    if stages is None:
        stages = list(ALL_STAGES)

    out_dir = Path(config.output.directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    ais_stem = Path(config.ais.raw_csv).stem

    def _save_stage_csv(df: pd.DataFrame, suffix: str) -> None:
        if config.output.save_stage_csv:
            path = out_dir / f"{ais_stem}_{suffix}.csv"
            df.to_csv(path, index=False)
            print(f"  \u2713 saved {path.name}")

    results: dict = {}

    if "filter" in stages:
        from aiswakepy.stages.filter import filter_ais
        print("Stage 1/4: AIS filtering...")
        t0 = time.perf_counter()
        results["df_filtered"] = filter_ais(
            csv_path=config.ais.raw_csv,
            coastline_shp=config.coastline.shapefile,
            gap_s=config.ais.traj_gap_s,
            max_velocity_knots=config.ais.max_velocity_knots,
            max_acceleration_ms2=config.ais.max_acceleration_ms2,
            interval_s=config.ais.interp_interval_s,
            study_area_shp=config.ais.study_area_shp,
        )
        print(
            f"  \u2192 {len(results['df_filtered'])} rows after filtering "
            f"({time.perf_counter() - t0:.1f}s)"
        )
        _save_stage_csv(results["df_filtered"], "01_filtered")

    if "depth" in stages:
        from aiswakepy.stages.depth import assign_depth
        print("Stage 2/4: Depth assignment...")
        t0 = time.perf_counter()
        df_in = results.get("df_filtered")
        if df_in is None:
            raise RuntimeError("Stage 'depth' requires 'filter' to have run first")
        results["df_depth"] = assign_depth(
            df=df_in,
            bathy_path=config.bathymetry.source,
            tide_dfs0_path=config.bathymetry.tide_dfs0,
            tide_item=config.bathymetry.tide_item,
            underkeel_margin_m=config.bathymetry.underkeel_margin_m,
        )
        print(
            f"  \u2192 {len(results['df_depth'])} rows after depth filter "
            f"({time.perf_counter() - t0:.1f}s)"
        )
        _save_stage_csv(results["df_depth"], "02_depth")

    if "wave" in stages:
        from aiswakepy.stages.wave_params import compute_wave_params
        w = config.wave
        print(
            f"Stage 3/4: Wave parameters...\n"
            f"  filters: Fm=[{w.min_froude_m}, {w.max_froude_m}]"
            f"  BF\u2264{w.max_bf}"
            f"  SOG\u2264{w.max_sog_knots}kn"
            f"  B/L\u2264{w.max_bl_ratio}"
        )
        t0 = time.perf_counter()
        df_in = results.get("df_depth")
        if df_in is None:
            raise RuntimeError("Stage 'wave' requires 'depth' to have run first")
        results["df_wave"] = compute_wave_params(
            df=df_in,
            cb_method=config.vessel.cb_method,
            g=config.wave.gravity,
            rho=config.wave.rho_water,
            min_froude_m=config.wave.min_froude_m,
            max_froude_m=config.wave.max_froude_m,
            max_bf=config.wave.max_bf,
            max_sog_knots=config.wave.max_sog_knots,
            max_bl_ratio=config.wave.max_bl_ratio,
        )
        print(
            f"  \u2192 {len(results['df_wave'])} wake events "
            f"({time.perf_counter() - t0:.1f}s)"
        )
        _save_stage_csv(results["df_wave"], "03_wave")

    if "impact" in stages:
        from aiswakepy.stages.shore_impact import compute_shore_impact
        print("Stage 4/4: Shore impact...")
        t0 = time.perf_counter()
        df_in = results.get("df_wave")
        if df_in is None:
            raise RuntimeError("Stage 'impact' requires 'wave' to have run first")
        results["df_impact"] = compute_shore_impact(
            df_wave=df_in,
            coastline_shp=config.coastline.shapefile,
            max_propagation_m=config.impact.max_propagation_m,
            wake_cutoff_m=config.impact.wake_cutoff_m,
            g=config.wave.gravity,
        )
        print(
            f"  \u2192 {len(results['df_impact'])} shore impact events "
            f"({time.perf_counter() - t0:.1f}s)"
        )
        _save_stage_csv(results["df_impact"], "04_impact")

    if "viz" in stages and "df_impact" in results:
        from aiswakepy.viz.wave_map import plot_wave_height_map, plot_wave_period_map
        print("Visualisation...")
        t0 = time.perf_counter()
        df_impact = results["df_impact"]

        if config.output.plot_wave_height_map:
            plot_wave_height_map(
                df_impact, config.coastline.shapefile,
                out_dir / config.output.wave_height_map_name,
                max_points=config.output.plot_max_points,
            )
        if config.output.plot_period_map:
            plot_wave_period_map(
                df_impact, config.coastline.shapefile,
                out_dir / config.output.wave_period_map_name,
                max_points=config.output.plot_max_points,
            )
        if config.output.save_parquet and "df_wave" in results:
            results["df_wave"].to_parquet(
                out_dir / config.output.wave_params_name, index=False
            )
        df_impact.to_csv(out_dir / config.output.shore_impact_name, index=False)
        print(
            f"  \u2192 Outputs saved to {out_dir} "
            f"({time.perf_counter() - t0:.1f}s)"
        )

    return results
