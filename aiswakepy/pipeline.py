"""Pipeline orchestration: run all stages or a selected subset."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import pandas as pd
from rich.console import Console

from aiswakepy.config import ShipwakeConfig, load_config

Stage = Literal["filter", "depth", "wave", "impact", "viz"]
ALL_STAGES: list[Stage] = ["filter", "depth", "wave", "impact", "viz"]

_console = Console()


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

    results: dict = {}

    if "filter" in stages:
        from aiswakepy.stages.filter import filter_ais
        _console.print("[bold]Stage 1/4:[/bold] AIS filtering...")
        t0 = time.perf_counter()
        results["df_filtered"] = filter_ais(
            csv_path=config.ais.raw_csv,
            coastline_shp=config.coastline.shapefile,
            gap_s=config.ais.traj_gap_s,
            spacing_m=config.ais.interp_spacing_m,
            trigger_m=config.ais.interp_trigger_m,
        )
        _console.print(
            f"  → {len(results['df_filtered'])} rows after filtering "
            f"[dim]({time.perf_counter() - t0:.1f}s)[/dim]"
        )

    if "depth" in stages:
        from aiswakepy.stages.depth import assign_depth
        _console.print("[bold]Stage 2/4:[/bold] Depth assignment...")
        t0 = time.perf_counter()
        df_in = results.get("df_filtered")
        if df_in is None:
            raise RuntimeError("Stage 'depth' requires 'filter' to have run first")
        results["df_depth"] = assign_depth(
            df=df_in,
            bathy_path=config.bathymetry.source,
            tide_dfs0_path=config.bathymetry.tide_dfs0,
            underkeel_margin_m=config.bathymetry.underkeel_margin_m,
        )
        _console.print(
            f"  → {len(results['df_depth'])} rows after depth filter "
            f"[dim]({time.perf_counter() - t0:.1f}s)[/dim]"
        )

    if "wave" in stages:
        from aiswakepy.stages.wave_params import compute_wave_params
        _console.print("[bold]Stage 3/4:[/bold] Wave parameters...")
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
        _console.print(
            f"  → {len(results['df_wave'])} wake events "
            f"[dim]({time.perf_counter() - t0:.1f}s)[/dim]"
        )

    if "impact" in stages:
        from aiswakepy.stages.shore_impact import compute_shore_impact
        _console.print("[bold]Stage 4/4:[/bold] Shore impact...")
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
        _console.print(
            f"  → {len(results['df_impact'])} shore impact events "
            f"[dim]({time.perf_counter() - t0:.1f}s)[/dim]"
        )

    if "viz" in stages and "df_impact" in results:
        from aiswakepy.viz.wave_map import plot_wave_height_map, plot_wave_period_map
        _console.print("[bold]Visualisation...[/bold]")
        t0 = time.perf_counter()
        out_dir = Path(config.output.directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        df_impact = results["df_impact"]
        top_n = config.output.plot_top_n_per_bin or None

        if config.output.plot_wave_height_map:
            plot_wave_height_map(
                df_impact, config.coastline.shapefile,
                out_dir / "WaveHeightMap.png",
                top_n_per_bin=top_n,
            )
        if config.output.plot_period_map:
            plot_wave_period_map(
                df_impact, config.coastline.shapefile,
                out_dir / "WavePeriodMap.png",
                top_n_per_bin=top_n,
            )

        if config.output.save_parquet and "df_wave" in results:
            results["df_wave"].to_parquet(out_dir / "wave_params.parquet", index=False)
        df_impact.to_csv(out_dir / "shore_impact.csv", index=False)
        _console.print(
            f"  → Outputs saved to [cyan]{out_dir}[/cyan] "
            f"[dim]({time.perf_counter() - t0:.1f}s)[/dim]"
        )

    return results
