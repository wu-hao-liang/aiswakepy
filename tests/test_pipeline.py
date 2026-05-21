"""Integration tests for aiswakepy.pipeline."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import geopandas as gpd
import matplotlib
import pandas as pd
import pytest
from shapely.geometry import Polygon

matplotlib.use("Agg")

from aiswakepy.pipeline import run_pipeline


def _write_shp(tmp_path: Path, polygon: Polygon) -> Path:
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    p = tmp_path / "coast.shp"
    gdf.to_file(p)
    return p


def _write_ais_csv(tmp_path: Path) -> Path:
    """Write a minimal AIS CSV with 3 tanker records that should produce wake events."""
    rows = "\n".join([
        "mmsi,width,length,draught,obstime,longitude,latitude,sog,cog,typecargo",
        "111111111,30,200,8,2024-01-01 00:00:00,103.848,1.290,8.0,90,80",
        "111111111,30,200,8,2024-01-01 00:01:00,103.849,1.290,8.0,90,80",
        "111111111,30,200,8,2024-01-01 00:02:00,103.850,1.290,8.0,90,80",
    ])
    p = tmp_path / "ais.csv"
    p.write_text(rows)
    return p


def _make_bathy_stub(depth_value: float = 15.0):
    stub = MagicMock()
    import numpy as np
    def get_depth(lons, lats):
        return np.full(len(lons), depth_value)
    stub.get_depth.side_effect = get_depth
    return stub


def _make_config(tmp_path: Path, ais_path: Path, shp_path: Path) -> dict:
    return {
        "ais": {
            "raw_csv": str(ais_path),
            "land_shp": str(shp_path),  # land mask (separate from coastline)
        },
        "vessel": {"cb_method": "L_Le"},
        "bathymetry": {"source": "dummy.mesh"},
        "coastline": {"shapefile": str(shp_path)},  # wave-impact shore intersection
        "wave": {"gravity": 9.78},
        "impact": {"max_propagation_m": 5000.0, "wake_cutoff_m": 0.001},
        "output": {
            "directory": str(tmp_path / "output"),
            "save_parquet": False,
            "plot_wave_height_map": True,
            "plot_period_map": False,
            "plot_vessel_diagrams": False,
        },
    }


def test_pipeline_filter_only(tmp_path):
    """Run only the filter stage. The depth check inside filter_ais needs a
    stubbed bathymetry because BathymetryConfig.source is required by the
    schema."""
    poly = Polygon([(103.87, 1.27), (103.95, 1.27), (103.95, 1.32), (103.87, 1.32)])
    shp = _write_shp(tmp_path, poly)
    ais = _write_ais_csv(tmp_path)
    cfg = _make_config(tmp_path, ais, shp)

    stub = _make_bathy_stub(15.0)
    import aiswakepy.stages.depth as depth_mod
    original = depth_mod.load_bathymetry
    depth_mod.load_bathymetry = lambda _: stub
    try:
        results = run_pipeline(cfg, stages=["filter"])
    finally:
        depth_mod.load_bathymetry = original

    assert "df_filtered" in results
    assert len(results["df_filtered"]) > 0
    # depth check folded into filter — WaterDepth column should now be present
    assert "WaterDepth" in results["df_filtered"].columns


def test_pipeline_filter_vessel_with_stub(tmp_path):
    """Run filter + vessel stages (depth is folded into filter)."""
    poly = Polygon([(103.86, 1.27), (103.95, 1.27), (103.95, 1.32), (103.86, 1.32)])
    shp = _write_shp(tmp_path, poly)
    ais = _write_ais_csv(tmp_path)
    cfg = _make_config(tmp_path, ais, shp)

    stub = _make_bathy_stub(15.0)

    import aiswakepy.stages.depth as depth_mod
    original = depth_mod.load_bathymetry
    depth_mod.load_bathymetry = lambda _: stub
    try:
        results = run_pipeline(cfg, stages=["filter", "vessel"])
    finally:
        depth_mod.load_bathymetry = original

    assert "df_vessel" in results
    assert len(results["df_vessel"]) > 0


def test_pipeline_full_with_stub(tmp_path):
    """Full pipeline with stubbed bathymetry."""
    poly = Polygon([(103.86, 1.27), (103.95, 1.27), (103.95, 1.32), (103.86, 1.32)])
    shp = _write_shp(tmp_path, poly)
    ais = _write_ais_csv(tmp_path)
    cfg = _make_config(tmp_path, ais, shp)

    stub = _make_bathy_stub(15.0)

    import aiswakepy.stages.depth as depth_mod
    original = depth_mod.load_bathymetry
    depth_mod.load_bathymetry = lambda _: stub
    try:
        results = run_pipeline(cfg, stages=["filter", "vessel", "wave_impact", "viz"])
    finally:
        depth_mod.load_bathymetry = original

    assert "df_wave_impact" in results
    out_dir = Path(cfg["output"]["directory"])
    assert out_dir.exists()
    assert (out_dir / "shore_impact.csv").exists()
    assert (out_dir / "WaveHeightMap.png").exists()


def test_pipeline_missing_stage_dependency(tmp_path):
    """Running 'vessel' without 'filter' should raise."""
    poly = Polygon([(103.84, 1.27), (103.92, 1.27), (103.92, 1.32), (103.84, 1.32)])
    shp = _write_shp(tmp_path, poly)
    ais = _write_ais_csv(tmp_path)
    cfg = _make_config(tmp_path, ais, shp)
    with pytest.raises(RuntimeError, match="filter"):
        run_pipeline(cfg, stages=["vessel"])


def test_cli_help():
    """CLI --help exits cleanly."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0
    assert "config" in result.stdout
