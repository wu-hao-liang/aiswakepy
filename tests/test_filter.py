"""Tests for shipwake.stages.filter — Step 4."""

import io
import textwrap
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from aiswakepy.stages.filter import (
    filter_ais,
    interpolate_trajectories,
    load_ais,
    mask_land,
    segment_trajectories,
    validate_speed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ais_csv(rows: list[dict]) -> str:
    """Return CSV text with the required AIS columns."""
    cols = ["mmsi", "width", "length", "draught", "obstime",
            "longitude", "latitude", "sog", "cog", "typecargo"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    return "\n".join(lines)


def _base_row(mmsi=1, lon=103.85, lat=1.29, sog=5.0, t="2024-01-01 00:00:00"):
    return dict(mmsi=mmsi, width=10, length=50, draught=3,
                obstime=t, longitude=lon, latitude=lat,
                sog=sog, cog=90, typecargo=70)


def _df_from_rows(rows):
    csv = _make_ais_csv(rows)
    return load_ais(io.StringIO(csv))


# ---------------------------------------------------------------------------
# load_ais
# ---------------------------------------------------------------------------

def test_load_basic(tmp_path):
    rows = [_base_row()]
    p = tmp_path / "ais.csv"
    p.write_text(_make_ais_csv(rows))
    df = load_ais(p)
    assert len(df) == 1
    assert pd.api.types.is_datetime64_any_dtype(df["obstime"])


def test_load_drops_extra_columns(tmp_path):
    rows = [_base_row()]
    csv = _make_ais_csv(rows) + "\n"
    # add an extra column
    lines = csv.strip().split("\n")
    lines[0] += ",extra"
    lines[1] += ",foo"
    p = tmp_path / "ais.csv"
    p.write_text("\n".join(lines))
    df = load_ais(p)
    assert "extra" not in df.columns


def test_load_missing_column_raises(tmp_path):
    csv = "mmsi,longitude,latitude\n1,103.85,1.29\n"
    p = tmp_path / "ais.csv"
    p.write_text(csv)
    with pytest.raises(ValueError, match="missing required columns"):
        load_ais(p)


# ---------------------------------------------------------------------------
# segment_trajectories
# ---------------------------------------------------------------------------

def test_segment_same_vessel_short_gap():
    rows = [
        _base_row(t="2024-01-01 00:00:00"),
        _base_row(t="2024-01-01 00:05:00"),  # 5 min gap → same segment
    ]
    df = _df_from_rows(rows)
    df = segment_trajectories(df, gap_s=600)
    assert df["segment_id"].nunique() == 1


def test_segment_same_vessel_long_gap():
    rows = [
        _base_row(t="2024-01-01 00:00:00"),
        _base_row(t="2024-01-01 00:15:00"),  # 15 min gap → new segment
    ]
    df = _df_from_rows(rows)
    df = segment_trajectories(df, gap_s=600)
    assert df["segment_id"].nunique() == 2


def test_segment_two_vessels():
    rows = [
        _base_row(mmsi=1, t="2024-01-01 00:00:00"),
        _base_row(mmsi=1, t="2024-01-01 00:05:00"),
        _base_row(mmsi=2, t="2024-01-01 00:02:00"),
        _base_row(mmsi=2, t="2024-01-01 00:07:00"),
    ]
    df = _df_from_rows(rows)
    df = segment_trajectories(df, gap_s=600)
    assert df["segment_id"].nunique() == 2


# ---------------------------------------------------------------------------
# validate_speed
# ---------------------------------------------------------------------------

def test_validate_speed_clamps():
    """Vessel reports 10 kts but only moves ~50 m in 60 s → ~1.6 kts."""
    # 50 m in 60 s = 0.833 m/s = 1.619 kts
    rows = [
        _base_row(lon=103.850000, lat=1.290000, sog=10.0, t="2024-01-01 00:00:00"),
        _base_row(lon=103.850450, lat=1.290000, sog=10.0, t="2024-01-01 00:01:00"),
    ]
    df = _df_from_rows(rows)
    df = segment_trajectories(df)
    df = validate_speed(df)
    assert df.iloc[1]["sog"] < 10.0


def test_validate_speed_keeps_low_sog():
    """If reported SOG is lower than computed, keep reported SOG."""
    # ~100 m in 10 s = ~20 kts; reported = 5 kts → keep 5 kts
    rows = [
        _base_row(lon=103.850000, lat=1.290000, sog=5.0, t="2024-01-01 00:00:00"),
        _base_row(lon=103.850900, lat=1.290000, sog=5.0, t="2024-01-01 00:00:10"),
    ]
    df = _df_from_rows(rows)
    df = segment_trajectories(df)
    df = validate_speed(df)
    assert df.iloc[1]["sog"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# interpolate_trajectories
# ---------------------------------------------------------------------------

def test_interpolation_inserts_points():
    """Two points ~200 m apart should produce ~10+ interpolated rows at 20 m."""
    rows = [
        _base_row(lon=103.850000, lat=1.290000, t="2024-01-01 00:00:00"),
        _base_row(lon=103.851800, lat=1.290000, t="2024-01-01 00:01:00"),
    ]
    df = _df_from_rows(rows)
    df = segment_trajectories(df)
    df = validate_speed(df)
    df_interp = interpolate_trajectories(df, spacing_m=20, trigger_m=100)
    assert len(df_interp) > 5


def test_no_interpolation_short_gap():
    """Points < trigger_m apart → no extra rows inserted."""
    rows = [
        _base_row(lon=103.850000, lat=1.290000, t="2024-01-01 00:00:00"),
        _base_row(lon=103.850040, lat=1.290000, t="2024-01-01 00:00:10"),
    ]
    df = _df_from_rows(rows)
    df = segment_trajectories(df)
    df = validate_speed(df)
    df_interp = interpolate_trajectories(df, spacing_m=20, trigger_m=100)
    assert len(df_interp) == 2


# ---------------------------------------------------------------------------
# mask_land
# ---------------------------------------------------------------------------

def _write_coast_shp(tmp_path: Path, polygon: Polygon) -> Path:
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    p = tmp_path / "coast.shp"
    gdf.to_file(p)
    return p


def test_mask_land_removes_inside(tmp_path):
    # Polygon covers lon 103.84-103.86, lat 1.28-1.30
    poly = Polygon([(103.84, 1.28), (103.86, 1.28), (103.86, 1.30), (103.84, 1.30)])
    shp = _write_coast_shp(tmp_path, poly)

    rows = [
        _base_row(lon=103.850, lat=1.290),   # inside → removed
        _base_row(lon=103.900, lat=1.350),   # outside → kept
    ]
    df = _df_from_rows(rows)
    df_masked = mask_land(df, shp)
    assert len(df_masked) == 1
    assert df_masked.iloc[0]["longitude"] == pytest.approx(103.900)


def test_mask_land_keeps_all_outside(tmp_path):
    poly = Polygon([(103.84, 1.28), (103.86, 1.28), (103.86, 1.30), (103.84, 1.30)])
    shp = _write_coast_shp(tmp_path, poly)

    rows = [_base_row(lon=104.0, lat=1.5), _base_row(lon=104.1, lat=1.6)]
    df = _df_from_rows(rows)
    df_masked = mask_land(df, shp)
    assert len(df_masked) == 2


# ---------------------------------------------------------------------------
# Integration: filter_ais orchestrator
# ---------------------------------------------------------------------------

def test_filter_ais_integration(tmp_path):
    """End-to-end: 3-row CSV, simple coastline, verify pipeline runs."""
    rows = [
        _base_row(mmsi=1, lon=103.850, lat=1.290, t="2024-01-01 00:00:00"),
        _base_row(mmsi=1, lon=103.851, lat=1.290, t="2024-01-01 00:01:00"),
        _base_row(mmsi=1, lon=103.840, lat=1.285, t="2024-01-01 00:02:00"),  # inside land
    ]
    csv_path = tmp_path / "ais.csv"
    csv_path.write_text(_make_ais_csv(rows))

    # Coastline polygon covering the third point
    poly = Polygon([(103.83, 1.28), (103.85, 1.28), (103.85, 1.29), (103.83, 1.29)])
    shp = _write_coast_shp(tmp_path, poly)

    df = filter_ais(csv_path, shp, gap_s=600, spacing_m=20, trigger_m=50)
    assert len(df) > 0
    # Third point (inside land) should not appear
    assert not any(
        (df["longitude"] < 103.845) & (df["latitude"] < 1.286)
    )
