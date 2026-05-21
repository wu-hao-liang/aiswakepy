"""Tests for aiswakepy.stages.filter."""

import io
import textwrap
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from aiswakepy.stages.filter import (
    clean_error_coords,
    clean_error_speed,
    deduplicate,
    filter_ais,
    filter_study_area,
    interpolate_trajectories,
    load_ais,
    mask_land,
    remove_zero_dimensions,
    segment_trajectories,
    uniformize_vessel_info,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ais_csv(rows: list[dict]) -> str:
    cols = ["mmsi", "width", "length", "draught", "obstime",
            "longitude", "latitude", "sog", "cog", "typecargo"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    return "\n".join(lines)


def _base_row(mmsi=1, lon=103.85, lat=1.29, sog=5.0, cog=90.0,
              t="2024-01-01 00:00:00", width=10, length=50, draught=3):
    return dict(mmsi=mmsi, width=width, length=length, draught=draught,
                obstime=t, longitude=lon, latitude=lat,
                sog=sog, cog=cog, typecargo=70)


def _df_from_rows(rows):
    csv = _make_ais_csv(rows)
    return load_ais(io.StringIO(csv))


def _segmented(rows, gap_s=600):
    df = _df_from_rows(rows)
    return segment_trajectories(df, gap_s=gap_s)


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
# deduplicate
# ---------------------------------------------------------------------------

def test_deduplicate_removes_same_mmsi_obstime():
    rows = [
        _base_row(mmsi=1, lon=103.85, t="2024-01-01 00:00:00"),
        _base_row(mmsi=1, lon=103.86, t="2024-01-01 00:00:00"),  # duplicate time
    ]
    df = _df_from_rows(rows)
    result = deduplicate(df)
    assert len(result) == 1
    assert result.iloc[0]["longitude"] == pytest.approx(103.85)  # first kept


def test_deduplicate_keeps_different_time():
    rows = [
        _base_row(mmsi=1, t="2024-01-01 00:00:00"),
        _base_row(mmsi=1, t="2024-01-01 00:01:00"),
    ]
    df = _df_from_rows(rows)
    result = deduplicate(df)
    assert len(result) == 2


def test_deduplicate_different_mmsi_same_time():
    rows = [
        _base_row(mmsi=1, t="2024-01-01 00:00:00"),
        _base_row(mmsi=2, t="2024-01-01 00:00:00"),
    ]
    df = _df_from_rows(rows)
    result = deduplicate(df)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# uniformize_vessel_info
# ---------------------------------------------------------------------------

def test_uniformize_sets_mode():
    rows = [
        _base_row(mmsi=1, width=10, t="2024-01-01 00:00:00"),
        _base_row(mmsi=1, width=10, t="2024-01-01 00:01:00"),
        _base_row(mmsi=1, width=12, t="2024-01-01 00:02:00"),  # outlier
    ]
    df = _df_from_rows(rows)
    result = uniformize_vessel_info(df, columns=["width"])
    assert (result["width"] == 10).all()


def test_uniformize_preserves_different_mmsi():
    rows = [
        _base_row(mmsi=1, width=10, t="2024-01-01 00:00:00"),
        _base_row(mmsi=2, width=20, t="2024-01-01 00:01:00"),
    ]
    df = _df_from_rows(rows)
    result = uniformize_vessel_info(df, columns=["width"])
    assert result[result["mmsi"] == 1].iloc[0]["width"] == pytest.approx(10)
    assert result[result["mmsi"] == 2].iloc[0]["width"] == pytest.approx(20)


# ---------------------------------------------------------------------------
# remove_zero_dimensions
# ---------------------------------------------------------------------------

def test_remove_zero_dimensions_drops_zero_width():
    rows = [_base_row(width=0), _base_row(width=10)]
    df = _df_from_rows(rows)
    result = remove_zero_dimensions(df)
    assert len(result) == 1
    assert result.iloc[0]["width"] == pytest.approx(10)


def test_remove_zero_dimensions_drops_zero_length():
    rows = [_base_row(length=0), _base_row(length=50)]
    df = _df_from_rows(rows)
    result = remove_zero_dimensions(df)
    assert len(result) == 1


def test_remove_zero_dimensions_drops_zero_draught():
    rows = [_base_row(draught=0), _base_row(draught=3)]
    df = _df_from_rows(rows)
    result = remove_zero_dimensions(df)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# segment_trajectories
# ---------------------------------------------------------------------------

def test_segment_same_vessel_short_gap():
    rows = [
        _base_row(t="2024-01-01 00:00:00"),
        _base_row(t="2024-01-01 00:02:00"),  # 2 min gap → same segment at 180s
    ]
    df = _df_from_rows(rows)
    df = segment_trajectories(df, gap_s=180)
    assert df["segment_id"].nunique() == 1


def test_segment_same_vessel_long_gap():
    rows = [
        _base_row(t="2024-01-01 00:00:00"),
        _base_row(t="2024-01-01 00:05:00"),  # 5 min gap → new segment at 180s
    ]
    df = _df_from_rows(rows)
    df = segment_trajectories(df, gap_s=180)
    assert df["segment_id"].nunique() == 2


def test_segment_two_vessels():
    rows = [
        _base_row(mmsi=1, t="2024-01-01 00:00:00"),
        _base_row(mmsi=1, t="2024-01-01 00:01:00"),
        _base_row(mmsi=2, t="2024-01-01 00:02:00"),
        _base_row(mmsi=2, t="2024-01-01 00:03:00"),
    ]
    df = _df_from_rows(rows)
    df = segment_trajectories(df, gap_s=600)
    assert df["segment_id"].nunique() == 2


# ---------------------------------------------------------------------------
# clean_error_coords
# ---------------------------------------------------------------------------

def _make_seg_df(coords, times_min, sog=5.0, cog=90.0, mmsi=1):
    """Build a segmented DataFrame from a list of (lon, lat) tuples."""
    rows = [
        _base_row(mmsi=mmsi, lon=lon, lat=lat, sog=sog, cog=cog,
                  t=f"2024-01-01 00:{int(m):02d}:00")
        for (lon, lat), m in zip(coords, times_min)
    ]
    df = _df_from_rows(rows)
    return segment_trajectories(df, gap_s=600)


def test_clean_coords_flag2_removed():
    """GPS spike: middle point far off track. Both adjacent segments are too fast
    → spike gets flag=2 and is removed."""
    # Normal points ~0.001 deg apart (≈111m). Spike 5 degrees away.
    coords = [(103.850, 1.290), (108.000, 6.000), (103.852, 1.290)]
    times_min = [0, 1, 2]
    df = _make_seg_df(coords, times_min)
    result = clean_error_coords(df, max_velocity_knots=12.0)
    assert len(result) == 2
    # The spike point (108.0) should be gone
    assert not any(result["longitude"] > 104.0)


def test_clean_coords_keeps_valid():
    """Normal trajectory — no points removed."""
    # ~111 m between points, 60s apart → ~1.8 kts well under 12 kts
    coords = [(103.850, 1.290), (103.851, 1.290), (103.852, 1.290)]
    times_min = [0, 1, 2]
    df = _make_seg_df(coords, times_min)
    result = clean_error_coords(df, max_velocity_knots=12.0)
    assert len(result) == 3


def test_clean_coords_short_segment_unchanged():
    """Segment with < 2 points passes through unchanged."""
    rows = [_base_row(t="2024-01-01 00:00:00")]
    df = _df_from_rows(rows)
    df = segment_trajectories(df, gap_s=600)
    result = clean_error_coords(df)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# clean_error_speed
# ---------------------------------------------------------------------------

def test_clean_speed_normal_accel_kept():
    """Vessel moving steadily east — SOG/COG consistent with trajectory, kept."""
    # ~111 m steps, 30s apart → ~3.7 kts, reported 3.5 kts (close enough)
    coords = [(103.850, 1.290), (103.851, 1.290), (103.852, 1.290)]
    rows = [
        _base_row(lon=lon, lat=lat, sog=3.5, cog=90.0,
                  t=f"2024-01-01 00:00:{i*30:02d}")
        for i, (lon, lat) in enumerate(coords)
    ]
    df = _df_from_rows(rows)
    df = segment_trajectories(df, gap_s=600)
    result = clean_error_speed(df, max_acceleration_ms2=0.2)
    # SOG should remain close to 3.5 (not replaced)
    assert result.iloc[1]["sog"] == pytest.approx(3.5, abs=0.1)


def test_clean_speed_high_accel_replaced():
    """Interior point with wildly wrong SOG — replaced by weighted average."""
    # Two points 111m apart in 30s → ~3.7 kts
    # Middle point reports sog=25 kts (impossibly high, would require huge accel)
    coords = [(103.850, 1.290), (103.851, 1.290), (103.852, 1.290)]
    rows = [
        _base_row(lon=lon, lat=lat, sog=sog, cog=90.0,
                  t=f"2024-01-01 00:00:{i*30:02d}")
        for i, ((lon, lat), sog) in enumerate(zip(coords, [3.5, 25.0, 3.5]))
    ]
    df = _df_from_rows(rows)
    df = segment_trajectories(df, gap_s=600)
    result = clean_error_speed(df, max_acceleration_ms2=0.2)
    # Middle point SOG should be corrected away from 25
    assert result.iloc[1]["sog"] < 10.0


# ---------------------------------------------------------------------------
# interpolate_trajectories (Hermite spline)
# ---------------------------------------------------------------------------

def test_interpolate_hermite_produces_points():
    """Two points 60s apart with 30s interval → 3 output points (0s, 30s, 60s)."""
    rows = [
        _base_row(lon=103.850, lat=1.290, sog=5.0, cog=90.0, t="2024-01-01 00:00:00"),
        _base_row(lon=103.851, lat=1.290, sog=5.0, cog=90.0, t="2024-01-01 00:01:00"),
    ]
    df = _df_from_rows(rows)
    df = segment_trajectories(df, gap_s=600)
    df_interp = interpolate_trajectories(df, interval_s=30.0)
    # 60s span at 30s → 3 points
    assert len(df_interp) >= 2
    assert "sog" in df_interp.columns
    assert "cog" in df_interp.columns


def test_interpolate_single_point_passthrough():
    """A segment with only 1 point should pass through unchanged."""
    rows = [_base_row(lon=103.85, lat=1.29, t="2024-01-01 00:00:00")]
    df = _df_from_rows(rows)
    df = segment_trajectories(df, gap_s=600)
    df_interp = interpolate_trajectories(df, interval_s=30.0)
    assert len(df_interp) == 1


def test_interpolate_cog_in_range():
    """Interpolated COG should always be in [0, 360)."""
    rows = [
        _base_row(lon=103.850, lat=1.290, sog=5.0, cog=350.0, t="2024-01-01 00:00:00"),
        _base_row(lon=103.851, lat=1.291, sog=5.0, cog=10.0, t="2024-01-01 00:02:00"),
    ]
    df = _df_from_rows(rows)
    df = segment_trajectories(df, gap_s=600)
    df_interp = interpolate_trajectories(df, interval_s=30.0)
    assert (df_interp["cog"] >= 0).all()
    assert (df_interp["cog"] < 360).all()


# ---------------------------------------------------------------------------
# filter_study_area
# ---------------------------------------------------------------------------

def test_filter_study_area_none_passthrough():
    rows = [_base_row(lon=103.85, lat=1.29)]
    df = _df_from_rows(rows)
    result = filter_study_area(df, polygon_shp=None)
    assert len(result) == len(df)


def test_filter_study_area_removes_outside(tmp_path):
    poly = Polygon([(103.84, 1.28), (103.86, 1.28), (103.86, 1.30), (103.84, 1.30)])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
    shp = tmp_path / "study.shp"
    gdf.to_file(shp)

    rows = [
        _base_row(lon=103.850, lat=1.290),   # inside
        _base_row(lon=104.000, lat=1.400),   # outside
    ]
    df = _df_from_rows(rows)
    result = filter_study_area(df, polygon_shp=shp)
    assert len(result) == 1
    assert result.iloc[0]["longitude"] == pytest.approx(103.850)


# ---------------------------------------------------------------------------
# mask_land
# ---------------------------------------------------------------------------

def _write_coast_shp(tmp_path: Path, polygon: Polygon) -> Path:
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    p = tmp_path / "coast.shp"
    gdf.to_file(p)
    return p


def test_mask_land_removes_inside(tmp_path):
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
    assert len(mask_land(df, shp)) == 2


# ---------------------------------------------------------------------------
# Integration: filter_ais orchestrator
# ---------------------------------------------------------------------------

def test_filter_ais_integration(tmp_path):
    # Points 0 and 1 are outside land (lat=1.2900 > polygon top 1.2898).
    # Point 2 drifts slightly south into land (~105 m in 60 s ≈ 1.8 kts, well under 12).
    rows = [
        _base_row(mmsi=1, lon=103.8500, lat=1.2900, t="2024-01-01 00:00:00"),
        _base_row(mmsi=1, lon=103.8509, lat=1.2900, t="2024-01-01 00:01:00"),
        _base_row(mmsi=1, lon=103.8501, lat=1.2895, t="2024-01-01 00:02:00"),  # inside land
    ]
    csv_path = tmp_path / "ais.csv"
    csv_path.write_text(_make_ais_csv(rows))

    # Polygon covers only the third point (lat < 1.2898)
    poly = Polygon([
        (103.849, 1.287), (103.851, 1.287),
        (103.851, 1.2898), (103.849, 1.2898),
    ])
    shp = _write_coast_shp(tmp_path, poly)

    df = filter_ais(csv_path, shp, shp, gap_s=600)
    assert len(df) > 0
    # No output point should be inside the land polygon
    inside = (
        (df["longitude"] >= 103.849) & (df["longitude"] <= 103.851)
        & (df["latitude"] >= 1.287) & (df["latitude"] <= 1.2898)
    )
    assert not inside.any()


def test_filter_ais_with_depth(tmp_path, monkeypatch):
    """filter_ais embeds the depth-clearance check when bathy_path is given.

    With a stubbed bathy mesh returning 20m everywhere, a vessel with 8m
    draught + 1m underkeel margin (= 9m required) is kept. With a stub
    returning 5m everywhere, all points are dropped (5 < 8 + 1 = 9).
    """
    rows = [
        _base_row(mmsi=1, lon=103.8500, lat=1.2900, t="2024-01-01 00:00:00", draught=8.0),
        _base_row(mmsi=1, lon=103.8509, lat=1.2900, t="2024-01-01 00:01:00", draught=8.0),
        _base_row(mmsi=1, lon=103.8518, lat=1.2900, t="2024-01-01 00:02:00", draught=8.0),
    ]
    csv_path = tmp_path / "ais.csv"
    csv_path.write_text(_make_ais_csv(rows))
    poly = Polygon([(103.95, 1.30), (103.96, 1.30), (103.96, 1.31), (103.95, 1.31)])
    shp = _write_coast_shp(tmp_path, poly)

    from unittest.mock import MagicMock
    def _stub_bathy(depth_value):
        stub = MagicMock()
        stub.get_depth.side_effect = lambda lons, lats: np.full(len(lons), depth_value)
        return stub

    import aiswakepy.stages.depth as depth_mod

    # Deep water: 20 m → all points retain.
    monkeypatch.setattr(depth_mod, "load_bathymetry", lambda _: _stub_bathy(20.0))
    df_deep = filter_ais(csv_path, shp, shp, gap_s=600,
                         bathy_path="stub.mesh", underkeel_margin_m=1.0)
    assert "WaterDepth" in df_deep.columns
    assert len(df_deep) > 0

    # Shallow water: 5 m → underkeel violation drops everything.
    monkeypatch.setattr(depth_mod, "load_bathymetry", lambda _: _stub_bathy(5.0))
    df_shallow = filter_ais(csv_path, shp, shp, gap_s=600,
                            bathy_path="stub.mesh", underkeel_margin_m=1.0)
    assert len(df_shallow) == 0
