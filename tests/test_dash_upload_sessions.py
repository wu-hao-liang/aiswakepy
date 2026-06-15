from __future__ import annotations

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

import dash_app


def _reset_sessions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dash_app, "UPLOAD_FOLDER", tmp_path)
    dash_app._sessions.clear()


def _create_session(client) -> str:
    resp = client.post("/api/session")
    assert resp.status_code == 200
    sid = resp.get_json()["session_id"]
    assert sid
    return sid


def test_session_routes_are_isolated(tmp_path, monkeypatch):
    _reset_sessions(tmp_path, monkeypatch)
    client = dash_app.server.test_client()
    sid1 = _create_session(client)
    sid2 = _create_session(client)

    s1 = dash_app._sessions[sid1]
    s2 = dash_app._sessions[sid2]
    s1.ipc_vessels = b"session-one"
    s2.ipc_vessels = b"session-two"

    r1 = client.get(f"/api/vessels.arrow?session_id={sid1}")
    r2 = client.get(f"/api/vessels.arrow?session_id={sid2}")

    assert r1.data == b"session-one"
    assert r2.data == b"session-two"


def test_ais_upload_validates_required_columns(tmp_path, monkeypatch):
    _reset_sessions(tmp_path, monkeypatch)
    client = dash_app.server.test_client()
    sid = _create_session(client)

    bad = io.BytesIO(b"mmsi,longitude,latitude\n1,103.7,1.2\n")
    resp = client.post(
        f"/api/upload/ais?session_id={sid}",
        data={"file": (bad, "bad.csv")},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 400
    assert "missing required columns" in resp.get_json()["error"]


def test_shapefile_upload_accepts_selected_sidecars(tmp_path, monkeypatch):
    _reset_sessions(tmp_path / "sessions", monkeypatch)
    client = dash_app.server.test_client()
    sid = _create_session(client)

    source = tmp_path / "source"
    source.mkdir()
    shp = source / "layer.shp"
    gpd.GeoDataFrame(
        {"name": ["area"]},
        geometry=[Polygon([(103.0, 1.0), (103.1, 1.0), (103.1, 1.1), (103.0, 1.0)])],
        crs="EPSG:4326",
    ).to_file(shp)

    selected = []
    for suffix in (".shp", ".shx", ".dbf", ".cpg"):
        path = source / f"layer{suffix}"
        selected.append((io.BytesIO(path.read_bytes()), path.name))
    resp = client.post(
        f"/api/upload/coast?session_id={sid}",
        data={"files": selected},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 200, resp.get_json()
    payload = resp.get_json()
    assert payload["filename"] == "layer.shp"
    assert set(payload["files"]) == {"layer.shp", "layer.shx", "layer.dbf", "layer.cpg"}
    assert "assuming the coordinates are WGS84" in payload["warning"]


def test_shapefile_upload_requires_shx_and_dbf(tmp_path, monkeypatch):
    _reset_sessions(tmp_path, monkeypatch)
    client = dash_app.server.test_client()
    sid = _create_session(client)

    resp = client.post(
        f"/api/upload/land?session_id={sid}",
        data={"files": [(io.BytesIO(b"not-a-real-shapefile"), "layer.shp")]},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 400
    assert "missing required sidecars" in resp.get_json()["error"]


def test_export_filtered_returns_rerun_zip(tmp_path, monkeypatch):
    _reset_sessions(tmp_path, monkeypatch)
    client = dash_app.server.test_client()
    sid = _create_session(client)
    state = dash_app._sessions[sid]

    df = pd.DataFrame({
        "mmsi": [111, 111, 222],
        "segment_id": [7, 7, 8],
        "width": [20.0, 20.0, 10.0],
        "length": [100.0, 100.0, 50.0],
        "draught": [5.0, 5.0, 2.0],
        "obstime": pd.to_datetime(
            ["2024-01-01", "2024-01-01 00:01", "2024-01-02"],
            format="mixed",
        ),
        "longitude": [103.1, 103.2, 103.3],
        "latitude": [1.1, 1.2, 1.3],
        "sog": [5.0, 6.0, 7.0],
        "cog": [90.0, 91.0, 92.0],
        "typecargo": [70, 70, 80],
    })
    waves = pd.DataFrame({
        "ShLongitude": [103.4],
        "ShLatitude": [1.4],
        "MMSI": [111],
        "WaveHeight": [0.2],
        "WavePeriod": [3.0],
        "Side": ["P"],
        "DistLoc_km": [0.5],
        "SOG": [6.0],
        "VesselLength": [100.0],
        "VesselWidth": [20.0],
        "DateTime": ["2024-01-01 00:01:00"],
        "VesselLongitude": [103.2],
        "VesselLatitude": [1.2],
        "segment_id": [7],
        "VesselDraught": [5.0],
        "VesselCOG": [91.0],
    })
    state.last_results["df_filtered"] = df
    state.df_vessels = df
    state.df_waves = waves
    state.pipeline["cfg"] = {
        "ais": {"raw_csv": "uploaded.csv", "land_shp": "land.shp"},
        "bathymetry": {"source": "bathy.mesh"},
        "coastline": {"shapefile": "coast.shp"},
        "output": {"directory": "output", "save_stage_csv": True},
    }

    resp = client.post(
        f"/api/export/filtered?session_id={sid}",
        json={"dest_name": "subset", "seg_keys": [[111, 7]], "sel_ais": "uploaded.csv"},
    )

    assert resp.status_code == 200
    assert resp.mimetype == "application/zip"
    with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
        names = set(zf.namelist())
        assert "ais/uploaded.csv" in names
        assert "output/vessels.parquet" in names
        assert "output/waves.parquet" in names
        assert "output/wave_track_link.csv" in names
        assert "config.json" in names


def test_example_endpoint_loads_all_roles(tmp_path, monkeypatch):
    """POST /api/example should copy all example_data/ files into the session
    and return a roles dict with paths for all 5 input types."""
    _reset_sessions(tmp_path, monkeypatch)
    client = dash_app.server.test_client()
    sid = _create_session(client)

    resp = client.post(f"/api/example?session_id={sid}")
    assert resp.status_code == 200, resp.get_json()
    j = resp.get_json()
    assert "roles" in j, j

    roles = j["roles"]
    # All required roles must be present.
    for role in ("ais", "coast", "land", "bathy"):
        assert role in roles, f"Missing role: {role}"
        assert roles[role].get("path"), f"No path for role: {role}"
        assert roles[role].get("filename"), f"No filename for role: {role}"

    # Tide is optional but should be present when example_data/tide/ exists.
    if "tide" in roles:
        assert roles["tide"].get("path")
        # chosen_item must be populated (the dfs0 has at least one item).
        assert roles["tide"].get("chosen_item"), "Tide chosen_item should be set"

    # Verify the session state.files was actually populated.
    state = dash_app._sessions[sid]
    assert state.files.get("ais") is not None
    assert state.files.get("coast") is not None
    assert state.files.get("land") is not None
    assert state.files.get("bathy") is not None
