"""Tests for shipwake.config — Step 1."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from aiswakepy.config import ShipwakeConfig, load_config

_MINIMAL = {
    "ais": {"raw_csv": "data/test.csv", "land_shp": "shp/land.shp"},
    "bathymetry": {"source": "bathy/test.mesh"},
    "coastline": {"shapefile": "shp/coast.shp"},
}


def test_load_from_dict():
    cfg = load_config(_MINIMAL)
    assert isinstance(cfg, ShipwakeConfig)
    assert cfg.ais.raw_csv == "data/test.csv"


def test_load_from_json_string():
    cfg = load_config(json.dumps(_MINIMAL))
    assert cfg.bathymetry.source == "bathy/test.mesh"


def test_load_from_file(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(_MINIMAL))
    cfg = load_config(str(p))
    assert cfg.coastline.shapefile == "shp/coast.shp"


def test_load_from_path_object(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(_MINIMAL))
    cfg = load_config(p)
    assert cfg.ais.min_speed_knots == 0.0   # default


def test_defaults_applied():
    cfg = load_config(_MINIMAL)
    assert cfg.wave.gravity == 9.78
    assert cfg.wave.rho_water == 1026.0
    assert cfg.impact.max_propagation_m == 2000.0
    assert cfg.vessel.cb_method == "L_Le"
    assert cfg.output.save_parquet is True


def test_missing_required_field_raises():
    bad = {"ais": {"raw_csv": "x.csv"}, "coastline": {"shapefile": "c.shp"}}
    # missing bathymetry
    with pytest.raises(ValidationError):
        load_config(bad)


def test_unknown_field_raises():
    bad = dict(_MINIMAL, unknown_key="oops")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_invalid_gravity_raises():
    bad = dict(_MINIMAL, wave={"gravity": -1.0})
    with pytest.raises(ValidationError):
        load_config(bad)


def test_invalid_cb_method_raises():
    bad = dict(_MINIMAL, vessel={"cb_method": "magic"})
    with pytest.raises(ValidationError):
        load_config(bad)


def test_load_real_config():
    """Load the default config.json from project root."""
    root = Path(__file__).parent.parent / "config.json"
    cfg = load_config(root)
    assert cfg.wave.gravity == 9.78
    assert cfg.vessel.cb_method == "L_Le"
