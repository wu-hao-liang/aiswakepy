"""Configuration schema and loader for ShipwakeAIS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class AisConfig(BaseModel):
    raw_csv: str
    min_speed_knots: float = 0.5
    interp_spacing_m: float = 20.0
    traj_gap_s: float = 600.0
    interp_trigger_m: float = 100.0


class VesselConfig(BaseModel):
    cb_method: Literal["L_Le", "B_Le", "table"] = "L_Le"
    block_coeff_csv: str = "aiswakepy/vessel/ship_data.csv"
    waterline_factor: float = 0.8


class BathymetryConfig(BaseModel):
    source: str
    tide_dfs0: str | None = None
    underkeel_margin_m: float = 1.0


class CoastlineConfig(BaseModel):
    shapefile: str


class WaveConfig(BaseModel):
    max_froude_m: float = 0.5
    min_froude_m: float = 0.1
    max_bf: float = 0.4
    max_sog_knots: float = 12.0
    max_bl_ratio: float = 0.3
    rho_water: float = 1026.0
    gravity: float = 9.78

    @field_validator("gravity")
    @classmethod
    def gravity_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("gravity must be positive")
        return v


class ImpactConfig(BaseModel):
    max_propagation_m: float = 2000.0
    wake_cutoff_m: float = 0.01


class OutputConfig(BaseModel):
    directory: str = "output/"
    save_parquet: bool = True
    plot_wave_height_map: bool = True
    plot_period_map: bool = True
    plot_vessel_diagrams: bool = False


class ShipwakeConfig(BaseModel):
    ais: AisConfig
    vessel: VesselConfig = VesselConfig()
    bathymetry: BathymetryConfig
    coastline: CoastlineConfig
    wave: WaveConfig = WaveConfig()
    impact: ImpactConfig = ImpactConfig()
    output: OutputConfig = OutputConfig()

    model_config = {"extra": "forbid"}


def load_config(source: str | dict | Path) -> ShipwakeConfig:
    """Load config from a JSON file path, JSON string, Path, or plain dict.

    Examples
    --------
    >>> cfg = load_config("config.json")
    >>> cfg = load_config('{"ais": {...}, "bathymetry": {...}, "coastline": {...}}')
    >>> cfg = load_config({"ais": {...}, "bathymetry": {...}, "coastline": {...}})
    """
    if isinstance(source, dict):
        return ShipwakeConfig.model_validate(source)

    source = str(source)

    # Treat as file path if it ends with .json or the path exists
    path = Path(source)
    if path.suffix == ".json" or path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return ShipwakeConfig.model_validate(data)

    # Otherwise try to parse as inline JSON string
    try:
        data = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"source is neither a valid file path nor a valid JSON string: {exc}"
        ) from exc
    return ShipwakeConfig.model_validate(data)
