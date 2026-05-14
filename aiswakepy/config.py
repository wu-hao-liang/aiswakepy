"""Configuration schema and loader for ShipwakeAIS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class AisConfig(BaseModel):
    raw_csv: str
    # Shapefile used to mask out AIS points on land (separate from the coastline
    # shapefile used for wave-impact shore intersection).
    land_shp: str
    min_speed_knots: float = 0.0
    traj_gap_s: float = 180.0
    max_velocity_knots: float = 36.0
    max_acceleration_ms2: float = 10.0
    max_draught_to_width: float = 1.0
    interp_interval_s: float = 30.0
    # Threshold (m/s) below which a vessel is considered stationary / low-speed.
    # Used by the mixed interpolation method (linear when both endpoints below
    # this value) and by the kinematic error-coord check (extra flag when both
    # endpoints are low-speed but the computed displacement speed exceeds
    # 2× this threshold).
    low_sog_threshold_ms: float = 1.0
    # Velocity-consistency ratio for error-coord detection.  A consecutive pair
    # (i, i+1) is flagged when the displacement-derived speed exceeds
    # ``velocity_ratio_threshold`` × the average of the two reported SOG values.
    # Default 2.0 means "positions moved more than twice as fast as the AIS
    # transponder claims".  Lower values are more aggressive; 2.0 is conservative.
    velocity_ratio_threshold: float = 2.0
    # Speed-consistency ratio for clean_error_speed.  A consecutive pair (i,i+1)
    # is flagged when the position-derived speed (dl/dt) is less than
    # ``speed_consistency_ratio`` × the magnitude of the vector-averaged AIS
    # velocity ((v_i+v_{i+1})/2).  Default 0.5 means "positions moved at less
    # than half the speed the transponder claims" → SOG/COG is likely erroneous.
    speed_consistency_ratio: float = 0.5
    # "linear": straight-line between consecutive raw points (default; conservative,
    # no overshoot). "hermite": CubicHermiteSpline using SOG/COG as velocity
    # constraints (smoother, but can produce spikes near noisy SOG/COG values).
    interp_method: Literal["linear", "hermite", "mixed"] = "linear"
    study_area_shp: str | None = None


class VesselConfig(BaseModel):
    cb_method: Literal["L_Le", "B_Le", "table"] = "L_Le"
    block_coeff_csv: str = "aiswakepy/vessel/ship_data.csv"
    waterline_factor: float = 0.8


class BathymetryConfig(BaseModel):
    source: str
    tide_dfs0: str | None = None
    tide_item: str | None = None
    underkeel_margin_m: float = 1.0


class CoastlineConfig(BaseModel):
    shapefile: str


class WaveConfig(BaseModel):
    max_Froude_M: float = 0.5
    min_Froude_M: float = 0.1
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
    plot_max_points: int = 100_000
    # Configurable output filenames (relative to directory)
    wave_height_map_name: str = "WaveHeightMap.png"
    wave_period_map_name: str = "WavePeriodMap.png"
    wave_params_name: str = "wave_params.parquet"
    shore_impact_name: str = "shore_impact.csv"
    # Stage CSV backups: saved as {ais_stem}_{stage}.csv in output directory
    save_stage_csv: bool = True


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
