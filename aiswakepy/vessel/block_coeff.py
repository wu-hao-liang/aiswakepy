"""Block coefficient (Cb) and bow entry length (Le) lookup.

Three methods, selectable via the ``method`` argument:

* ``"L_Le"``  — fixed Cb + L/Le ratio by AIS ship type (default, matches improved_version)
* ``"B_Le"``  — fixed Cb + B/Le ratio by AIS ship type
* ``"table"`` — type-filtered nearest-neighbour lookup from ShipDataEDnew.csv
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# AIS ship-type category helpers
# ---------------------------------------------------------------------------

_TANKER_TYPES = set(range(80, 90))          # 80-89
_CARGO_TYPES = {33} | set(range(70, 80))    # 33, 70-79

# Row ranges in ShipDataEDnew.csv (1-indexed, inclusive) per vessel category
# Derived from func_cb_tablelooking.m flag_exclude logic:
#   rows 1-22:  tankers
#   rows 23-59: general cargo / carriers (70-79)
#   rows 60-72: ferries (60-69)
#   rows 73-75: fast ferries (37, 40-49)
#   rows 76-84: fishing (30)
#   row 85:     tug (31,34,50,52-54,56-59)
#   row 86:     heavy lifter (32)
#   row 87:     navy frigate (35,51,55)
#   row 88:     sailing (36)
#   row 89:     dredger (33) — also catches unknown
_TYPE_ROW_RANGES: list[tuple[set[int], range]] = [
    (_TANKER_TYPES,                                    range(0, 22)),
    (set(range(70, 80)),                               range(22, 59)),
    (set(range(60, 70)),                               range(59, 72)),
    ({37} | set(range(40, 50)),                        range(72, 75)),
    ({30},                                             range(75, 84)),
    ({31, 34, 50, 52, 53, 54, 56, 57, 58, 59},        range(84, 85)),
    ({32},                                             range(85, 86)),
    ({35, 51, 55},                                     range(86, 87)),
    ({36},                                             range(87, 88)),
    ({33},                                             range(88, 89)),  # dredger + unknown
]


def _category(ship_type: int) -> str:
    if ship_type in _TANKER_TYPES:
        return "tanker"
    if ship_type in _CARGO_TYPES:
        return "cargo"
    return "other"


# ---------------------------------------------------------------------------
# Method A: L/Le ratio
# ---------------------------------------------------------------------------

_L_LE_TABLE = {
    "tanker": (0.86, 7.0),
    "cargo":  (0.80, 5.0),
    "other":  (0.67, 3.0),
}


def _get_params_L_Le(length_m: float, ship_type: int) -> dict:
    cb, ratio = _L_LE_TABLE[_category(ship_type)]
    return {"block_coeff": cb, "bow_entry_m": length_m / ratio}


# ---------------------------------------------------------------------------
# Method B: B/Le ratio
# ---------------------------------------------------------------------------

_B_LE_TABLE = {
    "tanker": (0.80, 1.0),
    "cargo":  (0.70, 0.7),
    "other":  (0.60, 0.4),
}


def _get_params_B_Le(beam_m: float, ship_type: int) -> dict:
    cb, ratio = _B_LE_TABLE[_category(ship_type)]
    return {"block_coeff": cb, "bow_entry_m": beam_m / ratio}


# ---------------------------------------------------------------------------
# Method C: type-filtered table lookup
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_ship_table() -> pd.DataFrame:
    csv_path = Path(__file__).parent / "ShipDataEDnew.csv"
    return pd.read_csv(csv_path)


def _row_indices_for_type(ship_type: int) -> range:
    for types, rows in _TYPE_ROW_RANGES:
        if ship_type in types:
            return rows
    return range(88, 89)  # fallback: dredger/unknown row


def _get_params_table(length_m: float, beam_m: float, ship_type: int) -> dict:
    df = _load_ship_table()
    rows = _row_indices_for_type(ship_type)
    subset = df.iloc[rows]
    loa = subset["LOA"].to_numpy()
    beam = subset["Beam"].to_numpy()
    tree = KDTree(np.column_stack([loa, beam]))
    _, idx = tree.query([length_m, beam_m])
    row = subset.iloc[idx]
    cb = float(row["CB"])
    leb = float(row["LeB"])   # LeB = Le/B ratio
    return {"block_coeff": cb, "bow_entry_m": beam_m * leb}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_vessel_params(
    length_m: float,
    beam_m: float,
    ship_type: int,
    method: Literal["L_Le", "B_Le", "table"] = "L_Le",
) -> dict:
    """Return Cb and bow entry length for a single vessel.

    Parameters
    ----------
    length_m:  overall vessel length (m)
    beam_m:    vessel beam/width (m)
    ship_type: AIS ship type code (``typecargo`` column)
    method:    one of ``"L_Le"`` (default), ``"B_Le"``, or ``"table"``

    Returns
    -------
    dict with keys ``block_coeff`` (float) and ``bow_entry_m`` (float)
    """
    if method == "L_Le":
        return _get_params_L_Le(length_m, ship_type)
    if method == "B_Le":
        return _get_params_B_Le(beam_m, ship_type)
    if method == "table":
        return _get_params_table(length_m, beam_m, ship_type)
    raise ValueError(f"Unknown method: {method!r}. Choose 'L_Le', 'B_Le', or 'table'.")


def get_vessel_params_df(
    df: pd.DataFrame,
    method: Literal["L_Le", "B_Le", "table"] = "L_Le",
) -> pd.DataFrame:
    """Vectorised version: add ``block_coeff`` and ``bow_entry_m`` columns to df.

    Input DataFrame must have columns: ``length``, ``width``, ``typecargo``.
    Returns a copy with two new columns appended.
    """
    lengths = df["length"].to_numpy(dtype=float)
    beams = df["width"].to_numpy(dtype=float)
    types = df["typecargo"].to_numpy(dtype=int)

    cbs = np.empty(len(df), dtype=float)
    les = np.empty(len(df), dtype=float)

    if method == "L_Le":
        cats = np.array([_category(t) for t in types])
        for cat, (cb, ratio) in _L_LE_TABLE.items():
            mask = cats == cat
            cbs[mask] = cb
            les[mask] = lengths[mask] / ratio

    elif method == "B_Le":
        cats = np.array([_category(t) for t in types])
        for cat, (cb, ratio) in _B_LE_TABLE.items():
            mask = cats == cat
            cbs[mask] = cb
            les[mask] = beams[mask] / ratio

    elif method == "table":
        for i, (l, b, t) in enumerate(zip(lengths, beams, types)):
            p = _get_params_table(l, b, int(t))
            cbs[i] = p["block_coeff"]
            les[i] = p["bow_entry_m"]

    else:
        raise ValueError(f"Unknown method: {method!r}")

    out = df.copy()
    out["block_coeff"] = cbs
    out["bow_entry_m"] = les
    # Volumetric displacement W = B * d * L * 0.95 * Cb (m³)
    # Only computed when draught is available (AIS pipeline always has it;
    # some unit tests create minimal DataFrames without draught).
    if "draught" in out.columns:
        out["displacement_m3"] = (
            out["width"].to_numpy(dtype=float)
            * out["draught"].to_numpy(dtype=float)
            * out["length"].to_numpy(dtype=float)
            * 0.95
            * cbs
        )
    return out
