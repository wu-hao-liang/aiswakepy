"""OSSI wave gauge data loading and event matching."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _matlab_datenum_to_datetime(dn: float) -> pd.Timestamp:
    """Convert MATLAB datenum (days since January 0, year 0000) to Timestamp.

    MATLAB datenum(2000,1,1) = 730486; Python ordinal for 2000-01-01 = 730120.
    The fixed offset is 366, so Python_ordinal = matlab_dn - 366.
    Example: 738946.652582176 → 2023-03-01 15:39.
    """
    return pd.Timestamp.fromordinal(int(dn) - 366) + pd.Timedelta(days=dn % 1)


def load_ossi(path: str | Path) -> pd.DataFrame:
    """Load OSSI wave gauge events from Excel.

    Expected sheet: 'SHIPWAKE'
    Column B (index 1): time (MATLAB datenum float or datetime)
    Column C (index 2): Hmax (m)
    Column E (index 4): T (s)

    Returns a DataFrame with columns: time, Hmax, T.
    """
    path = Path(path)
    raw = pd.read_excel(path, sheet_name="SHIPWAKE", header=0)

    time_col = raw.iloc[:, 1]
    hmax_col = raw.iloc[:, 2].to_numpy(dtype=float)
    t_col    = raw.iloc[:, 4].to_numpy(dtype=float)

    if pd.api.types.is_float_dtype(time_col) or pd.api.types.is_integer_dtype(time_col):
        times = pd.to_datetime(
            [_matlab_datenum_to_datetime(v) for v in time_col.to_numpy(dtype=float)]
        )
    else:
        times = pd.to_datetime(time_col)

    ossi = pd.DataFrame({"time": times, "Hmax": hmax_col, "T": t_col})
    return ossi.dropna(subset=["time", "Hmax"]).reset_index(drop=True)


def match_event_indices(
    ais_times: pd.Series,
    ossi: pd.DataFrame,
    window_min: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """For each AIS timestamp pick the nearest OSSI event within ±window_min.

    Returns ``(idx, dt_min)`` aligned to ``ais_times``:
      * ``idx``    — matched OSSI row index (int), or ``-1`` if none in window
      * ``dt_min`` — absolute time difference in minutes, NaN if no match
    """
    window_td = pd.Timedelta(minutes=window_min)
    ossi_time = ossi["time"].to_numpy()
    n = len(ais_times)
    idx_arr = np.full(n, -1, dtype=np.int64)
    dt_min  = np.full(n, np.nan)

    for i, t in enumerate(ais_times):
        dt = np.abs(ossi_time - np.datetime64(t))
        in_win = dt <= window_td.to_numpy()
        if in_win.any():
            idx = int(np.argmin(np.where(in_win, dt, np.iinfo(np.int64).max)))
            idx_arr[i] = idx
            dt_min[i]  = dt[idx] / np.timedelta64(1, "s") / 60.0

    return idx_arr, dt_min


def match_events(
    ais_times: pd.Series,
    ossi: pd.DataFrame,
    window_min: float = 0.5,
    return_dt: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """For each AIS timestamp pick the closest OSSI event within ±window_min.

    Returns an array of OSSI Hmax values aligned to ais_times.
    NaN where no OSSI event falls in the window.

    If ``return_dt`` is True, also returns the absolute time difference
    (in minutes) between each AIS timestamp and its matched OSSI event;
    NaN where no match was found.
    """
    idx_arr, dt_min = match_event_indices(ais_times, ossi, window_min)
    ossi_hmax = ossi["Hmax"].to_numpy(dtype=float)
    matched = np.where(idx_arr >= 0, ossi_hmax[np.maximum(idx_arr, 0)], np.nan)

    if return_dt:
        return matched, dt_min
    return matched
