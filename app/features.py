"""
Feature engineering for solar generation forecasting.

All functions are pure and stateless — no I/O, no global state.
FEATURE_COLUMNS is the single source of truth shared by both the training
script and the inference handler. Column order must never drift between them.
"""

from __future__ import annotations

import numpy as np

from app.schemas import FeatureRow

FEATURE_COLUMNS: list[str] = [
    "ghi",
    "temperature",
    "cloud_cover",
    "wind_speed",
    "panel_tilt",
    "panel_azimuth",
    "hour_of_day",
    "day_of_year",
    "clear_sky_index",
    "ghi_rolling_3h",
    "ghi_delta_1h",
    "latitude",
    "longitude",
]


def compute_clear_sky_index(ghi: float, ghi_clearsky: float) -> float:
    """Return GHI / clear-sky GHI. Returns 0.0 when clear-sky GHI is zero (night-time)."""
    if ghi_clearsky == 0.0:
        return 0.0
    return ghi / ghi_clearsky


def compute_rolling_ghi_3h(ghi_series: list[float]) -> list[float]:
    """3-element expanding-then-rolling mean of GHI.

    Index 0: mean of [ghi[0]]
    Index 1: mean of [ghi[0], ghi[1]]
    Index k>=2: mean of [ghi[k-2], ghi[k-1], ghi[k]]
    """
    result: list[float] = []
    for i, _ in enumerate(ghi_series):
        window = ghi_series[max(0, i - 2) : i + 1]
        result.append(sum(window) / len(window))
    return result


def compute_ghi_delta_1h(ghi_series: list[float]) -> list[float]:
    """Difference from the previous hour. First element is always 0.0."""
    if not ghi_series:
        return []
    result = [0.0]
    for i in range(1, len(ghi_series)):
        result.append(ghi_series[i] - ghi_series[i - 1])
    return result


def build_feature_matrix(
    rows: list[FeatureRow],
    clear_sky_series: list[float],
    lat: float,
    lon: float,
) -> np.ndarray:
    """Build an (n, 13) feature matrix in FEATURE_COLUMNS order.

    Args:
        rows: List of FeatureRow objects from the predict request.
        clear_sky_series: Clear-sky GHI values, one per row, computed via PVLIB.
        lat: Site latitude (degrees).
        lon: Site longitude (degrees).

    Returns:
        numpy array of shape (n, 13), dtype float64.
    """
    n = len(rows)
    ghi_series = [r.ghi for r in rows]

    rolling = compute_rolling_ghi_3h(ghi_series)
    delta = compute_ghi_delta_1h(ghi_series)

    matrix = np.empty((n, len(FEATURE_COLUMNS)), dtype=np.float64)

    for i, row in enumerate(rows):
        csi = compute_clear_sky_index(row.ghi, clear_sky_series[i])
        matrix[i] = [
            row.ghi,
            row.temperature,
            row.cloud_cover,
            row.wind_speed,
            row.panel_tilt,
            row.panel_azimuth,
            float(row.hour_of_day),
            float(row.day_of_year),
            csi,
            rolling[i],
            delta[i],
            lat,
            lon,
        ]

    return matrix
