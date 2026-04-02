from datetime import UTC, datetime

import numpy as np
import pytest

from app.features import (
    FEATURE_COLUMNS,
    build_feature_matrix,
    compute_clear_sky_index,
    compute_ghi_delta_1h,
    compute_rolling_ghi_3h,
)
from app.schemas import FeatureRow

# ── FEATURE_COLUMNS ───────────────────────────────────────────────────────────


def test_feature_columns_count():
    assert len(FEATURE_COLUMNS) == 13


def test_feature_columns_contains_location():
    assert "latitude" in FEATURE_COLUMNS
    assert "longitude" in FEATURE_COLUMNS


# ── compute_clear_sky_index ───────────────────────────────────────────────────


def test_clear_sky_index_normal():
    result = compute_clear_sky_index(ghi=650.0, ghi_clearsky=800.0)
    assert pytest.approx(result, rel=1e-6) == 650.0 / 800.0


def test_clear_sky_index_zero_denominator():
    # Night-time: clearsky = 0, must not divide by zero
    result = compute_clear_sky_index(ghi=0.0, ghi_clearsky=0.0)
    assert result == 0.0


def test_clear_sky_index_zero_ghi():
    result = compute_clear_sky_index(ghi=0.0, ghi_clearsky=500.0)
    assert result == 0.0


# ── compute_rolling_ghi_3h ────────────────────────────────────────────────────


def test_rolling_ghi_single_element():
    result = compute_rolling_ghi_3h([300.0])
    assert result == [300.0]


def test_rolling_ghi_two_elements():
    result = compute_rolling_ghi_3h([100.0, 200.0])
    assert result == pytest.approx([100.0, 150.0])


def test_rolling_ghi_three_elements():
    result = compute_rolling_ghi_3h([100.0, 200.0, 300.0])
    assert result == pytest.approx([100.0, 150.0, 200.0])


def test_rolling_ghi_four_elements():
    # Window is 3 — fourth element rolls: mean(200, 300, 400) = 300
    result = compute_rolling_ghi_3h([100.0, 200.0, 300.0, 400.0])
    assert result == pytest.approx([100.0, 150.0, 200.0, 300.0])


# ── compute_ghi_delta_1h ──────────────────────────────────────────────────────


def test_ghi_delta_first_element_is_zero():
    result = compute_ghi_delta_1h([500.0, 600.0, 400.0])
    assert result[0] == 0.0


def test_ghi_delta_subsequent_diffs():
    result = compute_ghi_delta_1h([500.0, 600.0, 400.0])
    assert result == pytest.approx([0.0, 100.0, -200.0])


def test_ghi_delta_single_element():
    result = compute_ghi_delta_1h([300.0])
    assert result == [0.0]


# ── build_feature_matrix ──────────────────────────────────────────────────────


def _make_rows(n: int) -> list[FeatureRow]:
    return [
        FeatureRow(
            timestamp=datetime(2024, 6, 1, h, 0, 0, tzinfo=UTC),
            ghi=float(600 + h * 10),
            temperature=20.0,
            wind_speed=3.0,
            cloud_cover=0.2,
            hour_of_day=h,
            day_of_year=153,
            panel_tilt=30.0,
            panel_azimuth=180.0,
        )
        for h in range(n)
    ]


def test_feature_matrix_shape():
    rows = _make_rows(24)
    clear_sky = [800.0] * 24
    X = build_feature_matrix(rows, clear_sky, lat=52.52, lon=13.42)
    assert X.shape == (24, 13)


def test_feature_matrix_column_order():
    rows = _make_rows(3)
    clear_sky = [800.0, 900.0, 700.0]
    X = build_feature_matrix(rows, clear_sky, lat=52.52, lon=13.42)
    # latitude column should be constant across rows
    lat_idx = FEATURE_COLUMNS.index("latitude")
    assert np.all(X[:, lat_idx] == pytest.approx(52.52))


def test_feature_matrix_clear_sky_index_column():
    rows = _make_rows(3)
    clear_sky = [800.0, 800.0, 800.0]
    X = build_feature_matrix(rows, clear_sky, lat=52.52, lon=13.42)
    csi_idx = FEATURE_COLUMNS.index("clear_sky_index")
    expected_csi = rows[0].ghi / 800.0
    assert X[0, csi_idx] == pytest.approx(expected_csi)
