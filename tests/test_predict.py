import pytest

SITE_ID = "550e8400-e29b-41d4-a716-446655440000"
LAT, LON = 52.52, 13.42

# ── helpers ───────────────────────────────────────────────────────────────────


def _feature_row(hour: int) -> dict:
    return {
        "timestamp": f"2024-06-01T{hour:02d}:00:00Z",
        "ghi": 600.0 + hour * 10,
        "temperature": 20.0,
        "wind_speed": 3.0,
        "cloud_cover": 0.2,
        "hour_of_day": hour,
        "day_of_year": 153,
        "panel_tilt": 30.0,
        "panel_azimuth": 180.0,
    }


def _payload(n: int = 24) -> dict:
    return {
        "site_id": SITE_ID,
        "latitude": LAT,
        "longitude": LON,
        "features": [_feature_row(h) for h in range(n)],
    }


# ── tests ─────────────────────────────────────────────────────────────────────


def test_predict_returns_200(client):
    response = client.post("/predict", json=_payload())
    assert response.status_code == 200


def test_predict_response_shape(client):
    response = client.post("/predict", json=_payload(24))
    body = response.json()
    assert body["site_id"] == SITE_ID
    assert len(body["predictions"]) == 24


def test_predict_stub_values(client):
    response = client.post("/predict", json=_payload(3))
    preds = response.json()["predictions"]
    assert preds[0]["p10_kwh"] == pytest.approx(710.0)
    assert preds[0]["p50_kwh"] == pytest.approx(842.5)
    assert preds[0]["p90_kwh"] == pytest.approx(960.0)


def test_predict_timestamp_preserved(client):
    response = client.post("/predict", json=_payload(3))
    preds = response.json()["predictions"]
    assert preds[0]["timestamp"] == "2024-06-01T00:00:00Z"
    assert preds[2]["timestamp"] == "2024-06-01T02:00:00Z"


def test_predict_confidence_score_range(client):
    response = client.post("/predict", json=_payload(3))
    preds = response.json()["predictions"]
    for pred in preds:
        assert 0.0 <= pred["confidence_score"] <= 1.0


def test_predict_model_version(client):
    response = client.post("/predict", json=_payload(1))
    pred = response.json()["predictions"][0]
    assert pred["model_version"] == "solar_generic_v1"


def test_predict_missing_features_returns_422(client):
    payload = {"site_id": SITE_ID, "latitude": LAT}  # missing longitude + features
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_empty_features_returns_422(client):
    payload = {
        "site_id": SITE_ID,
        "latitude": LAT,
        "longitude": LON,
        "features": [],
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
