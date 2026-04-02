from app.main import app


def test_import():
    assert app is not None


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_body(client):
    response = client.get("/health")
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "solar_generic_v1"
