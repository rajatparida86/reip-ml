import pytest

from app.model_registry import load_registry


def test_load_registry_missing_dir():
    with pytest.raises(FileNotFoundError):
        load_registry("/nonexistent/path/models")


def test_load_registry_missing_file(tmp_path):
    # Directory exists but no .joblib files inside
    solar_dir = tmp_path / "solar"
    solar_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_registry(str(tmp_path))
