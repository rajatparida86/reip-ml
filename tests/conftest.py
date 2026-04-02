from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.model_registry import ModelRegistry


@pytest.fixture()
def stub_registry() -> MagicMock:
    """Deterministic stub for ModelRegistry — no .joblib files needed.

    Always returns (p10=710.0, p50=842.5, p90=960.0) as scalar arrays
    matching the length of the input X.
    """
    registry = MagicMock(spec=ModelRegistry)
    registry.version = "solar_generic_v1"

    def _predict(X: np.ndarray):
        n = X.shape[0]
        return (
            np.full(n, 710.0),
            np.full(n, 842.5),
            np.full(n, 960.0),
        )

    registry.predict.side_effect = _predict
    return registry


@pytest.fixture()
def client(stub_registry) -> TestClient:
    """TestClient with stub_registry injected into app.state."""
    from app.main import app

    app.state.registry = stub_registry
    return TestClient(app)
