"""
ModelRegistry: load-once XGBoost quantile models at startup.

The registry is stored on app.state.registry via the FastAPI lifespan
context manager and shared across all requests without copying.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np

MODEL_VERSION = "solar_generic_v1"

_MODEL_FILES = {
    "p10": "generic_v1_p10.joblib",
    "p50": "generic_v1_p50.joblib",
    "p90": "generic_v1_p90.joblib",
}


@dataclass
class ModelRegistry:
    p10_model: object
    p50_model: object
    p90_model: object
    version: str = MODEL_VERSION

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run all three quantile models and return (p10, p50, p90) arrays."""
        return (
            self.p10_model.predict(X),
            self.p50_model.predict(X),
            self.p90_model.predict(X),
        )


def load_registry(model_dir: str) -> ModelRegistry:
    """Load P10/P50/P90 models from model_dir/solar/.

    Raises FileNotFoundError if the directory or any model file is missing.
    """
    solar_dir = os.path.join(model_dir, "solar")

    if not os.path.isdir(solar_dir):
        raise FileNotFoundError(f"Model directory not found: {solar_dir}")

    models = {}
    for quantile, filename in _MODEL_FILES.items():
        path = os.path.join(solar_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        models[quantile] = joblib.load(path)

    return ModelRegistry(
        p10_model=models["p10"],
        p50_model=models["p50"],
        p90_model=models["p90"],
    )
