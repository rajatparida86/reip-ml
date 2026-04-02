from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.model_registry import load_registry
from app.routers import health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models once and store on app.state
    try:
        app.state.registry = load_registry(settings.model_dir)
    except FileNotFoundError:
        # In development/CI without model files, registry stays unset.
        # The predict endpoint will raise a 503 if called without models.
        app.state.registry = None
    yield
    # Shutdown: nothing to clean up for XGBoost models


app = FastAPI(
    title="reip-ml",
    description="REIP ML inference service — solar generation forecasting (P10/P50/P90)",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predict.router)
