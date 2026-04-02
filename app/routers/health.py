from fastapi import APIRouter

from app.schemas import HealthResponse

router = APIRouter()

MODEL_VERSION = "solar_generic_v1"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_version=MODEL_VERSION)
