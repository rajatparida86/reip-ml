import pandas as pd
import pvlib
from fastapi import APIRouter, HTTPException, Request

from app import features as feat
from app.model_registry import MODEL_VERSION
from app.schemas import Prediction, PredictRequest, PredictResponse

router = APIRouter()


def _clear_sky_ghi(lat: float, lon: float, timestamps: list) -> list[float]:
    """Compute clear-sky GHI for a list of UTC timestamps at the given location."""
    location = pvlib.location.Location(latitude=lat, longitude=lon)
    index = pd.DatetimeIndex(timestamps, tz="UTC")
    clear_sky = location.get_clearsky(index, model="ineichen")
    return clear_sky["ghi"].tolist()


@router.post("/predict", response_model=PredictResponse)
def predict(request: Request, body: PredictRequest) -> PredictResponse:
    registry = request.app.state.registry
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not loaded")

    timestamps = [row.timestamp for row in body.features]
    clear_sky_series = _clear_sky_ghi(body.latitude, body.longitude, timestamps)

    X = feat.build_feature_matrix(
        body.features, clear_sky_series, lat=body.latitude, lon=body.longitude
    )

    p10_arr, p50_arr, p90_arr = registry.predict(X)

    predictions = []
    for i, row in enumerate(body.features):
        p10 = float(p10_arr[i])
        p50 = float(p50_arr[i])
        p90 = float(p90_arr[i])
        confidence = 1.0 - (p90 - p10) / (p90 + 1e-9)
        predictions.append(
            Prediction(
                timestamp=row.timestamp,
                p10_kwh=p10,
                p50_kwh=p50,
                p90_kwh=p90,
                confidence_score=max(0.0, min(1.0, confidence)),
                model_version=MODEL_VERSION,
            )
        )

    return PredictResponse(site_id=body.site_id, predictions=predictions)
