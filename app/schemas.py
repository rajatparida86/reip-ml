from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field


class FeatureRow(BaseModel):
    timestamp: datetime
    ghi: float
    temperature: float
    wind_speed: float
    cloud_cover: float
    hour_of_day: int
    day_of_year: int
    panel_tilt: float
    panel_azimuth: float


class PredictRequest(BaseModel):
    site_id: str
    latitude: float
    longitude: float
    features: Annotated[list[FeatureRow], Field(min_length=1)]


class Prediction(BaseModel):
    timestamp: datetime
    p10_kwh: float
    p50_kwh: float
    p90_kwh: float
    confidence_score: float
    model_version: str


class PredictResponse(BaseModel):
    site_id: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_version: str
