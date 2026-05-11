"""Pydantic v2 request/response schemas for the FarmerSync ML serving API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class FeaturesNamed(BaseModel):
    """Named feature input matching the 9 model features."""
    year: int = Field(..., ge=1950, le=2100, examples=[2024])
    zone_id: int = Field(..., ge=0, le=10, description="Encoded climate zone", examples=[1])
    temperature: float = Field(..., description="°C", examples=[24.5])
    rainfall_mm: float = Field(..., ge=0, examples=[3.2])
    humidity: float = Field(..., ge=0, le=100, examples=[65.0])
    harvest_quantity: float = Field(..., ge=0, examples=[5000.0])
    transport_cost: float = Field(..., ge=0, examples=[140.0])
    market_demand_index: float = Field(..., ge=0, le=1, examples=[0.78])
    prev_day_price: float = Field(..., ge=0, examples=[230.0])


class PredictRequest(BaseModel):
    """Single-record prediction — either raw list or named dict."""
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Feature array (Southern zone, 2024)",
                    "value": {
                        "features": [2024, 1, 24.5, 3.2, 65.0, 5000.0, 140.0, 0.78, 230.0]
                    },
                },
                {
                    "summary": "Named dict (Kigali zone)",
                    "value": {
                        "data": {
                            "year": 2024, "zone_id": 0, "temperature": 27.0,
                            "rainfall_mm": 0.5, "humidity": 60.0,
                            "harvest_quantity": 3000.0, "transport_cost": 125.0,
                            "market_demand_index": 0.90, "prev_day_price": 250.0,
                        }
                    },
                },
            ]
        }
    }

    features: list[float] | None = Field(
        default=None,
        description="Raw feature array: [year, zone_id, temperature, rainfall_mm, "
                    "humidity, harvest_quantity, transport_cost, market_demand_index, prev_day_price]",
        examples=[[2024, 1, 24.5, 3.2, 65.0, 5000.0, 140.0, 0.78, 230.0]],
    )
    data: FeaturesNamed | None = Field(
        default=None,
        description="Named feature dict (alternative to `features`)",
    )

    @model_validator(mode="after")
    def exactly_one_input(self) -> "PredictRequest":
        if self.features is None and self.data is None:
            raise ValueError("Provide either `features` (list) or `data` (dict).")
        if self.features is not None and self.data is not None:
            raise ValueError("Provide only one of `features` or `data`, not both.")
        if self.features is not None and len(self.features) != 9:
            raise ValueError(f"Expected 9 features, got {len(self.features)}.")
        return self


class PredictResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted farm gate price (RWF/kg)")
    model_version: str
    feature_names: list[str]


class BatchPredictRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "value": {
                        "records": [
                            [2024, 1, 24.5, 3.2, 65.0, 5000.0, 140.0, 0.78, 230.0],
                            [2024, 0, 27.0, 0.5, 60.0, 3000.0, 125.0, 0.90, 250.0],
                            [2024, 2, 18.0, 8.0, 75.0, 8000.0, 220.0, 0.55, 180.0],
                        ]
                    }
                }
            ]
        }
    }

    records: list[list[float]] = Field(
        ...,
        description="List of feature arrays, each with 9 values.",
        examples=[[[2024, 1, 24.5, 3.2, 65.0, 5000.0, 140.0, 0.78, 230.0]]],
    )

    @model_validator(mode="after")
    def validate_records(self) -> "BatchPredictRequest":
        if not self.records:
            raise ValueError("`records` must not be empty.")
        bad = [i for i, r in enumerate(self.records) if len(r) != 9]
        if bad:
            raise ValueError(f"Records at indices {bad} do not have 9 features.")
        return self


class BatchPredictResponse(BaseModel):
    predictions: list[float]
    count: int
    model_version: str


# ---------------------------------------------------------------------------
# Training data feed
# ---------------------------------------------------------------------------

class TrainingRecord(BaseModel):
    features: list[float] = Field(..., min_length=9, max_length=9)
    label: float = Field(..., description="Actual price (RWF/kg)")


class DataFeedRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "value": {
                        "records": [
                            {
                                "features": [2024, 1, 24.5, 3.2, 65.0, 5000.0, 140.0, 0.78, 230.0],
                                "label": 245.0,
                            },
                            {
                                "features": [2024, 0, 27.0, 0.5, 60.0, 3000.0, 125.0, 0.90, 250.0],
                                "label": 260.0,
                            },
                        ]
                    }
                }
            ]
        }
    }

    records: list[TrainingRecord] = Field(..., min_length=1)


class DataFeedResponse(BaseModel):
    records_saved: int
    total_available: int


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

class RetrainResponse(BaseModel):
    status: str
    job_id: str
    message: str


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

class ModelReloadResponse(BaseModel):
    status: str
    model_version: str


class ModelInfoResponse(BaseModel):
    version: str
    trained_at: datetime | None
    r2_score: float | None
    training_samples: int | None
    features: list[str]
    objective: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_version: str
    total_predictions: int


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: Any = None
    status_code: int
