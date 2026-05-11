"""FarmerSync ML API — FastAPI application entrypoint."""
from __future__ import annotations

import csv
import io
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Security,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ModelVersion, PredictionLog, TrainingData, get_db
from serving.load_model import (
    FEATURE_NAMES,
    ModelBundle,
    get_bundle,
    increment_predictions,
    load_model_on_startup,
    reload_model,
    total_predictions,
)
from serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DataFeedRequest,
    DataFeedResponse,
    ErrorResponse,
    FeaturesNamed,
    HealthResponse,
    ModelInfoResponse,
    ModelReloadResponse,
    PredictRequest,
    PredictResponse,
    RetrainResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

API_TOKEN = os.environ.get("API_SECRET_TOKEN", "")
bearer_scheme = HTTPBearer()


def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Security(bearer_scheme)],
) -> str:
    if not API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_SECRET_TOKEN is not configured on the server.",
        )
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token.",
        )
    return credentials.credentials


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model...")
    try:
        await load_model_on_startup()
        logger.info("Model ready: version=%s", get_bundle().version)
    except Exception as exc:
        logger.critical("Failed to load model at startup: %s", exc)
        raise
    yield
    logger.info("Shutting down FarmerSync ML API.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_DESCRIPTION = """
## FarmerSync ML API

Real-time **farm gate price forecasting** for Rwandan agricultural markets,
powered by an XGBoost regressor (R² = 0.95).

### Features
- **Single & batch predictions** — submit raw arrays or named feature dicts
- **Continuous retraining** — feed labelled data via JSON or CSV; trigger or schedule retraining
- **Hot model reload** — swap the active model from Cloudflare R2 without a restart
- **Full audit trail** — every prediction logged to PostgreSQL

### Authentication
All `POST` endpoints require a Bearer token:
```
Authorization: Bearer <API_SECRET_TOKEN>
```
`GET` endpoints (`/health`, `/api/model/info`) are public.

### Feature schema (9 inputs → 1 predicted price)
| # | Name | Type | Description |
|---|------|------|-------------|
| 1 | `year` | int | Calendar year |
| 2 | `zone_id` | int | Climate zone (0=Kigali 1=South 2=North 3=West 4=East) |
| 3 | `temperature` | float | °C |
| 4 | `rainfall_mm` | float | Daily rainfall (mm) |
| 5 | `humidity` | float | Relative humidity (%) |
| 6 | `harvest_quantity` | float | kg |
| 7 | `transport_cost` | float | RWF |
| 8 | `market_demand_index` | float | 0 – 1 |
| 9 | `prev_day_price` | float | Previous day price (RWF/kg) |
"""

_TAGS: list[dict] = [
    {
        "name": "Prediction",
        "description": (
            "Run price-forecasting inference. Send a single record or a batch. "
            "Accepts either a raw feature **array** or a **named dict**."
        ),
    },
    {
        "name": "Data",
        "description": (
            "Feed labelled training samples back into the system. "
            "Accepts JSON records or a CSV file upload. "
            "Saved rows become available for the next retraining run."
        ),
    },
    {
        "name": "Training",
        "description": (
            "Trigger or schedule model retraining. "
            "The job runs asynchronously; monitor progress in the server logs. "
            "The Render Cron Job calls `python -m training.train` daily at 02:00 UTC."
        ),
    },
    {
        "name": "Model",
        "description": (
            "Inspect the active model's metadata or hot-reload a newer version "
            "from Cloudflare R2 without restarting the process."
        ),
    },
    {
        "name": "System",
        "description": "Liveness / readiness probe. No authentication required.",
    },
]

_RESPONSES_401_422 = {
    401: {
        "description": "Invalid or missing Bearer token",
        "content": {
            "application/json": {
                "example": {
                    "error": "unauthorized",
                    "detail": "Invalid or missing Bearer token.",
                    "status_code": 401,
                }
            }
        },
    },
    422: {
        "description": "Validation error — wrong number of features or bad field types",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "features"],
                            "msg": "Expected 9 features, got 3.",
                            "type": "value_error",
                        }
                    ]
                }
            }
        },
    },
}

app = FastAPI(
    title="FarmerSync ML API",
    description=_DESCRIPTION,
    version="1.0.0",
    contact={
        "name": "FarmerSync Engineering",
        "email": "benjaminwell250@gmail.com",
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=_TAGS,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error", detail=str(exc), status_code=500
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness probe",
    responses={
        200: {
            "description": "Service is up and model is loaded",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "model_version": "xgboost_v20240512_1400",
                        "total_predictions": 1024,
                    }
                }
            },
        },
        503: {
            "description": "Model not yet loaded",
            "content": {
                "application/json": {
                    "example": {
                        "status": "model_not_loaded",
                        "model_version": "none",
                        "total_predictions": 0,
                    }
                }
            },
        },
    },
)
async def health_check():
    """Returns `ok` when the model is loaded and ready to serve predictions."""
    try:
        bundle = get_bundle()
        return HealthResponse(
            status="ok",
            model_version=bundle.version,
            total_predictions=total_predictions(),
        )
    except RuntimeError:
        return JSONResponse(
            status_code=503,
            content={"status": "model_not_loaded", "model_version": "none", "total_predictions": 0},
        )


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _to_array(req: PredictRequest) -> np.ndarray:
    """Convert a PredictRequest to a 2D numpy array (1 × 9)."""
    if req.features is not None:
        return np.array([req.features], dtype=float)
    data: FeaturesNamed = req.data  # type: ignore[assignment]
    return np.array([[
        data.year, data.zone_id, data.temperature, data.rainfall_mm,
        data.humidity, data.harvest_quantity, data.transport_cost,
        data.market_demand_index, data.prev_day_price,
    ]], dtype=float)


async def _log_prediction(
    db: AsyncSession,
    features: list,
    prediction: float,
    version: str,
) -> None:
    record = PredictionLog(
        features={"values": features, "names": FEATURE_NAMES},
        prediction=prediction,
        model_version=version,
    )
    db.add(record)


# ---------------------------------------------------------------------------
# POST /api/predict
# ---------------------------------------------------------------------------

@app.post(
    "/api/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Single-record price forecast",
    description=(
        "Predict the farm gate price for **one record**. "
        "Supply either a raw `features` array (order matters — see feature table above) "
        "or a `data` named dict. The two fields are mutually exclusive."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "examples": {
                        "array_input": {
                            "summary": "Feature array",
                            "value": {
                                "predicted_price": 428.28,
                                "model_version": "xgboost_v20240512_1400",
                                "feature_names": [
                                    "year", "zone_id", "temperature", "rainfall_mm",
                                    "humidity", "harvest_quantity", "transport_cost",
                                    "market_demand_index", "prev_day_price",
                                ],
                            },
                        },
                    }
                }
            }
        },
        **_RESPONSES_401_422,
    },
)
async def predict(
    req: PredictRequest,
    db: AsyncSession = Depends(get_db),
    _token: str = Depends(verify_token),
):
    bundle: ModelBundle = get_bundle()
    X = _to_array(req)
    try:
        prediction: float = float(bundle.model.predict(X)[0])
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Model inference error: {exc}")

    increment_predictions()
    await _log_prediction(db, X[0].tolist(), prediction, bundle.version)

    return PredictResponse(
        predicted_price=round(prediction, 2),
        model_version=bundle.version,
        feature_names=FEATURE_NAMES,
    )


# ---------------------------------------------------------------------------
# POST /api/predict/batch
# ---------------------------------------------------------------------------

@app.post(
    "/api/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Prediction"],
    summary="Batch price forecast",
    description=(
        "Predict prices for **multiple records** in one call. "
        "Each element in `records` must be an array of exactly 9 features "
        "in the canonical order. Results are returned in the same order as the input."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [428.28, 632.14, 310.08],
                        "count": 3,
                        "model_version": "xgboost_v20240512_1400",
                    }
                }
            }
        },
        **_RESPONSES_401_422,
    },
)
async def predict_batch(
    req: BatchPredictRequest,
    db: AsyncSession = Depends(get_db),
    _token: str = Depends(verify_token),
):
    bundle: ModelBundle = get_bundle()
    X = np.array(req.records, dtype=float)
    try:
        preds = bundle.model.predict(X).tolist()
    except Exception as exc:
        logger.error("Batch prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Model inference error: {exc}")

    increment_predictions(len(preds))
    for row, pred in zip(req.records, preds):
        await _log_prediction(db, row, float(pred), bundle.version)

    return BatchPredictResponse(
        predictions=[round(p, 2) for p in preds],
        count=len(preds),
        model_version=bundle.version,
    )


# ---------------------------------------------------------------------------
# POST /api/data/feed  (JSON)
# ---------------------------------------------------------------------------

@app.post(
    "/api/data/feed",
    response_model=DataFeedResponse,
    tags=["Data"],
    summary="Feed labelled training records (JSON)",
    description=(
        "Submit new labelled samples as JSON. Each record needs the 9 input features "
        "plus the actual observed `label` (price in RWF/kg). "
        "Records are stored in `training_data` with `used_for_training=false` "
        "and become available for the next retraining run."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {"records_saved": 50, "total_available": 3250}
                }
            }
        },
        **_RESPONSES_401_422,
    },
)
async def feed_data_json(
    req: DataFeedRequest,
    db: AsyncSession = Depends(get_db),
    _token: str = Depends(verify_token),
):
    rows = [
        TrainingData(
            features={"values": r.features, "names": FEATURE_NAMES},
            label=r.label,
            source="api",
        )
        for r in req.records
    ]
    db.add_all(rows)
    await db.flush()

    total = await db.scalar(select(func.count()).select_from(TrainingData))
    return DataFeedResponse(records_saved=len(rows), total_available=int(total or 0))


# ---------------------------------------------------------------------------
# POST /api/data/feed/csv  (file upload)
# ---------------------------------------------------------------------------

@app.post(
    "/api/data/feed/csv",
    response_model=DataFeedResponse,
    tags=["Data"],
    summary="Feed labelled training records (CSV upload)",
    description=(
        "Upload a `.csv` file containing training data. "
        "Required columns (order-independent): "
        "`year, zone_id, temperature, rainfall_mm, humidity, harvest_quantity, "
        "transport_cost, market_demand_index, prev_day_price, price`. "
        "The `price` column is the label (target value)."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {"records_saved": 150, "total_available": 3400}
                }
            }
        },
        400: {
            "description": "File is not a .csv or contains no data rows",
            "content": {
                "application/json": {
                    "example": {"detail": "Only .csv files are accepted."}
                }
            },
        },
        **_RESPONSES_401_422,
    },
)
async def feed_data_csv(
    file: UploadFile = File(..., description="CSV with columns matching feature names + 'price'"),
    db: AsyncSession = Depends(get_db),
    _token: str = Depends(verify_token),
):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    raw = await file.read()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    rows: list[TrainingData] = []

    for i, row in enumerate(reader):
        try:
            features = [float(row[f]) for f in FEATURE_NAMES]
            label = float(row["price"])
        except (KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Row {i + 1}: missing or invalid column — {exc}",
            )
        rows.append(TrainingData(
            features={"values": features, "names": FEATURE_NAMES},
            label=label,
            source="csv",
        ))

    if not rows:
        raise HTTPException(status_code=400, detail="CSV file contained no data rows.")

    db.add_all(rows)
    await db.flush()
    total = await db.scalar(select(func.count()).select_from(TrainingData))
    return DataFeedResponse(records_saved=len(rows), total_available=int(total or 0))


# ---------------------------------------------------------------------------
# POST /api/retrain
# ---------------------------------------------------------------------------

def _run_retraining_job(job_id: str) -> None:
    """Synchronous wrapper executed inside a background thread."""
    import subprocess  # noqa: PLC0415
    logger.info("Retraining job %s started.", job_id)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "training.train", "--job-id", job_id],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        if result.returncode != 0:
            logger.error("Retraining job %s failed:\n%s", job_id, result.stderr)
        else:
            logger.info("Retraining job %s completed.\n%s", job_id, result.stdout)
    except Exception as exc:
        logger.error("Retraining job %s raised: %s", job_id, exc)


@app.post(
    "/api/retrain",
    response_model=RetrainResponse,
    tags=["Training"],
    summary="Trigger model retraining asynchronously",
    description=(
        "Start a retraining job in the background. The job: \n\n"
        "1. Fetches unused rows from `training_data` (needs ≥ `MIN_SAMPLES_TO_RETRAIN`)\n"
        "2. Trains a new XGBoost model\n"
        "3. Compares R² against the current model (needs ≥ `MIN_ACCURACY_IMPROVEMENT`)\n"
        "4. Uploads the winner to Cloudflare R2\n"
        "5. Calls `POST /api/model/reload` to hot-swap without restart\n\n"
        "Returns immediately with a `job_id`. Monitor progress in server logs."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "status": "retraining_started",
                        "job_id": "a3f7b2c1",
                        "message": "Retraining running in background. Check server logs for progress.",
                    }
                }
            }
        },
        **_RESPONSES_401_422,
    },
)
async def trigger_retrain(
    background_tasks: BackgroundTasks,
    _token: str = Depends(verify_token),
):
    job_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(_run_retraining_job, job_id)
    return RetrainResponse(
        status="retraining_started",
        job_id=job_id,
        message="Retraining running in background. Check server logs for progress.",
    )


# ---------------------------------------------------------------------------
# POST /api/model/reload
# ---------------------------------------------------------------------------

@app.post(
    "/api/model/reload",
    response_model=ModelReloadResponse,
    tags=["Model"],
    summary="Hot-reload model from R2/S3 without restart",
    description=(
        "Downloads the most-recently-modified `.joblib` file from the configured "
        "R2/S3 bucket and swaps it into memory atomically. "
        "The serving process does **not** restart. "
        "Called automatically by `training/train.py` after a successful retraining run."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "status": "reloaded",
                        "model_version": "xgboost_v20240512_1400",
                    }
                }
            }
        },
        **_RESPONSES_401_422,
    },
)
async def model_reload(_token: str = Depends(verify_token)):
    try:
        bundle = await reload_model()
        return ModelReloadResponse(status="reloaded", model_version=bundle.version)
    except Exception as exc:
        logger.error("Model reload failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")


# ---------------------------------------------------------------------------
# GET /api/model/info
# ---------------------------------------------------------------------------

@app.get(
    "/api/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Active model metadata",
    description=(
        "Returns metadata for the currently-loaded model. "
        "Pulls `r2_score`, `training_samples`, and `trained_at` from the "
        "`model_versions` table when available, otherwise falls back to in-memory values."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "version": "xgboost_v20240512_1400",
                        "trained_at": "2024-05-12T14:00:00Z",
                        "r2_score": 0.9497,
                        "training_samples": 5000,
                        "features": [
                            "year", "zone_id", "temperature", "rainfall_mm",
                            "humidity", "harvest_quantity", "transport_cost",
                            "market_demand_index", "prev_day_price",
                        ],
                        "objective": "reg:squarederror",
                    }
                }
            }
        }
    },
)
async def model_info(db: AsyncSession = Depends(get_db)):
    bundle = get_bundle()
    row = await db.scalar(
        select(ModelVersion)
        .where(ModelVersion.version == bundle.version)
        .limit(1)
    )
    return ModelInfoResponse(
        version=bundle.version,
        trained_at=row.trained_at if row else bundle.loaded_at,
        r2_score=row.r2_score if row else bundle.r2_score,
        training_samples=row.training_samples if row else bundle.training_samples,
        features=FEATURE_NAMES,
        objective=bundle.objective,
    )
