"""Model loader: reads from PostgreSQL (model_versions.model_data), falls back to local file."""
from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import asyncpg
import joblib

logger = logging.getLogger(__name__)

FEATURE_NAMES: list[str] = [
    "year", "zone_id", "temperature", "rainfall_mm",
    "humidity", "harvest_quantity", "transport_cost",
    "market_demand_index", "prev_day_price",
]

_LOCAL_FALLBACK = Path(__file__).parent.parent / "models" / "xgboost_v1.joblib"


def _asyncpg_url() -> str:
    """Convert any postgresql:// or postgresql+asyncpg:// URL to plain asyncpg format."""
    url = os.environ.get("DATABASE_URL", "")
    return url.replace("postgresql+asyncpg://", "postgresql://")


@dataclass
class ModelBundle:
    model: object
    version: str
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    r2_score: float | None = None
    training_samples: int | None = None
    trained_at: datetime | None = None
    objective: str = "reg:squarederror"
    feature_names: list[str] = field(default_factory=lambda: FEATURE_NAMES)


_bundle: ModelBundle | None = None
_prediction_counter: int = 0


def get_bundle() -> ModelBundle:
    if _bundle is None:
        raise RuntimeError("Model not loaded. Call load_model_on_startup() first.")
    return _bundle


def increment_predictions(n: int = 1) -> None:
    global _prediction_counter
    _prediction_counter += n


def total_predictions() -> int:
    return _prediction_counter


def _from_bytes(data: bytes, version: str) -> ModelBundle:
    model = joblib.load(io.BytesIO(data))
    logger.info("Model loaded from bytes, version=%s", version)
    return ModelBundle(model=model, version=version)


def _from_local(path: Path, version: str) -> ModelBundle:
    model = joblib.load(path)
    logger.info("Model loaded from local file: %s", path)
    return ModelBundle(model=model, version=version)


async def _fetch_from_db() -> tuple[bytes, str] | None:
    """Query model_versions for the active model's binary. Returns (data, version) or None."""
    url = _asyncpg_url()
    if not url:
        logger.warning("DATABASE_URL not set — cannot load model from DB.")
        return None
    try:
        conn = await asyncpg.connect(url)
        try:
            row = await conn.fetchrow(
                """
                SELECT version, model_data
                FROM model_versions
                WHERE is_active = TRUE AND model_data IS NOT NULL
                ORDER BY trained_at DESC NULLS LAST
                LIMIT 1
                """
            )
            if row and row["model_data"]:
                logger.info("Found active model in DB: version=%s", row["version"])
                return bytes(row["model_data"]), row["version"]
            logger.info("No active model with data found in DB.")
            return None
        finally:
            await conn.close()
    except Exception as exc:
        logger.error("DB model fetch failed: %s", exc)
        return None


async def save_to_db(
    model_path: Path,
    version: str,
    r2_score: float,
    training_samples: int,
) -> bool:
    """
    Upsert the model binary and metadata into model_versions,
    then mark all other rows is_active=FALSE.
    Called by training/train.py after a successful training run.
    """
    url = _asyncpg_url()
    if not url:
        logger.error("DATABASE_URL not set — cannot save model to DB.")
        return False
    try:
        data = model_path.read_bytes()
        conn = await asyncpg.connect(url)
        try:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO model_versions
                        (version, model_data, r2_score, training_samples, trained_at, is_active)
                    VALUES ($1, $2, $3, $4, NOW(), TRUE)
                    ON CONFLICT (version) DO UPDATE
                        SET model_data        = EXCLUDED.model_data,
                            r2_score          = EXCLUDED.r2_score,
                            training_samples  = EXCLUDED.training_samples,
                            trained_at        = EXCLUDED.trained_at,
                            is_active         = TRUE
                    """,
                    version, data, r2_score, training_samples,
                )
                await conn.execute(
                    "UPDATE model_versions SET is_active = FALSE WHERE version != $1",
                    version,
                )
            logger.info("Model saved to DB: version=%s  size=%d bytes", version, len(data))
            return True
        finally:
            await conn.close()
    except Exception as exc:
        logger.error("DB model save failed: %s", exc)
        return False


async def load_model_on_startup() -> ModelBundle:
    """
    Priority: PostgreSQL model_versions table → local fallback file.
    Called once from the FastAPI lifespan.
    """
    global _bundle
    default_version = os.getenv("MODEL_VERSION", "v1")

    result = await _fetch_from_db()
    if result:
        data, version = result
        _bundle = _from_bytes(data, version)
    else:
        logger.warning("No model in DB — using local fallback file.")
        if not _LOCAL_FALLBACK.exists():
            raise FileNotFoundError(f"Local fallback model not found: {_LOCAL_FALLBACK}")
        _bundle = _from_local(_LOCAL_FALLBACK, default_version)

    return _bundle


async def reload_model() -> ModelBundle:
    """
    Re-fetch the latest active model from PostgreSQL and hot-swap in memory.
    Called by POST /api/model/reload after retraining.
    """
    global _bundle
    result = await _fetch_from_db()
    if result:
        data, version = result
        _bundle = _from_bytes(data, version)
        logger.info("Model hot-reloaded from DB: version=%s", version)
    else:
        logger.warning("reload_model: no model found in DB, keeping current.")
    return _bundle
