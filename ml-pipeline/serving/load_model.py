"""Model loader: downloads from Cloudflare R2 (S3-compatible), falls back to local file."""
from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import boto3
import joblib
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

FEATURE_NAMES: list[str] = [
    "year", "zone_id", "temperature", "rainfall_mm",
    "humidity", "harvest_quantity", "transport_cost",
    "market_demand_index", "prev_day_price",
]

_LOCAL_FALLBACK = Path(__file__).parent.parent / "models" / "xgboost_v1.joblib"


def _s3_client():
    """Build boto3 client for Cloudflare R2 or standard S3."""
    endpoint = os.getenv("S3_ENDPOINT_URL")  # None → AWS S3
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.getenv("AWS_REGION", "auto"),
    )


@dataclass
class ModelBundle:
    """Holds the loaded model and its metadata."""
    model: object
    version: str
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    r2_score: float | None = None
    training_samples: int | None = None
    trained_at: datetime | None = None
    objective: str = "reg:squarederror"
    feature_names: list[str] = field(default_factory=lambda: FEATURE_NAMES)


# Module-level singleton —— updated by reload_model()
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


def _load_from_bytes(data: bytes, version: str) -> ModelBundle:
    model = joblib.load(io.BytesIO(data))
    logger.info("Model loaded from bytes, version=%s", version)
    return ModelBundle(model=model, version=version)


def _load_local(path: Path, version: str) -> ModelBundle:
    model = joblib.load(path)
    logger.info("Model loaded from local file: %s", path)
    return ModelBundle(model=model, version=version)


def download_latest_from_r2() -> tuple[bytes, str] | None:
    """
    Download the latest model from R2/S3.
    Returns (model_bytes, key) or None on failure.
    """
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        logger.warning("S3_BUCKET not set — skipping R2 download.")
        return None
    try:
        s3 = _s3_client()
        # List objects with prefix "models/" and pick the most-recent by LastModified
        resp = s3.list_objects_v2(Bucket=bucket, Prefix="models/xgboost_v")
        objects = resp.get("Contents", [])
        if not objects:
            logger.info("No models found in R2 bucket '%s'.", bucket)
            return None
        latest = max(objects, key=lambda o: o["LastModified"])
        key = latest["Key"]
        logger.info("Downloading model from R2: s3://%s/%s", bucket, key)
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        version = Path(key).stem  # e.g. "xgboost_v20240512_1400"
        return data, version
    except (ClientError, NoCredentialsError) as exc:
        logger.error("R2 download failed: %s", exc)
        return None


def upload_to_r2(model_path: Path, key: str) -> bool:
    """Upload a model file to R2/S3. Returns True on success."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        logger.warning("S3_BUCKET not set — skipping upload.")
        return False
    try:
        s3 = _s3_client()
        s3.upload_file(str(model_path), bucket, key)
        logger.info("Uploaded model to R2: s3://%s/%s", bucket, key)
        return True
    except (ClientError, NoCredentialsError) as exc:
        logger.error("R2 upload failed: %s", exc)
        return False


def load_model_on_startup() -> ModelBundle:
    """
    Called once at startup via FastAPI lifespan.
    Priority: R2/S3  →  local fallback.
    """
    global _bundle
    default_version = os.getenv("MODEL_VERSION", "v1")

    result = download_latest_from_r2()
    if result:
        data, version = result
        _bundle = _load_from_bytes(data, version)
        # Persist locally as cache
        local_cache = Path(__file__).parent.parent / "models" / f"{version}.joblib"
        local_cache.parent.mkdir(parents=True, exist_ok=True)
        local_cache.write_bytes(data)
    else:
        logger.warning("Falling back to local model file.")
        if not _LOCAL_FALLBACK.exists():
            raise FileNotFoundError(f"Local model not found: {_LOCAL_FALLBACK}")
        _bundle = _load_local(_LOCAL_FALLBACK, default_version)

    return _bundle


def reload_model() -> ModelBundle:
    """
    Re-download the latest model from R2 and hot-swap in memory.
    Called by POST /api/model/reload without restarting the process.
    """
    global _bundle
    result = download_latest_from_r2()
    if result:
        data, version = result
        _bundle = _load_from_bytes(data, version)
        logger.info("Model hot-reloaded: version=%s", version)
    else:
        logger.warning("reload_model: no new model in R2, keeping current.")
    return _bundle
