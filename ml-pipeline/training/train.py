"""
FarmerSync model retraining script.

Runs as a Render Cron Job (daily at 02:00 UTC) or triggered manually via
POST /api/retrain.  Requires DATABASE_URL, S3_*, and ML_API_URL env vars.

Usage:
    python -m training.train
    python -m training.train --job-id abc123 --csv /path/to/data.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ── ensure project root is on sys.path when run as __main__
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving.load_model import FEATURE_NAMES, download_latest_from_r2, upload_to_r2
from training.data_loader import (
    _sync_engine,
    load_csv,
    load_training_data,
    mark_as_used,
)
from training.evaluate import EvalResult, evaluate, is_improvement

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("training.train")

MIN_SAMPLES = int(os.getenv("MIN_SAMPLES_TO_RETRAIN", "100"))
MIN_DELTA = float(os.getenv("MIN_ACCURACY_IMPROVEMENT", "0.01"))
ML_API_URL = os.getenv("ML_API_URL", "")
API_TOKEN = os.getenv("API_SECRET_TOKEN", "")
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

XGBOOST_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


def _load_current_model() -> tuple[object, EvalResult | None]:
    """
    Download and evaluate the current production model from R2.
    Returns (model_object, eval_result_or_None).
    """
    result = download_latest_from_r2()
    if not result:
        logger.warning("No current model in R2 — any trained model will be deployed.")
        return None, None
    import io  # noqa: PLC0415
    data, _ = result
    model = joblib.load(io.BytesIO(data))
    return model, None  # we'll evaluate both on the same test split


def _save_model_metadata(
    version: str,
    result: EvalResult,
    n_train: int,
) -> None:
    """Write model version record to PostgreSQL."""
    from sqlalchemy import text  # noqa: PLC0415

    engine = _sync_engine()
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO model_versions
                        (version, r2_score, training_samples, trained_at, is_active)
                    VALUES
                        (:version, :r2, :samples, :trained_at, TRUE)
                    ON CONFLICT (version) DO UPDATE
                        SET r2_score = EXCLUDED.r2_score,
                            training_samples = EXCLUDED.training_samples,
                            trained_at = EXCLUDED.trained_at,
                            is_active = TRUE;
                """),
                {
                    "version": version,
                    "r2": result.r2,
                    "samples": n_train,
                    "trained_at": datetime.utcnow(),
                },
            )
            # Deactivate all other versions
            conn.execute(
                text("UPDATE model_versions SET is_active = FALSE WHERE version != :v"),
                {"v": version},
            )
        logger.info("Model metadata saved to DB: version=%s", version)
    except Exception as exc:
        logger.error("Failed to save model metadata: %s", exc)


def _reload_api() -> None:
    """Call POST /api/model/reload so the serving process hot-swaps the model."""
    if not ML_API_URL or not API_TOKEN:
        logger.warning("ML_API_URL or API_SECRET_TOKEN not set — skipping reload call.")
        return
    url = ML_API_URL.rstrip("/") + "/api/model/reload"
    try:
        resp = httpx.post(url, headers={"Authorization": f"Bearer {API_TOKEN}"}, timeout=30)
        resp.raise_for_status()
        logger.info("API reload response: %s", resp.json())
    except Exception as exc:
        logger.error("Failed to call /api/model/reload: %s", exc)


def train(job_id: str = "manual", csv_path: str | None = None) -> None:
    logger.info("=== Retraining job %s started ===", job_id)

    # ── 1. Load data
    if csv_path:
        logger.info("Loading data from CSV: %s", csv_path)
        X, y = load_csv(csv_path)
    else:
        logger.info("Loading data from PostgreSQL (unused rows only).")
        X, y = load_training_data(min_samples=MIN_SAMPLES, only_unused=True)

    logger.info("Dataset shape: X=%s  y=%s", X.shape, y.shape)

    # ── 2. Train/test split (time-ordered, so no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False
    )

    # ── 3. Train new model
    logger.info("Training XGBRegressor with params: %s", XGBOOST_PARAMS)
    new_model = XGBRegressor(**XGBOOST_PARAMS)
    new_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    new_result = evaluate(new_model, X_test, y_test)
    logger.info("New model metrics: %s", new_result)

    # ── 4. Compare against current production model
    current_model, _ = _load_current_model()
    if current_model is not None:
        current_result = evaluate(current_model, X_test, y_test)
        logger.info("Current model metrics: %s", current_result)
        if not is_improvement(new_result, current_result, MIN_DELTA):
            logger.info(
                "New model did not improve by %.1f%% — aborting deployment.",
                MIN_DELTA * 100,
            )
            return
    else:
        logger.info("No current model to compare — deploying unconditionally.")

    # ── 5. Save model locally
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    version = f"xgboost_v{timestamp}"
    local_path = MODELS_DIR / f"{version}.joblib"
    joblib.dump(new_model, local_path)
    logger.info("Model saved locally: %s", local_path)

    # ── 6. Upload to R2/S3
    r2_key = f"models/{version}.joblib"
    uploaded = upload_to_r2(local_path, r2_key)
    if not uploaded:
        logger.error("Upload to R2 failed — model not deployed.")
        return

    # ── 7. Mark training rows as used
    engine = _sync_engine()
    mark_as_used(engine)

    # ── 8. Save metadata to PostgreSQL
    _save_model_metadata(version, new_result, len(X_train))

    # ── 9. Hot-reload the serving process
    _reload_api()

    logger.info("=== Retraining job %s complete — deployed %s ===", job_id, version)


def main() -> None:
    parser = argparse.ArgumentParser(description="FarmerSync model retraining")
    parser.add_argument("--job-id", default="cron", help="Job identifier for logging")
    parser.add_argument("--csv", default=None, help="Path to a CSV file to use instead of DB")
    args = parser.parse_args()
    try:
        train(job_id=args.job_id, csv_path=args.csv)
    except ValueError as exc:
        logger.error("Training aborted: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("Training crashed: %s", exc, exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
