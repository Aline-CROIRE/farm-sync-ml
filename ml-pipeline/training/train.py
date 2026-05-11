"""
FarmerSync model retraining script.

Runs as a Render Cron Job (daily at 02:00 UTC) or triggered manually via
POST /api/retrain.  Requires DATABASE_URL, API_SECRET_TOKEN, ML_API_URL.

Usage:
    python -m training.train
    python -m training.train --job-id abc123 --csv /path/to/data.csv
"""
from __future__ import annotations

import argparse
import asyncio
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving.load_model import FEATURE_NAMES, save_to_db
from training.data_loader import _sync_engine, load_csv, load_training_data, mark_as_used
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


def _load_current_model_from_db() -> object | None:
    """Fetch the active model binary from DB using the sync engine."""
    from sqlalchemy import text  # noqa: PLC0415

    engine = _sync_engine()
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT model_data FROM model_versions
                    WHERE is_active = TRUE AND model_data IS NOT NULL
                    ORDER BY trained_at DESC NULLS LAST LIMIT 1
                """)
            ).fetchone()
        if row and row[0]:
            import io  # noqa: PLC0415
            return joblib.load(io.BytesIO(bytes(row[0])))
        return None
    except Exception as exc:
        logger.warning("Could not fetch current model from DB: %s", exc)
        return None


def _reload_api() -> None:
    """Call POST /api/model/reload so the serving process hot-swaps."""
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

    # ── 2. Train/test split (time-ordered, no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False
    )

    # ── 3. Train new model
    logger.info("Training XGBRegressor: %s", XGBOOST_PARAMS)
    new_model = XGBRegressor(**XGBOOST_PARAMS)
    new_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    new_result = evaluate(new_model, X_test, y_test)
    logger.info("New model: %s", new_result)

    # ── 4. Compare against current model from DB
    current_model = _load_current_model_from_db()
    if current_model is not None:
        current_result = evaluate(current_model, X_test, y_test)
        logger.info("Current model: %s", current_result)
        if not is_improvement(new_result, current_result, MIN_DELTA):
            logger.info(
                "No improvement of %.1f%% — aborting deployment.", MIN_DELTA * 100
            )
            return
    else:
        logger.info("No current model in DB — deploying unconditionally.")

    # ── 5. Save model to disk temporarily
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    version = f"xgboost_v{timestamp}"
    local_path = MODELS_DIR / f"{version}.joblib"
    joblib.dump(new_model, local_path)
    logger.info("Model saved locally: %s", local_path)

    # ── 6. Upload binary to PostgreSQL model_versions table
    saved = asyncio.run(
        save_to_db(local_path, version, new_result.r2, len(X_train))
    )
    if not saved:
        logger.error("DB save failed — model not deployed.")
        return

    # ── 7. Mark training rows as used
    mark_as_used(_sync_engine())

    # ── 8. Hot-reload the serving process
    _reload_api()

    logger.info("=== Retraining job %s complete — deployed %s ===", job_id, version)


def main() -> None:
    parser = argparse.ArgumentParser(description="FarmerSync model retraining")
    parser.add_argument("--job-id", default="cron")
    parser.add_argument("--csv", default=None)
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
