"""
Standalone prediction tests — no running server or DB required.
Patches DB calls so only the model + API logic is exercised.

Run:
    cd ml-pipeline
    python tests/test_prediction_live.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ── env must be set before imports ──────────────────────────────────────────
os.environ.setdefault("API_SECRET_TOKEN", "test-token")
os.environ.setdefault("MODEL_VERSION", "v1")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")
os.environ.setdefault("S3_BUCKET", "")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "dummy")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "dummy")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    symbol = PASS if passed else FAIL
    line = f"  {symbol} {name}"
    if detail:
        line += f"  →  {detail}"
    print(line)


# ────────────────────────────────────────────────────────────────────────────
# 1. Raw model prediction
# ────────────────────────────────────────────────────────────────────────────
print("\n── 1. Raw XGBoost model ──────────────────────────────────────────")

import joblib
import numpy as np

try:
    model = joblib.load(ROOT / "models" / "xgboost_v1.joblib")
    check("Model loads from disk", True)
except Exception as e:
    check("Model loads from disk", False, str(e))
    sys.exit(1)

SAMPLES = {
    "Normal day (Southern, 2024)":    [2024, 1, 24.5, 3.2, 65.0, 5000.0, 140.0, 0.78, 230.0],
    "Kigali, hot, low harvest":        [2024, 0, 27.0, 0.5, 60.0, 3000.0, 125.0, 0.90, 250.0],
    "Northern, rainy, high harvest":   [2024, 2, 18.0, 8.0, 75.0, 8000.0, 220.0, 0.55, 180.0],
    "Western zone":                    [2024, 3, 21.0, 5.5, 70.0, 6500.0, 175.0, 0.68, 210.0],
    "Historical 1950":                 [1950, 1, 23.0, 3.0, 68.0, 4000.0, 145.0, 0.70, 200.0],
}

for label, feats in SAMPLES.items():
    X = np.array([feats], dtype=float)
    pred = float(model.predict(X)[0])
    ok = pred > 0
    check(f"Predict: {label}", ok, f"price = {pred:.2f} RWF/kg")

# Batch
X_batch = np.array(list(SAMPLES.values()), dtype=float)
preds = model.predict(X_batch)
check("Batch predict (5 rows)", len(preds) == 5, f"predictions: {[round(p,2) for p in preds]}")

# Edge: single row
p_single = float(model.predict(np.array([[2024, 1, 24.5, 3.2, 65.0, 5000.0, 140.0, 0.78, 230.0]]))[0])
check("Single-row array predict", p_single > 0, f"{p_single:.2f}")


# ────────────────────────────────────────────────────────────────────────────
# 2. Pydantic schema validation
# ────────────────────────────────────────────────────────────────────────────
print("\n── 2. Schema validation ────────────────────────────────────────────")

from serving.schemas import BatchPredictRequest, PredictRequest

try:
    req = PredictRequest(features=[2024, 1, 24.5, 3.2, 65.0, 5000.0, 140.0, 0.78, 230.0])
    check("PredictRequest (feature list) valid", True)
except Exception as e:
    check("PredictRequest (feature list) valid", False, str(e))

try:
    req = PredictRequest(data={
        "year": 2024, "zone_id": 1, "temperature": 24.5, "rainfall_mm": 3.2,
        "humidity": 65.0, "harvest_quantity": 5000.0, "transport_cost": 140.0,
        "market_demand_index": 0.78, "prev_day_price": 230.0,
    })
    check("PredictRequest (named dict) valid", True)
except Exception as e:
    check("PredictRequest (named dict) valid", False, str(e))

# wrong feature count
try:
    PredictRequest(features=[1.0, 2.0, 3.0])
    check("Wrong feature count raises ValidationError", False, "should have failed")
except Exception:
    check("Wrong feature count raises ValidationError", True)

# both fields at once
try:
    PredictRequest(
        features=[2024, 1, 24.5, 3.2, 65.0, 5000.0, 140.0, 0.78, 230.0],
        data={"year": 2024, "zone_id": 1, "temperature": 24.5, "rainfall_mm": 3.2,
              "humidity": 65.0, "harvest_quantity": 5000.0, "transport_cost": 140.0,
              "market_demand_index": 0.78, "prev_day_price": 230.0},
    )
    check("Both features + data rejected", False, "should have failed")
except Exception:
    check("Both features + data rejected", True)

# batch
try:
    BatchPredictRequest(records=[[2024,1,24.5,3.2,65.0,5000.0,140.0,0.78,230.0]] * 3)
    check("BatchPredictRequest (3 rows) valid", True)
except Exception as e:
    check("BatchPredictRequest (3 rows) valid", False, str(e))


# ────────────────────────────────────────────────────────────────────────────
# 3. FastAPI TestClient (DB patched out)
# ────────────────────────────────────────────────────────────────────────────
print("\n── 3. FastAPI endpoints (DB mocked) ────────────────────────────────")

from fastapi.testclient import TestClient

# Patch DB session so no real Postgres connection is needed
mock_session = AsyncMock()
mock_session.scalar = AsyncMock(return_value=100)
mock_session.add = MagicMock()
mock_session.add_all = MagicMock()
mock_session.flush = AsyncMock()
mock_session.commit = AsyncMock()
mock_session.rollback = AsyncMock()
mock_session.close = AsyncMock()
mock_session.__aenter__ = AsyncMock(return_value=mock_session)
mock_session.__aexit__ = AsyncMock(return_value=False)

from contextlib import asynccontextmanager

@asynccontextmanager
async def mock_get_db_ctx():
    yield mock_session

async def mock_get_db():
    yield mock_session

import db.models as db_models
db_models.get_db = mock_get_db  # patch before app import

from serving.main import app
from db.models import get_db

app.dependency_overrides[get_db] = mock_get_db

AUTH = {"Authorization": "Bearer test-token"}
FEATS = [2024, 1, 24.5, 3.2, 65.0, 5000.0, 140.0, 0.78, 230.0]
NAMED = {
    "year": 2024, "zone_id": 1, "temperature": 24.5, "rainfall_mm": 3.2,
    "humidity": 65.0, "harvest_quantity": 5000.0, "transport_cost": 140.0,
    "market_demand_index": 0.78, "prev_day_price": 230.0,
}

with TestClient(app, raise_server_exceptions=True) as client:

    # Health
    r = client.get("/health")
    check("GET /health → 200", r.status_code == 200, r.json().get("status"))

    # Predict — feature list
    r = client.post("/api/predict", json={"features": FEATS}, headers=AUTH)
    ok = r.status_code == 200 and "predicted_price" in r.json()
    check("POST /api/predict (list)", ok, f"price={r.json().get('predicted_price')}")

    # Predict — named dict
    r = client.post("/api/predict", json={"data": NAMED}, headers=AUTH)
    ok = r.status_code == 200 and "predicted_price" in r.json()
    check("POST /api/predict (named dict)", ok, f"price={r.json().get('predicted_price')}")

    # Batch
    r = client.post("/api/predict/batch", json={"records": [FEATS, FEATS, FEATS]}, headers=AUTH)
    ok = r.status_code == 200 and r.json().get("count") == 3
    check("POST /api/predict/batch (3 rows)", ok, str(r.json().get("predictions")))

    # Wrong token → 401
    r = client.post("/api/predict", json={"features": FEATS},
                    headers={"Authorization": "Bearer wrong"})
    check("Wrong token → 401", r.status_code == 401)

    # No token → 403 (HTTPBearer raises 403 when header is absent)
    r = client.post("/api/predict", json={"features": FEATS})
    check("No token → 403 or 401", r.status_code in (401, 403))

    # Wrong feature count → 422
    r = client.post("/api/predict", json={"features": [1.0, 2.0]}, headers=AUTH)
    check("Wrong feature count → 422", r.status_code == 422)

    # Empty batch → 422
    r = client.post("/api/predict/batch", json={"records": []}, headers=AUTH)
    check("Empty batch → 422", r.status_code == 422)

    # Feed JSON
    payload = {"records": [{"features": FEATS, "label": 245.0}] * 5}
    r = client.post("/api/data/feed", json=payload, headers=AUTH)
    check("POST /api/data/feed (JSON)", r.status_code == 200,
          f"saved={r.json().get('records_saved')}")

    # Retrain trigger
    r = client.post("/api/retrain", headers=AUTH)
    ok = r.status_code == 200 and r.json().get("status") == "retraining_started"
    check("POST /api/retrain → starts job", ok, r.json().get("job_id"))


# ────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────
print()
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)
print(f"{'─'*60}")
print(f"  Results: {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} FAILED)")
    for name, ok, detail in results:
        if not ok:
            print(f"    {FAIL} {name}: {detail}")
else:
    print("  — all good!")
print()
sys.exit(0 if failed == 0 else 1)
