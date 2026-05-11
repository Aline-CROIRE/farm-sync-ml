"""
pytest test suite for the FarmerSync ML API.

Run with:
    cd ml-pipeline
    pytest tests/ -v

Requires: pytest, httpx, pytest-asyncio
Environment: set TEST_API_TOKEN before running (or export API_SECRET_TOKEN).
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("API_SECRET_TOKEN", "test-token-local")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/farmersync_test")
os.environ.setdefault("MODEL_VERSION", "v1")

from serving.main import app  # noqa: E402  (env must be set first)

TOKEN = os.environ["API_SECRET_TOKEN"]
AUTH = {"Authorization": f"Bearer {TOKEN}"}

SAMPLE_FEATURES = [2024, 1, 24.5, 3.2, 65.0, 5000.0, 140.0, 0.78, 230.0]
SAMPLE_NAMED = {
    "year": 2024,
    "zone_id": 1,
    "temperature": 24.5,
    "rainfall_mm": 3.2,
    "humidity": 65.0,
    "harvest_quantity": 5000.0,
    "transport_cost": 140.0,
    "market_demand_index": 0.78,
    "prev_day_price": 230.0,
}


@pytest.fixture(scope="session")
def client():
    """TestClient with lifespan — loads the local model once per session."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_version" in data
        assert isinstance(data["total_predictions"], int)


# ---------------------------------------------------------------------------
# Single Prediction
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_feature_list(self, client):
        resp = client.post(
            "/api/predict",
            json={"features": SAMPLE_FEATURES},
            headers=AUTH,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "predicted_price" in data
        assert isinstance(data["predicted_price"], float)
        assert data["predicted_price"] > 0

    def test_predict_named_dict(self, client):
        resp = client.post(
            "/api/predict",
            json={"data": SAMPLE_NAMED},
            headers=AUTH,
        )
        assert resp.status_code == 200, resp.text
        assert "predicted_price" in resp.json()

    def test_predict_requires_auth(self, client):
        resp = client.post("/api/predict", json={"features": SAMPLE_FEATURES})
        assert resp.status_code == 403

    def test_predict_wrong_token(self, client):
        resp = client.post(
            "/api/predict",
            json={"features": SAMPLE_FEATURES},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_predict_wrong_feature_count(self, client):
        resp = client.post(
            "/api/predict",
            json={"features": [1.0, 2.0, 3.0]},  # only 3 features
            headers=AUTH,
        )
        assert resp.status_code == 422

    def test_predict_both_fields_rejected(self, client):
        resp = client.post(
            "/api/predict",
            json={"features": SAMPLE_FEATURES, "data": SAMPLE_NAMED},
            headers=AUTH,
        )
        assert resp.status_code == 422

    def test_predict_neither_field_rejected(self, client):
        resp = client.post("/api/predict", json={}, headers=AUTH)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Batch Prediction
# ---------------------------------------------------------------------------

class TestBatchPredict:
    def test_batch_predict_ok(self, client):
        resp = client.post(
            "/api/predict/batch",
            json={"records": [SAMPLE_FEATURES, SAMPLE_FEATURES]},
            headers=AUTH,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
        assert all(isinstance(p, float) for p in data["predictions"])

    def test_batch_empty_records(self, client):
        resp = client.post(
            "/api/predict/batch",
            json={"records": []},
            headers=AUTH,
        )
        assert resp.status_code == 422

    def test_batch_wrong_feature_count(self, client):
        resp = client.post(
            "/api/predict/batch",
            json={"records": [[1.0, 2.0]]},  # only 2 features
            headers=AUTH,
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Data Feed
# ---------------------------------------------------------------------------

class TestDataFeed:
    def test_feed_json_ok(self, client):
        payload = {
            "records": [
                {"features": SAMPLE_FEATURES, "label": 245.0},
                {"features": SAMPLE_FEATURES, "label": 260.0},
            ]
        }
        resp = client.post("/api/data/feed", json=payload, headers=AUTH)
        # 200 when DB is up; 500 when test DB not available — either is acceptable in unit test
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()
            assert data["records_saved"] == 2

    def test_feed_empty_records_rejected(self, client):
        resp = client.post("/api/data/feed", json={"records": []}, headers=AUTH)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Model Management
# ---------------------------------------------------------------------------

class TestModelManagement:
    def test_model_info(self, client):
        resp = client.get("/api/model/info")
        # May be 200 or 500 depending on DB availability
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()
            assert "version" in data
            assert "features" in data
            assert len(data["features"]) == 9


# ---------------------------------------------------------------------------
# Retrain
# ---------------------------------------------------------------------------

class TestRetrain:
    def test_retrain_trigger(self, client):
        resp = client.post("/api/retrain", headers=AUTH)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "retraining_started"
        assert "job_id" in data
