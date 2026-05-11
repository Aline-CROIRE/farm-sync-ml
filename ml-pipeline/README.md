# FarmerSync ML Pipeline

Production-ready price-forecasting API for Rwandan agricultural markets.
Serves an XGBoost regressor (R²=0.95) trained on weather, soil, and market data.

---

## Architecture

```
Render Web Service  ──►  FastAPI (serving/)     ──►  PostgreSQL
                                │                         │
Render Cron Job     ──►  train.py (training/)  ◄──  training_data table
                                │
                         Cloudflare R2  ──►  models/*.joblib
```

---

## Local Development

### 1. Prerequisites
```bash
python3.11+ and pip
PostgreSQL running locally (or use Docker)
```

### 2. Set up environment
```bash
cd ml-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your values (at minimum: DATABASE_URL, API_SECRET_TOKEN)
```

### 3. Run database migrations
```bash
export DATABASE_URL=postgresql://user:pass@localhost:5432/farmersync
alembic upgrade head
```

### 4. Copy the local model fallback
```bash
cp ../models/xgboost_v1.joblib models/
```

### 5. Start the API server
```bash
uvicorn serving.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

---

## API Endpoints

All POST endpoints require:
```
Authorization: Bearer <API_SECRET_TOKEN>
```

### Health check
```bash
curl http://localhost:8000/health
```

### Single prediction (feature array)
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Authorization: Bearer $API_SECRET_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [2024, 1, 24.5, 3.2, 65.0, 5000.0, 140.0, 0.78, 230.0]
  }'
```

**Feature order:** `year, zone_id, temperature, rainfall_mm, humidity, harvest_quantity, transport_cost, market_demand_index, prev_day_price`

### Single prediction (named fields)
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Authorization: Bearer $API_SECRET_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "year": 2024,
      "zone_id": 1,
      "temperature": 24.5,
      "rainfall_mm": 3.2,
      "humidity": 65.0,
      "harvest_quantity": 5000.0,
      "transport_cost": 140.0,
      "market_demand_index": 0.78,
      "prev_day_price": 230.0
    }
  }'
# → {"predicted_price": 248.32, "model_version": "xgboost_v20240512_1400", ...}
```

### Batch prediction
```bash
curl -X POST http://localhost:8000/api/predict/batch \
  -H "Authorization: Bearer $API_SECRET_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      [2024, 1, 24.5, 3.2, 65.0, 5000.0, 140.0, 0.78, 230.0],
      [2024, 2, 19.8, 1.1, 58.0, 6200.0, 130.0, 0.65, 195.0]
    ]
  }'
# → {"predictions": [248.32, 201.17], "count": 2, ...}
```

### Feed training data (JSON)
```bash
curl -X POST http://localhost:8000/api/data/feed \
  -H "Authorization: Bearer $API_SECRET_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {"features": [2024, 1, 24.5, 3.2, 65.0, 5000.0, 140.0, 0.78, 230.0], "label": 245.0}
    ]
  }'
```

### Feed training data (CSV upload)
CSV must have columns: `year,zone_id,temperature,rainfall_mm,humidity,harvest_quantity,transport_cost,market_demand_index,prev_day_price,price`

```bash
curl -X POST http://localhost:8000/api/data/feed/csv \
  -H "Authorization: Bearer $API_SECRET_TOKEN" \
  -F "file=@training_batch.csv"
```

### Trigger retraining (manual)
```bash
curl -X POST http://localhost:8000/api/retrain \
  -H "Authorization: Bearer $API_SECRET_TOKEN"
# → {"status": "retraining_started", "job_id": "a3f7b2c1", ...}
```

### Reload model from R2
```bash
curl -X POST http://localhost:8000/api/model/reload \
  -H "Authorization: Bearer $API_SECRET_TOKEN"
# → {"status": "reloaded", "model_version": "xgboost_v20240512_1400"}
```

### Model info
```bash
curl http://localhost:8000/api/model/info
# → {"version": "...", "r2_score": 0.9497, "features": [...], ...}
```

---

## Run Tests
```bash
cd ml-pipeline
export API_SECRET_TOKEN=test-token-local
pytest tests/ -v
```

---

## Render Deployment

### Step 1 — Create a Cloudflare R2 bucket
1. Go to Cloudflare dashboard → R2 → Create bucket → name: `farmersync-models`
2. Create an API token with R2 Object Read & Write permissions
3. Note your **Account ID**, **Access Key ID**, and **Secret Access Key**

### Step 2 — Upload the seed model to R2
```bash
pip install boto3
python3 - <<'EOF'
import boto3
s3 = boto3.client(
    "s3",
    endpoint_url="https://<account_id>.r2.cloudflarestorage.com",
    aws_access_key_id="<key>",
    aws_secret_access_key="<secret>",
)
s3.upload_file("models/xgboost_v1.joblib", "farmersync-models", "models/xgboost_v1.joblib")
print("Uploaded!")
EOF
```

### Step 3 — Deploy on Render using render.yaml (Blueprint)
1. Push this `ml-pipeline/` folder to a GitHub repository
2. Render dashboard → **New** → **Blueprint**
3. Connect your repo → Render reads `render.yaml` automatically
4. Set the **secret** env vars in the dashboard (marked `sync: false`):
   - `API_SECRET_TOKEN` — a long random string (your API key)
   - `S3_BUCKET` — `farmersync-models`
   - `S3_ENDPOINT_URL` — `https://<account_id>.r2.cloudflarestorage.com`
   - `AWS_ACCESS_KEY_ID` — R2 key
   - `AWS_SECRET_ACCESS_KEY` — R2 secret
   - `ML_API_URL` — `https://farmersync-ml-api.onrender.com` (your Render URL)
5. Click **Apply** — Render creates: Web Service + Cron Job + PostgreSQL

### Step 4 — Verify deployment
```bash
curl https://farmersync-ml-api.onrender.com/health
```

### Manual retraining (or change cron schedule)
The cron job runs `python -m training.train` daily at 02:00 UTC.
To change the schedule, edit `render.yaml` → `schedule` field (standard cron syntax).

---

## Zone ID mapping
| zone_id | Climate Zone |
|---------|--------------|
| 0       | Kigali       |
| 1       | Southern     |
| 2       | Northern     |
| 3       | Western      |
| 4       | Eastern      |

---

## Environment Variables Reference

| Variable                 | Required | Description                                        |
|--------------------------|----------|----------------------------------------------------|
| `API_SECRET_TOKEN`       | Yes      | Bearer token for all POST endpoints                |
| `DATABASE_URL`           | Yes      | PostgreSQL connection string                       |
| `S3_BUCKET`              | Yes      | R2/S3 bucket name                                  |
| `S3_ENDPOINT_URL`        | Yes      | R2 endpoint or omit for AWS S3                     |
| `AWS_ACCESS_KEY_ID`      | Yes      | R2/S3 access key                                   |
| `AWS_SECRET_ACCESS_KEY`  | Yes      | R2/S3 secret key                                   |
| `MODEL_VERSION`          | No       | Fallback label when R2 unavailable (default: `v1`) |
| `MIN_SAMPLES_TO_RETRAIN` | No       | Minimum new rows before retraining (default: 100)  |
| `MIN_ACCURACY_IMPROVEMENT`| No      | Minimum R² gain to deploy new model (default: 0.01)|
| `ML_API_URL`             | No       | Self-URL for hot-reload after retraining           |
| `CORS_ORIGINS`           | No       | Comma-separated allowed origins (default: `*`)     |
