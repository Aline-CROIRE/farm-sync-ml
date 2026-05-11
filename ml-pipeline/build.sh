#!/usr/bin/env bash
# Render build script — runs once before the web process starts.
set -euo pipefail

echo "=== FarmerSync ML build starting ==="

# 1. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 2. Run Alembic database migrations
echo "Running Alembic migrations..."
alembic upgrade head || echo "WARNING: Alembic migration failed — tables may not exist yet."

# 3. Pre-download the latest model from R2 (so startup is instant)
echo "Pre-downloading model from R2..."
python3 - <<'PYEOF'
import os, sys
sys.path.insert(0, ".")
try:
    from serving.load_model import download_latest_from_r2
    from pathlib import Path
    result = download_latest_from_r2()
    if result:
        data, version = result
        dest = Path("models") / f"{version}.joblib"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"Model cached: {dest}")
    else:
        print("No model in R2 — will use local fallback xgboost_v1.joblib")
except Exception as e:
    print(f"Model pre-download skipped: {e}")
PYEOF

echo "=== Build complete ==="
