#!/usr/bin/env bash
# Render build script — runs once before the web process starts.
set -euo pipefail

echo "=== FarmerSync ML build starting ==="

# 1. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 2. Run Alembic database migrations
echo "Running Alembic migrations..."
alembic upgrade head

# 3. Seed the model into PostgreSQL on first deploy
#    If model_versions is empty, read the local fallback and insert it.
echo "Seeding model into DB if not already present..."
python3 - <<'PYEOF'
import asyncio, os, sys
sys.path.insert(0, ".")

async def seed():
    from pathlib import Path
    from serving.load_model import save_to_db

    fallback = Path("models/xgboost_v1.joblib")
    if not fallback.exists():
        print("No local fallback model found — skipping seed.")
        return

    import asyncpg
    url = os.environ.get("DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    if not url:
        print("DATABASE_URL not set — skipping seed.")
        return

    conn = await asyncpg.connect(url)
    try:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM model_versions WHERE is_active = TRUE AND model_data IS NOT NULL"
        )
        if count and count > 0:
            print(f"Active model already in DB ({count} row(s)) — skipping seed.")
            return
        saved = await save_to_db(fallback, "v1", r2_score=0.9497, training_samples=0)
        print("Seed model inserted." if saved else "Seed failed — check logs.")
    finally:
        await conn.close()

asyncio.run(seed())
PYEOF

echo "=== Build complete ==="
