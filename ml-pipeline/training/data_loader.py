"""Fetch training data from PostgreSQL for the retraining pipeline."""
from __future__ import annotations

import logging
import os
from typing import Generator

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

FEATURE_NAMES: list[str] = [
    "year", "zone_id", "temperature", "rainfall_mm",
    "humidity", "harvest_quantity", "transport_cost",
    "market_demand_index", "prev_day_price",
]


def _sync_engine():
    """Synchronous engine for use in training scripts (psycopg2, not async)."""
    url = os.environ["DATABASE_URL"]
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    # psycopg2 accepts sslmode natively — no stripping needed
    return create_engine(url, pool_pre_ping=True)


def load_training_data(
    min_samples: int = 100,
    only_unused: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load labelled rows from the training_data table.

    Parameters
    ----------
    min_samples:
        Minimum number of rows required; raises ValueError if fewer found.
    only_unused:
        When True, only fetch rows where used_for_training = FALSE.

    Returns
    -------
    X : np.ndarray, shape (n, 9)
    y : np.ndarray, shape (n,)
    """
    engine = _sync_engine()
    clause = "WHERE used_for_training = FALSE" if only_unused else ""
    query = f"SELECT features, label FROM training_data {clause} ORDER BY created_at"
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(query)).fetchall()
    except Exception as exc:
        logger.error("DB query failed: %s", exc)
        raise

    if len(rows) < min_samples:
        raise ValueError(
            f"Only {len(rows)} usable rows found; need at least {min_samples}."
        )

    X_list, y_list = [], []
    for features_json, label in rows:
        values = features_json.get("values") if isinstance(features_json, dict) else features_json
        X_list.append(values)
        y_list.append(float(label))

    return np.array(X_list, dtype=float), np.array(y_list, dtype=float)


def mark_as_used(engine) -> None:
    """Mark all previously-unused training rows as used after a successful training run."""
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE training_data SET used_for_training = TRUE WHERE used_for_training = FALSE")
        )
    logger.info("Marked training rows as used.")


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a local CSV file with feature columns + 'price' column.
    Useful for bootstrapping before any API data exists.
    """
    df = pd.read_csv(path)
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    if "price" not in df.columns:
        raise ValueError("CSV must have a 'price' column for the target.")
    X = df[FEATURE_NAMES].values.astype(float)
    y = df["price"].values.astype(float)
    logger.info("Loaded CSV: %d rows from %s", len(X), path)
    return X, y


def load_from_postgres_direct(limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data directly from the farmersync dataset table (raw PostgreSQL).
    Used when training_data table is empty and we want to seed from raw data.
    """
    engine = _sync_engine()
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
        SELECT year, zone_id, temperature, rainfall_mm, humidity,
               harvest_quantity, transport_cost, market_demand_index,
               prev_day_price, price
        FROM farmersync_data
        ORDER BY date
        {limit_clause}
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
    except Exception as exc:
        logger.error("Direct DB load failed: %s", exc)
        raise
    X = df[FEATURE_NAMES].values.astype(float)
    y = df["price"].values.astype(float)
    logger.info("Loaded %d rows from farmersync_data table.", len(X))
    return X, y
