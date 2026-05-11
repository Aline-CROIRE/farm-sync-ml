"""Evaluate and compare XGBoost model candidates."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    r2: float
    mae: float
    rmse: float
    n_samples: int

    def __str__(self) -> str:
        return (
            f"R²={self.r2:.4f}  MAE={self.mae:.2f}  "
            f"RMSE={self.rmse:.2f}  n={self.n_samples}"
        )


def evaluate(model, X: np.ndarray, y: np.ndarray) -> EvalResult:
    """Compute regression metrics for a fitted model on a held-out split."""
    y_pred = model.predict(X)
    r2 = float(r2_score(y, y_pred))
    mae = float(mean_absolute_error(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    return EvalResult(r2=r2, mae=mae, rmse=rmse, n_samples=len(y))


def is_improvement(
    new: EvalResult,
    current: EvalResult,
    min_delta: float = 0.01,
) -> bool:
    """
    Return True when the new model's R² exceeds the current one by at least
    `min_delta` (default 1 percentage point).
    """
    improved = new.r2 >= current.r2 + min_delta
    logger.info(
        "Model comparison — new: %s | current: %s | delta=%.4f | improved=%s",
        new,
        current,
        new.r2 - current.r2,
        improved,
    )
    return improved


def cross_val_r2(model_cls, X: np.ndarray, y: np.ndarray, n_splits: int = 5, **kwargs) -> float:
    """
    Quick time-series-aware CV using the last fold as the test window.
    Returns mean R² across folds.
    """
    from sklearn.model_selection import TimeSeriesSplit  # noqa: PLC0415

    scores: list[float] = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(X):
        m = model_cls(**kwargs)
        m.fit(X[train_idx], y[train_idx])
        scores.append(float(r2_score(y[val_idx], m.predict(X[val_idx]))))
    mean_r2 = float(np.mean(scores))
    logger.info("Cross-val R² scores: %s  → mean=%.4f", scores, mean_r2)
    return mean_r2
