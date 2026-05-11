"""SQLAlchemy async ORM models for the FarmerSync ML pipeline."""
from __future__ import annotations

import os
import ssl
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, LargeBinary, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _prepare_db_url(url: str) -> tuple[str, dict]:
    """
    Convert a Render PostgreSQL URL for asyncpg:
      - swap postgresql:// → postgresql+asyncpg://
      - strip ?sslmode=require (asyncpg rejects it)
      - return ssl context in connect_args instead
    """
    needs_ssl = "sslmode=require" in url
    for token in ("?sslmode=require", "&sslmode=require", "sslmode=require&"):
        url = url.replace(token, "")
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    connect_args = {"ssl": ssl.create_default_context()} if needs_ssl else {}
    return url, connect_args


_raw_url = os.environ.get(
    "DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/farmersync"
)
DATABASE_URL, _connect_args = _prepare_db_url(_raw_url)

engine = create_async_engine(
    DATABASE_URL, echo=False, pool_pre_ping=True, connect_args=_connect_args
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


class TrainingData(Base):
    """Stores incoming labelled samples for future retraining runs."""
    __tablename__ = "training_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    label: Mapped[float] = mapped_column(Float, nullable=False, comment="Actual price RWF/kg")
    source: Mapped[str] = mapped_column(
        String(10), nullable=False, default="api",
        comment="api | csv | db"
    )
    used_for_training: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PredictionLog(Base):
    """Audit trail for every prediction served."""
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    prediction: Mapped[float] = mapped_column(Float, nullable=False)
    actual_outcome: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ModelVersion(Base):
    """Registry of every trained model. model_data holds the raw joblib bytes."""
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    r2_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    training_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Model binary stored directly — avoids needing external object storage
    model_data: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)


async def get_db() -> AsyncSession:  # type: ignore[override]
    """FastAPI dependency that yields an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables() -> None:
    """Create all tables (used in build.sh / startup when Alembic is not available)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
