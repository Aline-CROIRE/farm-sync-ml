"""Initial schema: training_data, prediction_logs, model_versions

Revision ID: 0001
Revises:
Create Date: 2025-05-11
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_data",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("features", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("label", sa.Float(), nullable=False, comment="Actual price RWF/kg"),
        sa.Column("source", sa.String(length=10), nullable=False, server_default="api"),
        sa.Column("used_for_training", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("features", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("prediction", sa.Float(), nullable=False),
        sa.Column("actual_outcome", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("r2_score", sa.Float(), nullable=True),
        sa.Column("training_samples", sa.Integer(), nullable=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version"),
    )

    # Index for quickly finding unused training rows
    op.create_index("ix_training_data_unused", "training_data", ["used_for_training"])


def downgrade() -> None:
    op.drop_index("ix_training_data_unused", table_name="training_data")
    op.drop_table("model_versions")
    op.drop_table("prediction_logs")
    op.drop_table("training_data")
