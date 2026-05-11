"""Add model_data BYTEA column to model_versions

Revision ID: 0002
Revises: 0001
Create Date: 2025-05-11
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_versions",
        sa.Column("model_data", sa.LargeBinary(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_versions", "model_data")
