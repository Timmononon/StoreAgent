"""Freeze RAG benchmark execution profiles on persistent Evaluation Jobs."""

import sqlalchemy as sa
from alembic import op


revision = "0019_rag_experiments"
down_revision = "0018_capability_control_plane"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_evaluation_jobs", sa.Column("experiment_json", sa.JSON(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("rag_evaluation_jobs", "experiment_json")
