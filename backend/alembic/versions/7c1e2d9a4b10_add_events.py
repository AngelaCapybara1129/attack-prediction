"""add events table

Revision ID: 7c1e2d9a4b10
Revises: 3650e0957d16
Create Date: 2026-11-14 12:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "7c1e2d9a4b10"
down_revision = "3650e0957d16"
branch_labels = None
depends_on = None


def upgrade():
    # 後端啟動時 Base.metadata.create_all 可能已建立此表
    if sa.inspect(op.get_bind()).has_table("events"):
        return
    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("alert_id", sa.String(length=32), nullable=False),
        sa.Column("task_id", sa.String(length=32), nullable=True),
        sa.Column("camera_id", sa.String(length=64), nullable=False),
        sa.Column("person_id", sa.Integer(), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=True),
        sa.Column("level", sa.String(length=4), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("rule_score", sa.Float(), nullable=True),
        sa.Column("model_score", sa.Float(), nullable=True),
        sa.Column("indices", sa.JSON(), nullable=False),
        sa.Column("indices_peak", sa.JSON(), nullable=True),
        sa.Column("predicted_next", sa.JSON(), nullable=True),
        sa.Column("triggers", sa.JSON(), nullable=True),
        sa.Column("aux", sa.JSON(), nullable=True),
        sa.Column("action", sa.String(length=32), nullable=True),
        sa.Column("action_conf", sa.Float(), nullable=True),
        sa.Column("bbox", sa.JSON(), nullable=True),
        sa.Column("trajectory", sa.JSON(), nullable=True),
        sa.Column("camera_path", sa.JSON(), nullable=True),
        sa.Column("score_history", sa.JSON(), nullable=True),
        sa.Column("appearance", sa.JSON(), nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("recommendations", sa.JSON(), nullable=True),
        sa.Column("similar", sa.JSON(), nullable=True),
        sa.Column("snapshot_path", sa.String(length=512), nullable=True),
        sa.Column("clip_dir", sa.String(length=512), nullable=True),
        sa.Column("clip_frames", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("handled_by", sa.String(length=255), nullable=True),
        sa.Column("handled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_index("ix_events_id", "events", ["id"], unique=False)
    op.create_index("ix_events_alert_id", "events", ["alert_id"], unique=True)
    op.create_index("ix_events_task_id", "events", ["task_id"], unique=False)
    op.create_index("ix_events_camera_id", "events", ["camera_id"], unique=False)
    op.create_index("ix_events_person_id", "events", ["person_id"], unique=False)
    op.create_index("ix_events_level", "events", ["level"], unique=False)
    op.create_index("ix_events_status", "events", ["status"], unique=False)
    op.create_index("ix_events_occurred_at", "events", ["occurred_at"], unique=False)


def downgrade():
    for name in ("ix_events_occurred_at", "ix_events_status", "ix_events_level", "ix_events_person_id",
                 "ix_events_camera_id", "ix_events_task_id", "ix_events_alert_id", "ix_events_id"):
        op.drop_index(name, table_name="events")
    op.drop_table("events")
