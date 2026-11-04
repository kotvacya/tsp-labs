"""add unique constraint to metric_records

Revision ID: a0101c233920
Revises: 539796477a60
Create Date: 2026-11-04 15:05:30.834308

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a0101c233920'
down_revision: Union[str, Sequence[str], None] = '539796477a60'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_unique_constraint('uq_run_sample', 'metric_records', ['run_id', 'sample_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('uq_run_sample', 'metric_records', type_='unique')
