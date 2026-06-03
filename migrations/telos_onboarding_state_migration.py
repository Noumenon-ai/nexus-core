"""V3.8 TELOS onboarding state migration.

Creates the `telos_onboarding_state` table that the V3.8 onboarding
flow tools (start, answer, view, cancel) and the briefing nudge
bookkeeping write to. One row per user; `user_id` is the primary key
and a foreign key to `users.id`.

Idempotent via `Table.create(checkfirst=True)` — same shape as the
V3.6 conversation_turns migration so re-running on every boot is a
no-op.
"""
from __future__ import annotations

import logging

from sqlalchemy import Engine

from models import TelosOnboardingState


logger = logging.getLogger(__name__)


def run_telos_onboarding_state_migration(engine: Engine) -> None:
    """Create the telos_onboarding_state table if absent."""
    TelosOnboardingState.__table__.create(bind=engine, checkfirst=True)
