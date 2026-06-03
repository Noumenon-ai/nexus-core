"""2026-05-27 audit_events table migration (Step 5).

Creates the audit_events table that captures every executed (or
attempted) action through the dispatcher / reasoning_adapter /
approval-confirm path. Delivery-truth layer: "if it's not in
audit_events, it didn't happen."

Idempotent: Table.create uses checkfirst=True. Safe to re-run.
"""
from __future__ import annotations

import logging

from sqlalchemy import Engine

from models import AuditEvent

logger = logging.getLogger(__name__)


def run_audit_events_migration(engine: Engine) -> None:
    AuditEvent.__table__.create(bind=engine, checkfirst=True)
    logger.info('audit_events_migration_complete')
