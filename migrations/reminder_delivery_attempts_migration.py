"""2026-05-31 bounded reminder redelivery schema migration.

Adds `delivery_attempts` to the reminders table so failed external-contact
deliveries can be retried a bounded number of times and then abandoned.
Idempotent and safe to run repeatedly.
"""
from __future__ import annotations

import logging

from sqlalchemy import Engine, inspect, text


logger = logging.getLogger(__name__)


def run_reminder_delivery_attempts_migration(engine: Engine) -> None:
    existing = {c['name'] for c in inspect(engine).get_columns('reminders')}
    if 'delivery_attempts' in existing:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                'ALTER TABLE reminders '
                'ADD COLUMN delivery_attempts INTEGER NOT NULL DEFAULT 0'
            )
        )
    logger.info('reminder_delivery_attempts_migration_added_column')
