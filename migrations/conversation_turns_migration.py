"""V3.6 forward-only conversation archive migration.

Creates the `conversation_turns` table that the V3.5+ dispatcher
will write to on every handle() invocation. Pre-V3 history is NOT
backfilled into this table — see HARDENING_PASS_V2.md H2-019/H2-020
for the spec re-scope and unrecoverable-history findings.

Idempotent: `Table.create(checkfirst=True)` is a no-op when the
table already exists. Index creation is also idempotent via SQLAlchemy's
internal handling.
"""
from __future__ import annotations

import logging

from sqlalchemy import Engine

from models import ConversationTurn


logger = logging.getLogger(__name__)


def run_conversation_turns_migration(engine: Engine) -> None:
    """Create the conversation_turns table + its indexes if absent."""
    ConversationTurn.__table__.create(bind=engine, checkfirst=True)
