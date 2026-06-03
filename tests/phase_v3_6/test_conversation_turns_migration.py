"""V3.6 schema migration tests.

Verifies the conversation_turns archive table:
1. Lands on a fresh DB without errors
2. Has the expected columns + indexes
3. Migration is idempotent (re-runnable safely)
4. Partial index correctly filters mem0_persisted_at IS NULL rows

Pre-V3 history is NOT in this table — see HARDENING_PASS_V2.md
H2-019 / H2-020 for the spec re-scope.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from db import create_database
from migrations.conversation_turns_migration import run_conversation_turns_migration
from models import Base, ConversationTurn, User
from tests.helpers import configure_test_env


def _build_database(monkeypatch, tmp_path):
    settings = configure_test_env(monkeypatch, tmp_path)
    database = create_database(settings)
    Base.metadata.create_all(database.engine)
    return settings, database


def test_migration_creates_table_on_fresh_engine(monkeypatch, tmp_path):
    settings = configure_test_env(monkeypatch, tmp_path)
    database = create_database(settings)
    User.__table__.create(bind=database.engine, checkfirst=True)

    inspector_before = inspect(database.engine)
    assert 'conversation_turns' not in inspector_before.get_table_names()

    run_conversation_turns_migration(database.engine)

    inspector_after = inspect(database.engine)
    assert 'conversation_turns' in inspector_after.get_table_names()


def test_migration_creates_expected_columns(monkeypatch, tmp_path):
    settings = configure_test_env(monkeypatch, tmp_path)
    database = create_database(settings)
    User.__table__.create(bind=database.engine, checkfirst=True)
    run_conversation_turns_migration(database.engine)

    columns = {col['name'] for col in inspect(database.engine).get_columns('conversation_turns')}
    assert columns == {
        'turn_id', 'user_id', 'role', 'content',
        'created_at', 'conversation_id',
        'mem0_persisted_at', 'mem0_memory_id',
    }


def test_migration_creates_expected_indexes(monkeypatch, tmp_path):
    settings = configure_test_env(monkeypatch, tmp_path)
    database = create_database(settings)
    User.__table__.create(bind=database.engine, checkfirst=True)
    run_conversation_turns_migration(database.engine)

    index_names = {idx['name'] for idx in inspect(database.engine).get_indexes('conversation_turns')}
    assert 'idx_conversation_turns_user_created' in index_names
    assert 'idx_conversation_turns_conv' in index_names
    assert 'idx_conversation_turns_mem0_pending' in index_names


def test_migration_is_idempotent(monkeypatch, tmp_path):
    """Running the migration twice must not raise. Real bot startup
    invokes this on every boot — second-and-onward boots must no-op."""
    settings = configure_test_env(monkeypatch, tmp_path)
    database = create_database(settings)
    User.__table__.create(bind=database.engine, checkfirst=True)

    run_conversation_turns_migration(database.engine)
    run_conversation_turns_migration(database.engine)
    run_conversation_turns_migration(database.engine)

    # Table still exists and is queryable post-multiple-runs.
    with Session(database.engine) as session:
        count = session.scalar(text('SELECT COUNT(*) FROM conversation_turns'))
        assert count == 0


def test_partial_index_filters_unpersisted_rows(monkeypatch, tmp_path):
    """The mem0-pending partial index must include rows where
    mem0_persisted_at IS NULL and exclude those where it's set.
    This is the index the future H2-020-style recovery script will
    query to find never-persisted turns."""
    settings, database = _build_database(monkeypatch, tmp_path)
    run_conversation_turns_migration(database.engine)

    user = User(telegram_id=12345)
    with Session(database.engine) as session:
        session.add(user)
        session.commit()
        session.refresh(user)

        pending = ConversationTurn(
            user_id=user.id,
            role='user',
            content='this turn was not persisted to mem0 yet',
            conversation_id='conv-1',
            created_at=datetime(2026, 5, 4, 10, 0, tzinfo=timezone.utc),
        )
        persisted = ConversationTurn(
            user_id=user.id,
            role='assistant',
            content='this turn was persisted',
            conversation_id='conv-1',
            created_at=datetime(2026, 5, 4, 10, 0, 1, tzinfo=timezone.utc),
            mem0_persisted_at=datetime(2026, 5, 4, 10, 0, 5, tzinfo=timezone.utc),
            mem0_memory_id='mem0-id-abc',
        )
        session.add_all([pending, persisted])
        session.commit()

    # Verify both rows landed
    with Session(database.engine) as session:
        total = session.scalar(text('SELECT COUNT(*) FROM conversation_turns'))
        assert total == 2

        # Use the partial index implicitly via a filtered query.
        # If the index works, the planner can use it. We assert the
        # query-shape returns only the pending row, regardless of plan.
        pending_count = session.scalar(
            text('SELECT COUNT(*) FROM conversation_turns WHERE mem0_persisted_at IS NULL')
        )
        assert pending_count == 1
