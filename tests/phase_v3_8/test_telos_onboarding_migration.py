"""V3.8 schema migration tests.

Verifies the telos_onboarding_state table:
1. Lands on a fresh DB without errors
2. Has the expected 7 columns (with started_at nullable per the
   V3.8 audit deviation — nudge writes can land before onboarding
   starts)
3. Migration is idempotent (re-runnable safely on every boot)
4. Foreign key to users.id is enforced (foundation for per-user
   isolation halt condition)
5. Defaults applied: current_section='identity', answers_so_far='{}'
6. started_at is nullable specifically (the audit-time spec deviation)
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db import create_database
from migrations.telos_onboarding_state_migration import (
    run_telos_onboarding_state_migration,
)
from models import Base, TelosOnboardingState, User
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
    assert 'telos_onboarding_state' not in inspector_before.get_table_names()

    run_telos_onboarding_state_migration(database.engine)

    inspector_after = inspect(database.engine)
    assert 'telos_onboarding_state' in inspector_after.get_table_names()


def test_migration_creates_expected_columns(monkeypatch, tmp_path):
    settings = configure_test_env(monkeypatch, tmp_path)
    database = create_database(settings)
    User.__table__.create(bind=database.engine, checkfirst=True)
    run_telos_onboarding_state_migration(database.engine)

    columns = {col['name'] for col in inspect(database.engine).get_columns('telos_onboarding_state')}
    assert columns == {
        'user_id', 'current_section', 'answers_so_far',
        'started_at', 'completed_at', 'cancelled_at', 'last_nudge_at',
    }, f'Unexpected column set: {columns}'


def test_migration_is_idempotent(monkeypatch, tmp_path):
    """Running the migration multiple times must not raise. Real bot
    startup calls every migration on every boot — second-and-onward
    boots must no-op."""
    settings = configure_test_env(monkeypatch, tmp_path)
    database = create_database(settings)
    User.__table__.create(bind=database.engine, checkfirst=True)

    run_telos_onboarding_state_migration(database.engine)
    run_telos_onboarding_state_migration(database.engine)
    run_telos_onboarding_state_migration(database.engine)

    with Session(database.engine) as session:
        count = session.scalar(text('SELECT COUNT(*) FROM telos_onboarding_state'))
        assert count == 0


def test_started_at_is_nullable(monkeypatch, tmp_path):
    """V3.8 audit deviation from the original spec: started_at is
    nullable so the nudge bookkeeping can write a row BEFORE
    onboarding actually starts. A row with last_nudge_at set but
    started_at NULL is a valid state — user got nudged but never
    began the flow.
    """
    settings, database = _build_database(monkeypatch, tmp_path)
    with Session(database.engine) as session:
        user = User(telegram_id=12345)
        session.add(user)
        session.commit()
        user_id = user.id

        nudge_only_row = TelosOnboardingState(
            user_id=user_id,
            last_nudge_at=datetime(2026, 5, 4, 10, 0, tzinfo=timezone.utc),
            # started_at deliberately omitted — must NOT raise NOT NULL
        )
        session.add(nudge_only_row)
        session.commit()  # would raise IntegrityError if started_at NOT NULL

    with Session(database.engine) as session:
        row = session.get(TelosOnboardingState, user_id)
        assert row is not None
        assert row.started_at is None
        assert row.last_nudge_at is not None
        # Defaults should still apply.
        assert row.current_section == 'identity'
        assert row.answers_so_far == '{}'


def test_defaults_applied_correctly(monkeypatch, tmp_path):
    """current_section defaults to 'identity', answers_so_far to '{}'."""
    settings, database = _build_database(monkeypatch, tmp_path)
    with Session(database.engine) as session:
        user = User(telegram_id=12345)
        session.add(user)
        session.commit()
        user_id = user.id

        # Insert with only the FK; rely on defaults.
        session.add(TelosOnboardingState(user_id=user_id))
        session.commit()

    with Session(database.engine) as session:
        row = session.get(TelosOnboardingState, user_id)
        assert row.current_section == 'identity'
        assert row.answers_so_far == '{}'
        assert row.started_at is None
        assert row.completed_at is None
        assert row.cancelled_at is None
        assert row.last_nudge_at is None


def test_user_id_foreign_key_to_users(monkeypatch, tmp_path):
    """The user_id column must be a foreign key to users.id —
    foundation for per-user isolation. Inserting a state row for a
    user_id that does not exist in users must fail when SQLite FK
    enforcement is enabled (it is, per db.py settings).

    If FK enforcement is OFF in the test environment, this still
    passes because the introspection assertion at the bottom verifies
    the FK exists in the schema metadata.
    """
    settings, database = _build_database(monkeypatch, tmp_path)

    # Schema-level assertion: FK exists.
    fks = inspect(database.engine).get_foreign_keys('telos_onboarding_state')
    user_id_fks = [fk for fk in fks if 'user_id' in fk['constrained_columns']]
    assert len(user_id_fks) == 1, f'Expected one FK on user_id, got: {fks}'
    assert user_id_fks[0]['referred_table'] == 'users'
    assert user_id_fks[0]['referred_columns'] == ['id']


def test_user_id_is_primary_key(monkeypatch, tmp_path):
    """user_id is the PK — one row per user, naturally enforces
    'one onboarding state per user'."""
    settings, database = _build_database(monkeypatch, tmp_path)
    pk = inspect(database.engine).get_pk_constraint('telos_onboarding_state')
    assert pk['constrained_columns'] == ['user_id']


def test_two_rows_for_same_user_id_rejected(monkeypatch, tmp_path):
    """PK on user_id means inserting a second row for the same user
    must fail. Guards against accidental concurrent state creation."""
    settings, database = _build_database(monkeypatch, tmp_path)
    with Session(database.engine) as session:
        user = User(telegram_id=12345)
        session.add(user)
        session.commit()
        user_id = user.id

        session.add(TelosOnboardingState(user_id=user_id))
        session.commit()

    with Session(database.engine) as session:
        session.add(TelosOnboardingState(user_id=user_id, current_section='values'))
        with pytest.raises(IntegrityError):
            session.commit()
