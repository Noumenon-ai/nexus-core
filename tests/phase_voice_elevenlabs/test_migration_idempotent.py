from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from db import create_database
from migrations.voice_credentials_migration import run_voice_credentials_migration
from models import Base, User
from tests.helpers import configure_test_env


def _build_database(monkeypatch, tmp_path, *, extra_env: dict[str, str] | None = None):
    settings = configure_test_env(monkeypatch, tmp_path, extra=extra_env)
    database = create_database(settings)
    Base.metadata.create_all(database.engine)
    return settings, database


def _insert_users(session: Session, count: int) -> list[User]:
    base_time = datetime(2026, 4, 27, 19, 0, tzinfo=timezone.utc)
    users = []
    for index in range(count):
        user = User(
            telegram_id=8000 + index,
            created_at=base_time + timedelta(minutes=index),
            updated_at=base_time + timedelta(minutes=index),
        )
        session.add(user)
        users.append(user)
    session.commit()
    for user in users:
        session.refresh(user)
    return users


def _snapshot_users(session: Session):
    rows = session.scalars(select(User).order_by(User.created_at.asc(), User.id.asc())).all()
    return [
        (row.telegram_id, row.elevenlabs_api_key, row.elevenlabs_voice_id, row.elevenlabs_migrated_at)
        for row in rows
    ]


def test_migration_first_run_populates_user_columns_in_created_at_order(monkeypatch, tmp_path):
    settings, database = _build_database(monkeypatch, tmp_path, extra_env={
        'ELEVENLABS_API_KEY_USER_1': 'key_1',
        'ELEVENLABS_VOICE_ID_USER_1': 'voice_1',
        'ELEVENLABS_API_KEY_USER_2': 'key_2',
        'ELEVENLABS_VOICE_ID_USER_2': 'voice_2',
    })
    with Session(database.engine) as session:
        _insert_users(session, 2)
    run_voice_credentials_migration(database.engine, settings)
    with Session(database.engine) as session:
        users = session.scalars(select(User).order_by(User.created_at.asc(), User.id.asc())).all()
        assert users[0].elevenlabs_api_key == 'key_1'
        assert users[0].elevenlabs_voice_id == 'voice_1'
        assert users[1].elevenlabs_api_key == 'key_2'
        assert users[1].elevenlabs_voice_id == 'voice_2'


def test_migration_second_run_no_overwrite(monkeypatch, tmp_path):
    settings, database = _build_database(monkeypatch, tmp_path, extra_env={
        'ELEVENLABS_API_KEY_USER_1': 'key_1',
        'ELEVENLABS_VOICE_ID_USER_1': 'voice_1',
        'ELEVENLABS_API_KEY_USER_2': 'key_2',
        'ELEVENLABS_VOICE_ID_USER_2': 'voice_2',
    })
    with Session(database.engine) as session:
        _insert_users(session, 2)
    run_voice_credentials_migration(database.engine, settings)
    with Session(database.engine) as session:
        before = _snapshot_users(session)
    run_voice_credentials_migration(database.engine, settings)
    with Session(database.engine) as session:
        after = _snapshot_users(session)
    assert before == after


def test_migration_skips_user_3_plus(monkeypatch, tmp_path):
    settings, database = _build_database(monkeypatch, tmp_path, extra_env={
        'ELEVENLABS_API_KEY_USER_1': 'key_1',
        'ELEVENLABS_VOICE_ID_USER_1': 'voice_1',
        'ELEVENLABS_API_KEY_USER_2': 'key_2',
        'ELEVENLABS_VOICE_ID_USER_2': 'voice_2',
        'ELEVENLABS_API_KEY_USER_3': 'key_3',
        'ELEVENLABS_VOICE_ID_USER_3': 'voice_3',
    })
    with Session(database.engine) as session:
        _insert_users(session, 3)
    run_voice_credentials_migration(database.engine, settings)
    with Session(database.engine) as session:
        users = session.scalars(select(User).order_by(User.created_at.asc(), User.id.asc())).all()
        assert users[2].elevenlabs_api_key is None
        assert users[2].elevenlabs_voice_id is None


def test_migration_safe_with_only_one_user(monkeypatch, tmp_path):
    settings, database = _build_database(monkeypatch, tmp_path, extra_env={
        'ELEVENLABS_API_KEY_USER_1': 'key_1',
        'ELEVENLABS_VOICE_ID_USER_1': 'voice_1',
        'ELEVENLABS_API_KEY_USER_2': 'key_2',
        'ELEVENLABS_VOICE_ID_USER_2': 'voice_2',
    })
    with Session(database.engine) as session:
        _insert_users(session, 1)
    run_voice_credentials_migration(database.engine, settings)
    with Session(database.engine) as session:
        user = session.scalar(select(User))
        assert user is not None
        assert user.elevenlabs_api_key == 'key_1'
        assert user.elevenlabs_voice_id == 'voice_1'


def test_migration_does_not_overwrite_already_set_values(monkeypatch, tmp_path):
    settings, database = _build_database(monkeypatch, tmp_path, extra_env={
        'ELEVENLABS_API_KEY_USER_1': 'key_1',
        'ELEVENLABS_VOICE_ID_USER_1': 'voice_1',
    })
    with Session(database.engine) as session:
        users = _insert_users(session, 1)
        current = session.get(User, users[0].id)
        assert current is not None
        current.elevenlabs_api_key = 'manual_key'
        current.elevenlabs_voice_id = 'manual_voice'
        session.commit()
    run_voice_credentials_migration(database.engine, settings)
    with Session(database.engine) as session:
        user = session.scalar(select(User))
        assert user is not None
        assert user.elevenlabs_api_key == 'manual_key'
        assert user.elevenlabs_voice_id == 'manual_voice'


def test_migration_fills_missing_voice_id_without_overwriting_manual_api_key(monkeypatch, tmp_path):
    settings, database = _build_database(monkeypatch, tmp_path, extra_env={
        'ELEVENLABS_API_KEY_USER_1': 'key_1',
        'ELEVENLABS_VOICE_ID_USER_1': 'voice_1',
    })
    with Session(database.engine) as session:
        users = _insert_users(session, 1)
        current = session.get(User, users[0].id)
        assert current is not None
        current.elevenlabs_api_key = 'manual_key'
        session.commit()
    run_voice_credentials_migration(database.engine, settings)
    with Session(database.engine) as session:
        user = session.scalar(select(User))
        assert user is not None
        assert user.elevenlabs_api_key == 'manual_key'
        assert user.elevenlabs_voice_id == 'voice_1'
