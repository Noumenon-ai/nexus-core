from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from config import ConfigError, Settings
from migrations.google_oauth_migration import run_google_oauth_migration
from migrations.voice_credentials_migration import run_voice_credentials_migration
from models import Base


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Database:
    engine: Engine
    session_factory: sessionmaker[Session]


def create_database(settings: Settings) -> Database:
    database_url = settings.core.database_url
    connect_args: dict[str, object] = {}
    if database_url.startswith('sqlite'):
        connect_args['check_same_thread'] = False
    elif settings.security.threat_model == 'hosted':
        raise ConfigError('Hosted mode currently requires a SQLCipher SQLite database URL')
    engine = create_engine(
        database_url,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
    )

    if database_url.startswith('sqlite'):
        @event.listens_for(engine, 'connect')
        def _set_sqlite_pragma(dbapi_connection: sqlite3.Connection, _connection_record: object) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute('PRAGMA journal_mode=WAL')
            cursor.execute('PRAGMA busy_timeout=5000')
            cursor.execute('PRAGMA synchronous=NORMAL')
            cursor.close()

    session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return Database(engine=engine, session_factory=session_factory)


def create_all(engine: Engine, settings: Settings | None = None) -> None:
    Base.metadata.create_all(engine)
    if settings is not None:
        run_voice_credentials_migration(engine, settings)
        run_google_oauth_migration(engine, settings)


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def run_with_retry(operation, *, retries: int = 5, base_delay: float = 0.05):
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return operation()
        except OperationalError as exc:
            last_error = exc
            if 'database is locked' not in str(exc).lower():
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning('database_lock_retry', extra={'attempt': attempt + 1, 'delay': delay})
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError('run_with_retry exhausted without running the operation')
