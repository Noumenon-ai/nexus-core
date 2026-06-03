from __future__ import annotations

import logging
import os

from sqlalchemy import Engine, inspect, text

from config import Settings


logger = logging.getLogger(__name__)


_USER_COLUMN_DEFINITIONS = {
    'google_connected': 'BOOLEAN NOT NULL DEFAULT 0',
    'google_email': 'TEXT',
    'google_calendar_primary_id': 'TEXT',
    'google_scopes_granted': 'TEXT',
    'google_connected_at': 'TIMESTAMP',
}


def run_google_oauth_migration(engine: Engine, settings: Settings) -> None:
    _ensure_user_columns(engine)
    _ensure_token_directory(settings)


def _ensure_user_columns(engine: Engine) -> None:
    existing_columns = {column['name'] for column in inspect(engine).get_columns('users')}
    missing = [name for name in _USER_COLUMN_DEFINITIONS if name not in existing_columns]
    if not missing:
        return
    with engine.begin() as connection:
        for column_name in missing:
            definition = _USER_COLUMN_DEFINITIONS[column_name]
            connection.execute(text(f'ALTER TABLE users ADD COLUMN {column_name} {definition}'))


def _ensure_token_directory(settings: Settings) -> None:
    if not settings.google.enabled:
        return
    token_dir = settings.google.token_dir
    token_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(token_dir, 0o700)
    except OSError as exc:
        logger.warning('google_token_dir_chmod_failed', extra={'path': str(token_dir), 'error': str(exc)})
