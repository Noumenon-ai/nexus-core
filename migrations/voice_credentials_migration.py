from __future__ import annotations

import logging

from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.orm import Session

from config import Settings
from models import ElevenLabsUsage, User, now_utc


logger = logging.getLogger(__name__)


_USER_COLUMN_DEFINITIONS = {
    'elevenlabs_voice_id': 'TEXT',
    'elevenlabs_api_key': 'TEXT',
    'elevenlabs_voice_settings': 'TEXT',
    'elevenlabs_migrated_at': 'TIMESTAMP',
}


def run_voice_credentials_migration(engine: Engine, settings: Settings) -> None:
    _ensure_user_columns(engine)
    ElevenLabsUsage.__table__.create(bind=engine, checkfirst=True)
    _warn_on_extra_env_slots(settings)
    if not any(config.api_key or config.voice_id for config in settings.voice.elevenlabs_users):
        return
    with Session(engine) as session:
        users = list(session.scalars(select(User).order_by(User.created_at.asc(), User.id.asc())).all())
        for slot, user in enumerate(users[:2], start=1):
            env_config = settings.voice.elevenlabs_user(slot)
            if env_config is None or not (env_config.api_key or env_config.voice_id):
                continue
            if user.elevenlabs_voice_id is not None:
                if user.elevenlabs_migrated_at is None:
                    user.elevenlabs_migrated_at = now_utc()
                    user.updated_at = now_utc()
                continue
            if user.elevenlabs_api_key is None and env_config.api_key:
                user.elevenlabs_api_key = env_config.api_key
            if env_config.voice_id:
                user.elevenlabs_voice_id = env_config.voice_id
            user.elevenlabs_migrated_at = now_utc()
            user.updated_at = now_utc()
        session.commit()


def _ensure_user_columns(engine: Engine) -> None:
    existing_columns = {column['name'] for column in inspect(engine).get_columns('users')}
    missing_columns = [name for name in _USER_COLUMN_DEFINITIONS if name not in existing_columns]
    if not missing_columns:
        return
    with engine.begin() as connection:
        for column_name in missing_columns:
            definition = _USER_COLUMN_DEFINITIONS[column_name]
            connection.execute(text(f'ALTER TABLE users ADD COLUMN {column_name} {definition}'))


def _warn_on_extra_env_slots(settings: Settings) -> None:
    for config in settings.voice.elevenlabs_users:
        if config.slot > 2 and (config.api_key or config.voice_id):
            logger.warning('elevenlabs_extra_env_slot_ignored', extra={'slot': config.slot})
