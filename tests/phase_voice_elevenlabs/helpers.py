from __future__ import annotations

from datetime import datetime, timezone

from models import User
from tests.helpers import build_test_container


DEFAULT_ELEVENLABS_ENV = {
    'VOICE_OUTPUT_BACKEND': 'elevenlabs',
    'ELEVENLABS_API_KEY_USER_1': 'sk_user_1',
    'ELEVENLABS_VOICE_ID_USER_1': 'voice_user_1',
    'ELEVENLABS_DAILY_CHAR_QUOTA_USER_1': '5000',
    'ELEVENLABS_MODEL': 'eleven_turbo_v2_5',
}


def build_elevenlabs_container(monkeypatch, tmp_path, *, extra_env: dict[str, str] | None = None):
    env = dict(DEFAULT_ELEVENLABS_ENV)
    if extra_env:
        env.update(extra_env)
    return build_test_container(monkeypatch, tmp_path, extra_env=env, validate_startup=False)


def create_user(container, telegram_id: int = 111, *, language: str = 'en') -> User:
    user = container.users_repository.get_or_create(telegram_id)
    with container.database.session_factory() as session:
        current = session.get(User, user.id)
        assert current is not None
        current.language = language
        session.commit()
        session.refresh(current)
        return current


def mark_user_migrated(container, user_id: str) -> None:
    container.users_repository.set_elevenlabs_profile(user_id, migrated_at=datetime.now(timezone.utc))
