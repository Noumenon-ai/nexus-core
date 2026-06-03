from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from models import User, now_utc
from repositories.base import BaseRepository


_UNSET = object()


class UsersRepository(BaseRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory)

    def get_or_create(self, telegram_id: int, *, username: str | None = None, full_name: str | None = None) -> User:
        def operation(session: Session) -> User:
            user = session.scalar(select(User).where(User.telegram_id == telegram_id))
            if user is not None:
                if username is not None:
                    user.username = username
                if full_name is not None:
                    user.full_name = full_name
                user.updated_at = now_utc()
                return user
            user = User(telegram_id=telegram_id, username=username, full_name=full_name)
            session.add(user)
            session.flush()
            return user
        return self._execute(operation)

    def get_by_id(self, user_id: str) -> User | None:
        return self._execute(lambda session: session.get(User, user_id))

    def list_all(self) -> list[User]:
        return self._execute(lambda session: list(session.scalars(select(User)).all()))

    def list_by_created_at(self) -> list[User]:
        return self._execute(lambda session: list(session.scalars(select(User).order_by(User.created_at.asc(), User.id.asc())).all()))

    def get_voice_env_slot(self, user_id: str) -> int | None:
        def operation(session: Session) -> int | None:
            users = list(session.scalars(select(User.id).order_by(User.created_at.asc(), User.id.asc())).all())
            for index, current_user_id in enumerate(users[:2], start=1):
                if current_user_id == user_id:
                    return index
            return None
        return self._execute(operation)

    def set_proactive_enabled(self, user_id: str, enabled: bool) -> User:
        def operation(session: Session) -> User:
            user = session.get(User, user_id)
            if user is None:
                raise ValueError('User not found')
            user.proactive_enabled = enabled
            user.updated_at = now_utc()
            session.flush()
            return user
        return self._execute(operation)

    def set_voice_output_enabled(self, user_id: str, enabled: bool) -> User:
        def operation(session: Session) -> User:
            user = session.get(User, user_id)
            if user is None:
                raise ValueError('User not found')
            user.voice_output_enabled = enabled
            user.updated_at = now_utc()
            session.flush()
            return user
        return self._execute(operation)

    def set_elevenlabs_profile(
        self,
        user_id: str,
        *,
        voice_id: str | None | object = _UNSET,
        api_key: str | None | object = _UNSET,
        voice_settings: str | None | object = _UNSET,
        migrated_at: object = _UNSET,
    ) -> User:
        def operation(session: Session) -> User:
            user = session.get(User, user_id)
            if user is None:
                raise ValueError('User not found')
            if voice_id is not _UNSET:
                user.elevenlabs_voice_id = voice_id
            if api_key is not _UNSET:
                user.elevenlabs_api_key = api_key
            if voice_settings is not _UNSET:
                user.elevenlabs_voice_settings = voice_settings
            if migrated_at is not _UNSET:
                user.elevenlabs_migrated_at = migrated_at
            user.updated_at = now_utc()
            session.flush()
            return user
        return self._execute(operation)

    def set_google_profile(
        self,
        user_id: str,
        *,
        connected: bool | object = _UNSET,
        email: str | None | object = _UNSET,
        scopes_granted: str | None | object = _UNSET,
        connected_at: object = _UNSET,
        primary_calendar_id: str | None | object = _UNSET,
    ) -> User:
        def operation(session: Session) -> User:
            user = session.get(User, user_id)
            if user is None:
                raise ValueError('User not found')
            if connected is not _UNSET:
                user.google_connected = bool(connected)
            if email is not _UNSET:
                user.google_email = email
            if scopes_granted is not _UNSET:
                user.google_scopes_granted = scopes_granted
            if connected_at is not _UNSET:
                user.google_connected_at = connected_at
            if primary_calendar_id is not _UNSET:
                user.google_calendar_primary_id = primary_calendar_id
            user.updated_at = now_utc()
            session.flush()
            return user
        return self._execute(operation)
