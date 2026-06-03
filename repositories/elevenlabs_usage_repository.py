from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from models import ElevenLabsUsage
from repositories.base import BaseRepository


class ElevenLabsUsageRepository(BaseRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory)

    def get_or_create(self, *, user_id: str | None, day: date) -> ElevenLabsUsage:
        def operation(session: Session) -> ElevenLabsUsage:
            self._ensure_row(session, user_id=user_id, day=day)
            usage = session.scalar(self._select_stmt(user_id=user_id, day=day))
            if usage is None:
                raise RuntimeError('Failed to load ElevenLabs usage row')
            return usage
        return self._execute(operation)

    def increment(self, *, user_id: str | None, day: date, chars: int = 0, api_calls: int = 0) -> ElevenLabsUsage:
        def operation(session: Session) -> ElevenLabsUsage:
            self._ensure_row(session, user_id=user_id, day=day)
            session.execute(
                self._update_sql(user_id=user_id, quota_guard=False),
                {'user_id': user_id, 'day': day, 'chars': chars, 'api_calls': api_calls},
            )
            usage = session.scalar(self._select_stmt(user_id=user_id, day=day))
            if usage is None:
                raise RuntimeError('Failed to update ElevenLabs usage row')
            return usage
        return self._execute(operation)

    def reserve_chars_with_quota(
        self,
        *,
        user_id: str | None,
        day: date,
        chars: int,
        quota: int,
    ) -> ElevenLabsUsage | None:
        def operation(session: Session) -> ElevenLabsUsage | None:
            self._ensure_row(session, user_id=user_id, day=day)
            result = session.execute(
                self._update_sql(user_id=user_id, quota_guard=True),
                {'user_id': user_id, 'day': day, 'chars': chars, 'api_calls': 0, 'quota': quota},
            )
            if result.rowcount != 1:
                return None
            usage = session.scalar(self._select_stmt(user_id=user_id, day=day))
            if usage is None:
                raise RuntimeError('Failed to reserve ElevenLabs usage row')
            return usage
        return self._execute(operation)

    def release_chars(self, *, user_id: str | None, day: date, chars: int) -> ElevenLabsUsage:
        def operation(session: Session) -> ElevenLabsUsage:
            self._ensure_row(session, user_id=user_id, day=day)
            session.execute(
                self._release_sql(user_id=user_id),
                {'user_id': user_id, 'day': day, 'chars': chars},
            )
            usage = session.scalar(self._select_stmt(user_id=user_id, day=day))
            if usage is None:
                raise RuntimeError('Failed to release ElevenLabs usage row')
            return usage
        return self._execute(operation)

    def increment_api_calls(self, *, user_id: str | None, day: date, api_calls: int) -> ElevenLabsUsage:
        return self.increment(user_id=user_id, day=day, api_calls=api_calls)

    def _ensure_row(self, session: Session, *, user_id: str | None, day: date) -> None:
        if user_id is None:
            usage = session.scalar(self._select_stmt(user_id=None, day=day))
            if usage is None:
                session.add(ElevenLabsUsage(user_id=None, day=day))
                session.flush()
            return
        session.execute(
            text(
                'INSERT INTO elevenlabs_usage (id, user_id, day, chars_count, api_calls) '
                'VALUES (:id, :user_id, :day, 0, 0) '
                'ON CONFLICT(user_id, day) DO NOTHING'
            ),
            {'id': str(uuid.uuid4()), 'user_id': user_id, 'day': day},
        )

    def _select_stmt(self, *, user_id: str | None, day: date):
        if user_id is None:
            return select(ElevenLabsUsage).where(ElevenLabsUsage.user_id.is_(None), ElevenLabsUsage.day == day)
        return select(ElevenLabsUsage).where(ElevenLabsUsage.user_id == user_id, ElevenLabsUsage.day == day)

    def _update_sql(self, *, user_id: str | None, quota_guard: bool):
        if user_id is None:
            where = 'user_id IS NULL AND day = :day'
        else:
            where = 'user_id = :user_id AND day = :day'
        quota_clause = ' AND chars_count + :chars <= :quota' if quota_guard else ''
        return text(
            'UPDATE elevenlabs_usage '
            'SET chars_count = chars_count + :chars, api_calls = api_calls + :api_calls '
            f'WHERE {where}{quota_clause}'
        )

    def _release_sql(self, *, user_id: str | None):
        if user_id is None:
            where = 'user_id IS NULL AND day = :day'
        else:
            where = 'user_id = :user_id AND day = :day'
        return text(
            'UPDATE elevenlabs_usage '
            'SET chars_count = CASE '
            'WHEN chars_count >= :chars THEN chars_count - :chars '
            'ELSE 0 '
            'END '
            f'WHERE {where}'
        )
