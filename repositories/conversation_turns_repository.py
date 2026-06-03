"""V3.6 forward-only conversation turn archive.

Writes every dispatcher user/assistant turn to `conversation_turns`.
Pre-V3.5 history is NOT in this table — see HARDENING_PASS_V2.md
H2-019 / H2-020 for the spec re-scope and unrecoverable-history note.

`resolve_conversation_id` implements the 2-hour silence rule: a user's
turns continue the same `conversation_id` until there has been at least
two hours of silence between user-turns from that user, after which the
next user-turn opens a fresh `conversation_id`. The grouping is
heuristic but stable enough for mem0 retrieval scoping.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterable

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from models import ConversationTurn, now_utc
from repositories.base import BaseRepository


_ALLOWED_ROLES = ('user', 'assistant')

CONVERSATION_SILENCE_GAP = timedelta(hours=2)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class ConversationTurnsRepository(BaseRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory)

    def insert(
        self,
        *,
        user_id: str,
        role: str,
        content: str,
        conversation_id: str,
        created_at: datetime | None = None,
    ) -> str:
        if role not in _ALLOWED_ROLES:
            raise ValueError(f'role must be one of {_ALLOWED_ROLES}, got {role!r}')
        timestamp = created_at or now_utc()

        def operation(session: Session) -> str:
            turn = ConversationTurn(
                turn_id=str(uuid.uuid4()),
                user_id=user_id,
                role=role,
                content=content,
                conversation_id=conversation_id,
                created_at=timestamp,
            )
            session.add(turn)
            session.flush()
            return turn.turn_id
        return self._execute(operation)

    def resolve_conversation_id(self, *, user_id: str, now: datetime | None = None) -> str:
        """Return the conversation_id this user's next user-turn belongs to.

        Same conversation_id as the user's most recent user-turn IF that
        turn happened within `CONVERSATION_SILENCE_GAP`. Otherwise a fresh
        UUID. Always scoped to the given user — never bridges users.
        """
        current = now or now_utc()
        threshold = current - CONVERSATION_SILENCE_GAP

        def operation(session: Session) -> str:
            stmt = (
                select(ConversationTurn)
                .where(
                    ConversationTurn.user_id == user_id,
                    ConversationTurn.role == 'user',
                )
                .order_by(ConversationTurn.created_at.desc())
                .limit(1)
            )
            last_user_turn = session.scalar(stmt)
            if last_user_turn is None:
                return str(uuid.uuid4())
            last_at = _ensure_utc(last_user_turn.created_at)
            if last_at < threshold:
                return str(uuid.uuid4())
            return last_user_turn.conversation_id
        return self._execute(operation)

    def list_recent_for_user(
        self,
        *,
        user_id: str,
        limit: int = 20,
        conversation_id: str | None = None,
    ) -> list[ConversationTurn]:
        """Return the user's most recent archived turns in chronological order.

        When `conversation_id` is provided, the result is scoped to that
        conversation only. The query still remains user-scoped so rows
        from another user can never bridge into the returned history.
        """
        if limit <= 0:
            return []

        def operation(session: Session) -> list[ConversationTurn]:
            stmt = select(ConversationTurn).where(ConversationTurn.user_id == user_id)
            if conversation_id is not None:
                stmt = stmt.where(ConversationTurn.conversation_id == conversation_id)
            stmt = stmt.order_by(
                ConversationTurn.created_at.desc(),
                ConversationTurn.turn_id.desc(),
            ).limit(limit)
            rows = list(session.scalars(stmt).all())
            rows.reverse()
            return rows

        return self._execute(operation)


    def mark_mem0_persisted(
        self,
        *,
        turn_ids: Iterable[str],
        memory_id: str | None,
        persisted_at: datetime | None = None,
    ) -> int:
        """Mark the given turn rows as persisted to mem0.

        Sets `mem0_persisted_at` (UTC) and `mem0_memory_id` for every row
        in `turn_ids`. Returns the number of rows updated. Caller invokes
        this only AFTER `mem0.add` returns successfully — failure leaves
        rows untouched so the partial index can find them later.

        `memory_id` may be None when the underlying mem0 client returns
        a shape we cannot extract an ID from; the persisted_at timestamp
        is still set so the recovery query can distinguish "we tried and
        succeeded but no id was reachable" from "we never tried."
        """
        ids = [tid for tid in turn_ids if tid]
        if not ids:
            return 0
        timestamp = persisted_at or now_utc()

        def operation(session: Session) -> int:
            stmt = (
                update(ConversationTurn)
                .where(ConversationTurn.turn_id.in_(ids))
                .values(mem0_persisted_at=timestamp, mem0_memory_id=memory_id)
            )
            result = session.execute(stmt)
            return result.rowcount or 0
        return self._execute(operation)
