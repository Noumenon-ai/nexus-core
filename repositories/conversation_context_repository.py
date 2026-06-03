from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from models import ConversationContext, now_utc
from repositories.base import BaseRepository


class ConversationContextRepository(BaseRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory)

    def get_active(self, user_id: str, *, now: datetime | None = None) -> ConversationContext | None:
        current = now or now_utc()
        def operation(session: Session) -> ConversationContext | None:
            stmt = select(ConversationContext).where(
                ConversationContext.user_id == user_id,
                ConversationContext.expires_at > current,
            ).order_by(ConversationContext.updated_at.desc())
            return session.scalar(stmt)
        return self._execute(operation)

    def upsert(self, *, user_id: str, last_topic: str | None, last_entity_type: str | None, last_entity_value: str | None, last_intent: str | None, context_json: str | None, ttl_minutes: int = 30) -> ConversationContext:
        expires_at = now_utc() + timedelta(minutes=ttl_minutes)
        def operation(session: Session) -> ConversationContext:
            stmt = select(ConversationContext).where(ConversationContext.user_id == user_id).order_by(ConversationContext.updated_at.desc())
            context = session.scalar(stmt)
            if context is None:
                context = ConversationContext(
                    user_id=user_id,
                    last_topic=last_topic,
                    last_entity_type=last_entity_type,
                    last_entity_value=last_entity_value,
                    last_intent=last_intent,
                    context_json=context_json,
                    expires_at=expires_at,
                )
                session.add(context)
            else:
                context.last_topic = last_topic
                context.last_entity_type = last_entity_type
                context.last_entity_value = last_entity_value
                context.last_intent = last_intent
                context.context_json = context_json
                context.expires_at = expires_at
                context.updated_at = now_utc()
            session.flush()
            return context
        return self._execute(operation)

    def clear(self, user_id: str) -> None:
        def operation(session: Session) -> None:
            stmt = select(ConversationContext).where(ConversationContext.user_id == user_id)
            for context in session.scalars(stmt).all():
                session.delete(context)
            session.flush()
        self._execute(operation)
