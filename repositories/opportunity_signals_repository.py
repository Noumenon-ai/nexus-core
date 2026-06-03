from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from models import OpportunitySignal, now_utc
from repositories.base import BaseRepository


class OpportunitySignalsRepository(BaseRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory)

    def create(self, *, user_id: str, category: str, title: str, summary: str, source: str, confidence: float, dedupe_hash: str, status: str) -> OpportunitySignal:
        def operation(session: Session) -> OpportunitySignal:
            signal = OpportunitySignal(
                user_id=user_id,
                category=category,
                title=title,
                summary=summary,
                source=source,
                confidence=confidence,
                dedupe_hash=dedupe_hash,
                status=status,
            )
            session.add(signal)
            session.flush()
            return signal
        return self._execute(operation)

    def list_recent_titles(self, *, user_id: str, days: int = 14) -> list[str]:
        cutoff = now_utc() - timedelta(days=days)
        def operation(session: Session) -> list[str]:
            stmt = select(OpportunitySignal.title).where(OpportunitySignal.user_id == user_id, OpportunitySignal.created_at >= cutoff)
            return list(session.scalars(stmt).all())
        return self._execute(operation)

    def count_recent_by_category(self, *, user_id: str, category: str, days: int = 7) -> int:
        cutoff = now_utc() - timedelta(days=days)
        def operation(session: Session) -> int:
            stmt = select(OpportunitySignal).where(
                OpportunitySignal.user_id == user_id,
                OpportunitySignal.category == category,
                OpportunitySignal.created_at >= cutoff,
            )
            return len(list(session.scalars(stmt).all()))
        return self._execute(operation)
