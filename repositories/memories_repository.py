from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from models import Memory, now_utc
from repositories.base import BaseRepository


class MemoriesRepository(BaseRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory)

    def upsert(self, *, user_id: str, memory_type: str, key: str, value: str, confidence: float, source: str) -> Memory:
        def operation(session: Session) -> Memory:
            stmt = select(Memory).where(Memory.user_id == user_id, Memory.memory_type == memory_type, Memory.key == key)
            memory = session.scalar(stmt)
            if memory is None:
                memory = Memory(user_id=user_id, memory_type=memory_type, key=key, value=value, confidence=confidence, source=source)
                session.add(memory)
            else:
                memory.value = value
                memory.confidence = confidence
                memory.source = source
                memory.updated_at = now_utc()
            session.flush()
            return memory
        return self._execute(operation)

    def list_by_user(self, user_id: str, *, source: str | None = None, memory_type: str | None = None) -> list[Memory]:
        def operation(session: Session) -> list[Memory]:
            stmt = select(Memory).where(Memory.user_id == user_id)
            if source is not None:
                stmt = stmt.where(Memory.source == source)
            if memory_type is not None:
                stmt = stmt.where(Memory.memory_type == memory_type)
            stmt = stmt.order_by(Memory.created_at)
            return list(session.scalars(stmt).all())
        return self._execute(operation)

    def delete(self, *, user_id: str, key: str) -> Memory | None:
        def operation(session: Session) -> Memory | None:
            stmt = select(Memory).where(Memory.user_id == user_id, Memory.key == key)
            memory = session.scalar(stmt)
            if memory is None:
                return None
            session.delete(memory)
            session.flush()
            return memory
        return self._execute(operation)

    def get(self, *, user_id: str, memory_type: str, key: str) -> Memory | None:
        def operation(session: Session) -> Memory | None:
            stmt = select(Memory).where(Memory.user_id == user_id, Memory.memory_type == memory_type, Memory.key == key)
            return session.scalar(stmt)
        return self._execute(operation)
