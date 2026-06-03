from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from models import Task, now_utc
from repositories.base import BaseRepository


def _ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class TasksRepository(BaseRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory)

    def _coerce(self, task: Task | None) -> Task | None:
        if task is None:
            return None
        task.due_at = _ensure_utc(task.due_at)
        task.completed_at = _ensure_utc(task.completed_at)
        return task

    def create(self, *, user_id: str, title: str, due_at: datetime | None, priority: int = 0, source: str = 'user') -> Task:
        def operation(session: Session) -> Task:
            task = Task(user_id=user_id, title=title, due_at=due_at, priority=priority, status='pending', source=source)
            session.add(task)
            session.flush()
            return self._coerce(task)
        return self._execute(operation)

    def list_pending(self, user_id: str) -> list[Task]:
        def operation(session: Session) -> list[Task]:
            stmt = select(Task).where(Task.user_id == user_id, Task.status == 'pending').order_by(Task.priority.desc(), Task.due_at)
            return [self._coerce(task) for task in session.scalars(stmt).all()]
        return self._execute(operation)

    def list_completed(self, user_id: str) -> list[Task]:
        def operation(session: Session) -> list[Task]:
            stmt = select(Task).where(Task.user_id == user_id, Task.status == 'done').order_by(Task.completed_at.desc())
            return [self._coerce(task) for task in session.scalars(stmt).all()]
        return self._execute(operation)

    def delete(self, user_id: str, query: str) -> Task | None:
        query_lower = query.lower().strip()
        if not query_lower:
            return None
        def operation(session: Session) -> Task | None:
            stmt = select(Task).where(Task.user_id == user_id, Task.status == 'pending').order_by(Task.created_at)
            for task in session.scalars(stmt).all():
                if task.id == query or query_lower in task.title.lower():
                    snapshot = self._coerce(task)
                    session.delete(task)
                    session.flush()
                    return snapshot
            return None
        return self._execute(operation)

    def mark_done(self, user_id: str, query: str) -> Task | None:
        query_lower = query.lower().strip()
        if not query_lower:
            return None
        def operation(session: Session) -> Task | None:
            stmt = select(Task).where(Task.user_id == user_id, Task.status == 'pending').order_by(Task.created_at)
            tasks = list(session.scalars(stmt).all())
            for task in tasks:
                if task.id == query or query_lower in task.title.lower():
                    task.status = 'done'
                    task.completed_at = now_utc()
                    task.updated_at = task.completed_at
                    session.flush()
                    return self._coerce(task)
            return None
        return self._execute(operation)
