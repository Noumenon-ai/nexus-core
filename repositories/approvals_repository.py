from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from models import Approval, now_utc
from repositories.base import BaseRepository


def _ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class ApprovalsRepository(BaseRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory)

    def _coerce(self, approval: Approval | None) -> Approval | None:
        if approval is None:
            return None
        approval.expires_at = _ensure_utc(approval.expires_at)
        approval.created_at = _ensure_utc(approval.created_at)
        approval.approved_at = _ensure_utc(approval.approved_at)
        approval.cancelled_at = _ensure_utc(approval.cancelled_at)
        approval.executed_at = _ensure_utc(approval.executed_at)
        return approval

    def create(self, *, user_id: str, action_type: str, preview_text: str, payload_json: str, expires_at: datetime) -> Approval:
        def operation(session: Session) -> Approval:
            approval = Approval(user_id=user_id, action_type=action_type, status='pending', preview_text=preview_text, payload_json=payload_json, expires_at=expires_at)
            session.add(approval)
            session.flush()
            return self._coerce(approval)
        return self._execute(operation)

    def get(self, approval_id: str) -> Approval | None:
        return self._execute(lambda session: self._coerce(session.get(Approval, approval_id)))

    def list_pending_expired(self, *, now: datetime | None = None) -> list[Approval]:
        current = now or now_utc()
        def operation(session: Session) -> list[Approval]:
            stmt = select(Approval).where(Approval.status == 'pending', Approval.expires_at <= current)
            return [self._coerce(approval) for approval in session.scalars(stmt).all()]
        return self._execute(operation)

    def list_active_pending_for_user(self, user_id: str, *, now: datetime | None = None) -> list[Approval]:
        current = now or now_utc()
        def operation(session: Session) -> list[Approval]:
            stmt = select(Approval).where(Approval.user_id == user_id, Approval.status == 'pending', Approval.expires_at > current).order_by(Approval.created_at.desc())
            return [self._coerce(approval) for approval in session.scalars(stmt).all()]
        return self._execute(operation)

    def claim_for_execution(self, approval_id: str, *, claimed_at: datetime | None = None) -> Approval | None:
        current = claimed_at or now_utc()
        def operation(session: Session) -> Approval | None:
            session.execute(update(Approval).where(Approval.id == approval_id, Approval.status == 'pending', Approval.expires_at > current).values(status='approved', approved_at=current))
            session.flush()
            return self._coerce(session.get(Approval, approval_id))
        return self._execute(operation)

    def mark_executed(self, approval_id: str, *, executed_at: datetime) -> Approval:
        def operation(session: Session) -> Approval:
            result = session.execute(update(Approval).where(Approval.id == approval_id, Approval.status == 'approved').values(status='executed', executed_at=executed_at))
            if result.rowcount == 0:
                raise ValueError('Approval not in executable state')
            session.flush()
            approval = session.get(Approval, approval_id)
            if approval is None:
                raise ValueError('Approval not found')
            return self._coerce(approval)
        return self._execute(operation)

    def update_status(self, approval_id: str, *, status: str, approved_at: datetime | None = None, cancelled_at: datetime | None = None, executed_at: datetime | None = None) -> Approval:
        def operation(session: Session) -> Approval:
            approval = session.get(Approval, approval_id)
            if approval is None:
                raise ValueError('Approval not found')
            approval.status = status
            approval.approved_at = approved_at
            approval.cancelled_at = cancelled_at
            approval.executed_at = executed_at
            session.flush()
            return self._coerce(approval)
        return self._execute(operation)
