"""Audit-events repository (Step 5).

Single writer + queries for the delivery-truth audit trail. Keep the
write path bullet-proof — every IntentExecutor branch calls write()
and an exception here must NEVER bubble back into the user pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from models import AuditEvent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AuditEventRow:
    id: str
    created_at: datetime
    user_id: str | None
    intent: str
    parameters: dict[str, Any]
    result: str
    response_time_ms: int | None
    provider: str | None
    error_text: str | None


class AuditEventsRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def write(
        self,
        *,
        user_id: str | None,
        intent: str,
        parameters: dict[str, Any] | None = None,
        result: str,
        response_time_ms: int | None = None,
        provider: str | None = None,
        error_text: str | None = None,
    ) -> str | None:
        """Insert an audit row. Returns the new row id, or None on failure.

        Never raises — audit must never break the user-facing call.
        """
        try:
            payload = json.dumps(parameters or {}, default=str, sort_keys=True)
        except (TypeError, ValueError) as exc:
            logger.warning('audit_events_param_serialize_failed', extra={'error': str(exc)})
            payload = '{}'
        try:
            with self._session_factory() as session:
                row = AuditEvent(
                    user_id=user_id,
                    intent=(intent or 'unknown')[:64],
                    parameters_json=payload,
                    result=(result or 'unknown')[:64],
                    response_time_ms=(
                        int(response_time_ms) if response_time_ms is not None else None
                    ),
                    provider=(provider or None),
                    error_text=(error_text or None),
                )
                session.add(row)
                session.commit()
                return row.id
        except Exception as exc:  # noqa: BLE001 — audit must never raise
            logger.warning(
                'audit_events_write_failed',
                extra={'intent': intent, 'result': result, 'error': str(exc)},
            )
            return None

    def list_recent(
        self,
        *,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[AuditEventRow]:
        """Most-recent-first. user_id=None returns global recent activity."""
        try:
            with self._session_factory() as session:
                stmt = select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(int(limit))
                if user_id is not None:
                    stmt = stmt.where(AuditEvent.user_id == user_id)
                rows = session.execute(stmt).scalars().all()
                return [self._to_dto(row) for row in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning('audit_events_read_failed', extra={'error': str(exc)})
            return []

    def count(self, *, user_id: str | None = None) -> int:
        try:
            with self._session_factory() as session:
                stmt = select(AuditEvent)
                if user_id is not None:
                    stmt = stmt.where(AuditEvent.user_id == user_id)
                return len(session.execute(stmt).scalars().all())
        except Exception as exc:  # noqa: BLE001
            logger.warning('audit_events_count_failed', extra={'error': str(exc)})
            return 0

    @staticmethod
    def _to_dto(row: AuditEvent) -> AuditEventRow:
        try:
            parameters = json.loads(row.parameters_json or '{}')
        except (TypeError, ValueError):
            parameters = {}
        return AuditEventRow(
            id=row.id,
            created_at=row.created_at,
            user_id=row.user_id,
            intent=row.intent,
            parameters=parameters if isinstance(parameters, dict) else {},
            result=row.result,
            response_time_ms=row.response_time_ms,
            provider=row.provider,
            error_text=row.error_text,
        )
