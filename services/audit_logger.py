"""Audit logger service (Step 5).

Thin wrapper that turns an IntentExecutor / dispatcher event into a
row in audit_events. Designed so each call site does one obvious
thing — no parameter juggling at the call site.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from repositories.audit_events_repository import AuditEventsRepository

logger = logging.getLogger(__name__)


class AuditLogger:
    """Single chokepoint for writing audit_events rows.

    Wired into the IntentExecutor's audit_log callback, into the
    direct reminder-read path, and into the destructive-approval
    confirm/cancel paths. Each call site supplies intent + result;
    this class handles serialization, truncation, and the never-raise
    contract.
    """

    def __init__(self, *, repository: AuditEventsRepository) -> None:
        self._repository = repository

    def write_event(
        self,
        *,
        user_id: str | None,
        intent: str,
        parameters: dict[str, Any] | None,
        result: str,
        response_time_ms: int | None = None,
        provider: str | None = None,
        error_text: str | None = None,
    ) -> str | None:
        return self._repository.write(
            user_id=user_id,
            intent=intent,
            parameters=parameters,
            result=result,
            response_time_ms=response_time_ms,
            provider=provider,
            error_text=error_text,
        )

    async def write_executor_event(self, payload: dict[str, Any]) -> None:
        """Adapter for IntentExecutor's `audit_log` callback signature.

        The executor passes a single dict; we map keys to the
        repository's named args. Missing keys default safely.
        """
        try:
            self._repository.write(
                user_id=payload.get('user_id'),
                intent=str(payload.get('intent') or 'unknown'),
                parameters=payload.get('parameters') or {},
                result=str(payload.get('result') or 'unknown'),
                response_time_ms=payload.get('response_time_ms'),
                provider=payload.get('provider'),
                error_text=payload.get('error_text'),
            )
        except Exception as exc:  # noqa: BLE001 — audit must never raise
            logger.warning('audit_logger_executor_event_failed', extra={'error': str(exc)})


class StopWatch:
    """Tiny helper for measuring response_time_ms around a call site.

        with StopWatch() as sw:
            ...
        audit.write_event(..., response_time_ms=sw.elapsed_ms)
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: int = 0

    def __enter__(self) -> StopWatch:
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed_ms = int((time.monotonic() - self._start) * 1000)
