"""V3.2.5.4 — Google Tasks service wrapper (read-only).

Mirrors GoogleCalendarService patterns:
  - GoogleAuthService dependency only (no rate_limiter — H-1 stays dropped)
  - blocking helper for sync googleapiclient .execute() calls
  - Shared exponential-backoff retry on 429/5xx; 401/403 → GoogleAuthError
  - Per-user isolation enforced: every public method requires user_id

Current scope: tasks.readonly. Only list_tasks is implemented. Write
methods (create_task, mark_done) deferred to a separate phase that
re-scopes OAuth from tasks.readonly to tasks (H2-011 cross-reference).
"""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from typing import Any, Callable, TypeVar

from googleapiclient import discovery
from googleapiclient.errors import HttpError

from services.google_auth_service import GoogleAuthError, GoogleAuthService
from services.google_types import GoogleTask


T = TypeVar('T')


async def _run_blocking_without_default_executor(fn, /, *args, **kwargs):
    """Run blocking Tasks API work without touching asyncio's default executor.

    In this environment, `asyncio.to_thread()` can leave a worker alive long
    enough to make short-lived async pytest processes hang on exit even after
    the awaited work already finished. Use a one-off daemon thread instead.
    """
    result_box: dict[str, object] = {}
    done = threading.Event()

    def worker():
        try:
            result_box['result'] = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - re-raised by awaiter
            result_box['exception'] = exc
        finally:
            done.set()

    threading.Thread(
        target=worker,
        daemon=True,
        name='google-tasks-blocking',
    ).start()
    while not done.is_set():
        await asyncio.sleep(0.01)
    if 'exception' in result_box:
        raise result_box['exception']  # type: ignore[misc]
    return result_box.get('result')


class GoogleTasksService:
    def __init__(self, auth_service: GoogleAuthService):
        self.auth = auth_service

    async def list_tasks(self, user_id: str, *, max_results: int = 20) -> list[GoogleTask]:
        creds = await _run_blocking_without_default_executor(
            self.auth.get_credentials, user_id,
        )
        if creds is None:
            raise GoogleAuthError('google_tasks_not_connected')

        def _call() -> dict[str, Any]:
            service = discovery.build('tasks', 'v1', credentials=creds, cache_discovery=False)
            return service.tasks().list(
                tasklist='@default',
                maxResults=max_results,
                showCompleted=False,
                showHidden=False,
            ).execute()

        response = await self._with_retry(_call)
        items = response.get('items', []) or []
        return [self._parse_task(item) for item in items]

    async def _with_retry(self, sync_fn: Callable[[], T], max_retries: int = 3) -> T:
        delay = 1.0
        for attempt in range(max_retries):
            try:
                return await _run_blocking_without_default_executor(sync_fn)
            except HttpError as exc:
                status = getattr(getattr(exc, 'resp', None), 'status', 0) or 0
                if status in (401, 403):
                    raise GoogleAuthError(f'google_tasks_auth_error status={status}') from exc
                if status == 429 or status >= 500:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError('retry loop exhausted unexpectedly')

    @staticmethod
    def _parse_task(raw: dict[str, Any]) -> GoogleTask:
        due_raw = raw.get('due')
        due = None
        if due_raw:
            due = datetime.fromisoformat(str(due_raw).replace('Z', '+00:00'))
        notes_raw = raw.get('notes')
        notes = str(notes_raw).strip() if notes_raw else None
        return GoogleTask(
            id=str(raw.get('id') or ''),
            title=str(raw.get('title') or ''),
            status=str(raw.get('status') or 'needsAction'),
            due=due,
            notes=notes or None,
        )
