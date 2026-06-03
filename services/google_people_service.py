"""V3.2.5.5 — Google People (Contacts) service wrapper.

Mirrors GoogleCalendarService / GoogleTasksService patterns:
  - GoogleAuthService dependency only
  - blocking helper for sync googleapiclient .execute()
  - Shared exponential-backoff retry on 429/5xx; 401/403 → GoogleAuthError
  - Per-user isolation: every public method takes user_id

Methods:
  - lookup_contact(user_id, query) -> list[GoogleContact]
  - create_contact(user_id, name, email=None, phone=None) -> GoogleContact
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, TypeVar

from googleapiclient import discovery
from googleapiclient.errors import HttpError

from services.google_auth_service import GoogleAuthError, GoogleAuthService
from services.google_types import GoogleContact


T = TypeVar('T')

_READ_MASK = 'names,emailAddresses,phoneNumbers'


async def _run_blocking_without_default_executor(fn, /, *args, **kwargs):
    """Run blocking People API work without touching asyncio's default executor.

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
        name='google-people-blocking',
    ).start()
    while not done.is_set():
        await asyncio.sleep(0.01)
    if 'exception' in result_box:
        raise result_box['exception']  # type: ignore[misc]
    return result_box.get('result')


class GooglePeopleService:
    def __init__(self, auth_service: GoogleAuthService):
        self.auth = auth_service

    async def lookup_contact(self, user_id: str, query: str) -> list[GoogleContact]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be non-empty')
        creds = await _run_blocking_without_default_executor(
            self.auth.get_credentials, user_id,
        )
        if creds is None:
            raise GoogleAuthError('google_people_not_connected')

        def _call() -> dict[str, Any]:
            service = discovery.build('people', 'v1', credentials=creds, cache_discovery=False)
            return service.people().searchContacts(
                query=query,
                readMask=_READ_MASK,
            ).execute()

        response = await self._with_retry(_call)
        results = response.get('results', []) or []
        return [self._parse_contact(item.get('person', {})) for item in results if item.get('person')]

    async def create_contact(
        self,
        user_id: str,
        *,
        name: str,
        email: str | None = None,
        phone: str | None = None,
    ) -> GoogleContact:
        if not isinstance(name, str) or not name.strip():
            raise ValueError('name must be non-empty')
        creds = await _run_blocking_without_default_executor(
            self.auth.get_credentials, user_id,
        )
        if creds is None:
            raise GoogleAuthError('google_people_not_connected')

        body: dict[str, Any] = {
            'names': [{'givenName': name.strip()}],
        }
        if email:
            body['emailAddresses'] = [{'value': email}]
        if phone:
            body['phoneNumbers'] = [{'value': phone}]

        def _call() -> dict[str, Any]:
            service = discovery.build('people', 'v1', credentials=creds, cache_discovery=False)
            return service.people().createContact(body=body).execute()

        response = await self._with_retry(_call)
        return self._parse_contact(response)

    async def _with_retry(self, sync_fn: Callable[[], T], max_retries: int = 3) -> T:
        delay = 1.0
        for attempt in range(max_retries):
            try:
                return await _run_blocking_without_default_executor(sync_fn)
            except HttpError as exc:
                status = getattr(getattr(exc, 'resp', None), 'status', 0) or 0
                if status in (401, 403):
                    raise GoogleAuthError(f'google_people_auth_error status={status}') from exc
                if status == 429 or status >= 500:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError('retry loop exhausted unexpectedly')

    @staticmethod
    def _parse_contact(person: dict[str, Any]) -> GoogleContact:
        names = person.get('names') or []
        display_name = ''
        if names:
            display_name = str(names[0].get('displayName') or names[0].get('givenName') or '')
        emails = [str(e.get('value')) for e in (person.get('emailAddresses') or []) if e.get('value')]
        phones = [str(p.get('value')) for p in (person.get('phoneNumbers') or []) if p.get('value')]
        return GoogleContact(
            resource_name=str(person.get('resourceName') or ''),
            display_name=display_name,
            emails=emails,
            phones=phones,
        )
