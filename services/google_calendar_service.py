from __future__ import annotations

import asyncio
import threading
from datetime import date, datetime, time, timezone
from typing import Any, Callable, TypeVar

from googleapiclient import discovery
from googleapiclient.errors import HttpError

from services.google_auth_service import GoogleAuthError, GoogleAuthService
from services.google_types import BusyBlock, CalendarEvent, CalendarEventCreate, CalendarEventUpdate


T = TypeVar('T')


async def _run_blocking_without_default_executor(fn, /, *args, **kwargs):
    """Run blocking calendar work without touching asyncio's default executor.

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
        name='google-calendar-blocking',
    ).start()
    while not done.is_set():
        await asyncio.sleep(0.01)
    if 'exception' in result_box:
        raise result_box['exception']  # type: ignore[misc]
    return result_box.get('result')


class GoogleCalendarService:
    def __init__(self, auth_service: GoogleAuthService):
        self.auth = auth_service

    async def freebusy(
        self,
        user_id: str,
        *,
        time_min: datetime,
        time_max: datetime,
    ) -> list[BusyBlock]:
        self._require_aware(time_min, field_name='time_min')
        self._require_aware(time_max, field_name='time_max')
        if time_max < time_min:
            raise ValueError('time_max must be greater than or equal to time_min')

        creds = await _run_blocking_without_default_executor(
            self.auth.get_credentials, user_id,
        )
        if creds is None:
            raise GoogleAuthError('google_calendar_not_connected')

        body = {
            'timeMin': time_min.isoformat(),
            'timeMax': time_max.isoformat(),
            'items': [{'id': 'primary'}],
        }

        def _call() -> dict[str, Any]:
            service = discovery.build('calendar', 'v3', credentials=creds, cache_discovery=False)
            return service.freebusy().query(body=body).execute()

        response = await self._with_retry(_call)
        busy_raw = response.get('calendars', {}).get('primary', {}).get('busy', [])
        return [self._parse_busy_block(item) for item in busy_raw]

    async def list_events(
        self,
        user_id: str,
        *,
        time_min: datetime,
        time_max: datetime,
        max_results: int = 10,
    ) -> list[CalendarEvent]:
        self._require_aware(time_min, field_name='time_min')
        self._require_aware(time_max, field_name='time_max')
        if time_max < time_min:
            raise ValueError('time_max must be greater than or equal to time_min')

        creds = await _run_blocking_without_default_executor(
            self.auth.get_credentials, user_id,
        )
        if creds is None:
            raise GoogleAuthError('google_calendar_not_connected')

        def _call() -> dict[str, Any]:
            service = discovery.build('calendar', 'v3', credentials=creds, cache_discovery=False)
            return service.events().list(
                calendarId='primary',
                timeMin=time_min.isoformat(),
                timeMax=time_max.isoformat(),
                singleEvents=True,
                orderBy='startTime',
                maxResults=max_results,
            ).execute()

        response = await self._with_retry(_call)
        items = response.get('items', []) or []
        return [self._parse_event(item) for item in items]

    async def create_event(
        self,
        user_id: str,
        *,
        event: CalendarEventCreate,
    ) -> CalendarEvent:
        creds = await _run_blocking_without_default_executor(
            self.auth.get_credentials, user_id,
        )
        if creds is None:
            raise GoogleAuthError('google_calendar_not_connected')

        body = event.to_body()

        def _call() -> dict[str, Any]:
            service = discovery.build('calendar', 'v3', credentials=creds, cache_discovery=False)
            return service.events().insert(calendarId='primary', body=body).execute()

        response = await self._with_retry(_call)
        return self._parse_event(response)

    async def update_event(
        self,
        user_id: str,
        *,
        event_id: str,
        patch: CalendarEventUpdate,
    ) -> CalendarEvent:
        if not event_id or not event_id.strip():
            raise ValueError('event_id must be non-empty')

        creds = await _run_blocking_without_default_executor(
            self.auth.get_credentials, user_id,
        )
        if creds is None:
            raise GoogleAuthError('google_calendar_not_connected')

        body = patch.to_patch_body()

        def _call() -> dict[str, Any]:
            service = discovery.build('calendar', 'v3', credentials=creds, cache_discovery=False)
            return service.events().patch(
                calendarId='primary',
                eventId=event_id,
                body=body,
            ).execute()

        response = await self._with_retry(_call)
        return self._parse_event(response)

    async def delete_event(
        self,
        user_id: str,
        *,
        event_id: str,
    ) -> None:
        if not event_id or not event_id.strip():
            raise ValueError('event_id must be non-empty')

        creds = await _run_blocking_without_default_executor(
            self.auth.get_credentials, user_id,
        )
        if creds is None:
            raise GoogleAuthError('google_calendar_not_connected')

        def _call() -> None:
            service = discovery.build('calendar', 'v3', credentials=creds, cache_discovery=False)
            service.events().delete(calendarId='primary', eventId=event_id).execute()
            return None

        await self._with_retry(_call)
        return None

    async def _with_retry(self, sync_fn: Callable[[], T], max_retries: int = 3) -> T:
        delay = 1.0
        for attempt in range(max_retries):
            try:
                return await _run_blocking_without_default_executor(sync_fn)
            except HttpError as exc:
                status = getattr(getattr(exc, 'resp', None), 'status', 0) or 0
                if status in (401, 403):
                    raise GoogleAuthError(f'google_calendar_auth_error status={status}') from exc
                if status == 429 or status >= 500:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError('retry loop exhausted unexpectedly')

    @staticmethod
    def _parse_event(raw: dict[str, Any]) -> CalendarEvent:
        return CalendarEvent(
            id=str(raw.get('id') or ''),
            summary=str(raw.get('summary') or ''),
            start=GoogleCalendarService._parse_event_time(raw.get('start')),
            end=GoogleCalendarService._parse_event_time(raw.get('end')),
            description=GoogleCalendarService._optional_text(raw.get('description')),
            location=GoogleCalendarService._optional_text(raw.get('location')),
            html_link=GoogleCalendarService._optional_text(raw.get('htmlLink')),
        )

    @staticmethod
    def _parse_busy_block(raw: dict[str, Any]) -> BusyBlock:
        return BusyBlock(
            start=GoogleCalendarService._parse_datetime(raw['start']),
            end=GoogleCalendarService._parse_datetime(raw['end']),
        )

    @staticmethod
    def _parse_event_time(raw: Any) -> datetime:
        if not isinstance(raw, dict):
            raise ValueError('calendar event time payload must be a dict')
        if raw.get('dateTime'):
            return GoogleCalendarService._parse_datetime(str(raw['dateTime']))
        if raw.get('date'):
            parsed = date.fromisoformat(str(raw['date']))
            return datetime.combine(parsed, time.min, tzinfo=timezone.utc)
        raise ValueError('calendar event missing dateTime/date')

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        GoogleCalendarService._require_aware(parsed, field_name='event_datetime')
        return parsed

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _require_aware(value: datetime, *, field_name: str) -> None:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{field_name} must be timezone-aware')
