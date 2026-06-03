"""V3.2 read tools — eight Nexus-internal read paths wrapped as @tool.

Each tool returns a ToolResult.ok carrying a JSON-serializable data dict so
the V3.5 dispatcher can hand results directly to Gemini. The four deferred
Google read tools live separately in services/read_tools_stubs.py (see
HARDENING_PASS_V2.md H2-011).

The eight tools:
    get_current_time
    list_active_reminders
    list_pending_tasks
    list_completed_tasks
    list_memories
    get_email_summary
    get_telos
    get_active_approvals
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from googleapiclient.errors import HttpError

from repositories.approvals_repository import ApprovalsRepository
from repositories.emails_ingested_repository import EmailsIngestedRepository
from repositories.memories_repository import MemoriesRepository
from repositories.reminders_repository import RemindersRepository
from repositories.tasks_repository import TasksRepository
from services.google_auth_service import GoogleAuthError
from services.google_calendar_service import GoogleCalendarService
from services.google_people_service import GooglePeopleService
from services.news_service import NewsService
from services.weather_service import WeatherService
from services.wikipedia_service import WikipediaService
from services.google_tasks_service import GoogleTasksService
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry, ToolResult, ToolSpec
from utils.dates import utc_now


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _serialize_reminder(r) -> dict[str, Any]:
    return {
        'id': r.id,
        'body': r.body,
        'next_fire_at': _iso(r.next_fire_at),
        'recurrence': r.recurrence,
        'status': r.status,
        'created_at': _iso(getattr(r, 'created_at', None)),
        'updated_at': _iso(getattr(r, 'updated_at', None)),
        'target_contact_id': getattr(r, 'target_contact_id', None),
    }


def _serialize_task(t) -> dict[str, Any]:
    return {
        'id': t.id,
        'title': t.title,
        'status': t.status,
        'due_at': _iso(t.due_at),
        'priority': t.priority,
        'completed_at': _iso(t.completed_at),
    }


def _serialize_memory(m) -> dict[str, Any]:
    return {
        'id': m.id,
        'memory_type': m.memory_type,
        'key': m.key,
        'value': m.value,
        'confidence': m.confidence,
        'source': m.source,
    }


def _serialize_email(e) -> dict[str, Any]:
    return {
        'id': e.id,
        'subject': e.subject,
        'sender': e.sender,
        'snippet': e.snippet,
        'received_at': _iso(e.received_at),
        'category': e.category,
    }


def _serialize_approval(a) -> dict[str, Any]:
    return {
        'id': a.id,
        'action_type': a.action_type,
        'status': a.status,
        'preview_text': a.preview_text,
        'expires_at': _iso(a.expires_at),
    }


_PARAMS_USER_SCOPED: dict[str, Any] = {
    'type': 'object',
    'properties': {},
    'required': [],
}
_PARAMS_EMAIL_SUMMARY: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'hours': {'type': 'integer', 'description': 'Lookback window in hours.', 'default': 24},
    },
    'required': [],
}
_PARAMS_NONE: dict[str, Any] = {
    'type': 'object',
    'properties': {},
    'required': [],
}
_PARAMS_CALENDAR: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'time_min': {'type': 'string', 'description': 'ISO 8601 lower bound for the window.'},
        'time_max': {'type': 'string', 'description': 'ISO 8601 upper bound for the window.'},
        'max_results': {'type': 'integer', 'description': 'Maximum number of events to return.', 'default': 10},
    },
    'required': ['time_min', 'time_max'],
}


def _parse_iso_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def make_list_calendar_events(calendar_service: GoogleCalendarService):
    async def list_calendar_events(*, user_id: str, time_min: str, time_max: str, max_results: int = 10) -> ToolResult:
        try:
            tmin = _parse_iso_datetime(time_min)
            tmax = _parse_iso_datetime(time_max)
        except ValueError:
            return ToolResult.fail('time_min and time_max must be valid ISO 8601 datetimes.')
        if isinstance(max_results, bool):
            return ToolResult.fail('max_results must be an integer between 1 and 50.')
        try:
            max_results_value = int(max_results)
        except (TypeError, ValueError):
            return ToolResult.fail('max_results must be an integer between 1 and 50.')
        if not 1 <= max_results_value <= 50:
            return ToolResult.fail('max_results must be an integer between 1 and 50.')

        try:
            events = await calendar_service.list_events(
                user_id,
                time_min=tmin,
                time_max=tmax,
                max_results=max_results_value,
            )
            return ToolResult.ok(data={
                'events': [event.to_dict() for event in events],
                'count': len(events),
            })
        except ValueError as exc:
            return ToolResult.fail(str(exc))
        except GoogleAuthError:
            return ToolResult.fail("Calendar access expired. Say 'reconnect google' to re-auth.")
        except HttpError as exc:
            status = getattr(getattr(exc, 'resp', None), 'status', 'unknown')
            return ToolResult.fail(f'Calendar service error (status={status}). Try again in a moment.')

    return list_calendar_events


def make_check_freebusy(calendar_service: GoogleCalendarService):
    async def check_freebusy(*, user_id: str, time_min: str, time_max: str) -> ToolResult:
        try:
            tmin = _parse_iso_datetime(time_min)
            tmax = _parse_iso_datetime(time_max)
            blocks = await calendar_service.freebusy(user_id, time_min=tmin, time_max=tmax)
            return ToolResult.ok(data={
                'busy': [block.to_dict() for block in blocks],
                'count': len(blocks),
            })
        except ValueError as exc:
            return ToolResult.fail(str(exc))
        except GoogleAuthError:
            return ToolResult.fail("Calendar access expired. Say 'reconnect google' to re-auth.")
        except HttpError as exc:
            status = getattr(getattr(exc, 'resp', None), 'status', 'unknown')
            return ToolResult.fail(f'Calendar service error (status={status}). Try again in a moment.')

    return check_freebusy


def make_read_tools(
    *,
    reminders_repository: RemindersRepository,
    tasks_repository: TasksRepository,
    memories_repository: MemoriesRepository,
    emails_repository: EmailsIngestedRepository,
    approvals_repository: ApprovalsRepository,
    telos_service: TelosService,
    app_timezone: str,
) -> list[tuple[Callable[..., ToolResult], dict[str, Any]]]:
    """Build the eight V3.2 read-tool closures bound to the given dependencies.

    Returns a list of (function, registration_metadata) pairs so a caller can
    register them into either the default registry or a test-scoped registry.
    """

    tz = ZoneInfo(app_timezone)

    def get_current_time() -> ToolResult:
        now = utc_now().astimezone(tz)
        return ToolResult.ok(data={'iso': now.isoformat(), 'timezone': app_timezone})

    def list_active_reminders(*, user_id: str) -> ToolResult:
        items = reminders_repository.list_active(user_id)
        return ToolResult.ok(data={'reminders': [_serialize_reminder(r) for r in items]})

    def list_pending_tasks(*, user_id: str) -> ToolResult:
        items = tasks_repository.list_pending(user_id)
        return ToolResult.ok(data={'tasks': [_serialize_task(t) for t in items]})

    def list_completed_tasks(*, user_id: str) -> ToolResult:
        items = tasks_repository.list_completed(user_id)
        return ToolResult.ok(data={'tasks': [_serialize_task(t) for t in items]})

    def list_memories(*, user_id: str) -> ToolResult:
        items = memories_repository.list_by_user(user_id)
        return ToolResult.ok(data={'memories': [_serialize_memory(m) for m in items]})

    def get_email_summary(*, user_id: str, hours: int = 24) -> ToolResult:
        if isinstance(hours, bool):
            return ToolResult.fail('hours must be an integer between 1 and 720.')
        try:
            hours_value = int(hours)
        except (TypeError, ValueError):
            return ToolResult.fail('hours must be an integer between 1 and 720.')
        if not 1 <= hours_value <= 720:
            return ToolResult.fail('hours must be an integer between 1 and 720.')
        items = emails_repository.list_recent(user_id, hours=hours_value)
        return ToolResult.ok(data={'emails': [_serialize_email(e) for e in items], 'window_hours': hours_value})

    def get_telos(*, user_id: str) -> ToolResult:
        content = telos_service.load(user_id)
        return ToolResult.ok(data={'present': content is not None, 'content': content})

    def get_active_approvals(*, user_id: str) -> ToolResult:
        items = approvals_repository.list_active_pending_for_user(user_id)
        return ToolResult.ok(data={'approvals': [_serialize_approval(a) for a in items]})

    return [
        (
            get_current_time,
            {
                'name': 'get_current_time',
                'description': 'Return the current time as ISO 8601 in the user app timezone.',
                'parameters': _PARAMS_NONE,
            },
        ),
        (
            list_active_reminders,
            {
                'name': 'list_active_reminders',
                'description': 'List the user\'s currently active reminders, ordered by next fire time.',
                'parameters': _PARAMS_USER_SCOPED,
            },
        ),
        (
            list_pending_tasks,
            {
                'name': 'list_pending_tasks',
                'description': 'List the user\'s pending Nexus-internal tasks (priority desc, then due date).',
                'parameters': _PARAMS_USER_SCOPED,
            },
        ),
        (
            list_completed_tasks,
            {
                'name': 'list_completed_tasks',
                'description': 'List the user\'s completed Nexus-internal tasks, most recent first.',
                'parameters': _PARAMS_USER_SCOPED,
            },
        ),
        (
            list_memories,
            {
                'name': 'list_memories',
                'description': 'List the user\'s explicit-memory key/value pairs (preferences, facts).',
                'parameters': _PARAMS_USER_SCOPED,
            },
        ),
        (
            get_email_summary,
            {
                'name': 'get_email_summary',
                'description': 'Return recently ingested emails for the user within the given lookback window.',
                'parameters': _PARAMS_EMAIL_SUMMARY,
            },
        ),
        (
            get_telos,
            {
                'name': 'get_telos',
                'description': 'Return the user\'s TELOS file contents (life-OS doc), or present=False if not yet written.',
                'parameters': _PARAMS_USER_SCOPED,
            },
        ),
        (
            get_active_approvals,
            {
                'name': 'get_active_approvals',
                'description': 'List pending, unexpired approval prompts for the user.',
                'parameters': _PARAMS_USER_SCOPED,
            },
        ),
    ]


def register_read_tools(registry: ToolRegistry, **deps: Any) -> list[ToolSpec]:
    """Register all eight V3.2 read tools into the given registry."""
    specs = []
    for fn, meta in make_read_tools(**deps):
        specs.append(registry.register(fn, **meta))
    return specs


def make_list_google_tasks(tasks_service: GoogleTasksService):
    async def list_google_tasks(*, user_id: str, max_results: int = 20) -> ToolResult:
        try:
            tasks = await tasks_service.list_tasks(user_id, max_results=max_results)
        except GoogleAuthError:
            return ToolResult.fail("Google Tasks access expired. Say 'reconnect google' to re-auth.")
        return ToolResult.ok(data={
            'tasks': [t.to_dict() for t in tasks],
            'count': len(tasks),
        })
    return list_google_tasks


def make_lookup_contact(people_service: GooglePeopleService):
    async def lookup_contact(*, user_id: str, query: str) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            return ToolResult.fail('Cannot look up contact: query is empty.')
        try:
            contacts = await people_service.lookup_contact(user_id, query)
        except GoogleAuthError:
            return ToolResult.fail("Google Contacts access expired. Say 'reconnect google' to re-auth.")
        return ToolResult.ok(data={
            'contacts': [c.to_dict() for c in contacts],
            'count': len(contacts),
        })
    return lookup_contact


def make_lookup_wikipedia(wiki: WikipediaService):
    async def lookup_wikipedia(*, topic: str) -> ToolResult:
        if not isinstance(topic, str) or not topic.strip():
            return ToolResult.fail('Cannot look up Wikipedia: topic is empty.')
        try:
            summary = await wiki.lookup(topic)
        except Exception as exc:
            return ToolResult.fail(f'Wikipedia lookup failed: {type(exc).__name__}.')
        if summary is None:
            return ToolResult.ok(data={'found': False, 'topic': topic})
        return ToolResult.ok(data={'found': True, **summary.to_dict()})
    return lookup_wikipedia


def make_get_weather(weather: WeatherService):
    async def get_weather(*, location: str) -> ToolResult:
        if not isinstance(location, str) or not location.strip():
            return ToolResult.fail('Cannot fetch weather: location is empty.')
        try:
            current = await weather.current(location)
        except Exception as exc:
            return ToolResult.fail(f'Weather lookup failed: {type(exc).__name__}.')
        if current is None:
            return ToolResult.ok(data={'found': False, 'location': location})
        return ToolResult.ok(data={'found': True, **current.to_dict()})
    return get_weather


def make_get_news_headlines(news: NewsService):
    async def get_news_headlines(*, max_results: int = 5) -> ToolResult:
        if not isinstance(max_results, int) or max_results < 1:
            max_results = 5
        max_results = min(max_results, 20)
        try:
            headlines = await news.top_headlines(max_results=max_results)
        except Exception as exc:
            return ToolResult.fail(f'News fetch failed: {type(exc).__name__}.')
        return ToolResult.ok(data={
            'headlines': [h.to_dict() for h in headlines],
            'count': len(headlines),
        })
    return get_news_headlines
