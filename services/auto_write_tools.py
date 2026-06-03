"""V3.3 auto-write tools — five Nexus-internal write paths wrapped as @tool.

Each tool returns ToolResult.ok with `announcement` set per the V3.3 contract
(every auto-write tool must return an announcement, never silent success).
The reminder hybrid parser stays internal to ReminderService; the @tool
surface accepts structured fields directly so Gemini can call without
intermediate NL parsing.

The five real tools:
    create_reminder
    create_task
    mark_task_done
    save_user_memory
    set_user_preference

Three deferred Google write tools (create_calendar_event,
update_calendar_event, create_contact) live in services/auto_write_tools_stubs.py
under the same V3.2.5 phase target tracked at HARDENING_PASS_V2.md H2-011.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from googleapiclient.errors import HttpError

from repositories.memories_repository import MemoriesRepository
from repositories.reminders_repository import RemindersRepository
from repositories.tasks_repository import TasksRepository
from services.google_auth_service import GoogleAuthError
from services.google_calendar_service import GoogleCalendarService
from services.google_people_service import GooglePeopleService
from services.google_types import CalendarEventCreate, CalendarEventUpdate
from services.tool_registry import ToolRegistry, ToolResult, ToolSpec
from utils.dates import format_local_datetime, utc_now
from utils.safety_rules import contains_secret


def _parse_iso_utc(value: str) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        from datetime import timezone
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def make_auto_write_tools(
    *,
    reminders_repository: RemindersRepository,
    tasks_repository: TasksRepository,
    memories_repository: MemoriesRepository,
    google_calendar_service: GoogleCalendarService | None = None,
    google_people_service: GooglePeopleService | None = None,
    scheduler: Any | None = None,
    habit_service: Any | None = None,
    app_timezone: str = 'UTC',
) -> list[tuple[Callable[..., ToolResult], dict[str, Any]]]:
    """Build the five V3.3 auto-write tool closures bound to dependencies.

    Returns a list of (function, registration_metadata) pairs.
    """

    def create_reminder(*, user_id: str, body: str, next_fire_at: str, recurrence: str | None = None) -> ToolResult:
        if not isinstance(body, str) or not body.strip():
            return ToolResult.ok(
                data={'created': False, 'reason': 'empty_body'},
                announcement='Cannot create reminder: body is empty.',
            )
        when = _parse_iso_utc(next_fire_at)
        if when is None:
            return ToolResult.ok(
                data={'created': False, 'reason': 'unparseable_datetime'},
                announcement=f'Cannot create reminder: {next_fire_at!r} is not a valid ISO datetime.',
            )
        if when <= utc_now():
            return ToolResult.ok(
                data={'created': False, 'reason': 'past_time'},
                announcement='Cannot create reminder: time is in the past.',
            )
        reminder = reminders_repository.create(user_id=user_id, body=body, next_fire_at=when, recurrence=recurrence)
        if scheduler is not None:
            scheduler.schedule_reminder(reminder.id, reminder.next_fire_at)
        if habit_service is not None:
            habit_service.record_reminder_creation(user_id, reminder.created_at)
        when_text = format_local_datetime(reminder.next_fire_at, app_timezone)
        return ToolResult.ok(
            data={'created': True, 'reminder_id': reminder.id, 'next_fire_at': reminder.next_fire_at.isoformat()},
            announcement=f'Reminder set for {when_text}: {body}',
        )

    def create_task(*, user_id: str, title: str, due_at: str | None = None, priority: int = 0, source: str = 'user') -> ToolResult:
        if not isinstance(title, str) or not title.strip():
            return ToolResult.ok(
                data={'created': False, 'reason': 'empty_title'},
                announcement='Cannot create task: title is empty.',
            )
        due = _parse_iso_utc(due_at) if due_at is not None else None
        if due_at is not None and due is None:
            return ToolResult.ok(
                data={'created': False, 'reason': 'unparseable_due_at'},
                announcement=f'Cannot create task: {due_at!r} is not a valid ISO datetime.',
            )
        task = tasks_repository.create(user_id=user_id, title=title.strip(), due_at=due, priority=priority, source=source)
        return ToolResult.ok(
            data={'created': True, 'task_id': task.id, 'title': task.title},
            announcement=f'Task added: {task.title}',
        )

    def mark_task_done(*, user_id: str, query: str) -> ToolResult:
        task = tasks_repository.mark_done(user_id, query)
        if task is None:
            return ToolResult.ok(
                data={'matched': False, 'query': query},
                announcement=f'No matching task found for {query!r}.',
            )
        if habit_service is not None and task.completed_at is not None:
            habit_service.record_task_completion(user_id, task.completed_at)
        return ToolResult.ok(
            data={'matched': True, 'task_id': task.id, 'title': task.title},
            announcement=f'Marked done: {task.title}',
        )

    def save_user_memory(*, user_id: str, memory_type: str, key: str, value: str, confidence: float = 1.0) -> ToolResult:
        if not isinstance(value, str) or not value.strip():
            return ToolResult.ok(
                data={'saved': False, 'reason': 'empty_value'},
                announcement='Cannot save memory: value is empty.',
            )
        if contains_secret(value):
            return ToolResult.ok(
                data={'saved': False, 'reason': 'forbidden_secret'},
                announcement='Refused to save: value looks like a secret (API key, token, etc.).',
            )
        memory = memories_repository.upsert(
            user_id=user_id,
            memory_type=memory_type,
            key=key,
            value=value,
            confidence=confidence,
            source='explicit',
        )
        return ToolResult.ok(
            data={'saved': True, 'memory_id': memory.id, 'memory_type': memory.memory_type, 'key': memory.key},
            announcement=f'Saved {memory.key} = {memory.value}',
        )

    def set_user_preference(*, user_id: str, key: str, value: str) -> ToolResult:
        if not isinstance(value, str) or not value.strip():
            return ToolResult.ok(
                data={'saved': False, 'reason': 'empty_value'},
                announcement='Cannot set preference: value is empty.',
            )
        memory = memories_repository.upsert(
            user_id=user_id,
            memory_type='preference',
            key=key,
            value=value,
            confidence=1.0,
            source='explicit',
        )
        return ToolResult.ok(
            data={'saved': True, 'memory_id': memory.id, 'key': memory.key, 'value': memory.value},
            announcement=f'Preference set: {memory.key} = {memory.value}',
        )

    async def create_calendar_event(
        *,
        user_id: str,
        summary: str,
        start: str,
        end: str,
        description: str | None = None,
        location: str | None = None,
    ) -> ToolResult:
        if google_calendar_service is None:
            return ToolResult.ok(
                data={'created': False, 'reason': 'calendar_not_configured'},
                announcement='Cannot create calendar event: Google Calendar is not connected.',
            )
        start_dt = _parse_iso_utc(start)
        end_dt = _parse_iso_utc(end)
        if start_dt is None or end_dt is None:
            return ToolResult.ok(
                data={'created': False, 'reason': 'unparseable_datetime'},
                announcement='Cannot create calendar event: start or end is not a valid ISO datetime.',
            )
        try:
            event_in = CalendarEventCreate(
                summary=summary,
                start=start_dt,
                end=end_dt,
                description=description,
                location=location,
            )
        except ValueError as exc:
            return ToolResult.ok(
                data={'created': False, 'reason': 'invalid_input', 'detail': str(exc)},
                announcement=f'Cannot create calendar event: {exc}.',
            )
        try:
            event = await google_calendar_service.create_event(user_id, event=event_in)
        except GoogleAuthError:
            return ToolResult.ok(
                data={'created': False, 'reason': 'auth_error'},
                announcement='Cannot create calendar event: Google Calendar auth needs re-connection.',
            )
        except HttpError as exc:
            return ToolResult.ok(
                data={'created': False, 'reason': 'api_error', 'detail': str(exc)},
                announcement='Cannot create calendar event: Google API returned an error.',
            )
        when_text = format_local_datetime(event.start, app_timezone)
        return ToolResult.ok(
            data={'created': True, 'event_id': event.id, 'summary': event.summary, 'html_link': event.html_link},
            announcement=f'Calendar event created for {when_text}: {event.summary}',
        )

    async def update_calendar_event(
        *,
        user_id: str,
        event_id: str,
        summary: str | None = None,
        start: str | None = None,
        end: str | None = None,
        description: str | None = None,
        location: str | None = None,
    ) -> ToolResult:
        if google_calendar_service is None:
            return ToolResult.ok(
                data={'updated': False, 'reason': 'calendar_not_configured'},
                announcement='Cannot update calendar event: Google Calendar is not connected.',
            )
        if not event_id or not event_id.strip():
            return ToolResult.ok(
                data={'updated': False, 'reason': 'empty_event_id'},
                announcement='Cannot update calendar event: event_id is required.',
            )
        start_dt = _parse_iso_utc(start) if start is not None else None
        end_dt = _parse_iso_utc(end) if end is not None else None
        if start is not None and start_dt is None:
            return ToolResult.ok(
                data={'updated': False, 'reason': 'unparseable_start'},
                announcement=f'Cannot update calendar event: {start!r} is not a valid ISO datetime.',
            )
        if end is not None and end_dt is None:
            return ToolResult.ok(
                data={'updated': False, 'reason': 'unparseable_end'},
                announcement=f'Cannot update calendar event: {end!r} is not a valid ISO datetime.',
            )
        try:
            patch_in = CalendarEventUpdate(
                summary=summary,
                start=start_dt,
                end=end_dt,
                description=description,
                location=location,
            )
        except ValueError as exc:
            return ToolResult.ok(
                data={'updated': False, 'reason': 'invalid_input', 'detail': str(exc)},
                announcement=f'Cannot update calendar event: {exc}.',
            )
        try:
            event = await google_calendar_service.update_event(user_id, event_id=event_id, patch=patch_in)
        except GoogleAuthError:
            return ToolResult.ok(
                data={'updated': False, 'reason': 'auth_error'},
                announcement='Cannot update calendar event: Google Calendar auth needs re-connection.',
            )
        except HttpError as exc:
            return ToolResult.ok(
                data={'updated': False, 'reason': 'api_error', 'detail': str(exc)},
                announcement='Cannot update calendar event: Google API returned an error.',
            )
        when_text = format_local_datetime(event.start, app_timezone)
        return ToolResult.ok(
            data={'updated': True, 'event_id': event.id, 'summary': event.summary, 'html_link': event.html_link},
            announcement=f'Calendar event updated for {when_text}: {event.summary}',
        )

    async def create_contact(
        *,
        user_id: str,
        name: str,
        email: str | None = None,
        phone: str | None = None,
    ) -> ToolResult:
        if google_people_service is None:
            return ToolResult.ok(
                data={'created': False, 'reason': 'people_not_configured'},
                announcement='Cannot create contact: Google Contacts is not connected.',
            )
        try:
            contact = await google_people_service.create_contact(user_id, name=name, email=email, phone=phone)
        except ValueError as exc:
            return ToolResult.ok(
                data={'created': False, 'reason': 'invalid_input', 'detail': str(exc)},
                announcement=f'Cannot create contact: {exc}.',
            )
        except GoogleAuthError:
            return ToolResult.ok(
                data={'created': False, 'reason': 'auth_error'},
                announcement='Cannot create contact: Google auth needs re-connection.',
            )
        except HttpError as exc:
            return ToolResult.ok(
                data={'created': False, 'reason': 'api_error', 'detail': str(exc)},
                announcement='Cannot create contact: Google API returned an error.',
            )
        return ToolResult.ok(
            data={'created': True, 'resource_name': contact.resource_name, 'display_name': contact.display_name},
            announcement=f'Contact created: {contact.display_name}',
        )

    return [
        (
            create_reminder,
            {
                'name': 'create_reminder',
                'description': 'Create a reminder for the user. body is the reminder text, next_fire_at is an ISO 8601 UTC datetime in the future, recurrence is an optional canonical RRULE string (FREQ=DAILY|WEEKLY).',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'body': {'type': 'string', 'description': 'Reminder text.'},
                        'next_fire_at': {'type': 'string', 'description': 'ISO 8601 UTC datetime (must be in the future).'},
                        'recurrence': {'type': 'string', 'description': 'Optional RRULE for recurring reminders.'},
                    },
                    'required': ['body', 'next_fire_at'],
                },
            },
        ),
        (
            create_task,
            {
                'name': 'create_task',
                'description': "Create a task in the user's Nexus-internal task list. title is required; due_at is an optional ISO datetime; priority defaults to 0.",
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'title': {'type': 'string', 'description': 'Task title.'},
                        'due_at': {'type': 'string', 'description': 'Optional ISO 8601 UTC due-by datetime.'},
                        'priority': {'type': 'integer', 'description': 'Priority (higher = more urgent).', 'default': 0},
                    },
                    'required': ['title'],
                },
            },
        ),
        (
            mark_task_done,
            {
                'name': 'mark_task_done',
                'description': "Mark one of the user's pending tasks as done by title fragment or task id.",
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string', 'description': 'Task title fragment or exact task id.'},
                    },
                    'required': ['query'],
                },
            },
        ),
        (
            save_user_memory,
            {
                'name': 'save_user_memory',
                'description': "Save an explicit memory for the user (fact, preference, etc.). Refuses values that look like secrets (API keys, tokens).",
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'memory_type': {'type': 'string', 'description': 'Memory category (fact, preference, etc.).'},
                        'key': {'type': 'string', 'description': 'Slug-style key.'},
                        'value': {'type': 'string', 'description': 'Memory value.'},
                    },
                    'required': ['memory_type', 'key', 'value'],
                },
            },
        ),
        (
            set_user_preference,
            {
                'name': 'set_user_preference',
                'description': "Convenience wrapper that saves a memory with memory_type='preference' (e.g. reminder_time_preference=morning).",
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'key': {'type': 'string', 'description': 'Preference key.'},
                        'value': {'type': 'string', 'description': 'Preference value.'},
                    },
                    'required': ['key', 'value'],
                },
            },
        ),
        (
            create_calendar_event,
            {
                'name': 'create_calendar_event',
                'description': "Create an event on the user's primary Google Calendar. start and end are ISO 8601 UTC datetimes. summary is required.",
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'summary': {'type': 'string', 'description': 'Event title.'},
                        'start': {'type': 'string', 'description': 'ISO 8601 UTC start datetime.'},
                        'end': {'type': 'string', 'description': 'ISO 8601 UTC end datetime.'},
                        'description': {'type': 'string', 'description': 'Optional event description.'},
                        'location': {'type': 'string', 'description': 'Optional event location.'},
                    },
                    'required': ['summary', 'start', 'end'],
                },
            },
        ),
        (
            update_calendar_event,
            {
                'name': 'update_calendar_event',
                'description': "Update an existing event on the user's primary Google Calendar. event_id is required; at least one other field must be provided.",
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'event_id': {'type': 'string', 'description': 'Google Calendar event id.'},
                        'summary': {'type': 'string', 'description': 'New event title.'},
                        'start': {'type': 'string', 'description': 'New ISO 8601 UTC start datetime.'},
                        'end': {'type': 'string', 'description': 'New ISO 8601 UTC end datetime.'},
                        'description': {'type': 'string', 'description': 'New event description.'},
                        'location': {'type': 'string', 'description': 'New event location.'},
                    },
                    'required': ['event_id'],
                },
            },
        ),
        (
            create_contact,
            {
                'name': 'create_contact',
                'description': "Create a contact in the user's Google Contacts. name is required; email and phone optional.",
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string', 'description': 'Display name (will be used as givenName).'},
                        'email': {'type': 'string', 'description': 'Optional email address.'},
                        'phone': {'type': 'string', 'description': 'Optional phone number.'},
                    },
                    'required': ['name'],
                },
            },
        ),
    ]


def register_auto_write_tools(registry: ToolRegistry, **deps: Any) -> list[ToolSpec]:
    """Register all five V3.3 auto-write tools into the given registry."""
    specs = []
    for fn, meta in make_auto_write_tools(**deps):
        specs.append(registry.register(fn, **meta))
    return specs
