"""V3.5 union ToolRegistry factory.

Builds a single ToolRegistry containing every V3.2/V3.3/V3.4/V3.8 tool —
real and deferred. The dispatcher is wired against this registry so:
  - The destructive-prefix safety invariant holds across the union
  - Gemini gets one unified tool catalog
  - Future V3.x tools register here and pick up the same gates

Total registered tools: 42.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from sqlalchemy import select

from config import get_settings
from db import run_with_retry, session_scope
from models import Reminder

# Dedup window for contact reminders — if the same user creates a reminder
# for the same contact with the same normalized body within this window,
# the second create returns the first reminder's id instead of inserting
# a duplicate row. Catches LLM hiccups, double-taps, and rapid-fire tests.
_REMINDER_DEDUP_WINDOW_SEC = 2.0


def _normalize_reminder_body(body: str) -> str:
    return ' '.join((body or '').lower().split())
from services.auto_write_tools import register_auto_write_tools
from services.auto_write_tools_stubs import register_google_write_stubs
from services.destructive_tools import register_destructive_tools
from services.destructive_tools_stubs import register_google_destructive_stubs
from services.file_system_tools import register_file_system_tools
from services.google_auth_service import GoogleAuthService
from services.google_calendar_service import GoogleCalendarService
from services.google_people_service import GooglePeopleService
from services.news_service import NewsService
from services.weather_service import WeatherService
from services.wikipedia_service import WikipediaService
from services.google_tasks_service import GoogleTasksService
from services.read_tools import make_check_freebusy, make_get_news_headlines, make_get_weather, make_list_calendar_events, make_list_google_tasks, make_lookup_contact, make_lookup_wikipedia, register_read_tools
from services.telos_onboarding_tools import register_telos_onboarding_tools
from services.tool_registry import ToolRegistry


_PARAMS_CALENDAR: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'time_min': {'type': 'string', 'description': 'ISO 8601 lower bound for the window.'},
        'time_max': {'type': 'string', 'description': 'ISO 8601 upper bound for the window.'},
        'max_results': {'type': 'integer', 'description': 'Maximum number of events to return.', 'default': 10},
    },
    'required': ['time_min', 'time_max'],
}
_PARAMS_FREEBUSY: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'time_min': {'type': 'string', 'description': 'ISO 8601 lower bound for the window.'},
        'time_max': {'type': 'string', 'description': 'ISO 8601 upper bound for the window.'},
    },
    'required': ['time_min', 'time_max'],
}
_PARAMS_GTASKS: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'max_results': {'type': 'integer', 'description': 'Maximum number of tasks to return.', 'default': 20},
    },
    'required': [],
}
_PARAMS_CONTACTS: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'query': {'type': 'string', 'description': 'Name, email, or phone fragment to search.'},
    },
    'required': ['query'],
}
_PARAMS_CONTACT_ALIAS_QUERY: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'query': {'type': 'string', 'description': 'Alias or free-form contact phrase to resolve.'},
    },
    'required': ['query'],
}
_PARAMS_CONTACT_REMINDER: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'alias': {'type': 'string', 'description': 'Existing local contact alias to notify.'},
        'body': {'type': 'string', 'description': 'Reminder text to send to that contact.'},
        'fire_at': {'type': 'string', 'description': 'ISO 8601 UTC datetime when the reminder should fire.'},
        'channel': {'type': 'string', 'description': "Optional delivery channel override ('whatsapp' or 'sms')."},
    },
    'required': ['alias', 'body', 'fire_at'],
}

_PARAMS_WIKIPEDIA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'topic': {'type': 'string', 'description': 'Topic to look up on Wikipedia.'},
    },
    'required': ['topic'],
}
_PARAMS_WEATHER: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'location': {'type': 'string', 'description': 'City, region, or "City, Country" string.'},
    },
    'required': ['location'],
}
_PARAMS_NEWS: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'max_results': {'type': 'integer', 'description': 'Number of headlines (1-20).', 'default': 5},
    },
    'required': [],
}


def _register_google_read_tools(
    registry: ToolRegistry,
    *,
    users_repository: Any,
    google_calendar_service: GoogleCalendarService | None,
    google_tasks_service: GoogleTasksService | None,
    google_people_service: GooglePeopleService | None,
) -> None:
    calendar_service = google_calendar_service
    if calendar_service is None:
        calendar_service = GoogleCalendarService(
            GoogleAuthService(get_settings(), users_repository)
        )
    tasks_service = google_tasks_service
    if tasks_service is None:
        tasks_service = GoogleTasksService(
            GoogleAuthService(get_settings(), users_repository)
        )
    people_service = google_people_service
    if people_service is None:
        people_service = GooglePeopleService(
            GoogleAuthService(get_settings(), users_repository)
        )

    registry.register(
        make_list_calendar_events(calendar_service),
        name='list_calendar_events',
        description='Read upcoming Google Calendar events for the user.',
        parameters=_PARAMS_CALENDAR,
    )
    registry.register(
        make_check_freebusy(calendar_service),
        name='check_freebusy',
        description='Check whether the user is busy in a given time window via Google Calendar.',
        parameters=_PARAMS_FREEBUSY,
    )
    registry.register(
        make_list_google_tasks(tasks_service),
        name='list_google_tasks',
        description='Read pending Google Tasks for the user (read-only).',
        parameters=_PARAMS_GTASKS,
    )
    registry.register(
        make_lookup_contact(people_service),
        name='lookup_contact',
        description='Look up a contact in the user\'s Google Contacts by name, email, or phone fragment.',
        parameters=_PARAMS_CONTACTS,
    )


def _register_content_tools(
    registry: ToolRegistry,
    *,
    wiki_service: WikipediaService,
    weather_service: WeatherService,
    news_service: NewsService,
) -> None:
    registry.register(
        make_lookup_wikipedia(wiki_service),
        name='lookup_wikipedia',
        description='Look up a Wikipedia article summary by topic. Returns the lead paragraph plus URL.',
        parameters=_PARAMS_WIKIPEDIA,
    )
    registry.register(
        make_get_weather(weather_service),
        name='get_weather',
        description="Get current weather for a location (city, region, or 'City, Country').",
        parameters=_PARAMS_WEATHER,
    )
    registry.register(
        make_get_news_headlines(news_service),
        name='get_news_headlines',
        description='Get top news headlines from Google News (up to 20).',
        parameters=_PARAMS_NEWS,
    )


def build_dispatcher_registry(
    *,
    reminders_repository,
    tasks_repository,
    memories_repository,
    emails_repository,
    approvals_repository,
    telos_service,
    scheduler: Any | None,
    habit_service: Any | None,
    google_disconnect: Callable[[str], None],
    google_calendar_service: GoogleCalendarService | None = None,
    google_tasks_service: GoogleTasksService | None = None,
    google_people_service: GooglePeopleService | None = None,
    app_timezone: str,
    onboarding_repository,
    users_repository,
) -> ToolRegistry:
    """Construct the dispatcher's union registry. Every tool registered
    here is callable; deferred tools return their deferral envelope
    rather than fake results.

    V3.2.5.1 promoted `list_calendar_events` from a deferred stub to a
    real Google Calendar read tool; the other three Google read tools
    remain deferred here.

    V3.8 added the four onboarding tools (`start_telos_onboarding`,
    `answer_telos_question`, `view_my_telos`, `cancel_telos_onboarding`)
    — `onboarding_repository` + `users_repository` are required. Test
    fixtures need to provide both so the V3.7 source-inspection
    invariant (TOOL_STAGE_MESSAGES coverage) checks against the full
    42-tool roster. Drift either way fails the invariant test, which
    is the test working as designed.
    """
    registry = ToolRegistry()

    register_read_tools(
        registry,
        reminders_repository=reminders_repository,
        tasks_repository=tasks_repository,
        memories_repository=memories_repository,
        emails_repository=emails_repository,
        approvals_repository=approvals_repository,
        telos_service=telos_service,
        app_timezone=app_timezone,
    )
    _register_google_read_tools(
        registry,
        users_repository=users_repository,
        google_calendar_service=google_calendar_service,
        google_tasks_service=google_tasks_service,
        google_people_service=google_people_service,
    )
    _register_content_tools(
        registry,
        wiki_service=WikipediaService(),
        weather_service=WeatherService(),
        news_service=NewsService(),
    )
    register_file_system_tools(registry)

    register_auto_write_tools(
        registry,
        reminders_repository=reminders_repository,
        tasks_repository=tasks_repository,
        memories_repository=memories_repository,
        google_calendar_service=google_calendar_service,
        google_people_service=google_people_service,
        scheduler=scheduler,
        habit_service=habit_service,
        app_timezone=app_timezone,
    )
    register_google_write_stubs(registry)

    register_destructive_tools(
        registry,
        reminders_repository=reminders_repository,
        tasks_repository=tasks_repository,
        memories_repository=memories_repository,
        telos_service=telos_service,
        google_disconnect=google_disconnect,
        google_calendar_service=google_calendar_service,
        scheduler=scheduler,
    )
    register_google_destructive_stubs(registry)

    register_telos_onboarding_tools(
        registry,
        telos_service=telos_service,
        onboarding_repository=onboarding_repository,
        users_repository=users_repository,
    )

    return registry
