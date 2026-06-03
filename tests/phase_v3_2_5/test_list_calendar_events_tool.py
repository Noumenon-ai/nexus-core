from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from googleapiclient.errors import HttpError

from services.google_auth_service import GoogleAuthError
from services.google_types import CalendarEvent
from services.read_tools import make_list_calendar_events
from services.tool_registry import ToolRegistry


def _http_error(status: int) -> HttpError:
    return HttpError(resp=Mock(status=status), content=b'{}')


def _user(container, telegram_id: int):
    return container.users_repository.get_or_create(telegram_id)


@pytest.fixture
def calendar_tool():
    service = Mock()
    service.list_events = AsyncMock(return_value=[])
    registry = ToolRegistry()
    tool_fn = make_list_calendar_events(service)
    spec = registry.register(
        tool_fn,
        name='list_calendar_events',
        description='Read upcoming Google Calendar events for the user.',
        parameters={
            'type': 'object',
            'properties': {
                'time_min': {'type': 'string'},
                'time_max': {'type': 'string'},
                'max_results': {'type': 'integer', 'default': 10},
            },
            'required': ['time_min', 'time_max'],
        },
    )
    return service, spec


@pytest.mark.asyncio
async def test_list_calendar_events_tool_uses_real_service_not_stub(calendar_tool):
    service, spec = calendar_tool

    result = await spec.fn(
        user_id='user-123',
        time_min='2026-05-06T09:00:00+00:00',
        time_max='2026-05-06T10:00:00+00:00',
    )

    service.list_events.assert_awaited_once()
    assert result.success is True
    assert result.data == {'events': [], 'count': 0}


@pytest.mark.asyncio
async def test_list_calendar_events_tool_returns_events_in_data(calendar_tool):
    service, spec = calendar_tool
    service.list_events.return_value = [
        CalendarEvent(
            id='evt-1',
            summary='Standup',
            start=datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc),
            end=datetime(2026, 5, 6, 9, 30, tzinfo=timezone.utc),
            description='Daily sync',
            location='Zoom',
            html_link='https://calendar.google.com/event?eid=1',
        ),
    ]

    result = await spec.fn(
        user_id='user-123',
        time_min='2026-05-06T09:00:00+00:00',
        time_max='2026-05-06T10:00:00+00:00',
    )

    assert result.success is True
    assert result.data == {
        'events': [{
            'id': 'evt-1',
            'summary': 'Standup',
            'start': '2026-05-06T09:00:00+00:00',
            'end': '2026-05-06T09:30:00+00:00',
            'description': 'Daily sync',
            'location': 'Zoom',
            'html_link': 'https://calendar.google.com/event?eid=1',
        }],
        'count': 1,
    }


@pytest.mark.asyncio
async def test_list_calendar_events_tool_handles_google_auth_error(calendar_tool):
    service, spec = calendar_tool
    service.list_events.side_effect = GoogleAuthError('expired')

    result = await spec.fn(
        user_id='user-123',
        time_min='2026-05-06T09:00:00+00:00',
        time_max='2026-05-06T10:00:00+00:00',
    )

    assert result.success is False
    assert result.error == "Calendar access expired. Say 'reconnect google' to re-auth."


@pytest.mark.asyncio
async def test_list_calendar_events_tool_passes_user_id_through(calendar_tool, container):
    service, spec = calendar_tool
    user = _user(container, 111)

    await spec.fn(
        user_id=user.id,
        time_min='2026-05-06T09:00:00+00:00',
        time_max='2026-05-06T10:00:00+00:00',
    )

    assert service.list_events.await_args.args[0] == user.id


@pytest.mark.asyncio
async def test_list_calendar_events_tool_parses_relative_time_range(calendar_tool):
    service, spec = calendar_tool

    await spec.fn(
        user_id='user-123',
        time_min='2026-05-06T09:00:00Z',
        time_max='2026-05-06T10:00:00Z',
        max_results=5,
    )

    assert service.list_events.await_args.kwargs == {
        'time_min': datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc),
        'time_max': datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc),
        'max_results': 5,
    }


@pytest.mark.asyncio
async def test_list_calendar_events_tool_maps_http_errors(calendar_tool):
    service, spec = calendar_tool
    service.list_events.side_effect = _http_error(503)

    result = await spec.fn(
        user_id='user-123',
        time_min='2026-05-06T09:00:00+00:00',
        time_max='2026-05-06T10:00:00+00:00',
    )

    assert result.success is False
    assert result.error == 'Calendar service error (status=503). Try again in a moment.'
