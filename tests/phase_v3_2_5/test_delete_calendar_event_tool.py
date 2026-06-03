"""V3.2.5.3 destructive — tool-level tests for the real delete_calendar_event.

The tool is approval-gated (V3.4 invariant: any name matching delete_/forget_/
send_/disconnect_ must have requires_approval=True). This file tests the
post-approval implementation: when the V3.5 dispatcher invokes the tool fn
after the user taps approve, the fn calls GoogleCalendarService.delete_event.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from googleapiclient.errors import HttpError

from services.destructive_tools import make_destructive_tools
from services.google_auth_service import GoogleAuthError
from services.google_calendar_service import GoogleCalendarService


def _build_tools(*, calendar_service=None):
    """Build the destructive tools dict, calendar service injectable."""
    reminders_repo = MagicMock()
    tasks_repo = MagicMock()
    memories_repo = MagicMock()
    telos_service = MagicMock()

    async def fake_disconnect(user_id):
        return None

    return {fn.__name__: fn for fn, _meta in make_destructive_tools(
        reminders_repository=reminders_repo,
        tasks_repository=tasks_repo,
        memories_repository=memories_repo,
        telos_service=telos_service,
        google_disconnect=fake_disconnect,
        google_calendar_service=calendar_service,
    )}


@pytest.mark.asyncio
async def test_delete_calendar_event_calls_service_with_event_id():
    fake_service = MagicMock(spec=GoogleCalendarService)
    fake_service.delete_event = AsyncMock(return_value=None)

    tools = _build_tools(calendar_service=fake_service)
    result = await tools['delete_calendar_event'](user_id='user-1', event_id='evt-abc')

    assert result.success is True
    assert result.data['deleted'] is True
    assert result.data['event_id'] == 'evt-abc'
    assert result.announcement and 'deleted' in result.announcement.lower()
    fake_service.delete_event.assert_awaited_once_with('user-1', event_id='evt-abc')


@pytest.mark.asyncio
async def test_delete_calendar_event_returns_calendar_not_configured_when_service_missing():
    tools = _build_tools(calendar_service=None)
    result = await tools['delete_calendar_event'](user_id='user-1', event_id='evt-abc')

    assert result.success is True
    assert result.data['deleted'] is False
    assert result.data['reason'] == 'calendar_not_configured'
    assert 'not connected' in result.announcement.lower()


@pytest.mark.asyncio
async def test_delete_calendar_event_rejects_empty_event_id():
    fake_service = MagicMock(spec=GoogleCalendarService)
    fake_service.delete_event = AsyncMock()

    tools = _build_tools(calendar_service=fake_service)
    result = await tools['delete_calendar_event'](user_id='user-1', event_id='   ')

    assert result.success is True
    assert result.data['deleted'] is False
    assert result.data['reason'] == 'empty_event_id'
    fake_service.delete_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_calendar_event_handles_google_auth_error():
    fake_service = MagicMock(spec=GoogleCalendarService)
    fake_service.delete_event = AsyncMock(side_effect=GoogleAuthError('expired'))

    tools = _build_tools(calendar_service=fake_service)
    result = await tools['delete_calendar_event'](user_id='user-1', event_id='evt-abc')

    assert result.success is True
    assert result.data['deleted'] is False
    assert result.data['reason'] == 'auth_error'
    assert 'reconnect' in result.announcement.lower() or 'auth' in result.announcement.lower()


@pytest.mark.asyncio
async def test_delete_calendar_event_handles_http_error():
    fake_service = MagicMock(spec=GoogleCalendarService)
    http_error = HttpError(resp=MagicMock(status=503), content=b'{}')
    fake_service.delete_event = AsyncMock(side_effect=http_error)

    tools = _build_tools(calendar_service=fake_service)
    result = await tools['delete_calendar_event'](user_id='user-1', event_id='evt-abc')

    assert result.success is True
    assert result.data['deleted'] is False
    assert result.data['reason'] == 'api_error'


@pytest.mark.asyncio
async def test_delete_calendar_event_passes_user_id_through():
    """Per-user isolation invariant — User A delete must request User A creds."""
    fake_service = MagicMock(spec=GoogleCalendarService)
    fake_service.delete_event = AsyncMock(return_value=None)

    tools = _build_tools(calendar_service=fake_service)
    await tools['delete_calendar_event'](user_id='user-A', event_id='evt-1')

    fake_service.delete_event.assert_awaited_once_with('user-A', event_id='evt-1')
