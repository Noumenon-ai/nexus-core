from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from services.google_auth_service import GoogleAuthError
from services.google_types import BusyBlock
from services.read_tools import make_check_freebusy
from services.tool_registry import ToolRegistry


def _user(container, telegram_id: int):
    return container.users_repository.get_or_create(telegram_id)


@pytest.fixture
def freebusy_tool():
    service = Mock()
    service.freebusy = AsyncMock(return_value=[])
    registry = ToolRegistry()
    tool_fn = make_check_freebusy(service)
    spec = registry.register(
        tool_fn,
        name='check_freebusy',
        description='Check whether the user is busy in a given time window via Google Calendar.',
        parameters={
            'type': 'object',
            'properties': {
                'time_min': {'type': 'string'},
                'time_max': {'type': 'string'},
            },
            'required': ['time_min', 'time_max'],
        },
    )
    return service, spec


@pytest.mark.asyncio
async def test_check_freebusy_tool_uses_real_service_not_stub(freebusy_tool):
    service, spec = freebusy_tool

    result = await spec.fn(
        user_id='user-123',
        time_min='2026-05-06T09:00:00+00:00',
        time_max='2026-05-06T10:00:00+00:00',
    )

    service.freebusy.assert_awaited_once()
    assert result.success is True
    assert result.data == {'busy': [], 'count': 0}


@pytest.mark.asyncio
async def test_check_freebusy_tool_returns_busy_blocks_in_data(freebusy_tool):
    service, spec = freebusy_tool
    service.freebusy.return_value = [
        BusyBlock(
            start=datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc),
            end=datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc),
        ),
    ]

    result = await spec.fn(
        user_id='user-123',
        time_min='2026-05-06T09:00:00+00:00',
        time_max='2026-05-06T10:00:00+00:00',
    )

    assert result.success is True
    assert result.data == {
        'busy': [{
            'start': '2026-05-06T09:00:00+00:00',
            'end': '2026-05-06T10:00:00+00:00',
        }],
        'count': 1,
    }


@pytest.mark.asyncio
async def test_check_freebusy_tool_handles_google_auth_error(freebusy_tool):
    service, spec = freebusy_tool
    service.freebusy.side_effect = GoogleAuthError('expired')

    result = await spec.fn(
        user_id='user-123',
        time_min='2026-05-06T09:00:00+00:00',
        time_max='2026-05-06T10:00:00+00:00',
    )

    assert result.success is False
    assert result.error == "Calendar access expired. Say 'reconnect google' to re-auth."


@pytest.mark.asyncio
async def test_check_freebusy_tool_passes_user_id_through(freebusy_tool, container):
    service, spec = freebusy_tool
    user = _user(container, 111)

    await spec.fn(
        user_id=user.id,
        time_min='2026-05-06T09:00:00+00:00',
        time_max='2026-05-06T10:00:00+00:00',
    )

    assert service.freebusy.await_args.args[0] == user.id


@pytest.mark.asyncio
async def test_check_freebusy_tool_parses_iso8601_time_range(freebusy_tool):
    service, spec = freebusy_tool

    await spec.fn(
        user_id='user-123',
        time_min='2026-05-06T09:00:00Z',
        time_max='2026-05-06T10:00:00Z',
    )

    assert service.freebusy.await_args.kwargs == {
        'time_min': datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc),
        'time_max': datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc),
    }
