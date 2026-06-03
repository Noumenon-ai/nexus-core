"""V3.2.5.4 — list_google_tasks tool-level tests."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.google_auth_service import GoogleAuthError
from services.google_tasks_service import GoogleTasksService
from services.google_types import GoogleTask
from services.read_tools import make_list_google_tasks


@pytest.mark.asyncio
async def test_list_google_tasks_returns_real_tasks():
    fake_service = MagicMock(spec=GoogleTasksService)
    fake_service.list_tasks = AsyncMock(return_value=[
        GoogleTask(id='t1', title='pay bill', status='needsAction', due=datetime(2026, 6, 1, tzinfo=timezone.utc), notes='urgent'),
        GoogleTask(id='t2', title='walk dog', status='needsAction'),
    ])
    tool = make_list_google_tasks(fake_service)
    result = await tool(user_id='user-1')
    assert result.success is True
    assert result.data['count'] == 2
    assert result.data['tasks'][0]['title'] == 'pay bill'


@pytest.mark.asyncio
async def test_list_google_tasks_passes_user_id():
    fake_service = MagicMock(spec=GoogleTasksService)
    fake_service.list_tasks = AsyncMock(return_value=[])
    tool = make_list_google_tasks(fake_service)
    await tool(user_id='user-A')
    fake_service.list_tasks.assert_awaited_once_with('user-A', max_results=20)


@pytest.mark.asyncio
async def test_list_google_tasks_handles_auth_error():
    fake_service = MagicMock(spec=GoogleTasksService)
    fake_service.list_tasks = AsyncMock(side_effect=GoogleAuthError('expired'))
    tool = make_list_google_tasks(fake_service)
    result = await tool(user_id='user-1')
    assert result.success is False
    assert 'reconnect' in result.error.lower() or 'expired' in result.error.lower()


@pytest.mark.asyncio
async def test_list_google_tasks_returns_empty_list():
    fake_service = MagicMock(spec=GoogleTasksService)
    fake_service.list_tasks = AsyncMock(return_value=[])
    tool = make_list_google_tasks(fake_service)
    result = await tool(user_id='user-1')
    assert result.success is True
    assert result.data['count'] == 0
    assert result.data['tasks'] == []
