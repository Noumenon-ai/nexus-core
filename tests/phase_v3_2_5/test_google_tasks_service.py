"""V3.2.5.4 — GoogleTasksService unit tests."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError

from services.google_auth_service import GoogleAuthError
from services.google_tasks_service import GoogleTasksService
from services.google_types import GoogleTask


def _http_error(status):
    return HttpError(resp=MagicMock(status=status), content=b'{}')


def _tasks_stack(response):
    request = MagicMock()
    request.execute = MagicMock(return_value=response)
    tasks_resource = MagicMock()
    tasks_resource.list.return_value = request
    service = MagicMock()
    service.tasks.return_value = tasks_resource
    return service, tasks_resource, request


@pytest.mark.asyncio
async def test_list_tasks_passes_user_credentials():
    creds = object()
    auth = MagicMock()
    auth.get_credentials = MagicMock(return_value=creds)
    svc = GoogleTasksService(auth)
    api_service, _, _ = _tasks_stack({'items': []})
    with patch('services.google_tasks_service.discovery.build', return_value=api_service):
        await svc.list_tasks('user-1')
    auth.get_credentials.assert_called_with('user-1')


@pytest.mark.asyncio
async def test_list_tasks_calls_tasks_v1_default_tasklist():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=object())
    svc = GoogleTasksService(auth)
    api_service, tasks_resource, _ = _tasks_stack({'items': []})
    with patch('services.google_tasks_service.discovery.build', return_value=api_service) as build_mock:
        await svc.list_tasks('user-1')
    build_mock.assert_called_with('tasks', 'v1', credentials=svc.auth.get_credentials.return_value, cache_discovery=False)
    list_calls = [c for c in tasks_resource.list.call_args_list if c.kwargs]
    assert any(c.kwargs.get('tasklist') == '@default' for c in list_calls)


@pytest.mark.asyncio
async def test_list_tasks_parses_response_correctly():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=object())
    svc = GoogleTasksService(auth)
    api_service, _, _ = _tasks_stack({
        'items': [
            {'id': 't1', 'title': 'pay bill', 'status': 'needsAction', 'due': '2026-06-01T00:00:00Z', 'notes': 'urgent'},
            {'id': 't2', 'title': 'walk dog', 'status': 'needsAction'},
        ]
    })
    with patch('services.google_tasks_service.discovery.build', return_value=api_service):
        tasks = await svc.list_tasks('user-1')
    assert len(tasks) == 2
    assert isinstance(tasks[0], GoogleTask)
    assert tasks[0].id == 't1'
    assert tasks[0].title == 'pay bill'
    assert tasks[0].notes == 'urgent'
    assert tasks[0].due == datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    assert tasks[1].due is None
    assert tasks[1].notes is None


@pytest.mark.asyncio
async def test_list_tasks_returns_empty_list_when_no_items():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=object())
    svc = GoogleTasksService(auth)
    api_service, _, _ = _tasks_stack({})  # no 'items' key
    with patch('services.google_tasks_service.discovery.build', return_value=api_service):
        tasks = await svc.list_tasks('user-1')
    assert tasks == []


@pytest.mark.asyncio
async def test_list_tasks_no_credentials_raises_auth_error():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=None)
    svc = GoogleTasksService(auth)
    with pytest.raises(GoogleAuthError):
        await svc.list_tasks('user-1')


@pytest.mark.asyncio
async def test_list_tasks_maps_401_to_google_auth_error():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=object())
    svc = GoogleTasksService(auth)
    api_service = MagicMock()
    api_service.tasks().list().execute.side_effect = _http_error(401)
    with patch('services.google_tasks_service.discovery.build', return_value=api_service):
        with pytest.raises(GoogleAuthError):
            await svc.list_tasks('user-1')


@pytest.mark.asyncio
async def test_list_tasks_handles_429_with_backoff():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=object())
    svc = GoogleTasksService(auth)
    api_service = MagicMock()
    api_service.tasks().list().execute.side_effect = [_http_error(429), {'items': []}]
    with patch('services.google_tasks_service.discovery.build', return_value=api_service):
        with patch('services.google_tasks_service.asyncio.sleep') as sleep_mock:
            tasks = await svc.list_tasks('user-1')
    assert tasks == []
    sleep_mock.assert_called()
