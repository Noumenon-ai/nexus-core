from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from googleapiclient.errors import HttpError

from services.google_auth_service import GoogleAuthError
from services.google_calendar_service import GoogleCalendarService
from services.google_types import BusyBlock


def _http_error(status: int) -> HttpError:
    return HttpError(resp=Mock(status=status), content=b'{}')


def _calendar_stack(response: dict):
    request = Mock()
    request.execute = Mock(return_value=response)
    events_resource = Mock()
    events_resource.list.return_value = request
    service = Mock()
    service.events.return_value = events_resource
    return service, events_resource, request


def _freebusy_stack(response: dict):
    request = Mock()
    request.execute = Mock(return_value=response)
    freebusy_resource = Mock()
    freebusy_resource.query.return_value = request
    service = Mock()
    service.freebusy.return_value = freebusy_resource
    return service, freebusy_resource, request


async def _run_to_thread(sync_fn, *args, **kwargs):
    return sync_fn(*args, **kwargs)


RUN_BLOCKING_TARGET = 'services.google_calendar_service._run_blocking_without_default_executor'


@pytest.mark.asyncio
async def test_list_events_passes_user_credentials():
    auth = Mock()
    creds = object()
    auth.get_credentials.return_value = creds
    service = GoogleCalendarService(auth)
    api_service, _events_resource, _request = _calendar_stack({'items': []})
    time_min = datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc)
    time_max = datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc)

    with patch('services.google_calendar_service.discovery.build', return_value=api_service), patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        await service.list_events('user-123', time_min=time_min, time_max=time_max)

    auth.get_credentials.assert_called_once_with('user-123')


@pytest.mark.asyncio
async def test_list_events_calls_calendar_v3_with_primary():
    auth = Mock()
    creds = object()
    auth.get_credentials.return_value = creds
    service = GoogleCalendarService(auth)
    api_service, events_resource, _request = _calendar_stack({'items': []})
    time_min = datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc)
    time_max = datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc)

    with patch('services.google_calendar_service.discovery.build', return_value=api_service) as build_mock, patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        await service.list_events('user-123', time_min=time_min, time_max=time_max)

    build_mock.assert_called_once_with('calendar', 'v3', credentials=creds, cache_discovery=False)
    events_resource.list.assert_called_once()
    assert events_resource.list.call_args.kwargs['calendarId'] == 'primary'


@pytest.mark.asyncio
async def test_list_events_passes_time_range_as_iso8601():
    auth = Mock()
    auth.get_credentials.return_value = object()
    service = GoogleCalendarService(auth)
    api_service, events_resource, _request = _calendar_stack({'items': []})
    time_min = datetime(2026, 5, 6, 9, 15, tzinfo=timezone.utc)
    time_max = datetime(2026, 5, 6, 11, 45, tzinfo=timezone.utc)

    with patch('services.google_calendar_service.discovery.build', return_value=api_service), patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        await service.list_events('user-123', time_min=time_min, time_max=time_max, max_results=7)

    assert events_resource.list.call_args.kwargs['timeMin'] == time_min.isoformat()
    assert events_resource.list.call_args.kwargs['timeMax'] == time_max.isoformat()
    assert events_resource.list.call_args.kwargs['maxResults'] == 7


@pytest.mark.asyncio
async def test_list_events_parses_timed_event_correctly():
    auth = Mock()
    auth.get_credentials.return_value = object()
    service = GoogleCalendarService(auth)
    api_service, _events_resource, _request = _calendar_stack({
        'items': [{
            'id': 'evt-1',
            'summary': 'Standup',
            'start': {'dateTime': '2026-05-06T09:30:00+00:00'},
            'end': {'dateTime': '2026-05-06T10:00:00+00:00'},
            'description': 'Daily sync',
            'location': 'Zoom',
            'htmlLink': 'https://calendar.google.com/event?eid=1',
        }],
    })

    with patch('services.google_calendar_service.discovery.build', return_value=api_service), patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        events = await service.list_events(
            'user-123',
            time_min=datetime(2026, 5, 6, 0, 0, tzinfo=timezone.utc),
            time_max=datetime(2026, 5, 7, 0, 0, tzinfo=timezone.utc),
        )

    assert len(events) == 1
    assert events[0].id == 'evt-1'
    assert events[0].summary == 'Standup'
    assert events[0].start == datetime(2026, 5, 6, 9, 30, tzinfo=timezone.utc)
    assert events[0].end == datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc)
    assert events[0].description == 'Daily sync'
    assert events[0].location == 'Zoom'
    assert events[0].html_link == 'https://calendar.google.com/event?eid=1'


@pytest.mark.asyncio
async def test_list_events_parses_all_day_event_correctly():
    auth = Mock()
    auth.get_credentials.return_value = object()
    service = GoogleCalendarService(auth)
    api_service, _events_resource, _request = _calendar_stack({
        'items': [{
            'id': 'evt-2',
            'summary': 'Offsite',
            'start': {'date': '2026-05-07'},
            'end': {'date': '2026-05-08'},
        }],
    })

    with patch('services.google_calendar_service.discovery.build', return_value=api_service), patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        events = await service.list_events(
            'user-123',
            time_min=datetime(2026, 5, 7, 0, 0, tzinfo=timezone.utc),
            time_max=datetime(2026, 5, 9, 0, 0, tzinfo=timezone.utc),
        )

    assert len(events) == 1
    assert events[0].start == datetime(2026, 5, 7, 0, 0, tzinfo=timezone.utc)
    assert events[0].end == datetime(2026, 5, 8, 0, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_list_events_returns_empty_list_when_no_items():
    auth = Mock()
    auth.get_credentials.return_value = object()
    service = GoogleCalendarService(auth)
    api_service, _events_resource, _request = _calendar_stack({})

    with patch('services.google_calendar_service.discovery.build', return_value=api_service), patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        events = await service.list_events(
            'user-123',
            time_min=datetime(2026, 5, 6, 0, 0, tzinfo=timezone.utc),
            time_max=datetime(2026, 5, 7, 0, 0, tzinfo=timezone.utc),
        )

    assert events == []


@pytest.mark.asyncio
async def test_list_events_uses_blocking_helper_for_execute():
    auth = Mock()
    auth.get_credentials.return_value = object()
    service = GoogleCalendarService(auth)
    order: list[str] = []

    request = Mock()
    request.execute = Mock(side_effect=lambda: order.append('execute') or {'items': []})
    events_resource = Mock()
    events_resource.list.return_value = request
    api_service = Mock()
    api_service.events.return_value = events_resource

    async def fake_run_blocking(sync_fn, *args, **kwargs):
        order.append('run_blocking')
        return sync_fn(*args, **kwargs)

    with patch('services.google_calendar_service.discovery.build', return_value=api_service), patch(
        RUN_BLOCKING_TARGET,
        new=AsyncMock(side_effect=fake_run_blocking),
    ) as run_blocking_mock:
        await service.list_events(
            'user-123',
            time_min=datetime(2026, 5, 6, 0, 0, tzinfo=timezone.utc),
            time_max=datetime(2026, 5, 7, 0, 0, tzinfo=timezone.utc),
        )

    assert run_blocking_mock.await_count >= 2
    assert order[-2:] == ['run_blocking', 'execute']


@pytest.mark.asyncio
async def test_with_retry_handles_429_with_backoff():
    service = GoogleCalendarService(Mock())
    sync_fn = Mock(side_effect=[_http_error(429), {'ok': True}])

    with patch(
        RUN_BLOCKING_TARGET,
        new=AsyncMock(side_effect=_run_to_thread),
    ) as run_blocking_mock, patch(
        'services.google_calendar_service.asyncio.sleep',
        new=AsyncMock(),
    ) as sleep_mock:
        result = await service._with_retry(sync_fn)

    assert result == {'ok': True}
    assert run_blocking_mock.await_count == 2
    sleep_mock.assert_awaited_once_with(1.0)


@pytest.mark.asyncio
async def test_with_retry_gives_up_after_max_retries():
    service = GoogleCalendarService(Mock())
    sync_fn = Mock(side_effect=[_http_error(500), _http_error(500), _http_error(500)])

    with patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ), patch(
        'services.google_calendar_service.asyncio.sleep',
        new=AsyncMock(),
    ) as sleep_mock:
        with pytest.raises(HttpError):
            await service._with_retry(sync_fn, max_retries=3)

    assert sync_fn.call_count == 3
    assert sleep_mock.await_count == 2


@pytest.mark.asyncio
async def test_with_retry_propagates_4xx_immediately():
    service = GoogleCalendarService(Mock())
    sync_fn = Mock(side_effect=_http_error(400))

    with patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ), patch(
        'services.google_calendar_service.asyncio.sleep',
        new=AsyncMock(),
    ) as sleep_mock:
        with pytest.raises(HttpError):
            await service._with_retry(sync_fn)

    assert sync_fn.call_count == 1
    sleep_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_with_retry_maps_401_to_google_auth_error():
    service = GoogleCalendarService(Mock())
    sync_fn = Mock(side_effect=_http_error(401))

    with patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        with pytest.raises(GoogleAuthError):
            await service._with_retry(sync_fn)


@pytest.mark.asyncio
async def test_with_retry_maps_403_to_google_auth_error():
    service = GoogleCalendarService(Mock())
    sync_fn = Mock(side_effect=_http_error(403))

    with patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        with pytest.raises(GoogleAuthError):
            await service._with_retry(sync_fn)


@pytest.mark.asyncio
async def test_freebusy_passes_user_credentials():
    auth = Mock()
    creds = object()
    auth.get_credentials.return_value = creds
    service = GoogleCalendarService(auth)
    api_service, _freebusy_resource, _request = _freebusy_stack({'calendars': {'primary': {'busy': []}}})
    time_min = datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc)
    time_max = datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc)

    with patch('services.google_calendar_service.discovery.build', return_value=api_service), patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        await service.freebusy('user-123', time_min=time_min, time_max=time_max)

    auth.get_credentials.assert_called_once_with('user-123')


@pytest.mark.asyncio
async def test_freebusy_calls_calendar_v3_freebusy_query():
    auth = Mock()
    creds = object()
    auth.get_credentials.return_value = creds
    service = GoogleCalendarService(auth)
    api_service, freebusy_resource, _request = _freebusy_stack({'calendars': {'primary': {'busy': []}}})
    time_min = datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc)
    time_max = datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc)

    with patch('services.google_calendar_service.discovery.build', return_value=api_service) as build_mock, patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        await service.freebusy('user-123', time_min=time_min, time_max=time_max)

    build_mock.assert_called_once_with('calendar', 'v3', credentials=creds, cache_discovery=False)
    freebusy_resource.query.assert_called_once()


@pytest.mark.asyncio
async def test_freebusy_passes_time_range_and_primary_calendar_in_body():
    auth = Mock()
    auth.get_credentials.return_value = object()
    service = GoogleCalendarService(auth)
    api_service, freebusy_resource, _request = _freebusy_stack({'calendars': {'primary': {'busy': []}}})
    time_min = datetime(2026, 5, 6, 9, 15, tzinfo=timezone.utc)
    time_max = datetime(2026, 5, 6, 11, 45, tzinfo=timezone.utc)

    with patch('services.google_calendar_service.discovery.build', return_value=api_service), patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        await service.freebusy('user-123', time_min=time_min, time_max=time_max)

    assert freebusy_resource.query.call_args.kwargs['body'] == {
        'timeMin': time_min.isoformat(),
        'timeMax': time_max.isoformat(),
        'items': [{'id': 'primary'}],
    }


@pytest.mark.asyncio
async def test_freebusy_parses_busy_blocks_correctly():
    auth = Mock()
    auth.get_credentials.return_value = object()
    service = GoogleCalendarService(auth)
    api_service, _freebusy_resource, _request = _freebusy_stack({
        'calendars': {
            'primary': {
                'busy': [{
                    'start': '2026-05-06T09:00:00Z',
                    'end': '2026-05-06T10:00:00Z',
                }],
            },
        },
    })

    with patch('services.google_calendar_service.discovery.build', return_value=api_service), patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        blocks = await service.freebusy(
            'user-123',
            time_min=datetime(2026, 5, 6, 0, 0, tzinfo=timezone.utc),
            time_max=datetime(2026, 5, 7, 0, 0, tzinfo=timezone.utc),
        )

    assert blocks == [
        BusyBlock(
            start=datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc),
            end=datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc),
        ),
    ]


@pytest.mark.asyncio
async def test_freebusy_returns_empty_list_when_no_busy_blocks():
    auth = Mock()
    auth.get_credentials.return_value = object()
    service = GoogleCalendarService(auth)
    api_service, _freebusy_resource, _request = _freebusy_stack({'calendars': {'primary': {'busy': []}}})

    with patch('services.google_calendar_service.discovery.build', return_value=api_service), patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        blocks = await service.freebusy(
            'user-123',
            time_min=datetime(2026, 5, 6, 0, 0, tzinfo=timezone.utc),
            time_max=datetime(2026, 5, 7, 0, 0, tzinfo=timezone.utc),
        )

    assert blocks == []


@pytest.mark.asyncio
async def test_freebusy_returns_empty_list_when_no_calendars_key():
    auth = Mock()
    auth.get_credentials.return_value = object()
    service = GoogleCalendarService(auth)
    api_service, _freebusy_resource, _request = _freebusy_stack({})

    with patch('services.google_calendar_service.discovery.build', return_value=api_service), patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        blocks = await service.freebusy(
            'user-123',
            time_min=datetime(2026, 5, 6, 0, 0, tzinfo=timezone.utc),
            time_max=datetime(2026, 5, 7, 0, 0, tzinfo=timezone.utc),
        )

    assert blocks == []


@pytest.mark.asyncio
async def test_freebusy_uses_blocking_helper_for_execute():
    auth = Mock()
    auth.get_credentials.return_value = object()
    service = GoogleCalendarService(auth)
    order: list[str] = []

    request = Mock()
    request.execute = Mock(side_effect=lambda: order.append('execute') or {'calendars': {'primary': {'busy': []}}})
    freebusy_resource = Mock()
    freebusy_resource.query.return_value = request
    api_service = Mock()
    api_service.freebusy.return_value = freebusy_resource

    async def fake_run_blocking(sync_fn, *args, **kwargs):
        order.append('run_blocking')
        return sync_fn(*args, **kwargs)

    with patch('services.google_calendar_service.discovery.build', return_value=api_service), patch(
        RUN_BLOCKING_TARGET,
        new=AsyncMock(side_effect=fake_run_blocking),
    ) as run_blocking_mock:
        await service.freebusy(
            'user-123',
            time_min=datetime(2026, 5, 6, 0, 0, tzinfo=timezone.utc),
            time_max=datetime(2026, 5, 7, 0, 0, tzinfo=timezone.utc),
        )

    assert run_blocking_mock.await_count >= 2
    assert order[-2:] == ['run_blocking', 'execute']


@pytest.mark.asyncio
async def test_freebusy_propagates_429_via_with_retry():
    auth = Mock()
    auth.get_credentials.return_value = object()
    service = GoogleCalendarService(auth)
    time_min = datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc)
    time_max = datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc)

    with patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ), patch.object(
        service,
        '_with_retry',
        new=AsyncMock(side_effect=_http_error(429)),
    ) as with_retry_mock:
        with pytest.raises(HttpError):
            await service.freebusy('user-123', time_min=time_min, time_max=time_max)

    with_retry_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_freebusy_maps_401_to_google_auth_error():
    auth = Mock()
    auth.get_credentials.return_value = object()
    service = GoogleCalendarService(auth)
    api_service, _freebusy_resource, request = _freebusy_stack({'calendars': {'primary': {'busy': []}}})
    request.execute.side_effect = _http_error(401)

    with patch('services.google_calendar_service.discovery.build', return_value=api_service), patch(
        'services.google_calendar_service.asyncio.to_thread',
        new=AsyncMock(side_effect=_run_to_thread),
    ):
        with pytest.raises(GoogleAuthError):
            await service.freebusy(
                'user-123',
                time_min=datetime(2026, 5, 6, 0, 0, tzinfo=timezone.utc),
                time_max=datetime(2026, 5, 7, 0, 0, tzinfo=timezone.utc),
            )


# ============================================================
# V3.2.5.3 — calendar writes (create_event, update_event, delete_event)
# ============================================================

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from services.google_calendar_service import GoogleCalendarService
from services.google_auth_service import GoogleAuthError
from services.google_types import (
    CalendarEvent,
    CalendarEventCreate,
    CalendarEventUpdate,
)


def _v3253_make_creds():
    """Fake credentials object — only needs to be non-None."""
    return MagicMock()


def _v3253_make_auth(creds=None):
    """Fake GoogleAuthService that returns the given creds (or None)."""
    auth = MagicMock()
    auth.get_credentials = MagicMock(return_value=creds)
    return auth


def _v3253_event_response(event_id='evt_abc', summary='Test', tz_offset=0):
    """Shape of a Google Calendar v3 event response (insert/patch return)."""
    base = datetime(2026, 6, 1, 14, 0, tzinfo=timezone(timedelta(hours=tz_offset)))
    return {
        'id': event_id,
        'summary': summary,
        'start': {'dateTime': base.isoformat()},
        'end': {'dateTime': (base + timedelta(hours=1)).isoformat()},
        'htmlLink': f'https://calendar.google.com/event?eid={event_id}',
    }


# ---------- create_event ----------

@pytest.mark.asyncio
async def test_create_event_success_returns_calendar_event():
    creds = _v3253_make_creds()
    auth = _v3253_make_auth(creds)
    svc = GoogleCalendarService(auth)

    fake_service = MagicMock()
    fake_service.events().insert().execute.return_value = _v3253_event_response('evt_new', 'NEXUS_TEST: lunch')

    with patch('services.google_calendar_service.discovery.build', return_value=fake_service):
        start = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=1)
        event_in = CalendarEventCreate(
            summary='NEXUS_TEST: lunch',
            start=start,
            end=end,
            description='auto-created by NEXUS test probe',
        )
        result = await svc.create_event('user-1', event=event_in)

    assert isinstance(result, CalendarEvent)
    assert result.id == 'evt_new'
    assert result.summary == 'NEXUS_TEST: lunch'


@pytest.mark.asyncio
async def test_create_event_no_credentials_raises_auth_error():
    auth = _v3253_make_auth(creds=None)
    svc = GoogleCalendarService(auth)

    start = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
    event_in = CalendarEventCreate(summary='x', start=start, end=start + timedelta(hours=1))

    with pytest.raises(GoogleAuthError):
        await svc.create_event('user-1', event=event_in)


@pytest.mark.asyncio
async def test_create_event_passes_correct_body_to_insert():
    creds = _v3253_make_creds()
    auth = _v3253_make_auth(creds)
    svc = GoogleCalendarService(auth)

    fake_service = MagicMock()
    fake_service.events().insert().execute.return_value = _v3253_event_response()

    with patch('services.google_calendar_service.discovery.build', return_value=fake_service):
        start = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=1)
        event_in = CalendarEventCreate(
            summary='NEXUS_TEST: meeting',
            start=start,
            end=end,
            location='Room 5',
        )
        await svc.create_event('user-1', event=event_in)

    # Find the insert() call that was made with kwargs (calendarId + body)
    insert_calls = [c for c in fake_service.events().insert.call_args_list if c.kwargs]
    assert len(insert_calls) >= 1, 'expected at least one insert() call with kwargs'
    last_call = insert_calls[-1]
    assert last_call.kwargs['calendarId'] == 'primary'
    body = last_call.kwargs['body']
    assert body['summary'] == 'NEXUS_TEST: meeting'
    assert body['location'] == 'Room 5'
    assert 'description' not in body  # not set, must be omitted
    assert body['start']['dateTime'] == start.isoformat()


# ---------- update_event ----------

@pytest.mark.asyncio
async def test_update_event_success_returns_updated_event():
    creds = _v3253_make_creds()
    auth = _v3253_make_auth(creds)
    svc = GoogleCalendarService(auth)

    fake_service = MagicMock()
    fake_service.events().patch().execute.return_value = _v3253_event_response('evt_xyz', 'NEXUS_TEST: updated')

    with patch('services.google_calendar_service.discovery.build', return_value=fake_service):
        patch_in = CalendarEventUpdate(summary='NEXUS_TEST: updated')
        result = await svc.update_event('user-1', event_id='evt_xyz', patch=patch_in)

    assert result.id == 'evt_xyz'
    assert result.summary == 'NEXUS_TEST: updated'


@pytest.mark.asyncio
async def test_update_event_empty_event_id_raises_value_error():
    auth = _v3253_make_auth(_v3253_make_creds())
    svc = GoogleCalendarService(auth)

    patch_in = CalendarEventUpdate(summary='x')
    with pytest.raises(ValueError, match='event_id must be non-empty'):
        await svc.update_event('user-1', event_id='   ', patch=patch_in)


@pytest.mark.asyncio
async def test_update_event_no_credentials_raises_auth_error():
    auth = _v3253_make_auth(creds=None)
    svc = GoogleCalendarService(auth)

    patch_in = CalendarEventUpdate(summary='x')
    with pytest.raises(GoogleAuthError):
        await svc.update_event('user-1', event_id='evt_1', patch=patch_in)


@pytest.mark.asyncio
async def test_update_event_passes_sparse_patch_body():
    creds = _v3253_make_creds()
    auth = _v3253_make_auth(creds)
    svc = GoogleCalendarService(auth)

    fake_service = MagicMock()
    fake_service.events().patch().execute.return_value = _v3253_event_response()

    with patch('services.google_calendar_service.discovery.build', return_value=fake_service):
        new_start = datetime(2026, 6, 2, 15, 0, tzinfo=timezone.utc)
        patch_in = CalendarEventUpdate(start=new_start)
        await svc.update_event('user-1', event_id='evt_1', patch=patch_in)

    patch_calls = [c for c in fake_service.events().patch.call_args_list if c.kwargs]
    assert len(patch_calls) >= 1
    last_call = patch_calls[-1]
    assert last_call.kwargs['calendarId'] == 'primary'
    assert last_call.kwargs['eventId'] == 'evt_1'
    body = last_call.kwargs['body']
    # Only start should be in the body — sparse patch
    assert 'start' in body
    assert 'summary' not in body
    assert 'end' not in body
    assert 'description' not in body
    assert 'location' not in body


# ---------- delete_event ----------

@pytest.mark.asyncio
async def test_delete_event_success_returns_none():
    creds = _v3253_make_creds()
    auth = _v3253_make_auth(creds)
    svc = GoogleCalendarService(auth)

    fake_service = MagicMock()
    fake_service.events().delete().execute.return_value = None

    with patch('services.google_calendar_service.discovery.build', return_value=fake_service):
        result = await svc.delete_event('user-1', event_id='evt_to_delete')

    assert result is None


@pytest.mark.asyncio
async def test_delete_event_calls_delete_with_correct_event_id():
    creds = _v3253_make_creds()
    auth = _v3253_make_auth(creds)
    svc = GoogleCalendarService(auth)

    fake_service = MagicMock()
    fake_service.events().delete().execute.return_value = None

    with patch('services.google_calendar_service.discovery.build', return_value=fake_service):
        await svc.delete_event('user-1', event_id='evt_target')

    delete_calls = [c for c in fake_service.events().delete.call_args_list if c.kwargs]
    assert len(delete_calls) >= 1
    last_call = delete_calls[-1]
    assert last_call.kwargs['calendarId'] == 'primary'
    assert last_call.kwargs['eventId'] == 'evt_target'


@pytest.mark.asyncio
async def test_delete_event_empty_event_id_raises_value_error():
    auth = _v3253_make_auth(_v3253_make_creds())
    svc = GoogleCalendarService(auth)

    with pytest.raises(ValueError, match='event_id must be non-empty'):
        await svc.delete_event('user-1', event_id='')


@pytest.mark.asyncio
async def test_delete_event_no_credentials_raises_auth_error():
    auth = _v3253_make_auth(creds=None)
    svc = GoogleCalendarService(auth)

    with pytest.raises(GoogleAuthError):
        await svc.delete_event('user-1', event_id='evt_1')


# ---------- per-user isolation invariant ----------

@pytest.mark.asyncio
async def test_create_event_uses_user_specific_credentials():
    """User A's create_event must request creds for user A, not bleed to user B."""
    creds_a = _v3253_make_creds()
    auth = MagicMock()
    auth.get_credentials = MagicMock(return_value=creds_a)
    svc = GoogleCalendarService(auth)

    fake_service = MagicMock()
    fake_service.events().insert().execute.return_value = _v3253_event_response()

    with patch('services.google_calendar_service.discovery.build', return_value=fake_service):
        start = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
        event_in = CalendarEventCreate(summary='x', start=start, end=start + timedelta(hours=1))
        await svc.create_event('user-A', event=event_in)

    # auth.get_credentials must have been called with 'user-A', not anything else
    auth.get_credentials.assert_called_with('user-A')
