from __future__ import annotations

import pytest

from tests.phase_10_google.helpers import (
    build_google_auth_service,
    configure_google_test_env,
    pasted_url_for,
)


_CALENDAR = 'https://www.googleapis.com/auth/calendar'
_TASKS = 'https://www.googleapis.com/auth/tasks.readonly'
_CONTACTS = 'https://www.googleapis.com/auth/contacts'


async def _complete(service, user_id):
    pending = service._pending[user_id]
    return await service.submit_pasted_url(user_id, pasted_url_for(state=pending.state))


@pytest.mark.asyncio
async def test_partial_grant_records_only_granted_scopes(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=(_CALENDAR, _CONTACTS))

    _url, _future = await service.begin_connect(user.id)
    result = await _complete(service, user.id)

    assert set(result.scopes_granted) == {_CALENDAR, _CONTACTS}
    refreshed = container.users_repository.get_by_id(user.id)
    assert refreshed.google_connected is True
    assert set((refreshed.google_scopes_granted or '').split(',')) == {_CALENDAR, _CONTACTS}


@pytest.mark.asyncio
async def test_has_scope_true_for_granted_scope(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=(_CALENDAR, _CONTACTS))
    await service.begin_connect(user.id)
    await _complete(service, user.id)

    assert service.has_scope(user.id, _CALENDAR) is True
    assert service.has_scope(user.id, _CONTACTS) is True


@pytest.mark.asyncio
async def test_has_scope_false_for_denied_scope(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=(_CALENDAR, _CONTACTS))
    await service.begin_connect(user.id)
    await _complete(service, user.id)

    assert service.has_scope(user.id, _TASKS) is False


@pytest.mark.asyncio
async def test_partial_grant_keeps_user_connected_for_other_features(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=(_CALENDAR,))

    await service.begin_connect(user.id)
    await _complete(service, user.id)

    refreshed = container.users_repository.get_by_id(user.id)
    assert refreshed.google_connected is True
    assert service.granted_scopes(user.id) == {_CALENDAR}
    assert service.has_scope(user.id, _TASKS) is False
    assert service.has_scope(user.id, _CONTACTS) is False


@pytest.mark.asyncio
async def test_reconnect_replaces_scope_set_with_new_grant(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)

    first_service, _client, _holder = build_google_auth_service(container, granted_scopes=(_CALENDAR,))
    await first_service.begin_connect(user.id)
    await _complete(first_service, user.id)
    assert container.users_repository.get_by_id(user.id).google_scopes_granted == _CALENDAR

    second_service, _client2, _holder2 = build_google_auth_service(container, granted_scopes=(_CALENDAR, _TASKS, _CONTACTS))
    await second_service.begin_connect(user.id, reconnect=True)
    result = await _complete(second_service, user.id)

    assert set(result.scopes_granted) == {_CALENDAR, _TASKS, _CONTACTS}
    refreshed = container.users_repository.get_by_id(user.id)
    assert set((refreshed.google_scopes_granted or '').split(',')) == {_CALENDAR, _TASKS, _CONTACTS}
