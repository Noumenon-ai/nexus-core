from __future__ import annotations

import os
import stat

import pytest

from services.google_auth_service import GoogleAuthError, GoogleAuthService, scopes_missing
from tests.phase_10_google.helpers import (
    build_google_auth_service,
    configure_google_test_env,
    pasted_url_for,
)


_FULL_SCOPES = (
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/tasks.readonly',
    'https://www.googleapis.com/auth/contacts',
)


@pytest.mark.asyncio
async def test_connect_marks_user_connected_and_records_all_scopes(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, flow_holder = build_google_auth_service(container, granted_scopes=_FULL_SCOPES)

    url, future = await service.begin_connect(user.id)
    assert 'state=' in url
    pasted = pasted_url_for(state=service._pending[user.id].state)
    result = await service.submit_pasted_url(user.id, pasted)
    assert future.result() == result

    assert result.email == 'test@example.com'
    assert set(result.scopes_granted) == set(_FULL_SCOPES)

    refreshed = container.users_repository.get_by_id(user.id)
    assert refreshed.google_connected is True
    assert refreshed.google_email == 'test@example.com'
    assert refreshed.google_connected_at is not None
    assert set((refreshed.google_scopes_granted or '').split(',')) == set(_FULL_SCOPES)


@pytest.mark.asyncio
async def test_connect_persists_token_with_mode_600(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=_FULL_SCOPES)

    _, future = await service.begin_connect(user.id)
    state = service._pending[user.id].state
    await service.submit_pasted_url(user.id, pasted_url_for(state=state))
    await future

    token_path = container.settings.google.token_dir / f'{user.id}.json'
    assert token_path.exists()
    mode = stat.S_IMODE(os.stat(token_path).st_mode)
    assert mode == 0o600
    parent_mode = stat.S_IMODE(os.stat(container.settings.google.token_dir).st_mode)
    assert parent_mode == 0o700


@pytest.mark.asyncio
async def test_begin_connect_invokes_on_url_callback(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=_FULL_SCOPES)
    captured: list[str] = []

    def callback(url: str) -> None:
        captured.append(url)

    await service.begin_connect(user.id, on_url=callback)

    assert len(captured) == 1
    assert captured[0].startswith('https://accounts.google.com/o/oauth2/auth')
    assert 'state=' in captured[0]


@pytest.mark.asyncio
async def test_disconnect_revokes_and_clears_db(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, fake_client, _holder = build_google_auth_service(container, granted_scopes=_FULL_SCOPES)
    _, future = await service.begin_connect(user.id)
    state = await _state_of(service, user.id)
    await service.submit_pasted_url(user.id, pasted_url_for(state=state))
    await future
    token_path = container.settings.google.token_dir / f'{user.id}.json'
    assert token_path.exists()

    await service.disconnect(user.id)

    assert not token_path.exists()
    refreshed = container.users_repository.get_by_id(user.id)
    assert refreshed.google_connected is False
    assert refreshed.google_email is None
    assert refreshed.google_scopes_granted is None
    assert refreshed.google_connected_at is None
    assert any('oauth2.googleapis.com/revoke' in url for url, _hdr in fake_client.post_calls)


@pytest.mark.asyncio
async def test_get_credentials_returns_none_when_not_connected(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=_FULL_SCOPES)

    creds = service.get_credentials(user.id)

    assert creds is None


@pytest.mark.asyncio
async def test_begin_connect_raises_when_disabled(monkeypatch, tmp_path):
    container = configure_google_test_env(
        monkeypatch,
        tmp_path,
        extra={'GOOGLE_INTEGRATION_ENABLED': 'false'},
    )
    user = container.users_repository.get_or_create(111)
    service = GoogleAuthService(container.settings, container.users_repository)

    with pytest.raises(GoogleAuthError):
        await service.begin_connect(user.id)


def test_scopes_missing_helper_full_match_returns_empty():
    assert scopes_missing(_FULL_SCOPES, _FULL_SCOPES) == set()


def test_scopes_missing_helper_returns_diff():
    granted = ('https://www.googleapis.com/auth/calendar',)
    required = (
        'https://www.googleapis.com/auth/calendar',
        'https://www.googleapis.com/auth/tasks.readonly',
    )
    assert scopes_missing(granted, required) == {'https://www.googleapis.com/auth/tasks.readonly'}


async def _state_of(service, user_id: str) -> str:
    pending = service._pending.get(user_id)
    assert pending is not None
    return pending.state
