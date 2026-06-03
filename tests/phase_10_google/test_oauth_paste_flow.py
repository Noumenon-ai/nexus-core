from __future__ import annotations

from datetime import timedelta

import pytest

from models import now_utc
from services.google_auth_service import GoogleAuthError
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


# -------------------------------- happy paths


@pytest.mark.asyncio
async def test_paste_full_url_extracts_code_and_state(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=_FULL_SCOPES)
    _, future = await service.begin_connect(user.id)
    pending_state = service._pending[user.id].state

    result = await service.submit_pasted_url(
        user.id,
        f'http://localhost:8080/?code=4/0AeaYS_hello&scope=foo&state={pending_state}',
    )

    assert result.email == 'test@example.com'
    assert future.done()


@pytest.mark.asyncio
async def test_paste_url_with_surrounding_text_still_extracts_code(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=_FULL_SCOPES)
    await service.begin_connect(user.id)
    pending_state = service._pending[user.id].state

    pasted = (
        f'here it is: http://localhost:9999/?code=4/0Aabc&state={pending_state} thanks'
    )
    result = await service.submit_pasted_url(user.id, pasted)
    assert result.email == 'test@example.com'


@pytest.mark.asyncio
async def test_paste_raw_code_value_works_when_no_state_present(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=_FULL_SCOPES)
    await service.begin_connect(user.id)

    raw_code = '4/0AbCdEfGhIjKlMnOpQrStUv'
    result = await service.submit_pasted_url(user.id, raw_code)
    assert result.email == 'test@example.com'


# -------------------------------- rejection paths


@pytest.mark.asyncio
async def test_paste_with_no_pending_flow_rejected(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=_FULL_SCOPES)

    with pytest.raises(GoogleAuthError) as exc_info:
        await service.submit_pasted_url(user.id, 'http://localhost/?code=4/abc&state=anything')
    assert str(exc_info.value) == 'no_pending_flow'


@pytest.mark.asyncio
async def test_paste_after_ttl_rejected(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=_FULL_SCOPES, pending_ttl_seconds=600)
    await service.begin_connect(user.id)
    pending = service._pending[user.id]
    pending.started_at = now_utc() - timedelta(seconds=601)

    with pytest.raises(GoogleAuthError) as exc_info:
        await service.submit_pasted_url(user.id, pasted_url_for(state=pending.state))
    assert str(exc_info.value) == 'flow_expired'
    assert user.id not in service._pending


@pytest.mark.asyncio
async def test_paste_with_state_mismatch_rejected_csrf(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=_FULL_SCOPES)
    await service.begin_connect(user.id)

    with pytest.raises(GoogleAuthError) as exc_info:
        await service.submit_pasted_url(
            user.id,
            'http://localhost:8080/?code=4/0AeaYS_attack&state=ATTACKER_STATE',
        )
    assert str(exc_info.value) == 'state_mismatch'
    # Pending flow must remain — user can retry
    assert user.id in service._pending


@pytest.mark.asyncio
async def test_paste_no_code_in_input_rejected(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=_FULL_SCOPES)
    await service.begin_connect(user.id)

    with pytest.raises(GoogleAuthError) as exc_info:
        await service.submit_pasted_url(user.id, 'just some random text without a code')
    assert str(exc_info.value) == 'no_code_in_paste'


@pytest.mark.asyncio
async def test_paste_replay_rejected(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=_FULL_SCOPES)
    await service.begin_connect(user.id)
    state = service._pending[user.id].state
    await service.submit_pasted_url(user.id, pasted_url_for(state=state))

    # Same paste again — pending was cleaned up after first redemption, so the second
    # paste sees no pending flow rather than a "code already used" lock error. Both are
    # legitimate replay rejections; we assert one of the rejection codes.
    with pytest.raises(GoogleAuthError) as exc_info:
        await service.submit_pasted_url(user.id, pasted_url_for(state=state))
    assert str(exc_info.value) in {'no_pending_flow', 'code_already_redeemed'}


@pytest.mark.asyncio
async def test_paste_url_from_user_a_rejected_when_pasted_by_user_b(monkeypatch, tmp_path):
    container = configure_google_test_env(monkeypatch, tmp_path)
    user_a = container.users_repository.get_or_create(111)
    user_b = container.users_repository.get_or_create(222)
    service, _client, _holder = build_google_auth_service(container, granted_scopes=_FULL_SCOPES)

    await service.begin_connect(user_a.id)
    a_state = service._pending[user_a.id].state
    await service.begin_connect(user_b.id)

    pasted_with_a_state = pasted_url_for(state=a_state)

    with pytest.raises(GoogleAuthError) as exc_info:
        await service.submit_pasted_url(user_b.id, pasted_with_a_state)
    assert str(exc_info.value) == 'state_mismatch'

    refreshed_b = container.users_repository.get_by_id(user_b.id)
    assert refreshed_b.google_connected is False
    refreshed_a = container.users_repository.get_by_id(user_a.id)
    assert refreshed_a.google_connected is False


# -------------------------------- parser unit tests


def test_parse_pasted_code_from_full_url_returns_code_and_state():
    from services.google_auth_service import GoogleAuthService
    code, state = GoogleAuthService._parse_pasted_code(
        'http://localhost:1234/?code=4/0AeaYS_xxx&scope=foo&state=abc123'
    )
    assert code == '4/0AeaYS_xxx'
    assert state == 'abc123'


def test_parse_pasted_code_from_query_string_only():
    from services.google_auth_service import GoogleAuthService
    code, state = GoogleAuthService._parse_pasted_code('?code=4/abc&state=xyz')
    assert code == '4/abc'
    assert state == 'xyz'


def test_parse_pasted_code_empty_input():
    from services.google_auth_service import GoogleAuthService
    code, state = GoogleAuthService._parse_pasted_code('   ')
    assert code is None
    assert state is None


def test_parse_pasted_code_no_code_returns_none():
    from services.google_auth_service import GoogleAuthService
    code, state = GoogleAuthService._parse_pasted_code('http://example.com/?foo=bar')
    assert code is None
    assert state is None
