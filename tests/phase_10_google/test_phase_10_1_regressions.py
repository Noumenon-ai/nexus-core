"""Phase 10.1 live-OAuth regression tests.

Three bugs surfaced during the first live OAuth deploy on 2026-05-02:

1. Confirmation said "unknown email" — userinfo endpoint returned 401 because
   default GOOGLE_SCOPES did not include `openid`/`email`.
2. Confirmation was sent twice — both `_handle_paste` (ServiceResponse) and
   `_await_completion` (notifier) emitted the same success text.
3. 6-minute event-loop freeze — `Flow.fetch_token` is a sync `requests` call
   that blocked the asyncio loop while the network stalled.

These tests guard against regressions on each.
"""
from __future__ import annotations

import asyncio
import time
from typing import Iterable

import pytest

from models import User
from pipeline.google_intent import GoogleIntentHandler
from services.google_auth_service import GoogleAuthService
from tests.phase_10_google.helpers import (
    FakeFlow,
    _ScopeAwareUserinfoClient,
    build_google_auth_service,
    configure_google_test_env,
    pasted_url_for,
)
from utils.i18n import Translator


_OPENID = 'openid'
_EMAIL_SCOPE = 'https://www.googleapis.com/auth/userinfo.email'
_PROFILE_SCOPE = 'https://www.googleapis.com/auth/userinfo.profile'
_CALENDAR = 'https://www.googleapis.com/auth/calendar'
_TASKS = 'https://www.googleapis.com/auth/tasks.readonly'
_CONTACTS = 'https://www.googleapis.com/auth/contacts'

_FULL_SCOPES = (
    _OPENID,
    _EMAIL_SCOPE,
    _PROFILE_SCOPE,
    _CALENDAR,
    _TASKS,
    _CONTACTS,
)


def _service_with_scope_aware_client(container, granted: Iterable[str], *, email: str = 'real@example.com'):
    fake_client = _ScopeAwareUserinfoClient(granted, email=email)
    flow_holder: dict[str, FakeFlow] = {}

    def flow_factory(scopes):
        flow = FakeFlow(granted)
        flow_holder['last'] = flow
        return flow

    def fake_local_server_starter(pending):
        return 0, None, None

    service = GoogleAuthService(
        container.settings,
        container.users_repository,
        http_client_factory=lambda: fake_client,
        flow_factory=flow_factory,
        local_server_starter=fake_local_server_starter,
        pending_ttl_seconds=600,
    )
    return service, fake_client, flow_holder


# ──────────────────────────────────────────────────── Issue 1 regression


@pytest.mark.asyncio
async def test_userinfo_returns_email_when_openid_scope_granted(monkeypatch, tmp_path):
    """When openid + userinfo.email are granted, users.google_email is populated
    with the real address — never NULL, never 'unknown email'."""
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, client, _ = _service_with_scope_aware_client(
        container,
        granted=_FULL_SCOPES,
        email='owner@example.com',
    )

    _, future = await service.begin_connect(user.id)
    state = service._pending[user.id].state
    result = await service.submit_pasted_url(user.id, pasted_url_for(state=state))
    await future

    assert result.email == 'owner@example.com'
    refreshed = container.users_repository.get_by_id(user.id)
    assert refreshed.google_email == 'owner@example.com'
    assert refreshed.google_email is not None
    assert any('openidconnect.googleapis.com/v1/userinfo' in url for url, _h in client.get_calls)


@pytest.mark.asyncio
async def test_userinfo_returns_none_when_openid_scope_missing(monkeypatch, tmp_path):
    """Negative companion: without openid+email, userinfo 401s and email stays NULL.
    Reproduces the 2026-05-02 production symptom."""
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    legacy_scopes = (_CALENDAR, _TASKS, _CONTACTS)
    service, _client, _ = _service_with_scope_aware_client(container, granted=legacy_scopes)

    await service.begin_connect(user.id)
    state = service._pending[user.id].state
    result = await service.submit_pasted_url(user.id, pasted_url_for(state=state))

    assert result.email is None
    refreshed = container.users_repository.get_by_id(user.id)
    assert refreshed.google_email is None


# ──────────────────────────────────────────────────── Issue 2 regression


class _MessageCounter:
    """Captures every Telegram-side message: both ServiceResponse texts that
    the bot would dispatch as replies, and notifier calls fired from the
    `_await_completion` background task."""

    def __init__(self) -> None:
        self.notifier_messages: list[tuple[str, str]] = []
        self.service_response_texts: list[str] = []

    async def notifier(self, user_id: str, text: str) -> None:
        self.notifier_messages.append((user_id, text))

    @property
    def total_count(self) -> int:
        return len(self.notifier_messages) + len(self.service_response_texts)


@pytest.mark.asyncio
async def test_paste_completion_sends_exactly_one_telegram_message(monkeypatch, tmp_path):
    """Paste path: user sends URL in Telegram → exactly ONE confirmation
    (the inline ServiceResponse). The background `_await_completion` task
    must skip its notify because the paste handler already replied."""
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _ = _service_with_scope_aware_client(container, granted=_FULL_SCOPES)
    counter = _MessageCounter()
    handler = GoogleIntentHandler(service, notifier=counter.notifier)
    translator = Translator('en')

    start_response = await handler.handle(user, 'connect google', translator)
    counter.service_response_texts.append(start_response.text)
    state = service._pending[user.id].state

    paste_response = await handler.handle(user, pasted_url_for(state=state), translator)
    counter.service_response_texts.append(paste_response.text)

    # Yield until the background _await_completion task has settled.
    for _ in range(50):
        await asyncio.sleep(0)
    pending_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    success_messages = [
        text for text in counter.service_response_texts
        if 'Google connected for' in text
    ] + [text for _uid, text in counter.notifier_messages if 'Google connected for' in text]

    assert len(success_messages) == 1, f'expected 1 success message, got {len(success_messages)}: {success_messages}'
    assert counter.notifier_messages == [], f'paste path must not notify; got {counter.notifier_messages}'


@pytest.mark.asyncio
async def test_localhost_completion_sends_exactly_one_telegram_message(monkeypatch, tmp_path):
    """Localhost-redirect path: wsgiref captures code → `_finalize` fires with
    via_paste=False → background `_await_completion` notifies. ServiceResponse
    is irrelevant here (user is in browser, not Telegram). Exactly ONE message."""
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)
    service, _client, _ = _service_with_scope_aware_client(container, granted=_FULL_SCOPES)
    counter = _MessageCounter()
    handler = GoogleIntentHandler(service, notifier=counter.notifier)
    translator = Translator('en')

    start_response = await handler.handle(user, 'connect google', translator)
    counter.service_response_texts.append(start_response.text)
    pending = service._pending[user.id]

    # Simulate localhost wsgiref hitting `_finalize` without via_paste.
    await service._finalize(pending, code='4/0AeaYS_localhost')

    for _ in range(50):
        await asyncio.sleep(0)
    pending_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    success_via_response = [text for text in counter.service_response_texts if 'Google connected for' in text]
    success_via_notifier = [text for _uid, text in counter.notifier_messages if 'Google connected for' in text]

    assert success_via_response == [], 'localhost path must not return inline success'
    assert len(success_via_notifier) == 1, f'expected exactly 1 notifier message, got {success_via_notifier}'


# ──────────────────────────────────────────────────── Issue 3 regression


class _SlowFakeFlow(FakeFlow):
    """FakeFlow whose fetch_token sleeps SYNCHRONOUSLY — simulates the live
    behavior where `Flow.fetch_token` (google-auth-oauthlib) blocks on a
    `requests.post` for seconds. If our `_finalize` ever calls fetch_token
    directly on the event loop, this sleep will freeze every other coroutine.
    """

    def __init__(self, granted_scopes: Iterable[str], sleep_seconds: float) -> None:
        super().__init__(granted_scopes)
        self._sleep = sleep_seconds

    def fetch_token(self, *, code: str) -> None:
        time.sleep(self._sleep)
        super().fetch_token(code=code)


@pytest.mark.asyncio
async def test_fetch_token_does_not_block_event_loop(monkeypatch, tmp_path):
    """`Flow.fetch_token` is synchronous and can stall for many seconds on a
    bad network. Wrapping it in `asyncio.to_thread` keeps the event loop
    responsive — an unrelated coroutine must complete promptly while the
    token exchange is in flight."""
    container = configure_google_test_env(monkeypatch, tmp_path)
    user = container.users_repository.get_or_create(111)

    fake_client = _ScopeAwareUserinfoClient(_FULL_SCOPES)
    slow_flow_holder: dict[str, _SlowFakeFlow] = {}

    def slow_flow_factory(scopes):
        flow = _SlowFakeFlow(_FULL_SCOPES, sleep_seconds=2.0)
        slow_flow_holder['last'] = flow
        return flow

    service = GoogleAuthService(
        container.settings,
        container.users_repository,
        http_client_factory=lambda: fake_client,
        flow_factory=slow_flow_factory,
        local_server_starter=lambda pending: (0, None, None),
        pending_ttl_seconds=600,
    )

    await service.begin_connect(user.id)
    state = service._pending[user.id].state

    parallel_done_at: dict[str, float] = {}
    parallel_started_at = time.monotonic()

    async def parallel_probe() -> None:
        await asyncio.sleep(0.05)
        parallel_done_at['t'] = time.monotonic()

    finalize_task = asyncio.create_task(
        service.submit_pasted_url(user.id, pasted_url_for(state=state))
    )
    probe_task = asyncio.create_task(parallel_probe())

    await probe_task
    probe_elapsed = parallel_done_at['t'] - parallel_started_at
    finalize_done_when_probe_done = finalize_task.done()

    await finalize_task

    assert probe_elapsed < 0.3, (
        f'event loop was blocked during fetch_token: probe took {probe_elapsed:.3f}s '
        f'(should be ~0.05s; >0.3s means fetch_token ran on the event loop)'
    )
    assert not finalize_done_when_probe_done, (
        'finalize completed before probe — sleep was not actually invoked, test is meaningless'
    )
