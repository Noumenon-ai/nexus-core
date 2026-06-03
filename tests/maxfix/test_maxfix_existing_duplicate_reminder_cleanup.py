from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import services.proactive_service as proactive_service_module
import services.task_service as task_service_module
import repositories.reminders_repository as reminders_repository_module
from services.dispatcher_registry import build_dispatcher_registry
from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.telos_service import TelosService
from utils.dates import utc_now
from utils.i18n import Translator


class _FailIfCalledLLM:
    async def generate_with_tools(self, **_kwargs):
        raise AssertionError('LLM should not be called for duplicate reminder cleanup tests')


class _StubMem0:
    def search(self, *args, **kwargs):
        return []

    def add(self, *args, **kwargs):
        return {'results': []}


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


@pytest.fixture
def full_registry(container, telos_service, tmp_path):
    return build_dispatcher_registry(
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        emails_repository=container.emails_repository,
        approvals_repository=container.approvals_repository,
        telos_service=telos_service,
        scheduler=container.scheduler,
        habit_service=container.habit_service,
        google_disconnect=lambda _user_id: None,
        app_timezone='America/New_York',
        onboarding_repository=container.onboarding_repository,
        users_repository=container.users_repository,
    )


def _build_dispatcher(container, telos_service, registry) -> ToolDispatcher:
    return ToolDispatcher(
        llm=_FailIfCalledLLM(),
        registry=registry,
        telos_service=telos_service,
        mem0=_StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        conversation_service=container.conversation_service,
        approvals_repository=container.approvals_repository,
        proactive_notifications_repository=container.proactive_repository,
        app_timezone='America/New_York',
    )


def _seed_reminder(
    container,
    *,
    user_id: str,
    body: str,
    next_fire_at: datetime,
    minutes_offset: int = 0,
):
    reminder = container.reminders_repository.create(
        user_id=user_id,
        body=body,
        next_fire_at=next_fire_at,
        recurrence=None,
    )
    reminder.created_at = reminder.created_at + timedelta(minutes=minutes_offset)
    reminder.updated_at = reminder.updated_at + timedelta(minutes=minutes_offset)
    with container.database.session_factory() as session:
        row = session.get(type(reminder), reminder.id)
        row.created_at = reminder.created_at
        row.updated_at = reminder.updated_at
        session.commit()
    return container.reminders_repository.get_by_id(reminder.id)


@pytest.mark.asyncio
async def test_duplicate_audit_finds_acme_rows_case_insensitively_and_clusters_generic_followup(
    container,
    full_registry,
    telos_service,
):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(container, telos_service, full_registry)
    base = datetime(2026, 5, 27, 13, 0, tzinfo=timezone.utc)
    _seed_reminder(container, user_id=user.id, body='Follow up with acme acme', next_fire_at=base, minutes_offset=0)
    _seed_reminder(container, user_id=user.id, body='Follow up with Acme Corp', next_fire_at=base, minutes_offset=1)
    _seed_reminder(container, user_id=user.id, body='Follow up', next_fire_at=base, minutes_offset=2)

    out = await dispatcher.handle(DispatcherInput(user=user, text='/reminders duplicates'))

    assert out.metadata.get('duplicate_reminder_audit') is True
    assert out.metadata.get('duplicate_cluster_count') == 1
    assert 'Follow up with acme acme' in out.text
    assert 'Follow up with Acme Corp' in out.text
    assert 'Follow up' in out.text
    assert 'Want me to keep the newest and cancel the others?' in out.text


@pytest.mark.asyncio
async def test_morning_digest_compresses_duplicate_reminders_today(
    container,
    monkeypatch,
):
    fixed_now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(proactive_service_module, 'utc_now', lambda: fixed_now)

    user = container.users_repository.get_or_create(111)
    fire_at = datetime(2026, 5, 28, 13, 0, tzinfo=timezone.utc)
    _seed_reminder(container, user_id=user.id, body='Follow up with acme acme', next_fire_at=fire_at, minutes_offset=0)
    _seed_reminder(container, user_id=user.id, body='Follow up with Acme Corp', next_fire_at=fire_at, minutes_offset=1)
    _seed_reminder(container, user_id=user.id, body='Follow up', next_fire_at=fire_at, minutes_offset=2)

    out = await container.proactive_service.morning_briefing(
        user,
        Translator('en'),
        explicit=False,
    )

    assert out.text.count('Follow up with Acme Corp') == 1
    assert '(2 duplicates found)' in out.text
    assert 'Want me to clean them up?' in out.text


@pytest.mark.asyncio
async def test_reminder_soon_section_compresses_duplicate_rows(
    container,
    monkeypatch,
):
    fixed_now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(task_service_module, 'utc_now', lambda: fixed_now)
    monkeypatch.setattr(reminders_repository_module, 'now_utc', lambda: fixed_now)

    user = container.users_repository.get_or_create(111)
    due_at = fixed_now + timedelta(minutes=45)
    _seed_reminder(container, user_id=user.id, body='Follow up with acme acme', next_fire_at=due_at, minutes_offset=0)
    _seed_reminder(container, user_id=user.id, body='Follow up with Acme Corp', next_fire_at=due_at, minutes_offset=1)
    _seed_reminder(container, user_id=user.id, body='Follow up', next_fire_at=due_at, minutes_offset=2)

    out = container.task_service.organize_day(user, Translator('en'))

    assert sum(line.startswith('Reminder ') for line in out.text.splitlines()) == 1
    assert 'Follow up with Acme Corp (2 duplicates found)' in out.text


@pytest.mark.asyncio
async def test_fire_time_duplicate_cluster_emits_one_notification(
    container,
    monkeypatch,
):
    fixed_now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr('services.reminder_service.utc_now', lambda: fixed_now)

    user = container.users_repository.get_or_create(111)
    due_at = fixed_now - timedelta(minutes=1)
    older = _seed_reminder(container, user_id=user.id, body='Follow up with acme acme', next_fire_at=due_at, minutes_offset=0)
    newer = _seed_reminder(container, user_id=user.id, body='Follow up with Acme Corp', next_fire_at=due_at, minutes_offset=2)
    generic = _seed_reminder(container, user_id=user.id, body='Follow up', next_fire_at=due_at, minutes_offset=1)
    sent: list[tuple[str, str]] = []

    async def _notifier(*, user_id: str, text: str):
        sent.append((user_id, text))

    container.reminder_service.notifier = _notifier

    await container.reminder_service.fire_reminder(older.id)
    await container.reminder_service.fire_reminder(generic.id)
    await container.reminder_service.fire_reminder(newer.id)

    assert len(sent) == 1
    assert sent[0][1] == 'Follow up with Acme Corp'


@pytest.mark.asyncio
async def test_cleanup_requires_approval_and_keeps_newest_after_approval(
    container,
    full_registry,
    telos_service,
):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(container, telos_service, full_registry)
    base = datetime(2026, 5, 27, 13, 0, tzinfo=timezone.utc)
    first = _seed_reminder(container, user_id=user.id, body='Follow up with acme acme', next_fire_at=base, minutes_offset=0)
    second = _seed_reminder(container, user_id=user.id, body='Follow up with Acme Corp', next_fire_at=base, minutes_offset=1)
    third = _seed_reminder(container, user_id=user.id, body='Follow up', next_fire_at=base, minutes_offset=2)

    audit_out = await dispatcher.handle(DispatcherInput(user=user, text='/reminders duplicates'))
    gate_out = await dispatcher.handle(DispatcherInput(user=user, text='keep newest'))
    pending = container.approvals_repository.list_active_pending_for_user(user.id)

    assert audit_out.metadata.get('duplicate_reminder_audit') is True
    assert gate_out.metadata.get('duplicate_cleanup_approval') is True
    assert gate_out.metadata.get('destructive_gate') is True
    assert len(pending) == 1

    approval_id = next(
        button.callback_data.split(':', 2)[2]
        for button in gate_out.buttons
        if 'approve' in button.callback_data
    )
    post_out = await dispatcher.handle(
        DispatcherInput(user=user, text=f'approval:approve:{approval_id}')
    )

    active_ids = {item.id for item in container.reminders_repository.list_active(user.id)}

    assert post_out.text == 'Cleaned up 2 duplicate reminders. I kept the newest row in each cluster.'
    assert third.id in active_ids
    assert first.id not in active_ids
    assert second.id not in active_ids


# ───────── 2026-05-27 — cross-process routing + safe thread binding ─────────


@pytest.mark.asyncio
async def test_duplicate_audit_persists_to_recovery_state(container, telos_service, full_registry):
    """`/reminders duplicates` should write a duplicate_audit active_thread
    to conversation_service recovery_state so a different process/worker
    can still find the audit clusters on a follow-up command."""
    user = container.users_repository.get_or_create(901)
    dispatcher = _build_dispatcher(container, telos_service, full_registry)
    base = datetime(2026, 5, 27, 13, 0, tzinfo=timezone.utc)
    _seed_reminder(container, user_id=user.id, body='Follow up with cross', next_fire_at=base, minutes_offset=0)
    _seed_reminder(container, user_id=user.id, body='Follow up with cross', next_fire_at=base, minutes_offset=1)

    await dispatcher.handle(DispatcherInput(user=user, text='/reminders duplicates'))

    recovery_state = container.conversation_service.get_recovery_context(user.id)
    active_thread = recovery_state.get('active_thread') or {}
    assert active_thread.get('thread_kind') == 'duplicate_audit'
    assert active_thread.get('status') == 'audited'
    # Clusters live at the recovery_state top level (not inside
    # active_thread) so the active_thread coercer doesn't strip them.
    audit_payload = recovery_state.get('duplicate_reminder_audit') or {}
    clusters = audit_payload.get('clusters') or []
    assert len(clusters) == 1, 'expected one cluster in duplicate_reminder_audit'
    assert audit_payload.get('audit_thread_id') == active_thread.get('thread_id')


@pytest.mark.asyncio
async def test_keep_newest_binds_to_audit_thread_after_process_restart(
    container, telos_service, full_registry,
):
    """Simulate a process restart between `/reminders duplicates` and
    `keep newest`: build TWO dispatchers, run the audit on the first,
    then run `keep newest` on the second (whose in-memory cache is
    empty). The second dispatcher must fall back to recovery_state,
    surface the clusters, and bind the cleanup_thread to the audit
    thread id from the first dispatcher."""
    user = container.users_repository.get_or_create(902)
    dispatcher_a = _build_dispatcher(container, telos_service, full_registry)
    dispatcher_b = _build_dispatcher(container, telos_service, full_registry)
    base = datetime(2026, 5, 27, 13, 0, tzinfo=timezone.utc)
    _seed_reminder(container, user_id=user.id, body='Follow up with bind', next_fire_at=base, minutes_offset=0)
    _seed_reminder(container, user_id=user.id, body='Follow up with bind', next_fire_at=base, minutes_offset=1)

    await dispatcher_a.handle(DispatcherInput(user=user, text='/reminders duplicates'))
    # Snapshot the persisted audit thread id BEFORE the cleanup call.
    pre_state = container.conversation_service.get_recovery_context(user.id)
    audit_thread_id = (pre_state.get('active_thread') or {}).get('thread_id')
    assert audit_thread_id, 'audit thread id should be persisted'

    # New process: dispatcher_b has empty _duplicate_reminder_audit_cache.
    assert not dispatcher_b._duplicate_reminder_audit_cache

    gate_out = await dispatcher_b.handle(DispatcherInput(user=user, text='keep newest'))

    assert gate_out.metadata.get('duplicate_cleanup_approval') is True, (
        'cross-process keep newest must surface the persisted audit, not '
        'the empty in-memory cache'
    )
    assert gate_out.metadata.get('audit_thread_id') == audit_thread_id, (
        'cleanup_thread must record the source audit thread id'
    )
    # The new cleanup_thread should be a fresh cleanup, not the audit one
    post_state = container.conversation_service.get_recovery_context(user.id)
    cleanup_thread = post_state.get('active_thread') or {}
    assert cleanup_thread.get('thread_kind') == 'cleanup'
    assert cleanup_thread.get('audit_thread_id') == audit_thread_id


@pytest.mark.asyncio
async def test_second_audit_supersedes_first(container, telos_service, full_registry):
    """A second `/reminders duplicates` call replaces the prior
    duplicate_audit active_thread (and bumps the revision). `keep
    newest` must bind to the LATEST audit, not the stale one."""
    user = container.users_repository.get_or_create(903)
    dispatcher = _build_dispatcher(container, telos_service, full_registry)
    base = datetime(2026, 5, 27, 13, 0, tzinfo=timezone.utc)
    _seed_reminder(container, user_id=user.id, body='Follow up with stale', next_fire_at=base, minutes_offset=0)
    _seed_reminder(container, user_id=user.id, body='Follow up with stale', next_fire_at=base, minutes_offset=1)

    await dispatcher.handle(DispatcherInput(user=user, text='/reminders duplicates'))
    first_state = container.conversation_service.get_recovery_context(user.id)
    first_id = (first_state.get('active_thread') or {}).get('thread_id')
    first_rev = (first_state.get('active_thread') or {}).get('thread_revision')

    # Clear in-memory cache to force the persisted path on the second call.
    dispatcher._duplicate_reminder_audit_cache.clear()
    await dispatcher.handle(DispatcherInput(user=user, text='/reminders duplicates'))
    second_state = container.conversation_service.get_recovery_context(user.id)
    second_id = (second_state.get('active_thread') or {}).get('thread_id')
    second_rev = (second_state.get('active_thread') or {}).get('thread_revision')

    assert second_id != first_id, 'second audit must mint a new thread id'
    assert second_rev > first_rev, 'second audit must bump the revision'

    gate_out = await dispatcher.handle(DispatcherInput(user=user, text='keep newest'))
    # keep newest must bind to the second (latest) audit, not the first.
    assert gate_out.metadata.get('audit_thread_id') == second_id
