from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.dispatcher_registry import build_dispatcher_registry
from services.telos_service import TelosService


class _FailIfCalledLLM:
    async def generate_with_tools(self, **_kwargs):
        raise AssertionError('LLM/provider path should not be called for reminder read commands')


class _StubMem0:
    def search(self, *args, **kwargs):
        return []

    def add(self, *args, **kwargs):
        return {'results': []}


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


@pytest.fixture
def full_registry(container, telos_service):
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
async def test_reminders_duplicates_with_no_duplicates_returns_direct_empty_result(
    container,
    full_registry,
    telos_service,
):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(container, telos_service, full_registry)

    out = await dispatcher.handle(DispatcherInput(user=user, text='/reminders duplicates'))

    assert out.text == 'No duplicate reminders found.'
    assert out.iterations == 0
    assert out.metadata.get('duplicate_reminder_audit') is True
    assert out.buttons == []
    assert container.approvals_repository.list_active_pending_for_user(user.id) == []
    # When there are no duplicates the audit produces zero clusters and
    # _store_duplicate_reminder_audit isn't called — so no recovery
    # state is persisted. (Behaviour unchanged for this branch.)
    assert container.context_repository.get_active(user.id) is None


@pytest.mark.asyncio
async def test_reminders_duplicates_with_duplicates_returns_groups_directly(
    container,
    full_registry,
    telos_service,
):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(container, telos_service, full_registry)
    fire_at = datetime(2026, 5, 28, 13, 0, tzinfo=timezone.utc)
    _seed_reminder(container, user_id=user.id, body='Follow up with acme acme', next_fire_at=fire_at, minutes_offset=0)
    _seed_reminder(container, user_id=user.id, body='Follow up with Acme Corp', next_fire_at=fire_at, minutes_offset=1)

    out = await dispatcher.handle(DispatcherInput(user=user, text='/reminders duplicates'))

    assert out.iterations == 0
    assert out.metadata.get('duplicate_reminder_audit') is True
    assert out.metadata.get('duplicate_cluster_count') == 1
    assert 'I found 1 duplicate reminder cluster:' in out.text
    assert 'Want me to keep the newest and cancel the others?' in out.text
    # The audit IS persisted to conversation context so a later
    # 'keep newest' from another process/worker can rehydrate the
    # clusters and bind a cleanup approval to this exact audit run.
    recovery_state = container.conversation_service.get_recovery_context(user.id)
    audit_payload = recovery_state.get('duplicate_reminder_audit') or {}
    assert (audit_payload.get('clusters') or [])  # non-empty cluster set


@pytest.mark.asyncio
@pytest.mark.parametrize('prompt', ['/reminders', '/reminders list', 'whats my reminders ?', 'show my reminders'])
async def test_reminder_list_commands_render_upcoming_reminders_without_loop(
    container,
    full_registry,
    telos_service,
    prompt,
):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(container, telos_service, full_registry)
    _seed_reminder(
        container,
        user_id=user.id,
        body='Call Mike',
        next_fire_at=datetime(2026, 5, 28, 14, 0, tzinfo=timezone.utc),
    )
    _seed_reminder(
        container,
        user_id=user.id,
        body='Check Unit 204',
        next_fire_at=datetime(2026, 5, 29, 13, 30, tzinfo=timezone.utc),
    )

    out = await dispatcher.handle(DispatcherInput(user=user, text=prompt))

    assert out.iterations == 0
    assert out.metadata.get('reminder_list') is True
    assert out.text.startswith('Upcoming reminders:\n')
    assert 'Call Mike' in out.text
    assert 'Check Unit 204' in out.text
    assert out.buttons == []
    assert container.approvals_repository.list_active_pending_for_user(user.id) == []


@pytest.mark.asyncio
async def test_read_only_reminder_commands_do_not_trigger_approval(
    container,
    full_registry,
    telos_service,
):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(container, telos_service, full_registry)
    _seed_reminder(
        container,
        user_id=user.id,
        body='Pay invoice',
        next_fire_at=datetime(2026, 5, 28, 15, 0, tzinfo=timezone.utc),
    )

    list_out = await dispatcher.handle(DispatcherInput(user=user, text='what are my reminders'))
    dup_out = await dispatcher.handle(DispatcherInput(user=user, text='/reminders duplicates'))

    assert list_out.metadata.get('destructive_gate') is not True
    assert dup_out.metadata.get('destructive_gate') is not True
    assert list_out.buttons == []
    assert dup_out.buttons == []
    assert container.approvals_repository.list_active_pending_for_user(user.id) == []


@pytest.mark.asyncio
async def test_malformed_combined_reminder_read_input_does_not_loop(
    container,
    full_registry,
    telos_service,
):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(container, telos_service, full_registry)
    fire_at = datetime(2026, 5, 28, 13, 0, tzinfo=timezone.utc)
    _seed_reminder(container, user_id=user.id, body='Follow up with acme acme', next_fire_at=fire_at, minutes_offset=0)
    _seed_reminder(container, user_id=user.id, body='Follow up', next_fire_at=fire_at, minutes_offset=1)

    out = await dispatcher.handle(
        DispatcherInput(user=user, text='/reminders duplicates whats my reminders')
    )

    assert out.iterations == 0
    assert 'iteration limit' not in out.text.lower()
    assert out.metadata.get('duplicate_reminder_audit') is True
    assert out.buttons == []
