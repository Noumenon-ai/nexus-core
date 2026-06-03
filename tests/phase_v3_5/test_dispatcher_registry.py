"""V3.5 + V3.8 union ToolRegistry: holds all 31 tools (27 V3.2/V3.3/V3.4
baseline + 4 V3.8 onboarding).

Verifies:
  - All real + stubbed tools register without name collisions
  - The V3.4 destructive-prefix invariant still holds across the union
  - Read tools are not approval-gated; destructive tools are
"""
from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from services.dispatcher_registry import build_dispatcher_registry
from services.telos_service import TelosService


_DESTRUCTIVE_PREFIXES = ('delete_', 'forget_', 'send_', 'disconnect_')


async def _async_noop_disconnect(uid):
    return None


_EXPECTED_NAMES = {
    # V3.2 reads (8 real)
    'get_current_time', 'list_active_reminders', 'list_pending_tasks',
    'list_completed_tasks', 'list_memories', 'get_email_summary',
    'get_telos', 'get_active_approvals',
    # Google read surface (1 real + 3 stubs)
    'list_calendar_events', 'check_freebusy', 'list_google_tasks',
    'lookup_contact',
    # Contact-reminder read helpers (2)
    # V3.3 writes (5 real)
    'create_reminder', 'create_task', 'mark_task_done',
    'save_user_memory', 'set_user_preference',
    # Contact-reminder write surface (1)
    # V3.3 stubs (3)
    'create_calendar_event', 'update_calendar_event', 'create_contact',
    # V3.4 destructive (5 real)
    'delete_reminder', 'delete_task', 'forget_user_memory',
    'disconnect_google', 'append_telos_update',
    # V3.4 stubs (2)
    'delete_calendar_event', 'send_telegram_message',
    # V3.8 onboarding (4)
    'start_telos_onboarding', 'answer_telos_question',
    'view_my_telos', 'cancel_telos_onboarding',
    # V3.upgrade content tools (3)
    'lookup_wikipedia', 'get_weather', 'get_news_headlines',
    # Phase 4 file-system reads (3)
    'read_file', 'list_directory', 'search_files',
    # Phase 4 file-system + terminal (2)
    'write_file', 'run_terminal_command',
}


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


@pytest.fixture
def calendar_service():
    service = Mock()
    service.list_events = AsyncMock(return_value=[])
    return service


@pytest.fixture
def registry(container, telos_service, calendar_service):
    return build_dispatcher_registry(
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        emails_repository=container.emails_repository,
        approvals_repository=container.approvals_repository,
        telos_service=telos_service,
        scheduler=container.scheduler,
        habit_service=container.habit_service,
        google_disconnect=_async_noop_disconnect,
        google_calendar_service=calendar_service,
        app_timezone='UTC',
        onboarding_repository=container.onboarding_repository,
        users_repository=container.users_repository,
    )


def test_dispatcher_registry_holds_all_42_tools(registry):
    names = set(registry.names())
    assert names == _EXPECTED_NAMES, (
        f'Missing: {_EXPECTED_NAMES - names}; Extra: {names - _EXPECTED_NAMES}'
    )
    assert len(_EXPECTED_NAMES) == 39


@pytest.mark.asyncio
async def test_dispatcher_registry_wires_real_list_calendar_events_tool(registry, calendar_service):
    spec = registry.get('list_calendar_events')
    assert spec is not None

    result = await spec.fn(
        user_id='user-123',
        time_min='2026-05-06T09:00:00+00:00',
        time_max='2026-05-06T10:00:00+00:00',
    )

    calendar_service.list_events.assert_awaited_once()
    assert result.success is True
    assert result.data == {'events': [], 'count': 0}


def test_dispatcher_registry_destructive_invariant(registry):
    """V3.4 SAFETY BOUNDARY enforced across the union."""
    violations = [
        s.name for s in registry.all()
        if any(s.name.startswith(p) for p in _DESTRUCTIVE_PREFIXES) and not s.requires_approval
    ]
    assert violations == [], f'Destructive-named tools missing requires_approval: {violations}'


def test_dispatcher_registry_no_duplicate_names(registry):
    names = registry.names()
    assert len(names) == len(set(names)), f'Duplicate tool names: {names}'


def test_dispatcher_registry_every_approval_gated_tool_has_template(registry):
    missing = [s.name for s in registry.all() if s.requires_approval and not s.approval_template]
    assert missing == [], f'Approval-gated tools missing approval_template: {missing}'


def test_dispatcher_registry_read_tools_not_approval_gated(registry):
    """Read tools are pure queries — never approval-gated."""
    read_names = {
        'get_current_time', 'list_active_reminders', 'list_pending_tasks',
        'list_completed_tasks', 'list_memories', 'get_email_summary',
        'get_telos', 'get_active_approvals',
        'list_calendar_events', 'check_freebusy', 'list_google_tasks',
    }
    gated_reads = [s.name for s in registry.all() if s.name in read_names and s.requires_approval]
    assert gated_reads == [], f'Read tools incorrectly approval-gated: {gated_reads}'
