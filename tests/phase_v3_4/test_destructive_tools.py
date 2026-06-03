"""V3.4 destructive (approval-gated) tools — five Nexus-internal write
paths wrapped as @tool(requires_approval=True).

Per V3 spec: when a tool has requires_approval=True, the V3.5 dispatcher
will NOT call its fn directly. Instead it creates an Approval row, returns
the approval id + preview text, and waits for the user tap. Tool fn is
called from approval_service.execute path post-tap. So at the V3.4 level,
the fn is the post-approval implementation: when invoked, it performs the
destructive op and returns a ToolResult with announcement.
"""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

from services.destructive_tools import (
    make_destructive_tools,
    register_destructive_tools,
)
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry, ToolResult
from utils.dates import utc_now


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


@pytest.fixture
def disconnect_calls():
    return []


@pytest.fixture
def google_disconnect(disconnect_calls):
    async def _disconnect(user_id):
        disconnect_calls.append(user_id)
    return _disconnect


@pytest.fixture
def tools(container, telos_service, google_disconnect):
    return {fn.__name__: fn for fn, _meta in make_destructive_tools(
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        telos_service=telos_service,
        google_disconnect=google_disconnect,
        scheduler=container.scheduler,
    )}


def _user(container, telegram_id):
    return container.users_repository.get_or_create(telegram_id)


# -------- delete_reminder ---------------------------------------------------

def test_delete_reminder_cancels_active_reminder_and_announces(tools, container):
    user = _user(container, 111)
    reminder = container.reminders_repository.create(user_id=user.id, body='take meds', next_fire_at=utc_now() + timedelta(hours=1), recurrence=None)
    result = tools['delete_reminder'](user_id=user.id, query='meds')
    assert result.success is True
    assert result.data.get('cancelled') is True
    assert 'meds' in result.announcement.lower() or 'cancelled' in result.announcement.lower()
    assert container.reminders_repository.list_active(user.id) == []
    assert reminder.id in container.scheduler.removed


def test_delete_reminder_no_match_returns_announcement(tools, container):
    user = _user(container, 111)
    result = tools['delete_reminder'](user_id=user.id, query='nothing')
    assert result.success is True
    assert result.data.get('cancelled') is False
    assert result.announcement is not None


def test_delete_reminder_user_isolation(tools, container):
    a = _user(container, 111)
    b = _user(container, 222)
    container.reminders_repository.create(user_id=b.id, body='B reminder', next_fire_at=utc_now() + timedelta(hours=1), recurrence=None)
    result = tools['delete_reminder'](user_id=a.id, query='B reminder')
    assert result.data.get('cancelled') is False
    assert len(container.reminders_repository.list_active(b.id)) == 1


# -------- delete_task -------------------------------------------------------

def test_delete_task_removes_pending_task_and_announces(tools, container):
    user = _user(container, 111)
    container.tasks_repository.create(user_id=user.id, title='go to gym', due_at=None)
    result = tools['delete_task'](user_id=user.id, query='gym')
    assert result.success is True
    assert result.data.get('deleted') is True
    assert result.announcement is not None
    assert container.tasks_repository.list_pending(user.id) == []


def test_delete_task_no_match_returns_announcement(tools, container):
    user = _user(container, 111)
    result = tools['delete_task'](user_id=user.id, query='nothing')
    assert result.data.get('deleted') is False
    assert result.announcement is not None


# -------- forget_user_memory ------------------------------------------------

def test_forget_user_memory_deletes_by_key_and_announces(tools, container):
    user = _user(container, 111)
    container.memories_repository.upsert(user_id=user.id, memory_type='preference', key='reminder_time_preference', value='morning', confidence=1.0, source='explicit')
    result = tools['forget_user_memory'](user_id=user.id, key='reminder_time_preference')
    assert result.success is True
    assert result.data.get('forgotten') is True
    assert result.announcement is not None
    assert container.memories_repository.list_by_user(user.id) == []


def test_forget_user_memory_no_match_returns_announcement(tools, container):
    user = _user(container, 111)
    result = tools['forget_user_memory'](user_id=user.id, key='ghost_key')
    assert result.data.get('forgotten') is False
    assert result.announcement is not None


def test_forget_user_memory_user_isolation(tools, container):
    a = _user(container, 111)
    b = _user(container, 222)
    container.memories_repository.upsert(user_id=b.id, memory_type='preference', key='shared_key', value='B value', confidence=1.0, source='explicit')
    result = tools['forget_user_memory'](user_id=a.id, key='shared_key')
    assert result.data.get('forgotten') is False
    remaining = container.memories_repository.list_by_user(b.id)
    assert [m.value for m in remaining] == ['B value']


# -------- disconnect_google -------------------------------------------------

@pytest.mark.asyncio
async def test_disconnect_google_invokes_callback_and_announces(tools, container, disconnect_calls):
    user = _user(container, 111)
    result = await tools['disconnect_google'](user_id=user.id)
    assert result.success is True
    assert result.data.get('disconnected') is True
    assert result.announcement is not None
    assert disconnect_calls == [user.id]


@pytest.mark.asyncio
async def test_disconnect_google_handles_callback_exception_and_announces(container, telos_service):
    """If the underlying disconnect callable raises (network blip, OAuth
    revoke fails), the tool returns an announcement-bearing failure-shaped
    success rather than propagating."""
    async def boom(user_id):
        raise RuntimeError('revoke failed')

    pairs = make_destructive_tools(
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        telos_service=telos_service,
        google_disconnect=boom,
        scheduler=container.scheduler,
    )
    fns = {fn.__name__: fn for fn, _meta in pairs}
    user = container.users_repository.get_or_create(111)
    result = await fns['disconnect_google'](user_id=user.id)
    assert result.success is True
    assert result.data.get('disconnected') is False
    assert result.announcement is not None


# -------- append_telos_update -----------------------------------------------

def test_append_telos_update_appends_content_and_announces(tools, telos_service):
    user_id = 'abc-123'
    telos_service.path_for(user_id).write_text('# Telos\n', encoding='utf-8')
    result = tools['append_telos_update'](user_id=user_id, content='\n## Update\nNew goal: ship V3.\n')
    assert result.success is True
    assert result.data.get('appended') is True
    assert result.announcement is not None
    final = telos_service.load(user_id)
    assert '# Telos\n' in final and 'New goal: ship V3.' in final


def test_append_telos_update_creates_file_when_absent(tools, telos_service):
    result = tools['append_telos_update'](user_id='new-user', content='# Fresh\n')
    assert result.success is True
    assert telos_service.load('new-user') == '# Fresh\n'


def test_append_telos_update_rejects_empty_content(tools):
    result = tools['append_telos_update'](user_id='abc-123', content='')
    assert result.success is True
    assert result.data.get('appended') is False
    assert result.announcement is not None


# -------- registration & shape gates ----------------------------------------

def test_register_destructive_tools_all_have_requires_approval(container, telos_service, google_disconnect):
    registry = ToolRegistry()
    specs = register_destructive_tools(
        registry,
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        telos_service=telos_service,
        google_disconnect=google_disconnect,
        scheduler=container.scheduler,
    )
    names = {s.name for s in specs}
    assert names == {
        'delete_reminder',
        'delete_task',
        'forget_user_memory',
        'disconnect_google',
        'append_telos_update',
        'delete_calendar_event',
        'write_file',
        'run_terminal_command',
    }
    for s in specs:
        assert s.requires_approval is True, f'{s.name} is destructive but is NOT approval-gated'
        assert s.approval_template, f'{s.name} missing approval_template'
        assert s.description


@pytest.mark.asyncio
async def test_all_destructive_tool_results_are_json_serializable(tools, container):
    user = _user(container, 111)
    container.reminders_repository.create(user_id=user.id, body='r', next_fire_at=utc_now() + timedelta(hours=1), recurrence=None)
    container.tasks_repository.create(user_id=user.id, title='t', due_at=None)
    container.memories_repository.upsert(user_id=user.id, memory_type='fact', key='k', value='v', confidence=1.0, source='explicit')

    results = [
        tools['delete_reminder'](user_id=user.id, query='r'),
        tools['delete_task'](user_id=user.id, query='t'),
        tools['forget_user_memory'](user_id=user.id, key='k'),
        await tools['disconnect_google'](user_id=user.id),
        tools['append_telos_update'](user_id=user.id, content='# new\n'),
    ]
    for r in results:
        assert isinstance(r, ToolResult)
        assert r.success is True
        json.dumps(r.data)
        assert isinstance(r.announcement, str) and r.announcement.strip()
