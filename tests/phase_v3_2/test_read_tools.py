"""V3.2 read tools — eight Nexus-internal read paths wrapped as @tool.

Each tool returns ToolResult.ok(data=<JSON-serializable dict>). Per spec gate:
"every read tool callable in isolation, results parseable."
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from services.read_tools import make_read_tools, register_read_tools
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry, ToolResult
from utils.dates import utc_now


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


@pytest.fixture
def tools(container, telos_service):
    return {fn.__name__: fn for fn, _meta in make_read_tools(
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        emails_repository=container.emails_repository,
        approvals_repository=container.approvals_repository,
        telos_service=telos_service,
        app_timezone='UTC',
    )}


def _user(container, telegram_id):
    return container.users_repository.get_or_create(telegram_id)


# -------- get_current_time --------------------------------------------------

def test_get_current_time_returns_iso_string_in_app_timezone(tools):
    result = tools['get_current_time']()
    assert result.success is True
    assert isinstance(result.data, dict)
    assert 'iso' in result.data
    assert 'timezone' in result.data
    parsed = datetime.fromisoformat(result.data['iso'])
    assert parsed.tzinfo is not None
    assert (utc_now() - parsed.astimezone(timezone.utc)).total_seconds() < 5


# -------- list_active_reminders ---------------------------------------------

def test_list_active_reminders_returns_only_active_per_user(tools, container):
    user = _user(container, 111)
    other = _user(container, 222)
    container.reminders_repository.create(user_id=user.id, body='take meds', next_fire_at=utc_now() + timedelta(hours=1), recurrence=None)
    container.reminders_repository.create(user_id=other.id, body='other reminder', next_fire_at=utc_now() + timedelta(hours=1), recurrence=None)

    result = tools['list_active_reminders'](user_id=user.id)
    assert result.success is True
    items = result.data['reminders']
    assert [r['body'] for r in items] == ['take meds']
    assert all('id' in r and 'next_fire_at' in r for r in items)


def test_list_active_reminders_empty_when_none(tools, container):
    user = _user(container, 111)
    result = tools['list_active_reminders'](user_id=user.id)
    assert result.data == {'reminders': []}


# -------- list_pending_tasks ------------------------------------------------

def test_list_pending_tasks_returns_nexus_internal_with_id_and_title(tools, container):
    user = _user(container, 111)
    container.tasks_repository.create(user_id=user.id, title='write report', due_at=None, priority=2)
    container.tasks_repository.create(user_id=user.id, title='call mom', due_at=None, priority=0)

    result = tools['list_pending_tasks'](user_id=user.id)
    assert result.success is True
    titles = [t['title'] for t in result.data['tasks']]
    assert 'write report' in titles
    assert 'call mom' in titles
    for t in result.data['tasks']:
        assert {'id', 'title', 'priority', 'status'} <= set(t.keys())
        assert t['status'] == 'pending'


# -------- list_completed_tasks ----------------------------------------------

def test_list_completed_tasks_uses_repo_and_returns_done(tools, container):
    user = _user(container, 111)
    container.tasks_repository.create(user_id=user.id, title='finished task', due_at=None)
    container.tasks_repository.mark_done(user.id, 'finished task')
    container.tasks_repository.create(user_id=user.id, title='still open', due_at=None)

    result = tools['list_completed_tasks'](user_id=user.id)
    titles = [t['title'] for t in result.data['tasks']]
    assert titles == ['finished task']
    assert all(t['status'] == 'done' for t in result.data['tasks'])


# -------- list_memories -----------------------------------------------------

def test_list_memories_returns_user_scoped_key_value_pairs(tools, container):
    user = _user(container, 111)
    other = _user(container, 222)
    container.memories_repository.upsert(user_id=user.id, memory_type='preference', key='reminder_time', value='morning', confidence=1.0, source='user')
    container.memories_repository.upsert(user_id=other.id, memory_type='preference', key='reminder_time', value='evening', confidence=1.0, source='user')

    result = tools['list_memories'](user_id=user.id)
    items = result.data['memories']
    assert len(items) == 1
    assert items[0]['key'] == 'reminder_time'
    assert items[0]['value'] == 'morning'
    assert items[0]['memory_type'] == 'preference'


# -------- get_email_summary -------------------------------------------------

def test_get_email_summary_returns_recent_emails_within_window(tools, container):
    user = _user(container, 111)
    now = utc_now()
    container.emails_repository.upsert(
        user_id=user.id, gmail_message_id='m1', subject='Hi', sender='alice@x.com',
        snippet='hello there', received_at=now - timedelta(hours=2),
        category=None, extracted_json=None,
    )
    container.emails_repository.upsert(
        user_id=user.id, gmail_message_id='m2', subject='Old', sender='bob@x.com',
        snippet='ancient', received_at=now - timedelta(hours=48),
        category=None, extracted_json=None,
    )

    result = tools['get_email_summary'](user_id=user.id, hours=24)
    subjects = [e['subject'] for e in result.data['emails']]
    assert subjects == ['Hi']
    assert result.data['window_hours'] == 24


# -------- get_telos ---------------------------------------------------------

def test_get_telos_returns_file_contents_when_present(tools, telos_service):
    user_id = 'abc-123'
    p = telos_service.path_for(user_id)
    p.write_text('# Telos\nI am building Nexus.', encoding='utf-8')

    result = tools['get_telos'](user_id=user_id)
    assert result.success is True
    assert result.data['present'] is True
    assert 'Nexus' in result.data['content']


def test_get_telos_returns_present_false_when_absent(tools):
    result = tools['get_telos'](user_id='ghost-user')
    assert result.success is True
    assert result.data['present'] is False
    assert result.data['content'] is None


# -------- get_active_approvals ----------------------------------------------

def test_get_active_approvals_returns_unexpired_pending_for_user(tools, container):
    user = _user(container, 111)
    other = _user(container, 222)
    now = utc_now()
    container.approvals_repository.create(
        user_id=user.id, action_type='delete_reminder', preview_text='delete X',
        payload_json='{}', expires_at=now + timedelta(minutes=5),
    )
    container.approvals_repository.create(
        user_id=user.id, action_type='send_email', preview_text='expired',
        payload_json='{}', expires_at=now - timedelta(minutes=1),
    )
    container.approvals_repository.create(
        user_id=other.id, action_type='delete_task', preview_text='other user',
        payload_json='{}', expires_at=now + timedelta(minutes=5),
    )

    result = tools['get_active_approvals'](user_id=user.id)
    items = result.data['approvals']
    previews = [a['preview_text'] for a in items]
    assert previews == ['delete X']
    assert all(a['status'] == 'pending' for a in items)


# -------- registration & shape gates ----------------------------------------

def test_register_read_tools_registers_eight_tools_no_approval(container, telos_service):
    registry = ToolRegistry()
    specs = register_read_tools(
        registry,
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        emails_repository=container.emails_repository,
        approvals_repository=container.approvals_repository,
        telos_service=telos_service,
        app_timezone='UTC',
    )
    names = {s.name for s in specs}
    assert names == {
        'get_current_time',
        'list_active_reminders',
        'list_pending_tasks',
        'list_completed_tasks',
        'list_memories',
        'get_email_summary',
        'get_telos',
        'get_active_approvals',
    }
    for s in specs:
        assert s.requires_approval is False, f'{s.name} is a read tool but is approval-gated'
        assert s.description, f'{s.name} missing description'
        assert isinstance(s.parameters, dict)


def test_all_read_tool_results_are_json_serializable(tools, container, telos_service):
    """V3.2 gate: results parseable. Tool data dicts must JSON-encode without
    custom encoders so the dispatcher can hand them to Gemini directly."""
    user = _user(container, 111)
    container.reminders_repository.create(user_id=user.id, body='r', next_fire_at=utc_now() + timedelta(hours=1), recurrence=None)
    container.tasks_repository.create(user_id=user.id, title='t', due_at=utc_now() + timedelta(hours=2), priority=1)
    container.memories_repository.upsert(user_id=user.id, memory_type='preference', key='k', value='v', confidence=1.0, source='user')
    container.emails_repository.upsert(user_id=user.id, gmail_message_id='m1', subject='s', sender='from', snippet='snip', received_at=utc_now() - timedelta(hours=1), category=None, extracted_json=None)
    container.approvals_repository.create(user_id=user.id, action_type='delete_reminder', preview_text='p', payload_json='{}', expires_at=utc_now() + timedelta(minutes=5))
    telos_service.path_for(user.id).write_text('# t', encoding='utf-8')

    results = [
        tools['get_current_time'](),
        tools['list_active_reminders'](user_id=user.id),
        tools['list_pending_tasks'](user_id=user.id),
        tools['list_completed_tasks'](user_id=user.id),
        tools['list_memories'](user_id=user.id),
        tools['get_email_summary'](user_id=user.id),
        tools['get_telos'](user_id=user.id),
        tools['get_active_approvals'](user_id=user.id),
    ]
    for r in results:
        assert isinstance(r, ToolResult)
        assert r.success is True
        json.dumps(r.data)


def test_user_isolation_across_all_read_tools(tools, container):
    """Cross-user negative — User B's data must not appear in any of User A's
    tool results. Defense against future drift where someone forgets a
    user_id filter."""
    a = _user(container, 111)
    b = _user(container, 222)
    container.reminders_repository.create(user_id=b.id, body='b reminder', next_fire_at=utc_now() + timedelta(hours=1), recurrence=None)
    container.tasks_repository.create(user_id=b.id, title='b task', due_at=None)
    container.tasks_repository.create(user_id=b.id, title='b done', due_at=None)
    container.tasks_repository.mark_done(b.id, 'b done')
    container.memories_repository.upsert(user_id=b.id, memory_type='preference', key='b_key', value='b_val', confidence=1.0, source='user')
    container.emails_repository.upsert(user_id=b.id, gmail_message_id='bm1', subject='b email', sender='b', snippet='b', received_at=utc_now() - timedelta(hours=1), category=None, extracted_json=None)
    container.approvals_repository.create(user_id=b.id, action_type='delete_reminder', preview_text='b approval', payload_json='{}', expires_at=utc_now() + timedelta(minutes=5))

    serialized = json.dumps([
        tools['list_active_reminders'](user_id=a.id).data,
        tools['list_pending_tasks'](user_id=a.id).data,
        tools['list_completed_tasks'](user_id=a.id).data,
        tools['list_memories'](user_id=a.id).data,
        tools['get_email_summary'](user_id=a.id).data,
        tools['get_active_approvals'](user_id=a.id).data,
    ])
    for forbidden in ('b reminder', 'b task', 'b done', 'b_key', 'b_val', 'b email', 'b approval'):
        assert forbidden not in serialized, f"User A's tools leaked {forbidden!r} from User B"
