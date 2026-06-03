"""V3.3 auto-write tools — five Nexus-internal write paths wrapped as @tool.

Each tool returns ToolResult.ok with `announcement` set (V3.3 contract: every
tool in the auto-write registry must return an announcement, never silent
success). The reminder hybrid parser is preserved internally; the @tool
surface takes structured fields (body / next_fire_at / recurrence).
"""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

from services.auto_write_tools import make_auto_write_tools, register_auto_write_tools
from services.tool_registry import ToolRegistry, ToolResult
from utils.dates import utc_now


@pytest.fixture
def tools(container):
    return {fn.__name__: fn for fn, _meta in make_auto_write_tools(
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        scheduler=container.scheduler,
        habit_service=container.habit_service,
        app_timezone='UTC',
    )}


def _user(container, telegram_id):
    return container.users_repository.get_or_create(telegram_id)


# -------- create_reminder ---------------------------------------------------

def test_create_reminder_persists_row_and_schedules_and_announces(tools, container):
    user = _user(container, 111)
    when = (utc_now() + timedelta(hours=2)).isoformat()

    result = tools['create_reminder'](user_id=user.id, body='take meds', next_fire_at=when)
    assert result.success is True
    assert result.announcement is not None
    assert 'take meds' in result.announcement or 'reminder' in result.announcement.lower()

    reminders = container.reminders_repository.list_active(user.id)
    assert [r.body for r in reminders] == ['take meds']
    assert container.scheduler.scheduled, 'scheduler.schedule_reminder was not called'
    assert container.scheduler.scheduled[0][0] == reminders[0].id


def test_create_reminder_rejects_past_time_and_announces_reason(tools, container):
    """H2-003 lesson — past-time reminders fire immediately or never. Tool
    rejects with announcement-bearing failure-shaped success (no DB row)."""
    user = _user(container, 111)
    past = (utc_now() - timedelta(hours=1)).isoformat()

    result = tools['create_reminder'](user_id=user.id, body='oops', next_fire_at=past)
    assert result.success is True
    assert result.data.get('created') is False
    assert result.announcement is not None
    assert 'past' in result.announcement.lower() or 'cannot' in result.announcement.lower()
    assert container.reminders_repository.list_active(user.id) == []
    assert container.scheduler.scheduled == []


def test_create_reminder_rejects_unparseable_datetime(tools, container):
    user = _user(container, 111)
    result = tools['create_reminder'](user_id=user.id, body='oops', next_fire_at='not-a-date')
    assert result.success is True
    assert result.data.get('created') is False
    assert result.announcement is not None


def test_create_reminder_propagates_recurrence(tools, container):
    user = _user(container, 111)
    when = (utc_now() + timedelta(hours=1)).isoformat()
    result = tools['create_reminder'](user_id=user.id, body='daily standup', next_fire_at=when, recurrence='FREQ=DAILY;BYHOUR=9;BYMINUTE=0')
    assert result.success is True
    reminders = container.reminders_repository.list_active(user.id)
    assert reminders[0].recurrence == 'FREQ=DAILY;BYHOUR=9;BYMINUTE=0'


# -------- create_task -------------------------------------------------------

def test_create_task_persists_row_and_announces(tools, container):
    user = _user(container, 111)
    result = tools['create_task'](user_id=user.id, title='write report', priority=2)
    assert result.success is True
    assert result.announcement is not None
    assert 'write report' in result.announcement or 'task' in result.announcement.lower()

    tasks = container.tasks_repository.list_pending(user.id)
    assert [t.title for t in tasks] == ['write report']
    assert tasks[0].priority == 2


def test_create_task_with_due_at_persists(tools, container):
    user = _user(container, 111)
    due = (utc_now() + timedelta(days=1)).isoformat()
    result = tools['create_task'](user_id=user.id, title='gym', due_at=due)
    assert result.success is True
    tasks = container.tasks_repository.list_pending(user.id)
    assert tasks[0].due_at is not None


def test_create_task_rejects_empty_title(tools, container):
    user = _user(container, 111)
    result = tools['create_task'](user_id=user.id, title='   ')
    assert result.success is True
    assert result.data.get('created') is False
    assert result.announcement is not None


# -------- mark_task_done ----------------------------------------------------

def test_mark_task_done_completes_existing_task_and_announces(tools, container):
    user = _user(container, 111)
    container.tasks_repository.create(user_id=user.id, title='go to gym', due_at=None)
    result = tools['mark_task_done'](user_id=user.id, query='gym')
    assert result.success is True
    assert result.data.get('matched') is True
    assert result.announcement is not None
    assert container.tasks_repository.list_pending(user.id) == []


def test_mark_task_done_no_match_still_returns_announcement(tools, container):
    user = _user(container, 111)
    result = tools['mark_task_done'](user_id=user.id, query='nonexistent')
    assert result.success is True
    assert result.data.get('matched') is False
    assert result.announcement is not None
    assert 'not found' in result.announcement.lower() or 'no match' in result.announcement.lower()


# -------- save_user_memory --------------------------------------------------

def test_save_user_memory_upserts_and_announces(tools, container):
    user = _user(container, 111)
    result = tools['save_user_memory'](user_id=user.id, memory_type='fact', key='favorite_color', value='blue')
    assert result.success is True
    assert result.data.get('saved') is True
    assert result.announcement is not None

    memories = container.memories_repository.list_by_user(user.id)
    assert [(m.memory_type, m.key, m.value) for m in memories] == [('fact', 'favorite_color', 'blue')]


def test_save_user_memory_refuses_secret_value_and_announces_reason(tools, container):
    """contains_secret() matches keyword patterns (password / token / secret /
    api[_]key / bearer ...). The LLM must not be able to spill those values
    into the memory store via the tool surface."""
    user = _user(container, 111)
    result = tools['save_user_memory'](user_id=user.id, memory_type='fact', key='note', value='my password is hunter2')
    assert result.success is True
    assert result.data.get('saved') is False
    assert result.announcement is not None
    assert 'secret' in result.announcement.lower() or 'refus' in result.announcement.lower()
    assert container.memories_repository.list_by_user(user.id) == []


def test_save_user_memory_overwrites_same_key(tools, container):
    user = _user(container, 111)
    tools['save_user_memory'](user_id=user.id, memory_type='preference', key='reminder_time_preference', value='morning')
    tools['save_user_memory'](user_id=user.id, memory_type='preference', key='reminder_time_preference', value='evening')
    memories = container.memories_repository.list_by_user(user.id)
    assert len(memories) == 1
    assert memories[0].value == 'evening'


# -------- set_user_preference -----------------------------------------------

def test_set_user_preference_persists_as_preference_memory(tools, container):
    user = _user(container, 111)
    result = tools['set_user_preference'](user_id=user.id, key='reminder_time_preference', value='morning')
    assert result.success is True
    assert result.announcement is not None

    memories = container.memories_repository.list_by_user(user.id, memory_type='preference')
    assert len(memories) == 1
    assert memories[0].key == 'reminder_time_preference'
    assert memories[0].value == 'morning'


# -------- structural / source-inspection / shape gates ----------------------

def test_register_auto_write_tools_registers_real_tools_no_approval(container):
    registry = ToolRegistry()
    specs = register_auto_write_tools(
        registry,
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        scheduler=container.scheduler,
        habit_service=container.habit_service,
        app_timezone='UTC',
    )
    names = {s.name for s in specs}
    assert names == {
        'create_reminder',
        'create_task',
        'mark_task_done',
        'save_user_memory',
        'set_user_preference',
        'create_calendar_event',
        'update_calendar_event',
        'create_contact',
    }
    for s in specs:
        assert s.requires_approval is False, f'{s.name} is a non-destructive write tool but is approval-gated'
        assert s.description, f'{s.name} missing description'
        assert isinstance(s.parameters, dict)


def test_all_auto_write_tool_results_are_json_serializable(tools, container):
    user = _user(container, 111)
    container.tasks_repository.create(user_id=user.id, title='dummy', due_at=None)
    when = (utc_now() + timedelta(hours=1)).isoformat()

    results = [
        tools['create_reminder'](user_id=user.id, body='r', next_fire_at=when),
        tools['create_task'](user_id=user.id, title='t'),
        tools['mark_task_done'](user_id=user.id, query='dummy'),
        tools['save_user_memory'](user_id=user.id, memory_type='fact', key='k', value='v'),
        tools['set_user_preference'](user_id=user.id, key='k2', value='v2'),
    ]
    for r in results:
        assert isinstance(r, ToolResult)
        assert r.success is True
        json.dumps(r.data)
        assert isinstance(r.announcement, str) and r.announcement.strip()


def test_create_reminder_user_isolation(tools, container):
    a = _user(container, 111)
    b = _user(container, 222)
    when = (utc_now() + timedelta(hours=1)).isoformat()
    tools['create_reminder'](user_id=a.id, body='a private reminder', next_fire_at=when)
    tools['create_reminder'](user_id=b.id, body='b private reminder', next_fire_at=when)
    a_reminders = container.reminders_repository.list_active(a.id)
    b_reminders = container.reminders_repository.list_active(b.id)
    assert [r.body for r in a_reminders] == ['a private reminder']
    assert [r.body for r in b_reminders] == ['b private reminder']
