"""V3.3 source-inspection test: every tool in the auto-write registry MUST
return ToolResult with announcement set (never silent success).

This is the new V3.3 structural gate, carrying forward the source-inspection
pattern V3.4 will use for requires_approval=True on destructive tools. It
introspects a fresh registry holding the union of:
  - 5 real auto-write tools  (services/auto_write_tools.py)
  - 3 stubbed Google write tools (services/auto_write_tools_stubs.py)
…and invokes each with a representative happy-path input, asserting:
  result.success is True
  result.announcement is a non-empty string

The stubs participate too — they MUST set announcement even though their
data carries the deferral envelope. Future drift where someone adds a
new auto-write tool but forgets the announcement is caught immediately
without any per-tool test having to enumerate the new tool.
"""
from __future__ import annotations

import asyncio
import inspect

from datetime import timedelta

import pytest

from services.auto_write_tools import register_auto_write_tools
from services.auto_write_tools_stubs import register_google_write_stubs
from services.tool_registry import ToolRegistry
from utils.dates import utc_now


_HAPPY_PATH_KWARGS_BY_NAME = {
    # real
    'create_reminder': lambda user_id: {'user_id': user_id, 'body': 'water plants', 'next_fire_at': (utc_now() + timedelta(hours=2)).isoformat()},
    'create_task': lambda user_id: {'user_id': user_id, 'title': 'do laundry'},
    'mark_task_done': lambda user_id: {'user_id': user_id, 'query': 'do laundry'},
    'save_user_memory': lambda user_id: {'user_id': user_id, 'memory_type': 'fact', 'key': 'pet_name', 'value': 'Mochi'},
    'set_user_preference': lambda user_id: {'user_id': user_id, 'key': 'reminder_time_preference', 'value': 'morning'},
    # stubs
    'create_calendar_event': lambda user_id: {'user_id': user_id, 'summary': 'standup', 'start': (utc_now() + timedelta(hours=1)).isoformat(), 'end': (utc_now() + timedelta(hours=2)).isoformat()},
    'update_calendar_event': lambda user_id: {'user_id': user_id, 'event_id': 'evt-stub'},
    'create_contact': lambda user_id: {'user_id': user_id, 'name': 'Alice'},
}


@pytest.fixture
def populated_auto_write_registry(container):
    registry = ToolRegistry()
    register_auto_write_tools(
        registry,
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        scheduler=container.scheduler,
        habit_service=container.habit_service,
        app_timezone='UTC',
    )
    register_google_write_stubs(registry)
    return registry


def test_auto_write_registry_has_eight_tools(populated_auto_write_registry):
    names = set(populated_auto_write_registry.names())
    assert names == set(_HAPPY_PATH_KWARGS_BY_NAME.keys())


def test_every_auto_write_tool_returns_announcement(populated_auto_write_registry, container):
    """Structural V3.3 contract: every tool in the auto-write registry must
    return ToolResult with success=True and a non-empty announcement.
    Counter-tests for failure modes live in the per-tool test files."""
    user = container.users_repository.get_or_create(111)
    # Pre-seed a task so mark_task_done has a match in its happy path.
    container.tasks_repository.create(user_id=user.id, title='do laundry', due_at=None)

    missing = []
    for spec in populated_auto_write_registry.all():
        kwargs_factory = _HAPPY_PATH_KWARGS_BY_NAME.get(spec.name)
        assert kwargs_factory is not None, f'No happy-path kwargs registered for tool {spec.name!r} — extend _HAPPY_PATH_KWARGS_BY_NAME'
        kwargs = kwargs_factory(user.id)
        result = spec.fn(**kwargs)
        if inspect.iscoroutine(result):
            result = asyncio.run(result)
        if not result.success or not (isinstance(result.announcement, str) and result.announcement.strip()):
            missing.append((spec.name, result.success, result.announcement))
    assert not missing, f'Tools missing announcement on happy path: {missing}'


def test_no_auto_write_tool_is_approval_gated(populated_auto_write_registry):
    """V3.3 is the non-destructive write phase — approval-gating is V3.4's
    job (delete_*, send_*, disconnect_*, telos-append). Any auto-write tool
    appearing here with requires_approval=True is a routing bug."""
    gated = [s.name for s in populated_auto_write_registry.all() if s.requires_approval]
    assert gated == [], f'Non-destructive write tools must not be approval-gated: {gated}'
