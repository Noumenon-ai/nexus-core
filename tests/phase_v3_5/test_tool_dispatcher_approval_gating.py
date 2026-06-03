"""V3.5 dispatcher SAFETY BOUNDARY: approval gating at runtime.

The V3.4 source-inspection invariant test asserts that destructive-named
tools have requires_approval=True at registration. This file asserts the
RUNTIME counterpart: the dispatcher must NOT call spec.fn directly when
spec.requires_approval=True. Instead it creates an Approval row, returns
the approval prompt with Approve/Cancel buttons, and ends the loop.

Per user (V3.5 directive): "If V3.5 wiring accidentally bypasses approval
gating in the dispatcher loop, the invariant test should fail. If it
doesn't fail, the invariant test isn't checking the dispatcher path —
flag that."

These tests close that gap with behavioral assertions on the dispatcher
itself.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from config import get_settings
from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.capability_registry import CapabilityRegistry
from services.destructive_tools import register_destructive_tools
from services.destructive_tools_stubs import register_google_destructive_stubs
from services.read_tools import register_read_tools
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
def fn_invocations():
    """Record every spec.fn invocation so tests can assert NEGATIVE — that
    a particular tool's fn was never called by the dispatcher."""
    return []


@pytest.fixture
def populated_registry(container, telos_service, disconnect_calls, fn_invocations):
    """Registry with the V3.4 destructive tools (real + stubs) plus reads.
    Each registered tool fn is wrapped with a recorder that appends to
    `fn_invocations` and then delegates."""
    registry = ToolRegistry()
    register_read_tools(
        registry,
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        emails_repository=container.emails_repository,
        approvals_repository=container.approvals_repository,
        telos_service=telos_service,
        app_timezone='UTC',
    )

    async def disconnect(uid):
        disconnect_calls.append(uid)

    register_destructive_tools(
        registry,
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        telos_service=telos_service,
        google_disconnect=disconnect,
        scheduler=container.scheduler,
    )
    register_google_destructive_stubs(registry)

    # Wrap every fn to record invocation.
    wrapped = {}
    for name, spec in list(registry._tools.items()):
        original = spec.fn

        def make_wrapper(real, n=name):
            def wrapper(*args, **kwargs):
                fn_invocations.append({'name': n, 'kwargs': kwargs})
                return real(*args, **kwargs)
            return wrapper

        registry._tools[name] = type(spec)(
            name=spec.name,
            description=spec.description,
            fn=make_wrapper(original),
            requires_approval=spec.requires_approval,
            parameters=spec.parameters,
            approval_template=spec.approval_template,
        )
    return registry


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
        self.calls.append({'system_prompt': system_prompt, 'contents': contents, 'tool_catalog': tool_catalog})
        return self.script.pop(0)


class StubMem0:
    def search(self, *a, **kw): return []
    def add(self, *a, **kw): pass


def _user(container, telegram_id):
    return container.users_repository.get_or_create(telegram_id)


@pytest.fixture
def approvals_disabled(monkeypatch):
    monkeypatch.setenv('DESTRUCTIVE_APPROVAL_ENABLED', 'false')
    get_settings.cache_clear()
    yield CapabilityRegistry(settings=get_settings())
    get_settings.cache_clear()


# -------- the runtime safety boundary ---------------------------------------

@pytest.mark.asyncio
async def test_dispatcher_does_not_invoke_approval_gated_tool_directly(populated_registry, container, telos_service, fn_invocations):
    """SAFETY BOUNDARY runtime check. Gemini emits a tool_call for
    delete_reminder. The dispatcher MUST refuse to call spec.fn directly
    and instead create an Approval row + return the approval prompt."""
    user = _user(container, 111)
    container.reminders_repository.create(user_id=user.id, body='take meds', next_fire_at=utc_now() + timedelta(hours=1), recurrence=None)

    llm = ScriptedLLM([
        {'tool_calls': [{'name': 'delete_reminder', 'arguments': {'query': 'meds'}}]},
        # The dispatcher should NOT loop after creating the approval — it
        # returns the approval prompt as the final reply. So no second
        # script entry is consumed. If it does loop, ScriptedLLM will
        # raise IndexError.
    ])
    dispatcher = ToolDispatcher(
        llm=llm,
        registry=populated_registry,
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )
    out = await dispatcher.handle(DispatcherInput(user=user, text='delete the meds reminder'))

    # Reminder still active — tool fn was NEVER called.
    invoked_names = [inv['name'] for inv in fn_invocations]
    assert 'delete_reminder' not in invoked_names, (
        f'SAFETY BOUNDARY VIOLATION: dispatcher invoked delete_reminder.fn '
        f'directly without approval. fn_invocations={fn_invocations}'
    )
    active = container.reminders_repository.list_active(user.id)
    assert len(active) == 1, 'reminder was destructively cancelled without approval'

    # Approval row was created.
    pending = container.approvals_repository.list_active_pending_for_user(user.id)
    assert len(pending) == 1
    assert pending[0].action_type == 'delete_reminder'
    assert 'meds' in pending[0].preview_text.lower()

    # Output text contains the preview.
    assert 'meds' in out.text.lower() or 'reminder' in out.text.lower()
    assert out.buttons, 'approval prompt must include Approve/Cancel buttons'


@pytest.mark.asyncio
async def test_dispatcher_creates_approval_for_each_destructive_tool_type(populated_registry, container, telos_service, fn_invocations):
    """Cover the V3.4 prefix set: delete_/forget_/send_/disconnect_ each
    gated. delete_calendar_event and send_telegram_message stubs included
    — they too must NOT be invoked directly even though they're stubs."""
    user = _user(container, 111)
    cases = [
        ('delete_task', {'query': 'gym'}),
        ('forget_user_memory', {'key': 'reminder_time_preference'}),
        ('disconnect_google', {}),
        ('append_telos_update', {'content': '## new\n'}),
        ('delete_calendar_event', {'event_id': 'evt-1'}),
        ('send_telegram_message', {'target': 'someone', 'text': 'hi'}),
    ]
    for tool_name, args in cases:
        container.tasks_repository.create(user_id=user.id, title='gym', due_at=None)
        llm = ScriptedLLM([
            {'tool_calls': [{'name': tool_name, 'arguments': args}]},
        ])
        dispatcher = ToolDispatcher(
            llm=llm,
            registry=populated_registry,
            telos_service=telos_service,
            mem0=StubMem0(),
            approval_service=container.approval_service,
            conversation_turns_repository=container.conversation_turns_repository,
            max_iterations=10,
        )
        await dispatcher.handle(DispatcherInput(user=user, text=f'do {tool_name}'))
        # fn must NOT have been invoked.
        assert tool_name not in [inv['name'] for inv in fn_invocations], (
            f'{tool_name} was invoked directly by dispatcher — SAFETY BOUNDARY violated'
        )
        # Cleanup pending tasks for next iteration.
        container.tasks_repository.delete(user.id, 'gym')


# -------- counter-test: read tools ARE invoked directly ---------------------

@pytest.mark.asyncio
async def test_dispatcher_does_invoke_read_tool_directly(populated_registry, container, telos_service, fn_invocations):
    """Counter-test. Without this the gating tests above pass trivially
    (e.g., if the dispatcher just refused to invoke ANY tool)."""
    user = _user(container, 111)
    container.reminders_repository.create(user_id=user.id, body='r', next_fire_at=utc_now() + timedelta(hours=1), recurrence=None)

    llm = ScriptedLLM([
        {'tool_calls': [{'name': 'list_active_reminders', 'arguments': {}}]},
        {'text': 'Done.'},
    ])
    dispatcher = ToolDispatcher(
        llm=llm,
        registry=populated_registry,
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )
    await dispatcher.handle(DispatcherInput(user=user, text='reminders'))
    invoked = [inv['name'] for inv in fn_invocations]
    assert 'list_active_reminders' in invoked, 'read tool was NOT invoked — dispatcher is broken in the other direction'


# -------- approval-prompt buttons round-trip via existing approval_service --

@pytest.mark.asyncio
async def test_dispatcher_approval_payload_carries_tool_name_and_args(populated_registry, container, telos_service):
    """The approval row's payload_json must carry enough info that the
    tap-to-approve callback path can re-dispatch the tool. Specifically
    it must contain the tool name and the original arguments."""
    import json

    user = _user(container, 111)
    container.tasks_repository.create(user_id=user.id, title='gym', due_at=None)
    llm = ScriptedLLM([{'tool_calls': [{'name': 'delete_task', 'arguments': {'query': 'gym'}}]}])
    dispatcher = ToolDispatcher(
        llm=llm,
        registry=populated_registry,
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )
    await dispatcher.handle(DispatcherInput(user=user, text='delete gym task'))
    pending = container.approvals_repository.list_active_pending_for_user(user.id)
    assert len(pending) == 1
    payload = json.loads(pending[0].payload_json)
    assert payload.get('tool_name') == 'delete_task'
    assert payload.get('arguments') == {'query': 'gym'}


@pytest.mark.asyncio
async def test_dispatcher_bypasses_requires_approval_when_flag_disabled(
    populated_registry,
    container,
    telos_service,
    fn_invocations,
    approvals_disabled,
):
    user = _user(container, 111)
    container.tasks_repository.create(user_id=user.id, title='gym', due_at=None)
    llm = ScriptedLLM([
        {'tool_calls': [{'name': 'delete_task', 'arguments': {'query': 'gym'}}]},
        {'text': 'Task deleted.'},
    ])
    dispatcher = ToolDispatcher(
        llm=llm,
        registry=populated_registry,
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        approvals_repository=container.approvals_repository,
        capability_registry=approvals_disabled,
        max_iterations=10,
    )

    out = await dispatcher.handle(DispatcherInput(user=user, text='delete gym task'))

    assert out.text == 'Task deleted.'
    assert 'delete_task' in [inv['name'] for inv in fn_invocations]
    assert container.approvals_repository.list_active_pending_for_user(user.id) == []
