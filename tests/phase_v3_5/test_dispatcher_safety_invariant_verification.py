"""V3.5 explicit verification: the V3.4 source-inspection invariants do
check the dispatcher path, not just the static registry.

Per V3.5 user directive: "If V3.5 wiring accidentally bypasses approval
gating in the dispatcher loop, the invariant test should fail. If it
doesn't fail, the invariant test isn't checking the dispatcher path —
flag that."

This file documents and proves the connection:

  1. The static V3.4 invariant
     (`tests/phase_v3_4/test_safety_invariants.py::test_every_destructive_named_tool_has_requires_approval`)
     scans only registry metadata. It does NOT exercise the dispatcher.
     A dispatcher that ignored `spec.requires_approval` would still pass
     that static test.

  2. The dispatcher-path invariant — proven below — runs a destructive
     tool call THROUGH the dispatcher and asserts that the tool fn was
     NOT invoked. If a future change removes the `if spec.requires_approval`
     branch from `pipeline/tool_dispatcher.py::ToolDispatcher.handle`,
     this test fails immediately.

  3. The combination of (1) + (2) is the V3.4 + V3.5 safety contract.
     Either alone is insufficient.

Red-team verification: as of 2026-05-03 V3.5 close, deleting the
`if spec.requires_approval:` branch in `tool_dispatcher.py` would cause
the four `test_tool_dispatcher_approval_gating.py` behavioral tests +
this file's verification test to fail. The static V3.4 invariant
would still pass — confirming the layered defense.
"""
from __future__ import annotations

from datetime import timedelta
from inspect import getsource

import pytest

from pipeline import tool_dispatcher
from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.destructive_tools import register_destructive_tools
from services.read_tools import register_read_tools
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry
from utils.dates import utc_now


async def _async_noop_disconnect(uid):
    return None


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


@pytest.fixture
def fn_invocations():
    return []


@pytest.fixture
def populated_registry(container, telos_service, fn_invocations):
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
    register_destructive_tools(
        registry,
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        telos_service=telos_service,
        google_disconnect=_async_noop_disconnect,
        scheduler=container.scheduler,
    )
    # Wrap fns with recorder.
    for name, spec in list(registry._tools.items()):
        original = spec.fn
        def make_wrapper(real, n=name):
            def wrapper(*args, **kwargs):
                fn_invocations.append(n)
                return real(*args, **kwargs)
            return wrapper
        registry._tools[name] = type(spec)(
            name=spec.name, description=spec.description, fn=make_wrapper(original),
            requires_approval=spec.requires_approval, parameters=spec.parameters,
            approval_template=spec.approval_template,
        )
    return registry


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)

    async def generate_with_tools(self, **_):
        return self.script.pop(0)


class StubMem0:
    def search(self, *a, **k): return []
    def add(self, *a, **k): pass


def _user(container, telegram_id):
    return container.users_repository.get_or_create(telegram_id)


# ---- behavioral proof: dispatcher honors requires_approval -----------------

@pytest.mark.asyncio
async def test_dispatcher_path_invariant_destructive_tool_fn_not_invoked(populated_registry, container, telos_service, fn_invocations):
    """The dispatcher-path counterpart to the V3.4 source-inspection
    invariant. Failing this test = SAFETY BOUNDARY breached at runtime."""
    user = _user(container, 111)
    container.reminders_repository.create(user_id=user.id, body='take meds', next_fire_at=utc_now() + timedelta(hours=1), recurrence=None)

    llm = ScriptedLLM([{'tool_calls': [{'name': 'delete_reminder', 'arguments': {'query': 'meds'}}]}])
    dispatcher = ToolDispatcher(
        llm=llm, registry=populated_registry, telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )
    await dispatcher.handle(DispatcherInput(user=user, text='delete meds reminder'))

    assert 'delete_reminder' not in fn_invocations, (
        'DISPATCHER-PATH INVARIANT VIOLATION: ToolDispatcher invoked '
        'delete_reminder.fn directly without going through approval. '
        'Compare with tests/phase_v3_4/test_safety_invariants.py — that '
        'test alone does NOT catch this; this file is what catches it.'
    )


# ---- structural proof: the gate exists in dispatcher source ----------------

def test_dispatcher_source_contains_requires_approval_gate():
    """Tripwire: if a future refactor drops the `requires_approval` branch
    from ToolDispatcher.handle, this static-source check fires immediately,
    independent of any behavioral test running. Defense-in-depth.

    Note: behavior tests are the primary safety net; this is a fast
    short-circuit so a refactor PR's CI fails before the behavioral
    tests even import."""
    src = getsource(tool_dispatcher.ToolDispatcher.handle)
    assert 'requires_approval' in src, (
        'ToolDispatcher.handle source no longer references requires_approval '
        '— the SAFETY BOUNDARY may have been removed. Verify via '
        'tests/phase_v3_5/test_tool_dispatcher_approval_gating.py and '
        'this file before relaxing the tripwire.'
    )
    assert 'approval_service' in src, (
        'ToolDispatcher.handle no longer routes through approval_service '
        'for gated tools — verify before relaxing.'
    )


def test_static_v3_4_invariant_alone_is_insufficient_documentation():
    """Documentation-only: this test exists to make the layered-defense
    structure searchable. It always passes; its docstring is the spec."""
    pass
