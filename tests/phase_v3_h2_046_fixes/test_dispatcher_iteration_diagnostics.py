"""H2-046 — dispatcher iteration-cap diagnostics + early-break on
consecutive unknown-tool iterations.

Pre-H2-046 the dispatcher's outer tool-call loop ran the full
``max_iterations=10`` even when Claude was stuck retrying tools that don't
exist in the registry. After H2-046:

  - Every iteration logs its tool-name list via
    ``dispatcher_tool_invocation_round`` so journalctl preserves the
    sequence for post-mortem.
  - Three consecutive iterations where EVERY tool_call hits the
    ``unknown_tool`` branch trigger an early break with a friendly user-
    facing message ("I'm having trouble figuring out which tool to use").
  - The legacy iteration-cap message still fires for the genuine
    long-loop case; ``dispatcher_iteration_cap_hit`` logs the last 3
    iterations + prompt size.
"""
from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from models import ConversationTurn
from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.read_tools import register_read_tools
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubMem0:
    """Drop-in mem0 placeholder for tests that don't care about archival."""

    def search(self, *args, **kwargs):
        return []

    def add(self, *args, **kwargs):
        return {'results': []}


class _UnknownToolLLM:
    """Always emits a single tool_call to a name absent from the registry.
    Used to drive the early-break path."""

    def __init__(self, unknown_name: str = 'mcp__nonexistent__nuke_universe'):
        self.calls = 0
        self.unknown_name = unknown_name

    async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
        self.calls += 1
        return {
            'tool_calls': [{'name': self.unknown_name, 'arguments': {}}],
        }


class _MixedLLM:
    """First N iterations emit a known tool, then unknown forever. Used to
    verify the consecutive counter resets when a known tool appears."""

    def __init__(self, known_tool: str, unknown_tool: str, known_iterations: int = 2):
        self.calls = 0
        self.known_tool = known_tool
        self.unknown_tool = unknown_tool
        self.known_iterations = known_iterations

    async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
        self.calls += 1
        tool = self.known_tool if self.calls <= self.known_iterations else self.unknown_tool
        return {'tool_calls': [{'name': tool, 'arguments': {}}]}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


@pytest.fixture
def registry(container, telos_service):
    reg = ToolRegistry()
    register_read_tools(
        reg,
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        emails_repository=container.emails_repository,
        approvals_repository=container.approvals_repository,
        telos_service=telos_service,
        app_timezone='UTC',
    )
    return reg


def _make_dispatcher(container, registry, telos_service, llm, *, max_iterations: int = 10) -> ToolDispatcher:
    return ToolDispatcher(
        llm=llm,
        registry=registry,
        telos_service=telos_service,
        mem0=_StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=max_iterations,
    )


# ---------------------------------------------------------------------------
# Early-break on consecutive unknowns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_consecutive_unknown_tools_break_early(container, registry, telos_service, caplog):
    """The H2-046 UX fix. After 3 iterations where every tool_call is to a
    name not in the registry, dispatcher breaks with the friendly message
    rather than waiting for max_iterations."""
    user = container.users_repository.get_or_create(111)
    llm = _UnknownToolLLM()
    dispatcher = _make_dispatcher(container, registry, telos_service, llm)

    with caplog.at_level('WARNING', logger='pipeline.tool_dispatcher'):
        out = await dispatcher.handle(DispatcherInput(user=user, text='do the impossible'))

    assert llm.calls == 3, (
        f'expected exactly 3 LLM calls before early break, got {llm.calls}'
    )
    assert "having trouble figuring out which tool" in out.text.lower(), (
        f'expected friendly early-break text, got: {out.text!r}'
    )
    assert 'tool-call iteration limit' not in out.text.lower(), (
        'old iteration-cap message leaked even though we broke early'
    )
    early_break_logs = [r for r in caplog.records
                        if r.message == 'dispatcher_early_break_unknown_tools']
    assert len(early_break_logs) == 1


@pytest.mark.asyncio
async def test_consecutive_counter_resets_on_known_tool(container, registry, telos_service):
    """A run of known tools must reset the consecutive-unknown counter so
    one bad LLM iteration in the middle of an otherwise-progressing loop
    doesn't trigger early-break."""
    user = container.users_repository.get_or_create(111)
    # 2 known + then unknown forever; we'd need 3 consecutive unknowns
    # starting AFTER the known run, so calls should be 2 + 3 = 5.
    llm = _MixedLLM(
        known_tool='get_current_time',
        unknown_tool='nope_does_not_exist',
        known_iterations=2,
    )
    dispatcher = _make_dispatcher(container, registry, telos_service, llm,
                                  max_iterations=20)

    await dispatcher.handle(DispatcherInput(user=user, text='start the loop'))
    # 2 known + 3 unknown = 5 iterations before early-break fires
    assert llm.calls == 5, f'expected 5 LLM calls, got {llm.calls}'


# ---------------------------------------------------------------------------
# Iteration-cap diagnostic logging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iteration_cap_hit_logs_diagnostic_context(container, registry, telos_service, caplog):
    """When the genuine iteration cap fires (loop runs to max_iterations
    without the early-break or empty-tool-calls escape), the dispatcher
    must emit `dispatcher_iteration_cap_hit` with the last 3 iterations'
    tool name lists. Without this, post-mortems on the iteration cap have
    no signal."""
    user = container.users_repository.get_or_create(111)
    # Alternate known/known/unknown so consecutive-unknown counter never
    # reaches 3 — forces the loop all the way to max_iterations.
    class _StriderLLM:
        def __init__(self):
            self.calls = 0

        async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
            self.calls += 1
            # Pattern: known, known, unknown, known, known, unknown, ...
            # Resets consecutive-unknown counter every 3rd iteration so
            # early-break never fires. Each iteration uses a known tool
            # so the loop legitimately runs through max_iterations.
            return {'tool_calls': [{'name': 'get_current_time', 'arguments': {}}]}

    llm = _StriderLLM()
    dispatcher = _make_dispatcher(container, registry, telos_service, llm,
                                  max_iterations=4)

    with caplog.at_level('WARNING', logger='pipeline.tool_dispatcher'):
        out = await dispatcher.handle(DispatcherInput(user=user, text='loop forever'))

    assert llm.calls == 4
    cap_logs = [r for r in caplog.records
                if r.message == 'dispatcher_iteration_cap_hit']
    assert len(cap_logs) == 1, f'expected one iteration_cap_hit log, got {len(cap_logs)}'
    assert hasattr(cap_logs[0], 'iterations')
    assert cap_logs[0].iterations == 4
    assert 'iteration limit' in out.text.lower()


@pytest.mark.asyncio
async def test_per_iteration_tool_invocation_round_logged(container, registry, telos_service, caplog):
    """The new per-iteration log makes it possible to read the dispatcher's
    tool sequence from journalctl. Verify the log fires once per LLM
    iteration with the tool names list."""
    user = container.users_repository.get_or_create(111)
    llm = _UnknownToolLLM(unknown_name='nope_does_not_exist')
    dispatcher = _make_dispatcher(container, registry, telos_service, llm)

    with caplog.at_level('INFO', logger='pipeline.tool_dispatcher'):
        await dispatcher.handle(DispatcherInput(user=user, text='trigger break'))

    round_logs = [r for r in caplog.records
                  if r.message == 'dispatcher_tool_invocation_round']
    # Exactly 3 — same as the LLM call count (early-break triggers on the 3rd)
    assert len(round_logs) == 3
    assert all(getattr(r, 'tools', None) == ['nope_does_not_exist'] for r in round_logs)
