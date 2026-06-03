"""V3.5 dispatcher loop semantics — text-only, single-tool, multi-tool, cap.

The dispatcher is constructed against a clean LLM Protocol (LLMClient) so
tests can drive scripted responses without hitting real Gemini. Each fake
turn returns either {'text': str} (terminal) or
{'tool_calls': [{'name': str, 'arguments': dict}, ...]} (continue loop).
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from pipeline.tool_dispatcher import DispatcherInput, DispatcherOutput, ToolDispatcher
from services.read_tools import register_read_tools
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry
from utils.dates import utc_now


class ScriptedLLM:
    """LLMClient stub. Each `generate_with_tools()` call pops the next
    scripted response. Records the system_prompt and tool catalog seen for
    assertions."""

    def __init__(self, script: list[dict[str, Any]]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def generate_with_tools(self, *, user_id: str, system_prompt: str, contents: list[dict[str, Any]], tool_catalog: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append({
            'user_id': user_id,
            'system_prompt': system_prompt,
            'contents': contents,
            'tool_catalog': tool_catalog,
        })
        if not self.script:
            raise AssertionError('ScriptedLLM exhausted; dispatcher made too many calls')
        return self.script.pop(0)


class StubMem0:
    def __init__(self, search_result=None):
        self._search_result = search_result or []
        self.adds: list[dict[str, Any]] = []

    def search(self, query: str, *, user_id, limit: int = 5):
        return self._search_result

    def add(self, messages, *, user_id):
        self.adds.append({'messages': messages, 'user_id': user_id})


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


@pytest.fixture
def populated_registry(container, telos_service):
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
    return registry


def _user(container, telegram_id):
    return container.users_repository.get_or_create(telegram_id)


# -------- text-only path ----------------------------------------------------

@pytest.mark.asyncio
async def test_text_only_response_terminates_loop(populated_registry, container, telos_service):
    user = _user(container, 111)
    llm = ScriptedLLM([{'text': 'Hello there.'}])
    mem0 = StubMem0()
    dispatcher = ToolDispatcher(
        llm=llm,
        registry=populated_registry,
        telos_service=telos_service,
        mem0=mem0,
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )
    out = await dispatcher.handle(DispatcherInput(user=user, text='hi'))
    assert isinstance(out, DispatcherOutput)
    assert out.text == 'Hello there.'
    assert out.iterations == 1
    assert len(llm.calls) == 1


# -------- single tool call --------------------------------------------------



# -------- multi-tool call in one turn ---------------------------------------

@pytest.mark.asyncio
async def test_multiple_tool_calls_in_one_turn_all_invoked(populated_registry, container, telos_service):
    user = _user(container, 111)
    container.reminders_repository.create(user_id=user.id, body='r1', next_fire_at=utc_now() + timedelta(hours=1), recurrence=None)
    container.tasks_repository.create(user_id=user.id, title='t1', due_at=None)
    llm = ScriptedLLM([
        {'tool_calls': [
            {'name': 'list_active_reminders', 'arguments': {}},
            {'name': 'list_pending_tasks', 'arguments': {}},
        ]},
        {'text': 'Reminders and tasks fetched.'},
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
    out = await dispatcher.handle(DispatcherInput(user=user, text='status'))
    assert out.text == 'Reminders and tasks fetched.'
    second_contents = repr(llm.calls[1]['contents'])
    assert 'list_active_reminders' in second_contents
    assert 'list_pending_tasks' in second_contents


# -------- iteration cap -----------------------------------------------------

@pytest.mark.asyncio
async def test_loop_hits_iteration_cap_and_returns_safe_reply(populated_registry, container, telos_service):
    """Hard cap must terminate even when the LLM keeps emitting tool calls.
    Output text must be a safe fallback, not the loop's last partial state."""
    user = _user(container, 111)
    container.reminders_repository.create(user_id=user.id, body='r', next_fire_at=utc_now() + timedelta(hours=1), recurrence=None)
    # Script: 11 turns of tool calls, never returning text.
    script = [{'tool_calls': [{'name': 'list_active_reminders', 'arguments': {}}]}] * 11
    llm = ScriptedLLM(script)
    dispatcher = ToolDispatcher(
        llm=llm,
        registry=populated_registry,
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )
    out = await dispatcher.handle(DispatcherInput(user=user, text='loop forever'))
    assert out.iterations == 10
    assert out.text and 'iteration' in out.text.lower() or 'too many' in out.text.lower() or 'limit' in out.text.lower()
    # LLM was called exactly 10 times (the cap), not 11.
    assert len(llm.calls) == 10


# -------- unknown tool name -------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_tool_name_returns_error_to_model_not_crash(populated_registry, container, telos_service):
    user = _user(container, 111)
    llm = ScriptedLLM([
        {'tool_calls': [{'name': 'no_such_tool', 'arguments': {}}]},
        {'text': 'Got it, never mind.'},
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
    out = await dispatcher.handle(DispatcherInput(user=user, text='do weird thing'))
    assert out.text == 'Got it, never mind.'
    second_contents = repr(llm.calls[1]['contents'])
    assert 'unknown' in second_contents.lower() or 'not_found' in second_contents.lower() or 'no_such_tool' in second_contents


# -------- tool catalog includes registered tools ----------------------------

@pytest.mark.asyncio
async def test_dispatcher_awaits_async_tool_fn_results(container, telos_service):
    """V3.5 lesson (H2-011 extension): the dispatcher must support both
    sync AND async tool functions. Sync-only would serialize coroutine
    objects via repr and never await them, so the tool silently no-ops
    while reporting success.

    This test registers an async tool, drives a tool_call for it, and
    asserts the awaited ToolResult flows back to Gemini — not the raw
    coroutine. Catches the bug class that motivated the fix and guards
    every future async tool (V3.2.5 Google wrappers in particular)."""
    import warnings
    from services.tool_registry import ToolRegistry, ToolResult

    registry = ToolRegistry()
    invocations = []

    async def async_tool(*, user_id: str) -> ToolResult:
        invocations.append(user_id)
        return ToolResult.ok(data={'awaited': True, 'user_id': user_id})

    registry.register(
        async_tool,
        name='async_tool',
        description='Async test tool that must be awaited.',
        parameters={'type': 'object', 'properties': {}, 'required': []},
    )

    user = _user(container, 111)
    llm = ScriptedLLM([
        {'tool_calls': [{'name': 'async_tool', 'arguments': {}}]},
        {'text': 'awaited cleanly'},
    ])
    dispatcher = ToolDispatcher(
        llm=llm, registry=registry, telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        out = await dispatcher.handle(DispatcherInput(user=user, text='hi'))

    coroutine_warnings = [w for w in captured if 'coroutine' in str(w.message).lower() and 'never awaited' in str(w.message).lower()]
    assert coroutine_warnings == [], (
        f"dispatcher emitted 'coroutine never awaited' warnings — async tool "
        f"was not awaited: {[str(w.message) for w in coroutine_warnings]}"
    )
    assert invocations == [user.id], 'async_tool was not actually invoked'
    assert out.text == 'awaited cleanly'

    # The second LLM call's contents must contain the awaited result, NOT
    # a coroutine repr.
    second_contents = repr(llm.calls[1]['contents'])
    assert '<coroutine object' not in second_contents, (
        f"dispatcher fed a coroutine object to Gemini instead of awaited "
        f"result. contents={second_contents}"
    )
    assert "'awaited': True" in second_contents


@pytest.mark.asyncio
async def test_tool_catalog_passed_to_llm_contains_all_registered_tools(populated_registry, container, telos_service):
    user = _user(container, 111)
    llm = ScriptedLLM([{'text': 'hi'}])
    dispatcher = ToolDispatcher(
        llm=llm,
        registry=populated_registry,
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )
    await dispatcher.handle(DispatcherInput(user=user, text='hello'))
    catalog = llm.calls[0]['tool_catalog']
    catalog_names = {entry['name'] for entry in catalog}
    assert {'get_current_time', 'list_active_reminders', 'list_pending_tasks'} <= catalog_names
