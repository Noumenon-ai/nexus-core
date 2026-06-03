"""V3.5 dispatcher: TELOS + mem0 wiring at entry, mem0.add scheduled
non-blocking after final reply. Cross-user isolation enforced.
"""
from __future__ import annotations

from typing import Any

import pytest

from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.read_tools import register_read_tools
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
        self.calls.append({'system_prompt': system_prompt, 'contents': contents})
        return self.script.pop(0)


class RecordingMem0:
    def __init__(self, search_result=None):
        self.search_calls: list[dict[str, Any]] = []
        self.add_calls: list[dict[str, Any]] = []
        self._search_result = search_result or []

    def search(self, query, *, user_id, limit: int = 5):
        self.search_calls.append({'query': query, 'user_id': user_id, 'limit': limit})
        return self._search_result

    def add(self, messages, *, user_id):
        self.add_calls.append({'messages': messages, 'user_id': user_id})


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


@pytest.fixture
def registry(container, telos_service):
    r = ToolRegistry()
    register_read_tools(
        r,
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        emails_repository=container.emails_repository,
        approvals_repository=container.approvals_repository,
        telos_service=telos_service,
        app_timezone='UTC',
    )
    return r


def _user(container, telegram_id):
    return container.users_repository.get_or_create(telegram_id)


# -------- TELOS load at entry -----------------------------------------------

@pytest.mark.asyncio
async def test_telos_content_present_appears_in_system_prompt(registry, container, telos_service):
    user = _user(container, 111)
    telos_service.path_for(user.id).write_text('# Telos\nI am building Nexus.\nGoal: ship V3.', encoding='utf-8')

    llm = ScriptedLLM([{'text': 'hi'}])
    mem0 = RecordingMem0()
    dispatcher = ToolDispatcher(
        llm=llm, registry=registry, telos_service=telos_service, mem0=mem0,
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )
    await dispatcher.handle(DispatcherInput(user=user, text='hello'))
    sp = llm.calls[0]['system_prompt']
    assert 'I am building Nexus.' in sp
    assert 'Goal: ship V3.' in sp


@pytest.mark.asyncio
async def test_telos_absent_handled_gracefully(registry, container, telos_service):
    """A user without a TELOS file must not crash the dispatcher; system
    prompt simply omits the TELOS section or marks it empty."""
    user = _user(container, 111)
    llm = ScriptedLLM([{'text': 'hi'}])
    dispatcher = ToolDispatcher(
        llm=llm, registry=registry, telos_service=telos_service, mem0=RecordingMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )
    out = await dispatcher.handle(DispatcherInput(user=user, text='hello'))
    assert out.text == 'hi'  # no crash


# -------- mem0 search at entry ----------------------------------------------

@pytest.mark.asyncio
async def test_mem0_search_runs_at_dispatcher_entry_with_user_message(registry, container, telos_service):
    user = _user(container, 111)
    mem0 = RecordingMem0(search_result=[
        {'memory': 'User mentioned a bank thing on 2026-04-30'},
        {'memory': 'User finished V3.4 on 2026-05-03'},
    ])
    llm = ScriptedLLM([{'text': 'About the bank thing — yes I remember.'}])
    dispatcher = ToolDispatcher(
        llm=llm, registry=registry, telos_service=telos_service, mem0=mem0,
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )
    await dispatcher.handle(DispatcherInput(user=user, text='the bank thing I mentioned last week'))
    assert len(mem0.search_calls) == 1
    assert mem0.search_calls[0]['user_id'] == user.id
    assert 'bank thing' in mem0.search_calls[0]['query'].lower()
    sp = llm.calls[0]['system_prompt']
    assert 'bank thing' in sp.lower()
    assert 'V3.4' in sp


# -------- mem0.add scheduled after final reply, non-blocking ---------------

@pytest.mark.asyncio
async def test_mem0_add_scheduled_after_final_reply(registry, container, telos_service):
    user = _user(container, 111)
    mem0 = RecordingMem0()
    llm = ScriptedLLM([{'text': 'reply text'}])
    dispatcher = ToolDispatcher(
        llm=llm, registry=registry, telos_service=telos_service, mem0=mem0,
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )
    out = await dispatcher.handle(DispatcherInput(user=user, text='hello world'))
    assert out.text == 'reply text'
    # H2-046 Part 0: archival is fire-and-forget. Drain the task before
    # asserting on mem0's side effects.
    await dispatcher.wait_for_archival_idle()
    # mem0.add should have been called with both the user message and
    # assistant reply, scoped to the right user.
    assert len(mem0.add_calls) == 1
    assert mem0.add_calls[0]['user_id'] == user.id
    serialized = repr(mem0.add_calls[0]['messages'])
    assert 'hello world' in serialized
    assert 'reply text' in serialized


@pytest.mark.asyncio
async def test_mem0_add_failure_does_not_break_reply(registry, container, telos_service):
    """mem0.add() is fire-and-forget; a failure must NOT prevent the user
    from getting their reply."""
    class FailingMem0:
        def search(self, *a, **kw): return []
        def add(self, *a, **kw): raise RuntimeError('mem0 down')

    user = _user(container, 111)
    llm = ScriptedLLM([{'text': 'reply text'}])
    dispatcher = ToolDispatcher(
        llm=llm, registry=registry, telos_service=telos_service, mem0=FailingMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )
    out = await dispatcher.handle(DispatcherInput(user=user, text='hello'))
    assert out.text == 'reply text'


# -------- cross-user isolation in dispatcher entry --------------------------

@pytest.mark.asyncio
async def test_user_a_cannot_see_user_b_telos(registry, container, telos_service):
    a = _user(container, 111)
    b = _user(container, 222)
    telos_service.path_for(a.id).write_text('A SECRET telos', encoding='utf-8')
    telos_service.path_for(b.id).write_text('B SECRET telos', encoding='utf-8')

    llm = ScriptedLLM([{'text': 'hi'}])
    dispatcher = ToolDispatcher(
        llm=llm, registry=registry, telos_service=telos_service, mem0=RecordingMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )
    await dispatcher.handle(DispatcherInput(user=a, text='hello'))
    sp = llm.calls[0]['system_prompt']
    assert 'A SECRET' in sp
    assert 'B SECRET' not in sp


@pytest.mark.asyncio
async def test_mem0_search_filtered_per_user(registry, container, telos_service):
    a = _user(container, 111)
    b = _user(container, 222)
    mem0 = RecordingMem0()
    llm = ScriptedLLM([{'text': 'hi'}, {'text': 'hi'}])
    dispatcher = ToolDispatcher(
        llm=llm, registry=registry, telos_service=telos_service, mem0=mem0,
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )
    await dispatcher.handle(DispatcherInput(user=a, text='hello'))
    await dispatcher.handle(DispatcherInput(user=b, text='hello'))
    user_ids = [call['user_id'] for call in mem0.search_calls]
    assert user_ids == [a.id, b.id]
