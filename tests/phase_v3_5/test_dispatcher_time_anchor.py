"""V3.5.1 — H2-014 follow-up patch coverage.

Verifies the dispatcher always grounds Gemini in current time so it
cannot hallucinate timestamps from training-data prior. Three tests:

1. test_dispatcher_passes_tool_results_to_followup_call — when
   get_current_time fires, its ToolResult.data reaches the second
   Gemini call's `contents` payload as a functionResponse part.
2. test_system_prompt_constrains_factual_invention — _build_system_prompt
   output always contains a Current time section AND a constraint clause
   instructing Gemini to use these values, not invent.
3. test_get_current_time_tool_result_format — the get_current_time tool
   returns ToolResult.data with parseable ISO 8601 'iso' field + 'timezone'
   string Gemini can use directly.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from pipeline.tool_dispatcher import (
    DispatcherInput,
    ToolDispatcher,
    _build_system_prompt,
)
from services.read_tools import register_read_tools
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry, ToolResult


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
        self.calls.append({
            'system_prompt': system_prompt,
            'contents': [c for c in contents],
            'tool_catalog': tool_catalog,
        })
        return self.script.pop(0)


class StubMem0:
    def search(self, query, *, user_id, limit: int = 5):
        return []

    def add(self, messages, *, user_id):
        pass


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
        app_timezone='Asia/Jerusalem',
    )
    return r


def _user(container, telegram_id):
    return container.users_repository.get_or_create(telegram_id)


# -------- Test 1: tool result reaches the followup LLM call -----------------

@pytest.mark.asyncio
async def test_dispatcher_passes_tool_results_to_followup_call(registry, container, telos_service):
    """When get_current_time fires, the ToolResult.data must reach the
    second LLM call as a functionResponse part with the actual ISO
    timestamp Gemini can read. This is the structural guard that makes
    the H2-014 fix actually plumbed end-to-end."""
    user = _user(container, 700)
    llm = ScriptedLLM([
        {'tool_calls': [{'name': 'get_current_time', 'arguments': {}}]},
        {'text': 'It is currently the time you provided.'},
    ])
    dispatcher = ToolDispatcher(
        llm=llm, registry=registry, telos_service=telos_service, mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
        app_timezone='Asia/Jerusalem',
    )

    out = await dispatcher.handle(DispatcherInput(user=user, text='what time is it'))

    assert out.text == 'It is currently the time you provided.'
    assert len(llm.calls) == 2

    second_call_contents = llm.calls[1]['contents']
    serialized = repr(second_call_contents)
    assert 'get_current_time' in serialized, 'tool name must reach followup call'
    assert 'functionResponse' in serialized, 'tool result must be packaged as functionResponse part'
    assert "'iso'" in serialized, "ToolResult.data must include 'iso' field"
    assert "'timezone'" in serialized, "ToolResult.data must include 'timezone' field"
    assert 'Asia/Jerusalem' in serialized, 'configured app_timezone must reach the followup'


# -------- Test 2: system prompt always constrains factual invention --------

def test_system_prompt_constrains_factual_invention():
    """_build_system_prompt output must always contain (a) a Current time
    section with both UTC and user-timezone timestamps, AND (b) an
    explicit constraint clause instructing Gemini to use these values
    rather than invent. Structural guard against future regressions
    that drop the time anchor or the constraint phrasing."""
    fixed_now = datetime(2026, 5, 3, 18, 4, 49, tzinfo=timezone.utc)
    prompt = _build_system_prompt(
        persona='You are Nexus.',
        telos=None,
        memories=[],
        language='en',
        now=fixed_now,
        app_timezone='Asia/Jerusalem',
    )

    assert '## Current time' in prompt, 'time section header missing'
    assert 'UTC: 2026-05-03T18:04:49+00:00' in prompt, 'UTC timestamp missing/wrong'
    assert 'Asia/Jerusalem' in prompt, 'user timezone label missing'
    assert '2026-05-03T21:04:49+03:00' in prompt, 'user-timezone-converted timestamp missing/wrong'
    assert 'Do NOT invent' in prompt, 'factual-invention constraint clause missing'
    assert 'get_current_time' in prompt, 'tool reference missing from constraint clause'


def test_system_prompt_requires_self_correction_echo():
    """When the user corrects themselves in a message, the reply must
    state the change explicitly instead of silently applying it. Guards
    the always-on self-correction instruction."""
    fixed_now = datetime(2026, 6, 2, 18, 0, 0, tzinfo=timezone.utc)
    prompt = _build_system_prompt(
        persona='You are Nexus.',
        telos=None,
        memories=[],
        language='en',
        now=fixed_now,
        app_timezone='America/New_York',
    )

    assert '## Self-corrections' in prompt, 'self-correction section header missing'
    assert 'Corrected' in prompt, 'correction echo format missing'
    assert '->' in prompt, 'old -> new arrow format missing'
    assert 'Never silently' in prompt, 'silent-drop prohibition missing'


def test_dispatcher_time_anchor_uses_app_timezone(registry, container, telos_service):
    """Server-tz leak guard: if the dispatcher is configured with a
    non-server timezone, the system prompt's user-timezone line must
    use that configured zone, not the server's default tz."""
    user = _user(container, 701)
    llm = ScriptedLLM([{'text': 'ok'}])
    dispatcher = ToolDispatcher(
        llm=llm, registry=registry, telos_service=telos_service, mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
        app_timezone='Asia/Jerusalem',
    )

    asyncio.run(dispatcher.handle(DispatcherInput(user=user, text='hi')))

    sp = llm.calls[0]['system_prompt']
    assert 'Asia/Jerusalem' in sp, 'configured app_timezone must appear in system prompt'
    # Asia/Jerusalem is +03:00 (IDT) or +02:00 (IST) — neither is the server's -04:00 EDT
    assert '-04:00' not in sp.split('## Current time')[1].split('## ')[0], (
        'server-tz EDT offset leaked into user-timezone line'
    )


# -------- Test 3: get_current_time ToolResult format ------------------------

def test_get_current_time_tool_result_format(registry):
    """get_current_time must return ToolResult.success=True with
    data={'iso': <ISO 8601 string>, 'timezone': <tz string>}. This is
    the contract the H2-014 fix relies on for both the prompt-injection
    path AND the tool-call path. Without this contract, both paths
    return junk Gemini will hallucinate against."""
    spec = registry.get('get_current_time')
    assert spec is not None, 'get_current_time must be registered in V3.2 read registry'
    assert spec.requires_approval is False, 'get_current_time must NOT require approval'

    result = spec.fn()
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.error is None
    assert isinstance(result.data, dict)
    assert 'iso' in result.data, "must expose 'iso' field"
    assert 'timezone' in result.data, "must expose 'timezone' field"

    # iso must be parseable ISO 8601
    parsed = datetime.fromisoformat(result.data['iso'])
    assert parsed.tzinfo is not None, 'iso must include timezone offset (not naive)'

    # timezone label must match the configured app_timezone the registry was built with
    assert result.data['timezone'] == 'Asia/Jerusalem'
