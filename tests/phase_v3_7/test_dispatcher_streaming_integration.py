"""V3.7 dispatcher integration: streaming hooks fire at the right beats.

Four spec-mandated tests:
1. Thinking-first — first update text is "Thinking..."
2. Tool-specific stage — registered tool name maps to its
   TOOL_STAGE_MESSAGES entry on its update beat
3. Unknown-tool fallback — tool name absent from mapping uses
   the generic "Working on it..." default
4. No streaming when streaming_session=None — pre-V3.7 dispatcher
   behavior preserved exactly; no edits happen anywhere

Halt-condition coverage:
- Existing 18 V3.6 sites that construct DispatcherInput WITHOUT
  streaming_session must still pass. Confirmed by full V3.5+V3.6
  suite re-run alongside this module's tests.
- StreamingSession instantiated but never reaches dispatcher: not
  possible by construction here — all fixtures pass it via
  DispatcherInput. Tests assert the bot saw the expected calls,
  which would fail if the dispatcher never invoked the session.
- finalize() called twice on same session: covered by
  test_dispatcher_calls_finalize_exactly_once (5th test, beyond spec
  floor — same "noticed a foot-gun while writing tests" pattern).
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from pipeline.tool_dispatcher import (
    DispatcherInput,
    ToolDispatcher,
    TOOL_STAGE_MESSAGES,
    _GENERIC_TOOL_STAGE,
)
from services.read_tools import register_read_tools
from services.telegram_streaming import StreamingSession
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry, ToolResult
from utils.dates import utc_now


class _ScriptedLLM:
    def __init__(self, replies: list[dict[str, Any]]):
        self._replies = list(replies)

    async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
        return self._replies.pop(0)


class _StubMem0:
    def search(self, query, *, user_id, limit=5):
        return []

    def add(self, messages, *, user_id):
        pass


class _RecorderBot:
    """Captures every send_text / edit_text call. Streaming session
    drives this in unit tests so we can read the exact sequence of
    placeholder events the dispatcher emitted."""

    def __init__(self):
        self.send_calls: list[dict[str, Any]] = []
        self.edit_calls: list[dict[str, Any]] = []
        self._next_id = 5000

    async def send_text(self, *, chat_id: int, text: str) -> int:
        self.send_calls.append({'chat_id': chat_id, 'text': text})
        msg_id = self._next_id
        self._next_id += 1
        return msg_id

    async def edit_text(self, *, chat_id: int, message_id: int, text: str) -> None:
        self.edit_calls.append({'chat_id': chat_id, 'message_id': message_id, 'text': text})


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


def _make_dispatcher(container, registry, telos_service) -> ToolDispatcher:
    return ToolDispatcher(
        llm=_ScriptedLLM([{'text': 'placeholder'}]),  # overridden per test
        registry=registry,
        telos_service=telos_service,
        mem0=_StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )


def _make_session(bot, *, time_provider=lambda: 100.0) -> StreamingSession:
    """Throttle-aware time provider that returns increasing values
    far apart enough that every update fires through. Tests that
    care about throttle behavior live in test_telegram_streaming.py."""
    counter = {'t': 100.0}
    def _ticking():
        counter['t'] += 1.0
        return counter['t']
    return StreamingSession(chat_id=42, telegram_bot=bot, time_provider=_ticking)


# ---- (1) Thinking-first ----------------------------------------------------

@pytest.mark.asyncio
async def test_dispatcher_emits_thinking_stage_first(container, registry, telos_service):
    bot = _RecorderBot()
    user = container.users_repository.get_or_create(111)
    dispatcher = _make_dispatcher(container, registry, telos_service)
    dispatcher.llm = _ScriptedLLM([{'text': 'Hi.'}])
    session = _make_session(bot)

    await dispatcher.handle(DispatcherInput(user=user, text='hello', streaming_session=session))

    # First wrapper call must be a SEND with text "Thinking..." — the
    # placeholder appearing tells the user the bot received their
    # message and is working.
    assert len(bot.send_calls) >= 1
    assert bot.send_calls[0]['text'] == 'Thinking...'


# ---- (2) Tool-specific stage ----------------------------------------------

@pytest.mark.asyncio
async def test_dispatcher_emits_tool_specific_stage(container, registry, telos_service):
    """When list_pending_tasks fires, the user must see
    'Looking up your tasks...' as a stage update — not the generic
    fallback."""
    bot = _RecorderBot()
    user = container.users_repository.get_or_create(111)
    dispatcher = _make_dispatcher(container, registry, telos_service)
    dispatcher.llm = _ScriptedLLM([
        {'tool_calls': [{'name': 'list_pending_tasks', 'arguments': {}}]},
        {'text': 'You have no pending tasks.'},
    ])
    session = _make_session(bot)

    await dispatcher.handle(DispatcherInput(user=user, text='whats my tasks', streaming_session=session))

    expected_stage = TOOL_STAGE_MESSAGES['list_pending_tasks']
    seen_stage_texts = [c['text'] for c in bot.edit_calls] + [c['text'] for c in bot.send_calls]
    assert expected_stage in seen_stage_texts, (
        f'Expected to see {expected_stage!r} on a stage update, but only '
        f'these texts hit the bot: {seen_stage_texts}'
    )


# ---- (3) Unknown-tool fallback --------------------------------------------

@pytest.mark.asyncio
async def test_dispatcher_emits_unknown_tool_fallback(container, registry, telos_service):
    """When the LLM emits a tool_call for a name not in the registry
    (and therefore not in TOOL_STAGE_MESSAGES), the dispatcher's
    streaming layer must still emit a stage update — the generic
    fallback. This covers a real failure mode: model hallucination
    of tool names should still feel like progress to the user."""
    bot = _RecorderBot()
    user = container.users_repository.get_or_create(111)
    dispatcher = _make_dispatcher(container, registry, telos_service)
    dispatcher.llm = _ScriptedLLM([
        {'tool_calls': [{'name': 'imaginary_tool_does_not_exist', 'arguments': {}}]},
        {'text': 'Sorry, I tried something I cannot do.'},
    ])
    session = _make_session(bot)

    await dispatcher.handle(DispatcherInput(user=user, text='hi', streaming_session=session))

    seen_texts = [c['text'] for c in bot.edit_calls] + [c['text'] for c in bot.send_calls]
    assert _GENERIC_TOOL_STAGE in seen_texts, (
        f'Expected generic fallback {_GENERIC_TOOL_STAGE!r} for unknown tool, '
        f'saw: {seen_texts}'
    )


# ---- (4) No streaming when session is None --------------------------------

@pytest.mark.asyncio
async def test_dispatcher_no_streaming_when_session_none(container, registry, telos_service):
    """streaming_session=None (or omitted) must produce zero bot
    interactions through the streaming path. This is the V3.5/V3.6
    backward-compat guarantee: existing 18 dispatcher test sites
    construct DispatcherInput without streaming_session and must
    keep working untouched.
    """
    user = container.users_repository.get_or_create(111)
    dispatcher = _make_dispatcher(container, registry, telos_service)
    dispatcher.llm = _ScriptedLLM([{'text': 'Hi.'}])

    out = await dispatcher.handle(DispatcherInput(user=user, text='hello'))

    assert out.text == 'Hi.'
    # `streamed` flag should be absent / falsy in metadata when no session.
    assert not out.metadata.get('streamed'), (
        f'streamed flag must be False/absent when no session: {out.metadata}'
    )


# ---- (5) finalize-exactly-once safety guard (above spec floor) -------------

@pytest.mark.asyncio
async def test_dispatcher_calls_finalize_exactly_once(container, registry, telos_service):
    """Halt condition: 'finalize() called twice on same session
    somehow (double-send to telegram) → halt'. Guard against that
    by asserting the session's final_sent flag is True after handle()
    returns AND the dispatcher does not call finalize again on a
    handle() that returns final_text.

    Implementation detail: a second call would either no-op (because
    final_sent==True is checked at top of finalize) OR error. Either
    way, we assert post-conditions.
    """
    bot = _RecorderBot()
    user = container.users_repository.get_or_create(111)
    dispatcher = _make_dispatcher(container, registry, telos_service)
    dispatcher.llm = _ScriptedLLM([{'text': 'Hi.'}])
    session = _make_session(bot)

    out = await dispatcher.handle(DispatcherInput(user=user, text='hello', streaming_session=session))

    assert session.final_sent is True
    assert out.metadata.get('streamed') is True
    # Exactly one wrapper event per chunk sent — the placeholder + final
    # short-message edit. For 'Hi.' (one chunk), expect 1 send + 1 edit.
    assert len(bot.send_calls) == 1, f'Expected exactly 1 send (placeholder), got {bot.send_calls}'
    assert len(bot.edit_calls) == 1, f'Expected exactly 1 edit (finalize), got {bot.edit_calls}'
    # The finalize edit must carry the final text, not the stage text.
    assert bot.edit_calls[-1]['text'] == 'Hi.'


# ---- (6) per-tool-call stage actually triggers PER tool call ---------------

@pytest.mark.asyncio
async def test_dispatcher_emits_stage_for_each_tool_in_multi_call_turn(container, registry, telos_service):
    """When the LLM returns multiple tool_calls in one turn, each one
    must trigger its own stage update. Without this, multi-tool
    queries (e.g. status briefings) feel frozen mid-turn."""
    bot = _RecorderBot()
    user = container.users_repository.get_or_create(111)
    dispatcher = _make_dispatcher(container, registry, telos_service)
    dispatcher.llm = _ScriptedLLM([
        {'tool_calls': [
            {'name': 'list_pending_tasks', 'arguments': {}},
            {'name': 'list_active_reminders', 'arguments': {}},
        ]},
        {'text': 'Done.'},
    ])
    session = _make_session(bot)

    await dispatcher.handle(DispatcherInput(user=user, text='status', streaming_session=session))

    seen_texts = [c['text'] for c in bot.edit_calls] + [c['text'] for c in bot.send_calls]
    assert TOOL_STAGE_MESSAGES['list_pending_tasks'] in seen_texts
    assert TOOL_STAGE_MESSAGES['list_active_reminders'] in seen_texts
