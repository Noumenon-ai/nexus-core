"""V3.6 dispatcher integration: user + assistant turn archive write.

Checkpoint 1 (2026-05-04): user-turn write only.
Checkpoint 2 (2026-05-04): assistant-turn write wired AND mem0
persistence path updates archive row on success — see
test_dispatcher_writes_both_turns_to_archive (the flipped-from-guard
test). H2-018 (sync-blocking mem0.add) is out of V3.6 scope.

See HARDENING_PASS_V2.md H2-019 / H2-020 for the underlying spec
re-scope and the "phantom internal data source" lesson.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from models import ConversationTurn
from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from repositories.conversation_turns_repository import (
    CONVERSATION_SILENCE_GAP,
    ConversationTurnsRepository,
)
from services.read_tools import register_read_tools
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry


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


def _make_dispatcher(container, registry, telos_service, llm) -> ToolDispatcher:
    return ToolDispatcher(
        llm=llm,
        registry=registry,
        telos_service=telos_service,
        mem0=_StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )


def _all_turns(container) -> list[ConversationTurn]:
    with Session(container.database.engine) as session:
        return list(session.scalars(select(ConversationTurn).order_by(ConversationTurn.created_at)).all())


@pytest.mark.asyncio
async def test_dispatcher_writes_user_turn_to_archive(container, registry, telos_service):
    user = container.users_repository.get_or_create(111)
    llm = _ScriptedLLM([{'text': 'hi back'}])
    dispatcher = _make_dispatcher(container, registry, telos_service, llm)

    await dispatcher.handle(DispatcherInput(user=user, text='hello nexus'))

    turns = _all_turns(container)
    user_turns = [t for t in turns if t.role == 'user']
    assert len(user_turns) == 1, f'expected 1 user-row, got {len(user_turns)} ({[t.role for t in turns]})'
    assert user_turns[0].user_id == user.id
    assert user_turns[0].content == 'hello nexus'
    assert user_turns[0].conversation_id  # any non-empty UUID is fine


@pytest.mark.asyncio
async def test_dispatcher_writes_both_turns_to_archive(container, registry, telos_service):
    """V3.6 checkpoint 2: ToolDispatcher.handle writes BOTH the user
    turn AND the assistant reply to conversation_turns. Both rows share
    the same conversation_id and reflect the same user_id. The user
    row is written before the LLM loop; the assistant row is written
    after final_text is composed.

    This test is the flipped form of the V3.6-checkpoint-1 guard
    `test_dispatcher_user_turn_includes_no_assistant_row_yet` — the
    flip is the visible signal that checkpoint 2's scope landed.
    """
    user = container.users_repository.get_or_create(111)
    llm = _ScriptedLLM([{'text': 'reply text'}])
    dispatcher = _make_dispatcher(container, registry, telos_service, llm)

    await dispatcher.handle(DispatcherInput(user=user, text='ping'))

    turns = _all_turns(container)
    assert [t.role for t in turns] == ['user', 'assistant']
    user_turn, assistant_turn = turns
    assert user_turn.user_id == user.id
    assert assistant_turn.user_id == user.id
    assert user_turn.content == 'ping'
    assert assistant_turn.content == 'reply text'
    assert user_turn.conversation_id == assistant_turn.conversation_id


@pytest.mark.asyncio
async def test_two_dispatcher_calls_within_2hr_share_conversation_id(container, registry, telos_service):
    user = container.users_repository.get_or_create(111)
    llm = _ScriptedLLM([{'text': 'r1'}, {'text': 'r2'}])
    dispatcher = _make_dispatcher(container, registry, telos_service, llm)

    await dispatcher.handle(DispatcherInput(user=user, text='first message'))
    await dispatcher.handle(DispatcherInput(user=user, text='second message'))

    turns = _all_turns(container)
    user_turns = [t for t in turns if t.role == 'user']
    assert len(user_turns) == 2
    assert user_turns[0].conversation_id == user_turns[1].conversation_id


@pytest.mark.asyncio
async def test_two_users_get_distinct_conversation_ids(container, registry, telos_service):
    a = container.users_repository.get_or_create(111)
    b = container.users_repository.get_or_create(222)
    llm = _ScriptedLLM([{'text': 'r1'}, {'text': 'r2'}])
    dispatcher = _make_dispatcher(container, registry, telos_service, llm)

    await dispatcher.handle(DispatcherInput(user=a, text='from A'))
    await dispatcher.handle(DispatcherInput(user=b, text='from B'))

    turns = _all_turns(container)
    by_user = {t.user_id: t for t in turns if t.role == 'user'}
    assert by_user[a.id].conversation_id != by_user[b.id].conversation_id
