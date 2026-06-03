from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

import pipeline.tool_dispatcher as dispatcher_module
from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry, ToolResult


class _ScriptedLLM:
    def __init__(self, script: list[dict[str, Any]]):
        self.script = list(script)

    async def generate_with_tools(self, **_kwargs):
        if not self.script:
            raise AssertionError('Script exhausted')
        return self.script.pop(0)


class _FailIfCalledLLM:
    async def generate_with_tools(self, **_kwargs):
        raise AssertionError('LLM should not be called for birthday prelude memory confirmation')


class _StubMem0:
    def search(self, *args, **kwargs):
        return []

    def add(self, *args, **kwargs):
        return {'results': []}


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


def _memory_registry(container) -> ToolRegistry:
    registry = ToolRegistry()

    def save_user_memory(*, user_id: str, memory_type: str, key: str, value: str) -> ToolResult:
        memory = container.memories_repository.upsert(
            user_id=user_id,
            memory_type=memory_type,
            key=key,
            value=value,
            confidence=1.0,
            source='explicit',
        )
        return ToolResult.ok(
            data={'saved': True, 'memory_id': memory.id, 'key': memory.key},
            announcement='Saved memory.',
        )

    registry.register(
        save_user_memory,
        name='save_user_memory',
        description='Save a user memory.',
        parameters={
            'type': 'object',
            'properties': {
                'memory_type': {'type': 'string'},
                'key': {'type': 'string'},
                'value': {'type': 'string'},
            },
            'required': ['memory_type', 'key', 'value'],
        },
    )
    return registry


def _build_dispatcher(container, telos_service, *, llm, registry: ToolRegistry) -> ToolDispatcher:
    return ToolDispatcher(
        llm=llm,
        registry=registry,
        telos_service=telos_service,
        mem0=_StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        conversation_service=container.conversation_service,
        approvals_repository=container.approvals_repository,
        max_iterations=5,
    )


@pytest.mark.asyncio
async def test_tomorrow_birthday_does_not_persist_without_confirmation(container, telos_service, monkeypatch):
    user = container.users_repository.get_or_create(111)
    monkeypatch.setattr(
        dispatcher_module,
        'app_now',
        lambda _timezone: datetime(2026, 5, 26, 9, 0, tzinfo=timezone.utc),
    )
    dispatcher = _build_dispatcher(
        container,
        telos_service,
        llm=_FailIfCalledLLM(),
        registry=_memory_registry(container),
    )

    out = await dispatcher.handle(DispatcherInput(user=user, text='Tomorrow its my bday'))

    memories = container.memories_repository.list_by_user(user.id)

    assert out.text == (
        "Got it — tomorrow's your birthday, May 27. "
        "Want me to remember that for next year? "
        "I can wish you tomorrow without saving it permanently."
    )
    assert memories == []


@pytest.mark.asyncio
async def test_explicit_remember_birthday_still_saves(container, telos_service):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(
        container,
        telos_service,
        llm=_ScriptedLLM([
            {
                'tool_calls': [{
                    'name': 'save_user_memory',
                    'arguments': {
                        'memory_type': 'fact',
                        'key': 'birthday',
                        'value': 'May 27',
                    },
                }],
            },
            {'text': 'Remembered.'},
        ]),
        registry=_memory_registry(container),
    )

    out = await dispatcher.handle(
        DispatcherInput(user=user, text='remember my birthday is May 27')
    )

    memories = container.memories_repository.list_by_user(user.id)

    assert out.text == 'Remembered.'
    assert [(m.memory_type, m.key, m.value) for m in memories] == [
        ('fact', 'birthday', 'May 27')
    ]
