"""Deterministic self-correction echo on reminder confirmations.

When the user revises themselves in one message ("June 2 no June 4") and the
turn creates a reminder, the dispatcher must append a 'Corrected ...' line even
if the reply model did not phrase one itself. Scoped to reminder-creating turns.
"""
from __future__ import annotations

from typing import Any

import pytest

from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry, ToolResult


class ScriptedLLM:
    def __init__(self, script: list[dict[str, Any]]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
        self.calls.append({'system_prompt': system_prompt})
        if not self.script:
            raise AssertionError('ScriptedLLM exhausted')
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
def reminder_registry():
    registry = ToolRegistry()

    def create_reminder(*, user_id: str, body: str = '', next_fire_at: str = '', recurrence=None) -> ToolResult:
        return ToolResult.ok(
            data={'created': True, 'reminder_id': 'r1', 'body': body},
            announcement='reminder created',
        )

    registry.register(
        create_reminder,
        name='create_reminder',
        description='Create a reminder.',
        parameters={'type': 'object', 'properties': {}, 'required': []},
    )
    return registry


def _dispatcher(container, registry, telos_service, llm):
    return ToolDispatcher(
        llm=llm,
        registry=registry,
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )


def _user(container, telegram_id=111):
    return container.users_repository.get_or_create(telegram_id)


@pytest.mark.asyncio
async def test_correction_line_appended_when_model_omits_it(reminder_registry, container, telos_service):
    user = _user(container)
    llm = ScriptedLLM([
        {'tool_calls': [{'name': 'create_reminder', 'arguments': {'body': 'take Kia to mechanic'}}]},
        {'text': "Done. I'll remind you to take Kia to the mechanic."},
    ])
    dispatcher = _dispatcher(container, reminder_registry, telos_service, llm)

    out = await dispatcher.handle(DispatcherInput(
        user=user, text='remind me to take Kia to the mechanic june 2 no june 4',
    ))

    assert out.text.endswith('Corrected date: June 2 -> June 4')
    assert out.metadata.get('self_correction_echoed') == {
        'field': 'date', 'old': 'June 2', 'new': 'June 4',
    }


@pytest.mark.asyncio
async def test_no_duplicate_when_model_already_echoed(reminder_registry, container, telos_service):
    user = _user(container)
    llm = ScriptedLLM([
        {'tool_calls': [{'name': 'create_reminder', 'arguments': {'body': 'call mom'}}]},
        {'text': 'Done. Corrected date: June 2 -> June 4.'},
    ])
    dispatcher = _dispatcher(container, reminder_registry, telos_service, llm)

    out = await dispatcher.handle(DispatcherInput(
        user=user, text='remind me to call mom june 2 no june 4',
    ))

    assert out.text.lower().count('corrected') == 1


@pytest.mark.asyncio
async def test_no_correction_line_without_self_correction(reminder_registry, container, telos_service):
    user = _user(container)
    llm = ScriptedLLM([
        {'tool_calls': [{'name': 'create_reminder', 'arguments': {'body': 'pay rent'}}]},
        {'text': "Done. I'll remind you to pay rent."},
    ])
    dispatcher = _dispatcher(container, reminder_registry, telos_service, llm)

    out = await dispatcher.handle(DispatcherInput(
        user=user, text='remind me to pay rent june 4',
    ))

    assert 'corrected' not in out.text.lower()
    assert 'self_correction_echoed' not in out.metadata
