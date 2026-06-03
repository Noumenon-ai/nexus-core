from __future__ import annotations

import asyncio

import pytest

import pipeline.tool_dispatcher as dispatcher_module
from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry
from utils.i18n import Translator


class _FailIfCalledLLM:
    async def generate_with_tools(self, **_kwargs):
        raise AssertionError('LLM should not be called for these recovery prelude checks')


class _HangingLLM:
    async def generate_with_tools(self, **_kwargs):
        await asyncio.Event().wait()
        raise AssertionError('unreachable')


class _StubMem0:
    def search(self, *args, **kwargs):
        return []

    def add(self, *args, **kwargs):
        return {'results': []}


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


def _unique_acme_alias(query: str) -> dict[str, object] | None:
    if query.strip().casefold() != 'acme':
        return None
    return {
        'ok': True,
        'match': 'unique',
        'alias_used': 'Acme Corp',
        'contact': {'aliases': ['Acme Corp']},
    }


def _alias_registry(resolve_fn=_unique_acme_alias) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        lambda *, user_id, query: resolve_fn(query),
        name='resolve_contact_alias',
        description='Resolve a local contact alias.',
        parameters={
            'type': 'object',
            'properties': {'query': {'type': 'string'}},
            'required': ['query'],
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


@pytest.mark.asyncio


@pytest.mark.asyncio
async def test_negated_recipient_blocks_send_before_approval(container, telos_service):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(
        container,
        telos_service,
        llm=_FailIfCalledLLM(),
        registry=_alias_registry(),
    )

    out = await dispatcher.handle(DispatcherInput(
        user=user,
        text='tell acme ill do it tmrrw nah not her the other one wait no friday',
    ))

    assert out.text == 'You said not Acme Corp — who should I send it to?'
    assert out.metadata.get('recovery_clarification') is True
    assert out.buttons == []
    assert container.approvals_repository.list_active_pending_for_user(user.id) == []


@pytest.mark.asyncio
async def test_send_with_weekday_correction_can_still_arm_approval(container, telos_service):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(
        container,
        telos_service,
        llm=_FailIfCalledLLM(),
        registry=_alias_registry(),
    )

    out = await dispatcher.handle(DispatcherInput(
        user=user,
        text='tell acme ill do it tmrrw wait no friday',
    ))

    pending = container.approvals_repository.list_active_pending_for_user(user.id)

    assert out.metadata.get('destructive_gate') is True
    assert len(pending) == 1
    assert 'Friday' in pending[0].preview_text
    assert 'tomorrow' not in pending[0].preview_text.lower()


@pytest.mark.asyncio
async def test_post_approval_timeout_with_negated_recipient_clarifies(container, telos_service, monkeypatch):
    monkeypatch.setattr(
        dispatcher_module,
        '_POST_APPROVAL_CONTINUATION_TIMEOUT_SEC',
        0.01,
        raising=False,
    )

    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(
        container,
        telos_service,
        llm=_HangingLLM(),
        registry=_alias_registry(),
    )

    sr = container.approval_service.request(
        user,
        action_type='destructive_message_gate',
        preview_text='Send a message? (double-confirm)',
        payload={
            'original_prompt': 'send message to Acme Corp: i will do it Friday',
            'recovery': {
                'recipient_negated_unresolved': True,
                'negated_recipient_label': 'Acme Corp',
                'negated_recipient_reason': 'other_one',
            },
            'user_id': user.id,
        },
        translator=Translator('en'),
    )
    approval_id = next(
        button.callback_data.split(':', 2)[2]
        for button in sr.buttons
        if 'approve' in button.callback_data
    )

    post_out = await dispatcher.handle(
        DispatcherInput(user=user, text=f'approval:approve:{approval_id}')
    )

    assert post_out.metadata.get('post_approval_timeout') is True
    assert post_out.metadata.get('post_approval_local_fallback') is True
    assert post_out.text == 'You said not Acme Corp — who should I send it to?'
