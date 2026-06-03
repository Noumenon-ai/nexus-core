from __future__ import annotations

import pytest

from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.conversational_recovery import ConversationalRecoveryLayer
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry


class _FailIfCalledLLM:
    async def generate_with_tools(self, **_kwargs):
        raise AssertionError('LLM should not be called for these recovery prelude checks')


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


def _ambiguous_acme_alias(query: str) -> dict[str, object] | None:
    if query.strip().casefold() != 'acme':
        return None
    return {
        'ok': True,
        'match': 'ambiguous',
        'candidates': [
            {'aliases': ['Acme Corp']},
            {'aliases': ['Bajo']},
        ],
    }


def _alias_registry(resolve_fn):
    registry = ToolRegistry()
    registry.register(
        lambda *, user_id, query: resolve_fn(query),
        name='resolve_contact_alias',
        description='Resolve a local contact alias.',
        parameters={
            'type': 'object',
            'properties': {
                'query': {'type': 'string'},
            },
            'required': ['query'],
        },
    )
    return registry


def test_recovery_normalizes_rental_shorthand_to_read_intent():
    layer = ConversationalRecoveryLayer()

    result = layer.recover(text='did u do the rentals m')

    assert result.outcome == 'auto_resolve'
    assert result.recovered_text == 'check rental update status for your rental records'
    assert result.suppress_vague_clarification is True
    assert result.risk_level == 'low'
    assert result.resolved_slots['action_kind'] == 'rental_status_check'


def test_recovery_reuses_last_recipient_and_message_body_for_generic_followup():
    layer = ConversationalRecoveryLayer()

    first = layer.recover(
        text='tell acme ill do it tmrrw',
        resolve_contact_alias=_unique_acme_alias,
    )

    second = layer.recover(
        text='send her the thing',
        context={'recovery_state': first.context_updates},
    )

    assert first.context_updates['last_recipient'] == 'Acme Corp'
    assert first.context_updates['last_message_body'] == 'i will do it tomorrow'
    assert second.outcome == 'auto_resolve'
    assert second.recovered_text == 'send message to Acme Corp: i will do it tomorrow'
    assert second.resolved_slots['recipient'] == 'Acme Corp'
    assert second.resolved_slots['message_body'] == 'i will do it tomorrow'


def test_recovery_uses_other_one_correction_memory():
    layer = ConversationalRecoveryLayer()

    result = layer.recover(
        text='nah not her the other one',
        context={'recovery_state': {
            'last_action_kind': 'outbound_message',
            'last_recipient': 'Sarah',
            'recipient_candidates': ['Sarah', 'Acme Corp'],
        }},
    )

    assert result.outcome == 'auto_resolve'
    assert result.recovered_text == 'use Acme Corp instead'
    assert result.resolved_slots['recipient'] == 'Acme Corp'
    assert 'recipient' in result.corrections_applied


def test_recovery_applies_latest_weekday_correction_from_short_reply():
    layer = ConversationalRecoveryLayer()

    result = layer.recover(
        text='wait no friday',
        context={'recovery_state': {
            'last_action_kind': 'follow_up',
            'last_time_label': 'Thursday',
        }},
    )

    assert result.outcome == 'auto_resolve'
    assert result.recovered_text == 'use Friday instead'
    assert result.resolved_slots['time_label'] == 'Friday'
    assert 'time_label' in result.corrections_applied


@pytest.mark.asyncio
async def test_dispatcher_uses_recovered_alias_text_before_destructive_gate(
    container,
    telos_service,
):
    user = container.users_repository.get_or_create(111)
    dispatcher = ToolDispatcher(
        llm=_FailIfCalledLLM(),
        registry=_alias_registry(_unique_acme_alias),
        telos_service=telos_service,
        mem0=_StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        conversation_service=container.conversation_service,
        approvals_repository=container.approvals_repository,
        max_iterations=5,
    )

    out = await dispatcher.handle(DispatcherInput(user=user, text='tell acme ill do it tmrrw'))

    pending = container.approvals_repository.list_active_pending_for_user(user.id)

    assert out.metadata.get('destructive_gate') is True
    assert out.metadata.get('recovery_applied') is True
    assert len(pending) == 1
    assert 'Acme Corp' in pending[0].preview_text
    assert 'tomorrow' in pending[0].preview_text
    assert 'tmrrw' not in pending[0].preview_text
    assert container.conversation_service.get_recovery_context(user.id)['last_recipient'] == 'Acme Corp'


@pytest.mark.asyncio
async def test_dispatcher_clarifies_specific_ambiguous_alias_before_gate(
    container,
    telos_service,
):
    user = container.users_repository.get_or_create(111)
    dispatcher = ToolDispatcher(
        llm=_FailIfCalledLLM(),
        registry=_alias_registry(_ambiguous_acme_alias),
        telos_service=telos_service,
        mem0=_StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        conversation_service=container.conversation_service,
        approvals_repository=container.approvals_repository,
        max_iterations=5,
    )

    out = await dispatcher.handle(DispatcherInput(user=user, text='send acme the lease'))

    assert out.text == 'You mean Acme Corp or Bajo?'
    assert out.metadata['recovery_clarification'] is True
    assert out.buttons == []
    assert container.approvals_repository.list_active_pending_for_user(user.id) == []
