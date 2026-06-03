from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry


class _FailIfCalledLLM:
    async def generate_with_tools(self, **_kwargs):
        raise AssertionError('LLM should not be called for clarification-state prelude tests')


class _StubMem0:
    def search(self, *args, **kwargs):
        return []

    def add(self, *args, **kwargs):
        return {'results': []}


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


def _build_dispatcher(
    container,
    telos_service,
    *,
    registry: ToolRegistry | None = None,
) -> ToolDispatcher:
    return ToolDispatcher(
        llm=_FailIfCalledLLM(),
        registry=registry or ToolRegistry(),
        telos_service=telos_service,
        mem0=_StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        conversation_service=container.conversation_service,
        approvals_repository=container.approvals_repository,
        max_iterations=5,
    )


def _ambiguous_alex_registry() -> ToolRegistry:
    def resolve_alias(*, user_id, query):
        del user_id
        lowered = query.strip().casefold()
        if lowered == 'alex':
            return {
                'ok': True,
                'match': 'ambiguous',
                'candidates': [
                    {'aliases': ['Alex One']},
                    {'aliases': ['Alex Two']},
                ],
            }
        if lowered == 'alex one':
            return {
                'ok': True,
                'match': 'unique',
                'alias_used': 'Alex One',
                'contact': {'aliases': ['Alex One']},
            }
        if lowered == 'alex two':
            return {
                'ok': True,
                'match': 'unique',
                'alias_used': 'Alex Two',
                'contact': {'aliases': ['Alex Two']},
            }
        return None

    registry = ToolRegistry()
    registry.register(
        resolve_alias,
        name='resolve_contact_alias',
        description='Resolve a local contact alias.',
        parameters={
            'type': 'object',
            'properties': {'query': {'type': 'string'}},
            'required': ['query'],
        },
    )
    return registry


async def _arm_rental_menu(container, telos_service) -> tuple[ToolDispatcher, object]:
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(container, telos_service)

    out = await dispatcher.handle(DispatcherInput(user=user, text='Update my rentals M'))

    assert out.metadata.get('vague_clarification') is True
    stored = container.conversation_service.get_active_clarification(user.id)
    assert stored is not None
    assert stored['clarification_id']
    assert stored['question'].startswith('Do you mean:')
    assert len(stored['options']) == 3
    return dispatcher, user


@pytest.mark.asyncio
async def test_numeric_answer_resolves_option_one(container, telos_service):
    dispatcher, user = await _arm_rental_menu(container, telos_service)

    out = await dispatcher.handle(DispatcherInput(user=user, text='1'))

    assert out.metadata.get('rental_status_auto_resolved') is True
    assert out.metadata.get('clarification_answer_resolved') is True
    assert out.metadata.get('clarification_option_id') == 'check_status'
    assert out.metadata.get('destructive_gate') is not True
    assert out.text.startswith(
        "You mean checking whether your rental records were updated. "
        "I'll check what I can see."
    )
    assert container.conversation_service.get_active_clarification(user.id) is None




@pytest.mark.asyncio
async def test_exact_option_text_resolves(container, telos_service):
    dispatcher, user = await _arm_rental_menu(container, telos_service)

    out = await dispatcher.handle(
        DispatcherInput(user=user, text='check whether they were updated')
    )

    assert out.metadata.get('rental_status_auto_resolved') is True
    assert out.metadata.get('clarification_option_id') == 'check_status'
    assert out.metadata.get('vague_clarification') is not True


@pytest.mark.asyncio
async def test_destructive_selected_option_still_triggers_approval(container, telos_service):
    dispatcher, user = await _arm_rental_menu(container, telos_service)

    out = await dispatcher.handle(DispatcherInput(user=user, text='update records'))
    pending = container.approvals_repository.list_active_pending_for_user(user.id)

    assert out.metadata.get('destructive_gate') is True
    assert out.metadata.get('clarification_answer_resolved') is True
    assert out.metadata.get('clarification_option_id') == 'update_records'
    assert len(out.buttons) == 2
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_read_only_selected_option_proceeds_without_approval(container, telos_service):
    dispatcher, user = await _arm_rental_menu(container, telos_service)

    out = await dispatcher.handle(DispatcherInput(user=user, text='option 1'))

    assert out.metadata.get('rental_status_auto_resolved') is True
    assert container.approvals_repository.list_active_pending_for_user(user.id) == []


@pytest.mark.asyncio
async def test_stale_clarification_does_not_resolve(container, telos_service):
    dispatcher, user = await _arm_rental_menu(container, telos_service)
    stored = container.conversation_service.get_active_clarification(user.id)
    assert stored is not None
    stored['created_at'] = (
        datetime.now(timezone.utc) - timedelta(minutes=45)
    ).isoformat()
    container.conversation_service.store_active_clarification(
        user.id,
        clarification=stored,
        topic=stored['question'],
    )

    out = await dispatcher.handle(DispatcherInput(user=user, text='1'))

    assert out.metadata.get('clarification_stale') is True
    assert out.metadata.get('rental_status_auto_resolved') is not True
    assert out.text == 'That earlier clarification is stale. Tell me the full request again.'


@pytest.mark.asyncio
async def test_unknown_answer_gets_targeted_follow_up_not_same_menu(container, telos_service):
    dispatcher, user = await _arm_rental_menu(container, telos_service)

    out = await dispatcher.handle(DispatcherInput(user=user, text='maybe'))

    assert out.metadata.get('clarification_follow_up') is True
    assert not out.text.startswith('Do you mean:')
    assert 'Which one do you want?' in out.text
    assert 'check whether your rental records were updated' in out.text


@pytest.mark.asyncio
async def test_no_the_other_one_resolves_alternate_candidate_when_two_exist(container, telos_service):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(
        container,
        telos_service,
        registry=_ambiguous_alex_registry(),
    )

    first_out = await dispatcher.handle(
        DispatcherInput(user=user, text="tell alex i'll do it friday")
    )

    assert first_out.metadata.get('recovery_clarification') is True
    assert first_out.text == 'You mean Alex One or Alex Two?'

    out = await dispatcher.handle(
        DispatcherInput(user=user, text='no the other one')
    )
    pending = container.approvals_repository.list_active_pending_for_user(user.id)

    assert out.metadata.get('destructive_gate') is True
    assert out.metadata.get('clarification_answer_resolved') is True
    assert out.metadata.get('clarification_option_id') == 'recipient_2'
    assert len(pending) == 1
    assert 'Alex Two' in pending[0].preview_text
