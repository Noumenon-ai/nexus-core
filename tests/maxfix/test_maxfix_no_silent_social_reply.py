from __future__ import annotations

import pytest

from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry


class _FailIfCalledLLM:
    async def generate_with_tools(self, **_kwargs):
        raise AssertionError('LLM should not be called for the social prelude path')


class _BlankLLM:
    async def generate_with_tools(self, **_kwargs):
        return {'text': ''}


class _StubMem0:
    def search(self, *args, **kwargs):
        return []

    def add(self, *args, **kwargs):
        return {'results': []}


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


def _build_dispatcher(container, telos_service, *, llm) -> ToolDispatcher:
    return ToolDispatcher(
        llm=llm,
        registry=ToolRegistry(),
        telos_service=telos_service,
        mem0=_StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        conversation_service=container.conversation_service,
        approvals_repository=container.approvals_repository,
        max_iterations=5,
    )


@pytest.mark.asyncio
async def test_birthday_social_turn_always_gets_a_reply(container, telos_service):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(container, telos_service, llm=_FailIfCalledLLM())

    out = await dispatcher.handle(
        DispatcherInput(
            user=user,
            text=(
                'I dont celebrate bday hahaha just told you to see how you react '
                'but it will be nice to geta bday wishes from you'
            ),
        )
    )

    assert out.text == (
        "Haha got it — I won't treat it like a celebration. "
        "I'll still wish you happy birthday tomorrow."
    )
    assert out.text != '(no response recorded)'
    assert out.metadata.get('social_reply') is True


@pytest.mark.asyncio
async def test_blank_llm_reply_gets_safe_non_empty_fallback(container, telos_service):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(container, telos_service, llm=_BlankLLM())

    out = await dispatcher.handle(DispatcherInput(user=user, text='just checking in'))

    assert out.text.strip()
    assert out.text != '(no response recorded)'
    assert out.metadata.get('silent_reply_fallback') is True
