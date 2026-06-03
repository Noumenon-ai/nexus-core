"""End-to-end tests for the destructive-message-gate in tool_dispatcher.

Covers:
  - Read-only prompt bypasses gate; LLM is invoked normally.
  - Destructive prompt: gate fires, ApprovalService row created, LLM NOT invoked.
  - Approval callback (approve) re-fires the dispatcher with bypass=True; LLM
    is then invoked with the original prompt.
  - Approval callback (cancel) returns the cancel sentinel; LLM NOT invoked.
  - bypass_destructive_approval=True at the input level skips the gate.
"""
from __future__ import annotations

from typing import Any

import pytest

from config import get_settings
from pipeline.tool_dispatcher import DispatcherInput, DispatcherOutput, ToolDispatcher
from services.capability_registry import CapabilityRegistry
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry


class ScriptedLLM:
    def __init__(self, script: list[dict[str, Any]]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
        self.calls.append({'contents': contents})
        if not self.script:
            raise AssertionError('ScriptedLLM exhausted')
        return self.script.pop(0)


class StubMem0:
    def search(self, *a, **kw): return []
    def add(self, *a, **kw): return None


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


def _build_dispatcher(container, telos_service, script):
    return ToolDispatcher(
        llm=ScriptedLLM(script),
        registry=ToolRegistry(),
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        approvals_repository=container.approvals_repository,
        max_iterations=5,
    )


@pytest.fixture
def approvals_disabled(monkeypatch):
    monkeypatch.setenv('DESTRUCTIVE_APPROVAL_ENABLED', 'false')
    get_settings.cache_clear()
    yield CapabilityRegistry(settings=get_settings())
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_read_only_prompt_skips_gate_and_invokes_llm(container, telos_service):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(container, telos_service,
                                   script=[{'text': 'You have 3 units.'}])
    out = await dispatcher.handle(DispatcherInput(user=user, text='show me my rental units'))
    assert out.text == 'You have 3 units.'
    assert out.metadata.get('destructive_gate') is not True
    assert len(dispatcher.llm.calls) == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_destructive_prompt_triggers_gate_no_llm_call(container, telos_service):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(container, telos_service, script=[])  # would error if invoked
    out = await dispatcher.handle(DispatcherInput(user=user, text='delete rental unit 1'))
    assert out.metadata.get('destructive_gate') is True
    assert 'delete_unit' in out.metadata.get('matched_tools', [])
    # Two buttons attached: approve + cancel.
    assert len(out.buttons) == 2
    # ApprovalService.request created exactly one row in the repo.
    rows = container.approvals_repository.list_active_pending_for_user(user.id)
    assert len(rows) == 1
    assert rows[0].action_type == 'destructive_message_gate'
    # LLM never called.
    assert dispatcher.llm.calls == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_approve_callback_refires_with_bypass(container, telos_service):
    user = container.users_repository.get_or_create(111)
    # First turn: gate fires.
    dispatcher = _build_dispatcher(container, telos_service,
                                   # Script for the post-approval re-fire only.
                                   script=[{'text': 'Unit 1 deleted.'}])
    gate_out = await dispatcher.handle(
        DispatcherInput(user=user, text='delete rental unit 1')
    )
    assert gate_out.metadata.get('destructive_gate') is True
    assert dispatcher.llm.calls == []  # type: ignore[attr-defined]

    # Locate the approval ID from the buttons.
    approve_button = next(b for b in gate_out.buttons if 'approve' in b.callback_data)
    approval_id = approve_button.callback_data.split(':', 2)[2]

    # Second turn: user taps approve → callback string arrives as text.
    callback_text = f'approval:approve:{approval_id}'
    post_out = await dispatcher.handle(DispatcherInput(user=user, text=callback_text))
    assert post_out.text == 'Unit 1 deleted.'
    # The re-fire invoked the LLM exactly once.
    assert len(dispatcher.llm.calls) == 1  # type: ignore[attr-defined]
    # Approval row transitioned to executed.
    row = container.approvals_repository.get(approval_id)
    assert row.status == 'executed'


@pytest.mark.asyncio
async def test_cancel_callback_returns_sentinel_no_llm_call(container, telos_service):
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(container, telos_service, script=[])
    gate_out = await dispatcher.handle(
        DispatcherInput(user=user, text='delete rental unit 1')
    )
    cancel_button = next(b for b in gate_out.buttons if 'cancel' in b.callback_data)
    approval_id = cancel_button.callback_data.split(':', 2)[2]
    cancel_out = await dispatcher.handle(
        DispatcherInput(user=user, text=f'approval:cancel:{approval_id}')
    )
    # The cancel response includes the localized "Cancelled." text.
    assert 'cancel' in cancel_out.text.lower()
    # LLM never called.
    assert dispatcher.llm.calls == []  # type: ignore[attr-defined]
    # Approval row transitioned to cancelled.
    row = container.approvals_repository.get(approval_id)
    assert row.status == 'cancelled'


@pytest.mark.asyncio
async def test_explicit_bypass_input_skips_gate(container, telos_service):
    """Direct bypass_destructive_approval=True (used by the re-fire path and
    by tests of downstream behavior) must skip the classifier entirely."""
    user = container.users_repository.get_or_create(111)
    dispatcher = _build_dispatcher(container, telos_service,
                                   script=[{'text': 'Done.'}])
    out = await dispatcher.handle(DispatcherInput(
        user=user, text='delete rental unit 1',
        bypass_destructive_approval=True,
    ))
    assert out.text == 'Done.'
    # LLM was invoked despite destructive intent.
    assert len(dispatcher.llm.calls) == 1  # type: ignore[attr-defined]
    # No approval row was created.
    assert container.approvals_repository.list_active_pending_for_user(user.id) == []


@pytest.mark.asyncio
async def test_unauthorized_approve_callback_is_rejected(container, telos_service):
    """If user B taps the approve button on user A's approval, the gate must
    NOT re-fire on behalf of user A. This is a security boundary."""
    user_a = container.users_repository.get_or_create(111)
    user_b = container.users_repository.get_or_create(222)
    dispatcher = _build_dispatcher(container, telos_service, script=[])

    # User A triggers the gate.
    gate_out = await dispatcher.handle(DispatcherInput(
        user=user_a, text='delete rental unit 1'
    ))
    approve_button = next(b for b in gate_out.buttons if 'approve' in b.callback_data)
    approval_id = approve_button.callback_data.split(':', 2)[2]

    # User B taps approve — should be rejected.
    out = await dispatcher.handle(DispatcherInput(
        user=user_b, text=f'approval:approve:{approval_id}'
    ))
    # No re-fire, no LLM call.
    assert dispatcher.llm.calls == []  # type: ignore[attr-defined]
    # Approval row should remain pending (not executed by user B).
    row = container.approvals_repository.get(approval_id)
    assert row.status == 'pending'


@pytest.mark.asyncio
async def test_gate_no_op_when_approvals_repository_missing(container, telos_service):
    """If a caller constructs ToolDispatcher without approvals_repository
    (legacy tests), the gate must degrade to a no-op rather than crash."""
    user = container.users_repository.get_or_create(111)
    dispatcher = ToolDispatcher(
        llm=ScriptedLLM([{'text': 'Hello'}]),
        registry=ToolRegistry(),
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        # approvals_repository deliberately omitted
        max_iterations=5,
    )
    out = await dispatcher.handle(DispatcherInput(user=user, text='delete rental unit 1'))
    assert out.text == 'Hello'  # LLM invoked normally; no gate fires


@pytest.mark.asyncio
async def test_destructive_prompt_skips_gate_when_flag_disabled(
    container,
    telos_service,
    approvals_disabled,
):
    user = container.users_repository.get_or_create(111)
    dispatcher = ToolDispatcher(
        llm=ScriptedLLM([{'text': 'Deleted without prompt.'}]),
        registry=ToolRegistry(),
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        approvals_repository=container.approvals_repository,
        capability_registry=approvals_disabled,
        max_iterations=5,
    )

    out = await dispatcher.handle(
        DispatcherInput(user=user, text='delete all my reminders')
    )

    assert out.text == 'Deleted without prompt.'
    assert out.metadata.get('destructive_gate') is not True
    assert len(dispatcher.llm.calls) == 1  # type: ignore[attr-defined]
    assert container.approvals_repository.list_active_pending_for_user(user.id) == []
