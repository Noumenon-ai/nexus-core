"""H2-047 Fix 2 — multi-intent prompts become advisory replies, NOT
approval prompts with Approve/Cancel buttons.

Pre-H2-047 a numbered multi-action message ("1. delete a reminder
2. cancel an event") armed approval_service.request with template
"Multiple actions detected — please send them one at a time." plus
Approve/Cancel buttons. Tapping Approve bypass-re-fired the original
prompt, defeating the whole point of the check. This fix routes
advisory intents to a plain-reply branch in the dispatcher.
"""
from __future__ import annotations

from typing import Any

import pytest

from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.destructive_intent_classifier import classify as classify_destructive
from services.read_tools import register_read_tools
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Classifier surface — is_advisory_only flag set for multi-intent
# ---------------------------------------------------------------------------


def test_multi_intent_sets_advisory_only_flag():
    intent = classify_destructive("1. delete a reminder 2. cancel an event")
    assert intent.is_destructive is True
    assert intent.is_advisory_only is True
    assert "one at a time" in intent.suggested_approval_template.lower()


def test_single_destructive_intent_is_NOT_advisory():
    """Don't regress real destructive gating: a single-action destructive
    prompt should arm approval as usual, NOT skip to advisory. (Reminders are
    now ungated per the 2026-06-02 directive, so use a file delete here — the
    point is the advisory_only flag, not the target.)"""
    intent = classify_destructive("delete file /tmp/junk.txt")
    assert intent.is_destructive is True
    assert intent.is_advisory_only is False


def test_non_destructive_prompt_is_NOT_advisory():
    intent = classify_destructive("what is on my calendar tomorrow")
    assert intent.is_destructive is False
    assert intent.is_advisory_only is False


# ---------------------------------------------------------------------------
# Dispatcher behaviour — advisory short-circuits to plain reply, no buttons
# ---------------------------------------------------------------------------


class _StubMem0:
    def search(self, *args, **kwargs):
        return []

    def add(self, *args, **kwargs):
        return {'results': []}


class _ScriptedLLM:
    def __init__(self):
        self.calls = 0

    async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
        self.calls += 1
        return {'text': 'should not be reached'}


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




@pytest.mark.asyncio
async def test_single_destructive_intent_still_arms_approval_with_buttons(
        container, registry, telos_service, caplog):
    """Contract preservation: a single-action destructive prompt still
    produces Approve/Cancel buttons. Without this assertion a refactor
    of the dispatcher's advisory branch could accidentally hide the gate
    for real destructive actions."""
    user = container.users_repository.get_or_create(111)
    llm = _ScriptedLLM()
    dispatcher = ToolDispatcher(
        llm=llm,
        registry=registry,
        telos_service=telos_service,
        mem0=_StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        approvals_repository=container.approvals_repository,
        max_iterations=10,
    )

    with caplog.at_level('INFO', logger='pipeline.tool_dispatcher'):
        out = await dispatcher.handle(DispatcherInput(
            user=user,
            # Reminders are ungated per the 2026-06-02 directive; use a file
            # delete so this test still exercises a genuinely gated action.
            text="delete file /tmp/junk.txt",
        ))

    # Approval gate armed, NOT advisory short-circuit
    assert out.metadata.get('destructive_gate') is True
    assert out.metadata.get('destructive_advisory') is None
    assert out.buttons, 'destructive gate must show Approve/Cancel buttons'
    gate_logs = [r for r in caplog.records
                 if r.message == 'dispatcher_destructive_gate_armed']
    advisory_logs = [r for r in caplog.records
                     if r.message == 'dispatcher_destructive_advisory_emitted']
    assert len(gate_logs) == 1
    assert len(advisory_logs) == 0
