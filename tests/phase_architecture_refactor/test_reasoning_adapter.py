"""Tests for the reasoning_adapter façade.

Verifies the contract laid out in NEXUS_ARCHITECTURE_REFACTOR.md step 1:
  - the system prompt embeds capabilities + thread + user_id
  - the adapter parses pure JSON, embedded-JSON, and plain-text responses
  - clarification falls through when fields are missing
  - IntentExecutor blocks on capability-missing / approval / safety
  - IntentExecutor writes an audit record on every branch
"""
from __future__ import annotations

import pytest

from services.reasoning_adapter import (
    IntentExecutionResult,
    IntentExecutor,
    ReasoningAdapter,
    ReasoningResult,
    build_reasoning_prompt,
    parse_reasoning_response,
)


# ── Prompt construction ─────────────────────────────────────────────────────


def test_prompt_embeds_capabilities_and_user_and_thread():
    prompt = build_reasoning_prompt(
        capabilities=['email_read', 'reminder_create'],
        thread=[
            {'role': 'user', 'content': 'remind me tomorrow'},
            {'role': 'assistant', 'content': 'ok, about what?'},
        ],
        user_id='owner',
    )
    assert 'email_read' in prompt
    assert 'reminder_create' in prompt
    assert 'owner' in prompt
    assert 'remind me tomorrow' in prompt
    # Constitutional NEVERs from the spec
    assert 'Send WhatsApp directly' in prompt
    assert 'Edit the database' in prompt


def test_prompt_handles_empty_thread_and_caps():
    prompt = build_reasoning_prompt(capabilities=[], thread=[], user_id='')
    assert '(none)' in prompt
    assert '(no prior turns)' in prompt
    assert 'unknown' in prompt


def test_thread_truncated_to_last_five_turns():
    long_thread = [
        {'role': 'user', 'content': f'message {i}'} for i in range(10)
    ]
    prompt = build_reasoning_prompt(
        capabilities=['x'], thread=long_thread, user_id='owner',
    )
    assert 'message 5' in prompt
    assert 'message 9' in prompt
    # The first five turns are dropped
    assert 'message 0' not in prompt
    assert 'message 4' not in prompt


# ── parse_reasoning_response ────────────────────────────────────────────────


def test_parse_pure_json_response():
    raw = """{
        "intent": "create_reminder",
        "confidence": 0.92,
        "target_user": "sam",
        "parameters": {"message": "take vitamins", "time": "9am"},
        "natural_response": "Got it.",
        "requires_clarification": false,
        "clarification_question": null
    }"""
    result = parse_reasoning_response(raw, provider_used='claude')
    assert result.intent == 'create_reminder'
    assert result.confidence == 0.92
    assert result.target_user == 'sam'
    assert result.parameters == {'message': 'take vitamins', 'time': '9am'}
    assert result.natural_response == 'Got it.'
    assert result.requires_clarification is False
    assert result.clarification_question is None
    assert result.provider_used == 'claude'


def test_parse_extracts_json_from_text_envelope():
    raw = (
        'sure, here is the structured intent:\n'
        '{"intent": "list_reminders", "confidence": 0.7, "target_user": null, '
        '"parameters": {}, "natural_response": "here you go", '
        '"requires_clarification": false, "clarification_question": null}\n'
        'thanks!'
    )
    result = parse_reasoning_response(raw, provider_used='codex')
    assert result.intent == 'list_reminders'
    assert result.target_user is None
    assert result.provider_used == 'codex'


def test_parse_plain_text_falls_back_to_natural_response():
    raw = 'Just chatting — no intent here.'
    result = parse_reasoning_response(raw, provider_used='claude')
    assert result.intent is None
    assert result.natural_response == 'Just chatting — no intent here.'
    assert result.confidence == 0.0


def test_parse_clamps_confidence_to_unit_interval():
    raw = '{"intent": "x", "confidence": 1.7, "natural_response": "ok"}'
    assert parse_reasoning_response(raw).confidence == 1.0

    raw = '{"intent": "x", "confidence": -0.4, "natural_response": "ok"}'
    assert parse_reasoning_response(raw).confidence == 0.0


def test_parse_treats_string_null_as_none():
    raw = '{"intent": "null", "target_user": "none", "natural_response": "k"}'
    out = parse_reasoning_response(raw)
    assert out.intent is None
    assert out.target_user is None


def test_parse_empty_input_falls_back_to_clarification():
    result = parse_reasoning_response('', provider_used='claude')
    assert result.intent is None
    assert result.requires_clarification is True


# ── ReasoningAdapter — async flow ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_adapter_passes_through_to_brain_and_parses():
    async def fake_brain(system_prompt: str, message: str):
        assert 'email_read' in system_prompt
        assert message == 'list my reminders'
        return (
            '{"intent": "list_reminders", "confidence": 0.8, '
            '"target_user": null, "parameters": {}, '
            '"natural_response": "Sure.", "requires_clarification": false, '
            '"clarification_question": null}',
            'claude',
        )

    adapter = ReasoningAdapter(brain_call=fake_brain)
    result = await adapter.reason(
        message='list my reminders',
        thread=[],
        capabilities=['email_read', 'reminder_list'],
        user_id='owner',
    )
    assert result.intent == 'list_reminders'
    assert result.natural_response == 'Sure.'
    assert result.provider_used == 'claude'


@pytest.mark.asyncio
async def test_adapter_swallows_brain_error_into_clarification_result():
    async def boom(system_prompt: str, message: str):
        raise RuntimeError('claude subprocess timed out')

    adapter = ReasoningAdapter(brain_call=boom)
    result = await adapter.reason(
        message='hi',
        thread=[],
        capabilities=['x'],
        user_id='owner',
    )
    assert result.intent is None
    assert result.requires_clarification is True


# ── IntentExecutor — branches ───────────────────────────────────────────────


@pytest.fixture
def audit_log():
    records: list[dict] = []

    async def _log(payload):
        records.append(dict(payload))

    return records, _log


def _make_intent(intent='create_reminder', params=None, natural='Done.'):
    return ReasoningResult(
        intent=intent,
        confidence=0.9,
        target_user='sam',
        parameters=dict(params or {}),
        natural_response=natural,
        requires_clarification=False,
        clarification_question=None,
        provider_used='claude',
    )


@pytest.mark.asyncio
async def test_executor_blocks_when_intent_missing(audit_log):
    records, log = audit_log
    executor = IntentExecutor(
        capability_available=lambda intent: True,
        requires_approval=lambda intent, params: False,
        request_approval=_unreachable,
        passes_safety=lambda **kw: True,
        execute_tool=_unreachable,
        audit_log=log,
    )
    out = await executor.execute(
        _make_intent(intent=None, natural='Not sure.'),
        user_id='owner',
    )
    assert out.executed is False
    assert out.blocked_reason == 'no_intent'
    assert out.text == 'Not sure.'
    assert records == []  # no_intent does not write audit


@pytest.mark.asyncio
async def test_executor_blocks_when_capability_unavailable(audit_log):
    records, log = audit_log
    executor = IntentExecutor(
        capability_available=lambda intent: False,
        requires_approval=lambda intent, params: False,
        request_approval=_unreachable,
        passes_safety=lambda **kw: True,
        execute_tool=_unreachable,
        audit_log=log,
    )
    # Use an empty natural_response so the executor falls back to its
    # canned "isn't connected" string — exercises the actual fallback.
    out = await executor.execute(
        _make_intent('messaging_sms', natural=''),
        user_id='owner',
    )
    assert out.executed is False
    assert out.blocked_reason == 'capability_unavailable'
    assert "isn't connected" in out.text
    assert records[0]['result'] == 'capability_unavailable'


@pytest.mark.asyncio
async def test_executor_routes_to_approval(audit_log):
    records, log = audit_log
    captured: dict = {}

    async def _approval(*, intent, user_id):
        captured['intent'] = intent.intent
        captured['user_id'] = user_id
        return 'Confirm cleanup of 3 reminders?'

    executor = IntentExecutor(
        capability_available=lambda intent: True,
        requires_approval=lambda intent, params: True,
        request_approval=_approval,
        passes_safety=lambda **kw: True,
        execute_tool=_unreachable,
        audit_log=log,
    )
    out = await executor.execute(_make_intent('delete_reminders'), user_id='owner')
    assert out.executed is False
    assert out.requires_approval is True
    assert out.text == 'Confirm cleanup of 3 reminders?'
    assert captured == {'intent': 'delete_reminders', 'user_id': 'owner'}
    assert records[0]['result'] == 'awaiting_approval'


@pytest.mark.asyncio
async def test_executor_blocks_on_safety_failure(audit_log):
    records, log = audit_log
    executor = IntentExecutor(
        capability_available=lambda intent: True,
        requires_approval=lambda intent, params: False,
        request_approval=_unreachable,
        passes_safety=lambda **kw: False,
        execute_tool=_unreachable,
        audit_log=log,
    )
    out = await executor.execute(_make_intent('send_email'), user_id='owner')
    assert out.executed is False
    assert out.blocked_reason == 'safety_blocked'
    assert records[0]['result'] == 'safety_blocked'


@pytest.mark.asyncio
async def test_executor_executes_when_all_gates_pass(audit_log):
    records, log = audit_log
    invocations: list[dict] = []

    async def _tool(*, intent, user_id):
        invocations.append({'intent': intent.intent, 'user_id': user_id})
        return {'status': 'ok'}

    executor = IntentExecutor(
        capability_available=lambda intent: True,
        requires_approval=lambda intent, params: False,
        request_approval=_unreachable,
        passes_safety=lambda **kw: True,
        execute_tool=_tool,
        audit_log=log,
    )
    out = await executor.execute(_make_intent('create_reminder', natural='Sure.'), user_id='owner')
    assert out.executed is True
    assert out.text == 'Sure.'
    assert invocations == [{'intent': 'create_reminder', 'user_id': 'owner'}]
    assert records[0]['result'] == 'ok'


async def _unreachable(*args, **kwargs):  # pragma: no cover
    raise AssertionError('this branch should not have run')
