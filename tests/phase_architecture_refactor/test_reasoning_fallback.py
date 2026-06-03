"""Step 6 — provider fallback chain tests."""
from __future__ import annotations

import pytest

from services.reasoning_adapter import ReasoningResult
from services.reasoning_fallback import (
    FallbackOutcome,
    ProviderUnavailable,
    reason_with_fallback,
    simple_intent_match,
)


# ── simple_intent_match ─────────────────────────────────────────────────────


@pytest.mark.parametrize('text, expected_intent, expected_target', [
    ('remind sam to take vitamins',          'create_reminder_sam', 'sam'),
    ('sam, remind me later',                 'create_reminder_sam', 'sam'),
    ('remind me to call mom at 5pm',           'create_reminder',       'owner'),
    ('set a reminder for tomorrow morning',    'create_reminder',       'owner'),
    ('what are my reminders',                  'list_reminders',        None),
    ('list reminders please',                  'list_reminders',        None),
    ('show me my reminders',                   'list_reminders',        None),
    ('any new email?',                         'email_summary',         None),
    ('inbox status',                           'email_summary',         None),
    ('how are the rentals',                    'rental_summary',        None),
    ('rent status please',                     'rental_summary',        None),
    ('what time is it',                        'get_time',              None),
    ('current time in tokyo',                  'get_time',              None),
])
def test_simple_pattern_matches(text, expected_intent, expected_target):
    result = simple_intent_match(text)
    assert result.intent == expected_intent
    assert result.target_user == expected_target
    assert result.confidence == 0.45
    assert result.provider_used == 'simple'
    assert result.parameters == {'raw_text': text}


def test_simple_unmatched_falls_back_to_unclear():
    out = simple_intent_match('hi how are you')
    assert out.intent is None
    assert out.natural_response.startswith('Having trouble connecting')
    assert out.provider_used == 'simple'


def test_simple_empty_returns_unclear():
    out = simple_intent_match('')
    assert out.intent is None


# ── reason_with_fallback ────────────────────────────────────────────────────


def _result(intent: str | None, provider: str = '') -> ReasoningResult:
    return ReasoningResult(
        intent=intent,
        confidence=0.9 if intent else 0.0,
        target_user=None,
        parameters={},
        natural_response='ok',
        requires_clarification=intent is None,
        clarification_question=None,
        provider_used=provider,
    )


@pytest.mark.asyncio
async def test_claude_success_returns_first():
    async def claude():
        return _result('list_reminders')

    async def codex():  # pragma: no cover - never reached
        raise AssertionError('codex should not be called')

    out = await reason_with_fallback(
        message='show reminders',
        claude=claude,
        codex=codex,
    )
    assert out.provider_used == 'claude'
    assert out.result.intent == 'list_reminders'
    assert out.fallback_used is False
    assert out.attempts[-1]['outcome'] == 'ok'


@pytest.mark.asyncio
async def test_claude_failure_advances_to_codex():
    async def claude():
        raise ProviderUnavailable('subprocess timeout')

    async def codex():
        return _result('list_reminders')

    out = await reason_with_fallback(
        message='show reminders',
        claude=claude,
        codex=codex,
    )
    assert out.provider_used == 'codex'
    assert out.fallback_used is True
    outcomes = [a['outcome'] for a in out.attempts]
    assert outcomes == ['unavailable', 'ok']


@pytest.mark.asyncio
async def test_both_llms_fail_simple_takes_over():
    async def claude():
        raise ProviderUnavailable('busy')

    async def codex():
        raise ProviderUnavailable('busy')

    out = await reason_with_fallback(
        message='remind sam tomorrow about vitamins',
        claude=claude,
        codex=codex,
    )
    assert out.provider_used == 'simple'
    assert out.fallback_used is True
    assert out.result.intent == 'create_reminder_sam'
    assert out.attempts[-1]['provider'] == 'simple'


@pytest.mark.asyncio
async def test_claude_exception_swallowed_and_advances():
    # An ordinary RuntimeError must NOT propagate — chain advances.
    async def claude():
        raise RuntimeError('claude crashed mid-call')

    async def codex():
        return _result('list_reminders')

    out = await reason_with_fallback(
        message='show reminders', claude=claude, codex=codex,
    )
    assert out.provider_used == 'codex'
    assert out.attempts[0]['outcome'] == 'error'


@pytest.mark.asyncio
async def test_claude_no_intent_treated_as_soft_failure():
    # Claude returned a result, but with no intent — count as "didn't
    # understand", move on to codex.
    async def claude():
        return _result(None)

    async def codex():
        return _result('list_reminders')

    out = await reason_with_fallback(
        message='show reminders', claude=claude, codex=codex,
    )
    assert out.provider_used == 'codex'
    assert out.attempts[0]['outcome'] == 'no_intent'


@pytest.mark.asyncio
async def test_chain_without_codex_skips_to_simple_after_claude_failure():
    async def claude():
        raise ProviderUnavailable('busy')

    out = await reason_with_fallback(
        message='set a reminder', claude=claude, codex=None,
    )
    assert out.provider_used == 'simple'
    assert out.result.intent == 'create_reminder'


@pytest.mark.asyncio
async def test_chain_with_no_providers_returns_simple_match():
    out = await reason_with_fallback(
        message='inbox status', claude=None, codex=None,
    )
    assert out.provider_used == 'simple'
    assert out.result.intent == 'email_summary'


@pytest.mark.asyncio
async def test_provider_used_is_set_when_provider_omits_it():
    # Some provider implementations forget to set provider_used.
    # The chain must stamp it before returning.
    async def claude():
        result = _result('list_reminders')
        result.provider_used = ''  # provider forgot to set it
        return result

    out = await reason_with_fallback(
        message='show reminders', claude=claude,
    )
    assert out.result.provider_used == 'claude'
