"""Step 6 — provider fallback for the reasoning brain.

claude → codex → simple-pattern. First success wins; each provider is
isolated so a flaky one never poisons the chain. The "simple" stage is
a regex matcher that produces a low-confidence ReasoningResult so the
user always gets a sensible reply — even when both LLMs are
unreachable.

The existing services/brain_router.py handles claude→codex internally
at the prompt level (BrainResult.fallback_used). This module operates
one level up — at the ReasoningAdapter contract — so the simple-pattern
stage can step in when BOTH LLM providers fail.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from services.reasoning_adapter import ReasoningResult

logger = logging.getLogger(__name__)


class ProviderUnavailable(Exception):
    """Raised by a provider stage when it cannot serve the call.

    The fallback chain catches this and moves to the next stage. Other
    exceptions are also caught (logged with traceback) — we never let
    the chain die on a flaky provider.
    """


# ---------------------------------------------------------------------------
# Simple pattern matcher
# ---------------------------------------------------------------------------


# NEXUS_ARCHITECTURE_REFACTOR.md step 6 examples + a couple of common
# extensions that fall out of the same shape. Patterns are compiled
# once at module load.
SIMPLE_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    # remind sam ... — bound to sam specifically
    (re.compile(r'\bremind(er)?\s.*\bsam\b', re.IGNORECASE), 'create_reminder_sam', 'sam'),
    (re.compile(r'\bsam\b.*\bremind', re.IGNORECASE), 'create_reminder_sam', 'sam'),
    # generic create_reminder
    (re.compile(r'\bremind\s+me\b', re.IGNORECASE), 'create_reminder', 'owner'),
    (re.compile(r'\bset\s+(a\s+)?reminder\b', re.IGNORECASE), 'create_reminder', 'owner'),
    # list reminders
    (re.compile(r'\b(list|show|what.*are)\s+.*reminders?\b', re.IGNORECASE), 'list_reminders', None),
    (re.compile(r'\bmy\s+reminders\b', re.IGNORECASE), 'list_reminders', None),
    # email
    (re.compile(r'\b(email|inbox|emails?)\b', re.IGNORECASE), 'email_summary', None),
    # rentals
    (re.compile(r'\b(rentals?|rent|tenants?|units?)\b', re.IGNORECASE), 'rental_summary', None),
    # time
    (re.compile(r'\bwhat.*time\b', re.IGNORECASE), 'get_time', None),
    (re.compile(r'\bcurrent\s+time\b', re.IGNORECASE), 'get_time', None),
)


def simple_intent_match(message: str) -> ReasoningResult:
    """Match `message` against SIMPLE_PATTERNS and return a low-confidence result.

    Always returns a ReasoningResult (never None) — if nothing matches
    we still produce a clarification result so the user gets a real
    reply. provider_used is always 'simple'.
    """
    text = (message or '').strip()
    if not text:
        return _unclear_result()
    for pattern, intent, target_user in SIMPLE_PATTERNS:
        if pattern.search(text):
            return ReasoningResult(
                intent=intent,
                confidence=0.45,  # explicit "this came from a regex, not the brain"
                target_user=target_user,
                parameters={'raw_text': text},
                natural_response=_canned_response_for(intent),
                requires_clarification=False,
                clarification_question=None,
                provider_used='simple',
            )
    return _unclear_result()


def _canned_response_for(intent: str) -> str:
    return {
        'create_reminder':       "I'll set that up — having trouble reaching the brain, so confirm if it sounds right.",
        'create_reminder_sam': "I'll set the reminder for Sam — quick check, brain's slow right now.",
        'list_reminders':        "Pulling your reminders — brain's slow, so this list may be a moment behind.",
        'email_summary':         "Checking the inbox — brain's slow right now, this might be a beat behind.",
        'rental_summary':        "Pulling rental status — brain's slow, this may be a beat behind.",
        'get_time':              'One sec — checking the time.',
    }.get(intent, "Got it — running on the simple path, brain's slow.")


def _unclear_result() -> ReasoningResult:
    return ReasoningResult(
        intent=None,
        confidence=0.0,
        target_user=None,
        parameters={},
        natural_response="Having trouble connecting right now. Please try again.",
        requires_clarification=False,
        clarification_question=None,
        provider_used='simple',
    )


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------


# Each provider in the chain is an async callable that returns a
# ReasoningResult. Raising ProviderUnavailable (or any exception) makes
# the chain move on to the next stage.
ProviderCall = Callable[[], Awaitable[ReasoningResult]]


@dataclass(slots=True)
class FallbackOutcome:
    result: ReasoningResult
    provider_used: str
    attempts: list[dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False


async def reason_with_fallback(
    *,
    message: str,
    claude: ProviderCall | None,
    codex: ProviderCall | None = None,
    simple: Callable[[str], ReasoningResult] = simple_intent_match,
) -> FallbackOutcome:
    """Run claude → codex → simple in order; return the first success.

    A "success" is a ReasoningResult with an intent set. A
    result-with-no-intent counts as a soft failure and the chain
    continues — the LLM either didn't understand or refused; the next
    stage gets a shot.

    Network / process exceptions are logged and the chain advances.
    The simple stage is synchronous and always produces a result.
    """
    attempts: list[dict[str, Any]] = []

    if claude is not None:
        candidate = await _try_provider('claude', claude, attempts)
        if candidate is not None:
            return FallbackOutcome(
                result=candidate,
                provider_used='claude',
                attempts=attempts,
                fallback_used=False,
            )

    if codex is not None:
        candidate = await _try_provider('codex', codex, attempts)
        if candidate is not None:
            return FallbackOutcome(
                result=candidate,
                provider_used='codex',
                attempts=attempts,
                fallback_used=True,
            )

    # Final stage. Simple is synchronous and always produces a result.
    final = simple(message)
    final.provider_used = 'simple'
    attempts.append({'provider': 'simple', 'outcome': 'used'})
    return FallbackOutcome(
        result=final,
        provider_used='simple',
        attempts=attempts,
        fallback_used=True,
    )


async def _try_provider(
    name: str,
    call: ProviderCall,
    attempts: list[dict[str, Any]],
) -> ReasoningResult | None:
    try:
        result = await call()
    except ProviderUnavailable as exc:
        attempts.append({'provider': name, 'outcome': 'unavailable', 'error': str(exc)})
        return None
    except Exception as exc:  # noqa: BLE001 — we never let the chain die
        attempts.append({'provider': name, 'outcome': 'error', 'error': str(exc)[:200]})
        logger.warning(
            'reasoning_fallback_provider_exception',
            extra={'provider': name, 'error': str(exc)[:200]},
        )
        return None
    if result is None:
        attempts.append({'provider': name, 'outcome': 'none'})
        return None
    if not result.intent:
        attempts.append({'provider': name, 'outcome': 'no_intent'})
        return None
    attempts.append({'provider': name, 'outcome': 'ok'})
    if not result.provider_used:
        result.provider_used = name
    return result
