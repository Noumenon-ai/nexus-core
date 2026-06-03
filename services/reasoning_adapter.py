"""Step 1 of the NEXUS architecture refactor (2026-05-27).

Reasoning adapter that draws the line between Claude's role and
NEXUS's role. Claude CLI is the *brain* — it sees the message, the
last few turns, the list of capabilities NEXUS has wired, and the
user's identity. It returns a structured intent + a natural reply.

Claude NEVER directly executes actions. NEXUS reads the intent and
decides what to do next: capability check, approval gate, safety
rules, tool execution, audit log, response. That's `IntentExecutor`.

This module composes with existing services (BrainRouter,
CapabilityRegistry, ApprovalService, audit_service) — it does not
replace them.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReasoningResult:
    """Structured output from the reasoning brain.

    Exact shape mandated by NEXUS_ARCHITECTURE_REFACTOR.md step 1.
    """
    intent: str | None
    confidence: float
    target_user: str | None
    parameters: dict[str, Any] = field(default_factory=dict)
    natural_response: str = ''
    requires_clarification: bool = False
    clarification_question: str | None = None
    provider_used: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'intent': self.intent,
            'confidence': float(self.confidence),
            'target_user': self.target_user,
            'parameters': dict(self.parameters),
            'natural_response': self.natural_response,
            'requires_clarification': bool(self.requires_clarification),
            'clarification_question': self.clarification_question,
            'provider_used': self.provider_used,
        }


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


_REASONING_SYSTEM_PROMPT_TEMPLATE = """You are the conversational brain of NEXUS.
Your ONLY job is to understand what the user wants and return structured intent.

You NEVER:
- Send WhatsApp directly
- Send email directly
- Edit the database
- Delete reminders
- Approve actions
- Bypass safety
- Run shell commands

You ONLY return:
- What the user likely wants (intent)
- Natural conversational response text
- Parameters needed for execution
- Whether clarification is needed

Available capabilities: {capabilities}

If a capability is not available, say so naturally.
Do not pretend you can do something NEXUS cannot do.

User identity: {user_id}

Conversation thread (last 5 turns):
{thread}

Return JSON with this exact structure:
{{
  "intent": "action_name or null",
  "confidence": 0.0-1.0,
  "target_user": "owner or sam or null",
  "parameters": {{}},
  "natural_response": "conversational response",
  "requires_clarification": true/false,
  "clarification_question": "question or null"
}}
"""


def build_reasoning_prompt(
    *,
    capabilities: list[str],
    thread: list[dict[str, str]],
    user_id: str,
) -> str:
    """Render the reasoning system prompt. Public for testing."""
    return _REASONING_SYSTEM_PROMPT_TEMPLATE.format(
        capabilities=', '.join(capabilities) if capabilities else '(none)',
        user_id=user_id or 'unknown',
        thread=_format_thread(thread),
    )


def _format_thread(thread: list[dict[str, str]]) -> str:
    if not thread:
        return '(no prior turns)'
    lines: list[str] = []
    for turn in thread[-5:]:
        role = str(turn.get('role') or 'user').strip() or 'user'
        content = str(turn.get('content') or '').strip()
        if not content:
            continue
        lines.append(f'  {role}: {content}')
    return '\n'.join(lines) or '(no prior turns)'


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


# Callable[[system_prompt, message], Awaitable[response_text + provider_used]]
BrainCallable = Callable[
    [str, str],
    Awaitable[tuple[str, str]],
]


class ReasoningAdapter:
    """Façade over BrainRouter that returns a structured ReasoningResult.

    Inject `brain_call` to keep the adapter testable without a real
    Claude CLI subprocess. Production wiring should provide a callable
    that delegates to `BrainRouter.generate(...)` and returns
    `(response_text, provider_used)`.
    """

    def __init__(self, *, brain_call: BrainCallable) -> None:
        self._brain_call = brain_call

    async def reason(
        self,
        *,
        message: str,
        thread: list[dict[str, str]],
        capabilities: list[str],
        user_id: str,
    ) -> ReasoningResult:
        system_prompt = build_reasoning_prompt(
            capabilities=list(capabilities or []),
            thread=list(thread or []),
            user_id=str(user_id or ''),
        )
        try:
            response_text, provider_used = await self._brain_call(
                system_prompt, message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'reasoning_adapter_brain_call_failed',
                extra={'user_id': user_id, 'error': str(exc)},
            )
            return _fallback_unclear_result(user_id=user_id)
        return parse_reasoning_response(response_text, provider_used=provider_used)


def parse_reasoning_response(
    response_text: str,
    *,
    provider_used: str = '',
) -> ReasoningResult:
    """Parse a brain response into a ReasoningResult.

    Tolerates two shapes:
      - pure JSON object (the contract)
      - JSON embedded inside a larger text body (we extract the first
        balanced {...} run)

    Anything else becomes a plain natural_response with no intent.
    """
    if not response_text:
        return _fallback_unclear_result(provider_used=provider_used)
    payload = _extract_first_json_object(response_text) or {}
    if not isinstance(payload, dict) or not payload:
        return ReasoningResult(
            intent=None,
            confidence=0.0,
            target_user=None,
            parameters={},
            natural_response=response_text.strip(),
            requires_clarification=False,
            clarification_question=None,
            provider_used=provider_used,
        )
    return ReasoningResult(
        intent=_normalize_intent(payload.get('intent')),
        confidence=_normalize_confidence(payload.get('confidence')),
        target_user=_normalize_target_user(payload.get('target_user')),
        parameters=dict(payload.get('parameters') or {})
        if isinstance(payload.get('parameters'), dict) else {},
        natural_response=str(payload.get('natural_response') or '').strip(),
        requires_clarification=bool(payload.get('requires_clarification')),
        clarification_question=_normalize_optional_str(payload.get('clarification_question')),
        provider_used=provider_used,
    )


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Return the first balanced JSON object found in `text`, or None."""
    if not text:
        return None
    text = text.strip()
    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            return candidate
    except (json.JSONDecodeError, TypeError):
        pass
    # Find the first balanced brace run
    depth = 0
    start = -1
    in_str = False
    escape = False
    for index, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == '\\' and in_str:
            escape = True
            continue
        if char == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if char == '{':
            if depth == 0:
                start = index
            depth += 1
        elif char == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    snippet = text[start:index + 1]
                    try:
                        candidate = json.loads(snippet)
                    except json.JSONDecodeError:
                        return None
                    if isinstance(candidate, dict):
                        return candidate
                    return None
    return None


def _normalize_intent(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {'null', 'none'}:
        return None
    return text


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _normalize_target_user(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {'null', 'none'}:
        return None
    return text


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {'null', 'none'}:
        return None
    return text


def _fallback_unclear_result(
    *,
    user_id: str = '',
    provider_used: str = '',
) -> ReasoningResult:
    return ReasoningResult(
        intent=None,
        confidence=0.0,
        target_user=None,
        parameters={},
        natural_response="I'm not sure what you'd like — could you rephrase?",
        requires_clarification=True,
        clarification_question='What would you like NEXUS to do?',
        provider_used=provider_used,
    )


# ---------------------------------------------------------------------------
# Intent executor (Claude provides intent → NEXUS executes)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IntentExecutionResult:
    """What the executor returns to the caller."""
    text: str
    executed: bool
    blocked_reason: str | None
    requires_approval: bool
    audit: dict[str, Any] = field(default_factory=dict)


class IntentExecutor:
    """Glue between Claude's intent and NEXUS's tools.

    The executor:
      1. Checks the intent's required capability is available.
      2. Checks if the intent requires approval (destructive actions).
      3. Checks safety rules.
      4. Executes via the wired tool.
      5. Writes an audit record.
      6. Returns the natural response + execution metadata.

    Each step is injected so the executor is testable without spinning
    up the full pipeline. In production the wiring lives in main.py.
    """

    def __init__(
        self,
        *,
        capability_available: Callable[[str], bool],
        requires_approval: Callable[[str, dict[str, Any]], bool],
        request_approval: Callable[..., Awaitable[str]],
        passes_safety: Callable[..., bool],
        execute_tool: Callable[..., Awaitable[dict[str, Any]]],
        audit_log: Callable[..., Awaitable[None]],
    ) -> None:
        self._capability_available = capability_available
        self._requires_approval = requires_approval
        self._request_approval = request_approval
        self._passes_safety = passes_safety
        self._execute_tool = execute_tool
        self._audit_log = audit_log

    async def execute(
        self,
        intent: ReasoningResult,
        *,
        user_id: str,
    ) -> IntentExecutionResult:
        audit: dict[str, Any] = {
            'user_id': user_id,
            'intent': intent.intent,
            'parameters': dict(intent.parameters),
            'provider': intent.provider_used,
        }
        if not intent.intent:
            text = intent.natural_response or "I don't have anything to do here."
            return IntentExecutionResult(
                text=text,
                executed=False,
                blocked_reason='no_intent',
                requires_approval=False,
                audit=audit,
            )
        if not self._capability_available(intent.intent):
            text = (
                intent.natural_response
                or f"That capability ({intent.intent}) isn't connected right now."
            )
            await self._audit_log(audit | {'result': 'capability_unavailable'})
            return IntentExecutionResult(
                text=text,
                executed=False,
                blocked_reason='capability_unavailable',
                requires_approval=False,
                audit=audit,
            )
        if self._requires_approval(intent.intent, intent.parameters):
            approval_text = await self._request_approval(
                intent=intent, user_id=user_id,
            )
            await self._audit_log(audit | {'result': 'awaiting_approval'})
            return IntentExecutionResult(
                text=approval_text or intent.natural_response,
                executed=False,
                blocked_reason=None,
                requires_approval=True,
                audit=audit,
            )
        if not self._passes_safety(intent=intent, user_id=user_id):
            await self._audit_log(audit | {'result': 'safety_blocked'})
            return IntentExecutionResult(
                text="I can't run that — it doesn't pass the safety rules.",
                executed=False,
                blocked_reason='safety_blocked',
                requires_approval=False,
                audit=audit,
            )
        result = await self._execute_tool(intent=intent, user_id=user_id)
        await self._audit_log(audit | {'result': result.get('status') or 'ok'})
        return IntentExecutionResult(
            text=intent.natural_response or 'Done.',
            executed=True,
            blocked_reason=None,
            requires_approval=False,
            audit=audit | {'result': result.get('status') or 'ok'},
        )
