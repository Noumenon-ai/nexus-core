from __future__ import annotations

import re
from typing import Any


_GENERIC_CLARIFICATION_PATTERNS = (
    re.compile(r"^can you clarify\b", re.IGNORECASE),
    re.compile(r"^please clarify\b", re.IGNORECASE),
    re.compile(r"^what would you like me to do with that\??$", re.IGNORECASE),
    re.compile(r"^i can't tell who/what this refers to\.?$", re.IGNORECASE),
    re.compile(r"^could you be more specific\b", re.IGNORECASE),
)
_NON_ANSWER_PATTERNS = (
    re.compile(r"^i(?:'m| am) not sure\b", re.IGNORECASE),
    re.compile(r"^it depends\b", re.IGNORECASE),
    re.compile(r"^can you clarify\b", re.IGNORECASE),
    re.compile(r"^please clarify\b", re.IGNORECASE),
    re.compile(r"^what would you like me to do with that\??$", re.IGNORECASE),
)
_QUESTION_START_RE = re.compile(
    r"^\s*(?:who|what|when|where|why|how|can|could|would|should|did|do|does|is|are|will)\b",
    re.IGNORECASE,
)
_ACTION_REQUEST_RE = re.compile(
    r"\b(?:send|text|message|email|reply|call|schedule|book|cancel|delete|remove|pay|update|edit|change|fix)\b",
    re.IGNORECASE,
)
_ACTIONABLE_REPLY_RE = re.compile(
    r"\b(?:next|first|then|open|click|check|confirm|review|reply|ask|restart|refresh|try|approve|send)\b",
    re.IGNORECASE,
)
_SPECIFIC_REPLY_RE = re.compile(
    r"(?:\b\d{1,4}\b|[:;]|https?://|\b(?:today|tomorrow|tonight|morning|afternoon|evening)\b)",
    re.IGNORECASE,
)
_OVERPROMISE_RE = re.compile(
    r"\b(?:already handled|already fixed|already sent|already paid|consider it done|it's done|it is done|resolved for you)\b",
    re.IGNORECASE,
)


def _word_count(text: str) -> int:
    return len([part for part in text.split() if part.strip()])


def _looks_generic_clarification(text: str) -> bool:
    cleaned = (text or "").strip()
    return any(pattern.match(cleaned) for pattern in _GENERIC_CLARIFICATION_PATTERNS)


def _looks_non_answer(text: str) -> bool:
    cleaned = (text or "").strip()
    return any(pattern.match(cleaned) for pattern in _NON_ANSWER_PATTERNS)


def _looks_like_question(text: str) -> bool:
    cleaned = (text or "").strip()
    return "?" in cleaned or bool(_QUESTION_START_RE.match(cleaned))


def _looks_like_action_request(text: str) -> bool:
    return bool(_ACTION_REQUEST_RE.search(text or ""))


def _looks_specific(text: str) -> bool:
    cleaned = (text or "").strip()
    if _SPECIFIC_REPLY_RE.search(cleaned):
        return True
    return _word_count(cleaned) >= 10 and not _looks_generic_clarification(cleaned)


def build_live_turn_review(
    *,
    user_text: str,
    assistant_text: str,
    source: str,
    provider: str | None,
    response_time_ms: int | None,
    fallback_used: bool,
    error_text: str | None = None,
    silent_reply: bool = False,
) -> dict[str, Any]:
    prompt = (user_text or "").strip()
    reply = (assistant_text or "").strip()
    flags: list[str] = []
    strengths: list[str] = []
    fix_hints: list[str] = []

    if error_text:
        flags.append("brain_error")
        fix_hints.append("Inspect the provider/router failure before trusting this turn.")
    if not reply and not silent_reply and not error_text:
        flags.append("empty_reply")
        fix_hints.append("Return a visible reply or explicit confirmation instead of silence.")
    if source == "brain" and provider in {None, "", "none"} and not error_text:
        flags.append("provider_unavailable")
        fix_hints.append("Check provider availability and fallback routing.")
    if fallback_used:
        flags.append("fallback_used")
        fix_hints.append("Review why the primary provider missed and whether the fallback reply is good enough.")
    if response_time_ms is not None and response_time_ms >= 20000:
        flags.append("slow_response")
        fix_hints.append("Trim the route or provider latency; this turn took too long.")
    if reply and _word_count(reply) > 120:
        flags.append("too_long")
        fix_hints.append("Tighten the reply. It is longer than it needs to be.")
    if reply and _looks_generic_clarification(reply):
        flags.append("generic_clarification")
        fix_hints.append("Ask a more specific clarification tied to the missing detail.")
    if _looks_like_question(prompt) and reply and _looks_non_answer(reply):
        flags.append("did_not_answer")
        fix_hints.append("Answer the user's question directly before asking for more detail.")
    if _looks_like_action_request(prompt) and reply and _OVERPROMISE_RE.search(reply):
        flags.append("unsafe_completion_claim")
        fix_hints.append("Do not claim an action already happened unless the system verified it.")

    if provider and provider not in {"none", "local"} and not error_text:
        strengths.append(f"provider:{provider}")
    if source != "brain":
        strengths.append(f"fast_path:{source}")
    if silent_reply:
        strengths.append("intentional_silence")
    if response_time_ms is not None and response_time_ms <= 4000:
        strengths.append("fast")
    if reply and _looks_specific(reply):
        strengths.append("specific")
    if reply and _ACTIONABLE_REPLY_RE.search(reply):
        strengths.append("actionable")

    critical = {"brain_error", "provider_unavailable", "unsafe_completion_claim", "empty_reply"}
    if any(flag in critical for flag in flags):
        health = "bad"
    elif flags:
        health = "warning"
    else:
        health = "good"

    damage = "none"
    if health == "warning":
        damage = "medium" if len(flags) >= 2 or "did_not_answer" in flags else "low"
    if health == "bad":
        damage = "high" if any(flag in {"brain_error", "unsafe_completion_claim"} for flag in flags) else "medium"

    if not fix_hints and health == "good":
        fix_hints.append("No obvious issue on this turn.")

    return {
        "health": health,
        "damage": damage,
        "flags": flags,
        "strengths": strengths,
        "fix_hints": fix_hints,
    }
