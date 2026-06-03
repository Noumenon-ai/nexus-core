"""Phase 2: pre-LLM intent classifier.

Routes messages to brain providers + tool-likely vs conversational paths.
Pure regex/keyword matching, no LLM calls.

H2-046: keyword matching now runs against the prompt with embedded
document blocks stripped (PDF body, audio transcript, attachment). A lease
PDF's body containing "personal" / "financial" / "confidential" used to
trip sensitive-route routing and force the prompt through Ollama, even
though the user's *instruction* was a benign "summarize this".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from services.prompt_content_filters import strip_embedded_content


class Provider(Enum):
    CLAUDE = 'claude'
    CODEX = 'codex'
    OLLAMA = 'ollama'


class Path(Enum):
    CONVERSATIONAL = 'conversational'
    TOOL_LIKELY = 'tool_likely'


@dataclass(frozen=True)
class IntentDecision:
    provider: Provider
    path: Path
    matched_keywords: tuple[str, ...]


_TOOL_KEYWORDS = (
    'remind', 'reminder',
    'schedule', 'calendar', 'event', 'meeting', 'appointment',
    'task', 'todo', 'tasks',
    'email', 'gmail', 'inbox', 'mail',
    'contact', 'contacts',
    'search', 'find', 'look up', 'lookup',
    'weather', 'forecast', 'temperature',
    'wikipedia', 'wiki',
    'news', 'headlines',
    'free', 'busy', 'available',
    'delete', 'cancel', 'remove',
    'update', 'change', 'modify',
    'create', 'add', 'new',
    'list', 'show me', 'whats on', "what's on",
    # Phase 4 file system + terminal vocabulary (2026-05-11): natural
    # phrasings like "Read the file at X" / "Show contents of Y" /
    # "Run X in terminal" must route TOOL_LIKELY so the dispatcher
    # delegates to the Gemini tool catalog (Claude CLI has no native
    # function calling and was hallucinating approval flows instead).
    'read', 'open', 'view', 'contents',
    'file', 'files', 'folder', 'directory',
    'run', 'execute', 'terminal', 'command',
    'write', 'save',
    # Phase 2.5b followup (2026-05-12): close the carry-forward gap from
    # H2-034 + H2-035 — "Summarize my unread emails" / "Draft a reply" /
    # "Compose a message" all classified conversational before this, so the
    # MCP tool layer was bypassed entirely for those phrasings.
    'summarize', 'summary', 'summaries',
    'emails', 'message', 'messages',
    'draft', 'drafts', 'compose', 'reply', 'forward',
    'archive', 'archived',
    'attachment', 'attachments',
)

_CODE_KEYWORDS = (
    'code', 'function', 'class', 'method', 'variable',
    'bug', 'error', 'exception', 'traceback', 'stack trace',
    'refactor', 'rewrite', 'implement',
    'git', 'commit', 'branch', 'merge',
    'unit test', 'pytest', 'jest',
    'compile', 'build', 'deploy',
    'regex', 'sql query',
    'algorithm', 'data structure',
)

_SENSITIVE_KEYWORDS = (
    'private', 'confidential', 'secret', 'personal',
    'medical', 'health', 'diagnosis', 'medication',
    'financial', 'bank account', 'ssn',
    'journal', 'diary',
)

# H2-047 Fix 1: explicit tool-invocation / capability-question patterns.
# These are unambiguous in intent ("use the X tool", "what X do I have
# access to") and must route TOOL_LIKELY regardless of the keyword set or
# the sensitive-route check. Without this, "What gmails do you have access
# to?" landed CONVERSATIONAL and never reached nexus-self.describe_my_access;
# "Use the calendar tool" similarly fell through to a plain conversational
# Claude CLI call where the dispatcher tool-loop early-break couldn't fire.
_EXPLICIT_TOOL_INVOCATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'\buse the \w+ tool\b', re.IGNORECASE),
    re.compile(r'\buse (?:the )?\w+(?:_\w+)+\b', re.IGNORECASE),  # snake_case names
    re.compile(r'\bcall the \w+(?:_\w+)*\b', re.IGNORECASE),
    re.compile(r'\binvoke \w+(?:_\w+)*\b', re.IGNORECASE),
    # "what X do I have access to" — allow 1+ words between "what" and the
    # verb phrase so "what private accounts do I have access to" matches.
    re.compile(r'\bwhat \w+(?:\s+\w+)* (?:do i|can i|do you) (?:have access to|access|see)\b', re.IGNORECASE),
    re.compile(r'\bwhat (?:capabilities|tools|accounts|gmails?) do (?:i|you) have\b', re.IGNORECASE),
    re.compile(r'\bwhat (?:can you do with|access do you have to)\b', re.IGNORECASE),
)


def _has_explicit_tool_invocation(text: str) -> tuple[str, ...]:
    """Return the lowercase matched substrings (for log payload) when the
    prompt unambiguously asks for tool invocation or capability surface."""
    matches: list[str] = []
    for pat in _EXPLICIT_TOOL_INVOCATION_PATTERNS:
        m = pat.search(text)
        if m:
            matches.append(m.group(0).lower())
    return tuple(matches)


def _has_keyword(text: str, keywords: tuple[str, ...]) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(k for k in keywords if re.search(rf'\b{re.escape(k)}\b', lowered))


def classify(text: str) -> IntentDecision:
    if not text or not text.strip():
        return IntentDecision(
            provider=Provider.CLAUDE,
            path=Path.CONVERSATIONAL,
            matched_keywords=(),
        )

    # H2-046: strip embedded document blocks so PDF/transcript body text
    # doesn't dominate the classifier. The user's surrounding instruction
    # text is what we want to match against, not 30KB of attached lease.
    text = strip_embedded_content(text)

    # H2-047 Fix 1: explicit tool-invocation / capability-question patterns
    # win unconditionally — they fire BEFORE the sensitive-keyword check so
    # "what gmails do I have access to" routes TOOL_LIKELY rather than
    # being dragged to OLLAMA on the word "gmails" or staying CONVERSATIONAL
    # and skipping nexus-self.describe_my_access entirely.
    explicit_tool_hits = _has_explicit_tool_invocation(text)
    if explicit_tool_hits:
        return IntentDecision(
            provider=Provider.CLAUDE,
            path=Path.TOOL_LIKELY,
            matched_keywords=explicit_tool_hits,
        )

    sensitive_hits = _has_keyword(text, _SENSITIVE_KEYWORDS)
    if sensitive_hits:
        return IntentDecision(
            provider=Provider.OLLAMA,
            path=Path.CONVERSATIONAL,
            matched_keywords=sensitive_hits,
        )

    code_hits = _has_keyword(text, _CODE_KEYWORDS)
    if code_hits:
        return IntentDecision(
            provider=Provider.CODEX,
            path=Path.CONVERSATIONAL,
            matched_keywords=code_hits,
        )

    tool_hits = _has_keyword(text, _TOOL_KEYWORDS)
    if tool_hits:
        return IntentDecision(
            provider=Provider.CLAUDE,
            path=Path.TOOL_LIKELY,
            matched_keywords=tool_hits,
        )

    return IntentDecision(
        provider=Provider.CLAUDE,
        path=Path.CONVERSATIONAL,
        matched_keywords=(),
    )
