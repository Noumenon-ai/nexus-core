"""H2-047 Fix 1 — explicit tool-invocation + capability-question patterns
route TOOL_LIKELY, bypassing the sensitive-keyword carve-out.

Pre-H2-047, "What gmails do you have access to?" landed CONVERSATIONAL
(no tool keywords matched). "Use the X tool" similarly fell through to a
plain conversational Claude CLI call. After this fix both route through
MCP so the dispatcher loop / nexus-self.describe_my_access can run.
"""
from __future__ import annotations

import pytest

from services.intent_classifier import Provider, Path, classify


# ---------------------------------------------------------------------------
# Phrasings the audit specifically called out as broken
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('prompt', [
    'Use the pussy tool',
    'Use the calendar tool',
    'use the search_email tool',
    'Call the search_email tool',
    'Invoke list_directory',
    'What gmails do you have access to',
    'What gmails do you have access to?',
    'What accounts can i access',
    'What capabilities do you have',
    'What tools do you have',
    'What can you do with email',
])
def test_explicit_tool_invocation_routes_tool_likely(prompt):
    decision = classify(prompt)
    assert decision.path is Path.TOOL_LIKELY, (
        f'{prompt!r} routed {decision.path.value!r}, expected tool_likely'
    )
    assert decision.provider is Provider.CLAUDE


def test_capability_question_overrides_sensitive_word_in_prompt():
    """'What private accounts do I have access to' contains 'private'
    (sensitive keyword) but the capability question regex must fire first
    and force TOOL_LIKELY. Pre-H2-047 this routed to OLLAMA — and ollama
    isn't configured in production, so the user saw the canned 'trouble
    reaching a provider' message instead of a real answer."""
    decision = classify('What private accounts do I have access to')
    assert decision.path is Path.TOOL_LIKELY
    assert decision.provider is Provider.CLAUDE


# ---------------------------------------------------------------------------
# Negative cases — must not regress
# ---------------------------------------------------------------------------


def test_sensitive_prompt_without_capability_pattern_still_routes_sensitive():
    """A bare sensitive prompt with no tool-invocation pattern still goes
    to OLLAMA-provider, CONVERSATIONAL-path (the historical contract)."""
    decision = classify('Tell me about my personal medical history.')
    assert decision.provider is Provider.OLLAMA
    assert decision.path is Path.CONVERSATIONAL


def test_plain_greeting_stays_conversational():
    decision = classify('hello there')
    assert decision.path is Path.CONVERSATIONAL


def test_existing_tool_keyword_still_fires():
    """Pre-H2-047 tool-keyword matches (list / delete / etc.) keep
    working — regex pre-check is additive."""
    decision = classify('list my home directory')
    assert decision.path is Path.TOOL_LIKELY


def test_empty_prompt_safe():
    decision = classify('')
    assert decision.path is Path.CONVERSATIONAL
    assert decision.matched_keywords == ()


def test_regex_runs_after_strip_embedded_content():
    """The regex pre-check must operate on the strip_embedded_content
    output, not the raw prompt — otherwise a PDF body containing 'use
    the X tool' would force-route TOOL_LIKELY on every PDF upload."""
    prompt = (
        "Summarize this:\n\n"
        "--- BEGIN PDF TEXT ---\n"
        "Use the signature tool to complete this contract.\n"
        "--- END PDF TEXT ---"
    )
    decision = classify(prompt)
    # 'Summarize' is in _TOOL_KEYWORDS, so the result IS tool_likely —
    # but via the standard tool-keyword path, not via the regex match
    # on the embedded PDF body.
    assert 'summarize' in decision.matched_keywords or 'summary' in decision.matched_keywords or \
           decision.matched_keywords  # don't pin the exact keyword set, just verify regex didn't pick up the PDF body
    # The forensic check: matched_keywords must not contain 'use the signature tool'
    for kw in decision.matched_keywords:
        assert 'signature tool' not in kw, (
            f'regex matched embedded PDF body content: {kw!r}'
        )
