"""H2-046 — brain_router ollama-unavailable fallthrough + clean error text.

Pre-H2-046 the literal '(brain_router error: ollama_url not configured)'
string was reaching Telegram chat replies (journal-confirmed at 23:21:12
EDT). Two production fixes:

  1. When the classifier routes a sensitive prompt to OLLAMA but
     ollama_url=None at construction time, transparently demote to
     Provider.CLAUDE rather than raising and bubbling to _fallback.
  2. _fallback now emits a clean canned message to the user regardless of
     which provider failed; the actual exception goes to journalctl only.
"""
from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.fallback_manager import PROVIDER_UNAVAILABLE_USER_TEXT
from services.brain_router import BrainRouter, BrainResult


def _make_completed(stdout: str, returncode: int = 0, stderr: str = '') -> MagicMock:
    m = MagicMock()
    m.returncode = returncode
    m.communicate.return_value = (stdout.encode(), stderr.encode())
    return m


# ---------------------------------------------------------------------------
# Ollama → Claude fallthrough when ollama_url is None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sensitive_route_with_no_ollama_demotes_to_claude(caplog):
    """The bug-1 fix. A sensitive-classified prompt (no MCP path) with
    ollama_url=None must NOT raise; it routes to Claude CLI instead.
    Log line `brain_router_ollama_unavailable_routing_to_claude` confirms
    the demotion happened."""
    router = BrainRouter(ollama_url=None)
    with caplog.at_level('INFO', logger='services.brain_router'):
        with patch('services.brain_router.subprocess.Popen',
                   return_value=_make_completed('claude reply')):
            result = await router.generate('this is my private medical journal')
    assert result.text == 'claude reply'
    assert result.provider_used == 'claude'
    demote_logs = [r for r in caplog.records
                   if r.message == 'brain_router_ollama_unavailable_routing_to_claude']
    assert len(demote_logs) == 1


@pytest.mark.asyncio
async def test_sensitive_route_with_real_ollama_url_uses_ollama():
    """Don't regress real Ollama routing: if ollama_url is configured the
    sensitive path still routes there. Asserted at the integration level
    (we monkeypatch the urlopen call)."""
    import json as _json
    router = BrainRouter(ollama_url='http://localhost:11434')
    fake = _json.dumps({'response': 'ollama reply'}).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = lambda *a: None
    mock_resp.read = lambda: fake
    with patch('urllib.request.urlopen', return_value=mock_resp):
        result = await router.generate('this is my private medical journal')
    assert result.provider_used == 'ollama'
    assert result.text == 'ollama reply'


# ---------------------------------------------------------------------------
# _fallback never leaks exception text to user
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_no_gemini_emits_canned_user_text():
    """No gemini_fallback configured → returns clean canned text, NOT the
    raw exception string."""
    router = BrainRouter(gemini_fallback=None)
    with patch.object(router, '_invoke_claude', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='claude', timeout=60))), \
         patch.object(router, '_invoke_codex', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='codex', timeout=120))):
        result = await router.generate('hello')
    assert result.fallback_used is True
    assert result.provider_used == 'none'
    text = result.text or ''
    assert text == PROVIDER_UNAVAILABLE_USER_TEXT
    assert 'brain_router error' not in text, (
        f'raw exception leaked to user: {text!r}'
    )
    assert 'ollama_url not configured' not in text


@pytest.mark.asyncio
async def test_fallback_logs_actual_error_to_journalctl(caplog):
    """The exception detail must end up in journalctl so we can debug
    production incidents — just not in the user-facing text."""
    router = BrainRouter(gemini_fallback=None)
    with caplog.at_level('WARNING', logger='services.brain_router'):
        with patch.object(router, '_invoke_claude', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='claude', timeout=60))), \
             patch.object(router, '_invoke_codex', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='codex', timeout=120))):
            await router.generate('hello')
    warn_logs = [r for r in caplog.records
                 if r.message == 'brain_router_fallback_engaged']
    assert len(warn_logs) == 1
    # The extra payload carries the last real provider failure in the chain.
    # Wave 2 now falls through Claude -> Codex before emitting the canned
    # fallback, so journalctl should still see a real timeout/provider detail
    # without hardcoding the old Claude-only wording.
    assert any(
        'timed out' in str(getattr(r, 'original_error', '')).lower()
        and any(
            provider in str(getattr(r, 'original_error', '')).lower()
            for provider in ('claude', 'codex')
        )
        for r in warn_logs
    )


@pytest.mark.asyncio
async def test_fallback_with_gemini_failure_still_emits_canned_text():
    """When Gemini fallback itself raises (e.g. 429 quota), the canned
    text is what reaches the user — NOT the gemini exception, NOT the
    original CLI exception."""
    fallback = MagicMock()

    async def _explode(**kwargs):
        raise RuntimeError('gemini 429: RESOURCE_EXHAUSTED')

    fallback.generate_text = _explode
    router = BrainRouter(gemini_fallback=fallback)
    with patch.object(router, '_invoke_claude', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='claude', timeout=60))), \
         patch.object(router, '_invoke_codex', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='codex', timeout=120))):
        result = await router.generate('hello')
    text = result.text or ''
    assert text == PROVIDER_UNAVAILABLE_USER_TEXT
    assert 'RESOURCE_EXHAUSTED' not in text
    assert 'TimeoutExpired' not in text
    assert 'brain_router error' not in text


@pytest.mark.asyncio
async def test_fallback_never_routes_to_gemini_after_wave_2():
    """H2-047 Wave 2 contract: Even when a working gemini_fallback object
    is constructed and passed to BrainRouter for back-compat, its
    generate_text MUST NOT be invoked. Pre-Wave-2 this test asserted the
    opposite — that a successful Gemini call returned its text. That
    surface is now retired."""
    fallback = MagicMock()
    fallback.generate_text = MagicMock()

    router = BrainRouter(gemini_fallback=fallback)
    with patch.object(router, '_invoke_claude', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='claude', timeout=60))), \
         patch.object(router, '_invoke_codex', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='codex', timeout=120))):
        result = await router.generate('hello')
    assert not fallback.generate_text.called, (
        'gemini_fallback was reached — Wave 2 should never invoke it'
    )
    assert result.provider_used == 'none'
    assert result.text == PROVIDER_UNAVAILABLE_USER_TEXT
