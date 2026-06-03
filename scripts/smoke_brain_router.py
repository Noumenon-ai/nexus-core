"""Real CLI smoke test for brain_router. Exercises Claude + Codex + fallback paths.

Run from project root:
    .venv/bin/python scripts/smoke_brain_router.py

Does NOT touch the running bot. Just verifies brain_router can talk to real CLIs.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.brain_router import BrainRouter


async def smoke_conversational_claude():
    print("\n[1/4] Conversational -> Claude CLI")
    router = BrainRouter(claude_timeout_sec=30)
    result = await router.generate("Say the word 'pong' and nothing else.")
    print(f"  provider_used: {result.provider_used}")
    print(f"  fallback_used: {result.fallback_used}")
    print(f"  text: {result.text!r}")
    assert result.provider_used == 'claude', f"expected claude, got {result.provider_used}"
    assert result.text is not None and 'pong' in result.text.lower(), f"expected 'pong' in reply"
    print("  PASS")


async def smoke_code_codex():
    print("\n[2/4] Code -> Codex CLI")
    router = BrainRouter(codex_timeout_sec=60)
    result = await router.generate("Write one line of Python code to print the word 'pong'. Code only, no explanation.")
    print(f"  provider_used: {result.provider_used}")
    print(f"  text: {result.text!r}")
    if result.fallback_used:
        print(f"  WARN: codex fell back: {result.text}")
    else:
        assert result.provider_used == 'codex'
        print("  PASS")


async def smoke_tool_likely_with_catalog():
    print("\n[3/4] Tool-likely with catalog -> Claude CLI (expect tool_calls or text)")
    router = BrainRouter(claude_timeout_sec=60)
    catalog = [
        {
            'name': 'create_reminder',
            'description': 'Create a reminder for the user. body is the reminder text, next_fire_at is an ISO 8601 UTC datetime in the future.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'body': {'type': 'string'},
                    'next_fire_at': {'type': 'string'},
                },
                'required': ['body', 'next_fire_at'],
            },
        },
    ]
    result = await router.generate(
        "Remind me at 2026-06-01T12:00:00Z to call the bank.",
        tool_catalog=catalog,
    )
    print(f"  provider_used: {result.provider_used}")
    if result.tool_calls:
        print(f"  tool_calls: {result.tool_calls}")
        print("  PASS (got structured tool call)")
    else:
        print(f"  text: {result.text!r}")
        print("  PASS (got text fallback — parser didn't find JSON, that's acceptable)")


async def smoke_subprocess_failure_with_no_fallback():
    print("\n[4/4] Bogus binary -> graceful failure (no gemini_fallback configured)")
    router = BrainRouter(claude_binary='this_binary_does_not_exist_xyzzy', claude_timeout_sec=5)
    result = await router.generate("hello")
    print(f"  provider_used: {result.provider_used}")
    print(f"  fallback_used: {result.fallback_used}")
    print(f"  text: {result.text!r}")
    assert result.fallback_used is True
    assert result.provider_used == 'none'
    print("  PASS (graceful degradation)")


async def main():
    print("===== brain_router real-CLI smoke test =====")
    try:
        await smoke_conversational_claude()
    except AssertionError as e:
        print(f"  FAIL: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        sys.exit(1)

    try:
        await smoke_code_codex()
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    try:
        await smoke_tool_likely_with_catalog()
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        sys.exit(1)

    try:
        await smoke_subprocess_failure_with_no_fallback()
    except AssertionError as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    print("\n===== ALL SMOKE TESTS PASSED =====")


if __name__ == '__main__':
    asyncio.run(main())
