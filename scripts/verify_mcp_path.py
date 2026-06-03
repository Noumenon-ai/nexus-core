"""Phase 2.5a live-verification harness.

Bypasses Telegram entirely. Spins up BrainRouter against the bot's same
config + .mcp.dev.json, runs 6 probes (one per MCP server's primary tool),
and prints (path_taken, reply) for each.

Usage:
    .venv/bin/python scripts/verify_mcp_path.py

The script uses the SAME brain_router + classifier the live bot uses, so the
results reflect production routing. NEXUS_MCP_CONFIG defaults to .mcp.dev.json.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `services.*` and `mcp_servers.*` resolve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.brain_router import BrainRouter
from services.intent_classifier import Path as RoutePath, classify

# Don't disable MCP — we want the live path to fire.
os.environ.setdefault("NEXUS_MCP_CONFIG", str(PROJECT_ROOT / ".mcp.dev.json"))

PROBES: list[tuple[str, str]] = [
    ("nexus-utils",            "Run do_math on 247 * 13 + 89."),
    ("nexus-filesystem",       "List the files in my Documents folder. Use the list_directory tool."),
    ("nexus-knowledge",        "Save my favorite color as blue using the remember tool. Tag it preferences."),
    ("nexus-pdf-docs",         "List the available PDF templates with the list_templates tool."),
    ("nexus-reminders-tasks",  "List my active reminders using the list_reminders tool."),
    ("nexus-calendar",         "Show me today's calendar schedule."),
    # Phase 2.5b probes
    ("nexus-email",            "Summarize my unread emails from the last 24 hours using the summarize_inbox tool."),
    ("nexus-contacts",         "List my contact groups using the list_groups tool."),
    # Phase 6 cron probe
    ("nexus-utils (cron)",     "Create a cron job that runs '* * * * *' with description 'phase6 verify' and action 'tick'. Use create_cron_job."),
    # Phase 2.5c messaging probe
    ("nexus-messaging",        "List my recent WhatsApp conversations using list_recent_conversations."),
    # Phase 2.5d rentals probe
    ("nexus-rentals",          "Show me a rental portfolio summary using the rental_summary tool."),
    # Phase 2.5e business probe
    ("nexus-business",         "List all my LLCs using the list_llcs tool."),
]


async def run() -> None:
    router = BrainRouter()
    print(f"MCP config: {os.environ.get('NEXUS_MCP_CONFIG')}", flush=True)
    print(f"File exists: {Path(os.environ['NEXUS_MCP_CONFIG']).exists()}", flush=True)
    print("=" * 80, flush=True)

    for server, prompt in PROBES:
        decision = classify(prompt)
        print(f"\n>>> {server}", flush=True)
        print(f"    prompt:           {prompt!r}", flush=True)
        print(f"    classifier path:  {decision.path.value}  (provider={decision.provider.value})", flush=True)
        print(f"    matched_keywords: {list(decision.matched_keywords)}", flush=True)
        if decision.path is not RoutePath.TOOL_LIKELY:
            print("    SKIP — not classified as TOOL_LIKELY, MCP path will not fire", flush=True)
            continue
        try:
            result = await router.generate_with_tools(
                user_id="verify-harness",
                system_prompt="You are Nexus. Use the available MCP tools when relevant.",
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                tool_catalog=[],
            )
        except Exception as exc:
            print(f"    EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
            continue
        text = result.get("text", "")
        if not text and result.get("tool_calls"):
            text = f"<tool_calls returned: {result['tool_calls']}>"
        # Cap reply to keep terminal readable
        if len(text) > 1500:
            text = text[:1500] + f"\n  ... [truncated, total {len(text)} chars]"
        print(f"    reply:\n{text}", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
