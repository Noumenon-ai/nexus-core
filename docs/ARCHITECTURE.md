# Architecture

Nexus Core is a layered, single-process assistant runtime. A message enters
through one channel (Telegram), is authorized, normalized, and handed to an
agentic tool loop that talks to an LLM and a fleet of MCP servers.

## Request lifecycle

1. **Channel** (`telegram_bot.py`) receives a text/voice/photo update and hands a
   normalized `PipelineInput` to the pipeline.
2. **Pipeline** (`pipeline/unified.py`) enforces the authorization gate
   (`ALLOWED_TELEGRAM_IDS`), normalizes voice→text, and dispatches.
3. **Tool dispatcher** (`pipeline/tool_dispatcher.py`) runs the agentic loop:
   build the system prompt → call the LLM → execute any proposed tool calls →
   feed results back → repeat until a final reply (bounded by a hard iteration
   cap). Destructive tool calls are intercepted by the approval gate.
4. **Brain router** (`services/brain_router.py`) is the single LLM entry point.
   It classifies the turn and routes to a local Claude/Codex CLI subprocess
   (with `--mcp-config` loading the MCP servers) or a local Ollama model, with
   circuit breakers, retries, and a deterministic fallback responder.
5. **MCP servers** (`mcp_servers/`) expose tools over the Model Context Protocol;
   each runs as its own subprocess and shares the SQLite store via `DATABASE_URL`.
6. **Services** (`services/`) hold domain logic: reminders, tasks, memory,
   approvals, conversational recovery, self-correction, voice, capability
   introspection.
7. **Repositories** (`repositories/`) are thin SQLAlchemy data-access objects,
   every user-scoped one filtering by `user_id`.

## Safety model

- **Authorization** happens before any routing — an unknown sender never reaches
  the dispatcher.
- **Destructive-intent classification** + an **approval gate**: a tool that
  deletes, sends, or overwrites is held as a pending `Approval` until the user
  confirms in-chat. Approvals are scoped to the requesting user.
- **Input validation** + parametrized queries throughout; the filesystem tools
  enforce an allow-list of roots and a denylist of sensitive paths/filenames.

## Conversational layer

- **Conversational recovery** (`services/conversational_recovery.py`) repairs
  messy multi-turn input (time-only follow-ups, corrections, vague references).
- **Self-correction** (`services/self_correction.py`) detects in-message
  revisions ("June 2 no June 4", "5pm change it to 6pm") and echoes the change so
  it is never silently applied.

## Persistence

A single SQLite database (WAL mode; optional SQLCipher in hosted mode) holds
users, reminders, tasks, memories, conversation context/turns, approvals, cron
jobs, audit events, and usage counters. See [DATA_MAP.md](DATA_MAP.md).

## Observability

Structured logging with a redaction filter (api keys, tokens, bearer headers),
a `/health` endpoint pattern, and request timing. No PII is logged.
