# Nexus Core — Data Map

What a running instance stores, where, who can access it, and retention. Nexus
Core is a single-owner self-hosted assistant; it stores real personal data on
the operator's own machine. This map exists for accountability and to guide
encryption/retention decisions.

## Storage

| Store | Path | Engine | Encryption |
|---|---|---|---|
| Primary DB | `.data/nexus.db` (WAL) | SQLite | Plaintext by default; SQLCipher when `THREAT_MODEL=hosted` |
| Local memory / embeddings | `.data/nexus_memory.db` | SQLite + sentence-transformers | Plaintext |
| OAuth tokens (optional) | `.data/google_tokens/`, `.data/gmail_tokens/` | JSON files | File perms `600`, dir `700` |
| OAuth client secrets (optional) | `secrets/*.json` | JSON files | File perms `600`, dir `700` |
| Voice audio (transient) | `.data/voice_in/`, `.data/voice_out/` | files | deleted after handling |
| Logs | `.data/*.log` | text | secret-redacted at write |

Secrets are read from `.env` / `secrets/` (both gitignored) — never from the
codebase. `.gitignore` blocks `.env*`, `secrets/`, `*_credentials*`, `*_tokens*`,
`*_keys*`, `*.pem`, `*.key`, `*.token`.

## Data inventory (DB tables)

| Data | Table | Contains PII? | Access | Retention |
|---|---|---|---|---|
| Users (telegram id, name, prefs) | `users` | yes | pipeline | not auto-deleted |
| Reminders | `reminders` | body may | pipeline, scheduler | no auto-purge |
| Tasks | `tasks` | possible | pipeline | no auto-purge |
| Memories (key/value) | `memories` | possible | pipeline | no TTL; forget tool |
| Conversation context | `conversation_context` | yes | pipeline | 30-min TTL (filtered) |
| Conversation turns (full text) | `conversation_turns` | yes | pipeline | no auto-purge |
| Emails ingested (optional) | `emails_ingested` | yes | pipeline | no auto-purge |
| Approvals | `approvals` | payload may | pipeline | expiry sweep |
| Audit events (prompt+response) | `audit_events` | yes | pipeline | no auto-purge |
| Proactive notifications | `proactive_notifications` | possible | scheduler | dedupe |
| Opportunity signals | `opportunity_signals` | low | pipeline | dedupe |
| Voice usage counters | `elevenlabs_usage` | no | voice path | per day |
| Onboarding state | `telos_onboarding_state` | possible | pipeline | no auto-purge |
| Cron jobs | `cron_jobs` | action text may | scheduler | soft-delete |

## Access control

- **Telegram:** every inbound message is gated by `ALLOWED_TELEGRAM_IDS`; the
  sender id is Telegram-server-authenticated, not client-spoofable. Per-user data
  is scoped by `user_id` in every repository.
- **MCP servers:** scoped via `NEXUS_MCP_DEFAULT_USER_ID` per subprocess;
  ownership is enforced on mutating tools.

## Notes for operators

- The default deployment stores the DB in plaintext (single-user-local threat
  model). Set `THREAT_MODEL=hosted` + `SQLCIPHER_KEY` for at-rest encryption.
- Several tables (`conversation_turns`, `audit_events`) grow unbounded — add a
  retention/purge job if you keep long histories.
