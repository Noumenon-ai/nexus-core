# Phase Test Gates

## Phase 0
What it tests: environment loading, startup validation, schema creation, and log redaction.
How to run: `pytest tests/phase_0/ -v`
Expected pass count: 4
Acceptance criteria: config loads typed settings, startup creates protected data directories, core tables exist, and sensitive logging terms are redacted.

## Phase 1
What it tests: unified pipeline authorization, assistant-first general replies, morning briefing routing, voice transcript normalization, and unauthorized-user rejection without persistence.
How to run: `pytest tests/phase_1/ -v`
Expected pass count: 5
Acceptance criteria: unauthorized users are rejected at entry, unauthorized input does not create a user row, normal speech stays conversational, morning greetings trigger the general briefing path, and voice input flows through the same pipeline.

## Phase 2
What it tests: reminder parsing, mandatory confirmation, active reminder listing/cancellation, and boot recovery behavior.
How to run: `pytest tests/phase_2/ -v`
Expected pass count: 6
Acceptance criteria: one-shot reminders confirm before saving, recurring reminders preserve explicit times and produce RRULEs, stale reminder confirmations are rejected, reminders can be listed and cancelled, missed one-shots fire once on recovery, and missed recurring reminders jump to the next future slot.

## Phase 3
What it tests: explicit memory, passive habit learning, task creation/completion, and day organization ordering.
How to run: `pytest tests/phase_3/ -v`
Expected pass count: 6
Acceptance criteria: explicit memories save and list correctly, habits accumulate observed patterns, live reminder/task flows update habit memory automatically, task due phrases are stripped from stored titles, tasks can be added and completed, and nearby reminders outrank later tasks.

## Phase 4
What it tests: approval prompt generation, expiry refusal, expiry sweeps, cross-user approval isolation, and approved action execution.
How to run: `pytest tests/phase_4/ -v`
Expected pass count: 5
Acceptance criteria: sensitive actions generate approval prompts, expired approvals cannot execute, pending expiries are swept and notified, users cannot execute each other's approvals, and approved memory deletion executes through the action layer.

## Phase 5
What it tests: low-confidence voice handling, transcription backend failure fallback, optional voice output, and TTS failure fallback.
How to run: `pytest tests/phase_5/ -v`
Expected pass count: 5
Acceptance criteria: unclear voice does not enter the LLM path, transcription backend failures degrade to the safe retry message, enabled voice output attaches synthesized audio, OpenAI TTS output is normalized to OGG for Telegram, and TTS failure still returns text.

## Phase 6
What it tests: Gmail read-only summarization and safe failure handling.
How to run: `pytest tests/phase_6/ -v`
Expected pass count: 2
Acceptance criteria: recent email metadata is categorized and summarized without storing bodies, and connection failures return the safe user-facing error.

## Phase 7
What it tests: proactive briefing composition, per-day dedupe, per-loop daily isolation, and downtime recovery timing policy.
How to run: `pytest tests/phase_7/ -v`
Expected pass count: 5
Acceptance criteria: morning briefings include reminders/tasks/email signals, repeated runs of the same loop dedupe per day, different loops do not suppress each other on the same day, recent missed loops recover on startup, and stale recoveries are skipped.

## Phase 8
What it tests: web search summarization, watch-interest persistence, and opportunity dedupe.
How to run: `pytest tests/phase_8/ -v`
Expected pass count: 4
Acceptance criteria: search results summarize into a concise answer with citations, watch interests persist, opportunity scans run on the configured cadence, and fuzzy title matching suppresses duplicate opportunity signals.

## Phase 9
What it tests: AI quota fallback, breaker fallback, half-open breaker recovery, DB lock retry, approval expiry through the router, and Telegram retry behavior.
How to run: `pytest tests/phase_9/ -v`
Expected pass count: 7
Acceptance criteria: quota exhaustion returns the explicit quota message, breaker-open requests short-circuit to fallback, a successful half-open probe closes the breaker, transient SQLite locks recover with retry, expired approval callbacks are refused, the live Telegram bot send path retries once on transient failure, and one retry delivers Telegram messages after a transient failure.
