"""V3.8 TELOS onboarding state repository.

Reads/writes the `telos_onboarding_state` table. Sync, follows the
`BaseRepository._execute` pattern shared by every other V3.x repo.

Two-tone access:
  - Tools call get/get_or_create/save/mark_started/mark_completed/
    mark_cancelled/record_answer to drive the onboarding flow.
  - ProactiveService calls get + mark_nudged to bookkeep the
    once-per-7-days briefing nudge cap.

Per-user isolation rests on the schema's PK on `user_id` (one row per
user) and FK to `users.id` (no orphan states).
"""
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from models import TelosOnboardingState, now_utc
from repositories.base import BaseRepository


class TelosOnboardingStateRepository(BaseRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory)

    def get(self, user_id: str) -> TelosOnboardingState | None:
        def operation(session: Session) -> TelosOnboardingState | None:
            return session.scalar(
                select(TelosOnboardingState).where(TelosOnboardingState.user_id == user_id)
            )
        return self._execute(operation)

    def get_or_create(self, user_id: str) -> TelosOnboardingState:
        """Returns the existing state row or creates a new one with
        defaults (current_section='identity', answers_so_far='{}').
        Does NOT set `started_at` — that's the responsibility of
        `mark_started`, called from `start_telos_onboarding`. The split
        lets the briefing nudge bookkeeping write rows BEFORE the user
        has actually started onboarding."""
        def operation(session: Session) -> TelosOnboardingState:
            row = session.scalar(
                select(TelosOnboardingState).where(TelosOnboardingState.user_id == user_id)
            )
            if row is None:
                row = TelosOnboardingState(user_id=user_id)
                session.add(row)
                session.flush()
            return row
        return self._execute(operation)

    def mark_started(self, user_id: str, *, when: datetime | None = None) -> None:
        """Sets `started_at` if it is currently NULL. Idempotent —
        re-calling on a row that already has `started_at` does NOT
        overwrite it (resume should preserve the original start
        timestamp)."""
        timestamp = when or now_utc()
        def operation(session: Session) -> None:
            row = session.scalar(
                select(TelosOnboardingState).where(TelosOnboardingState.user_id == user_id)
            )
            if row is None:
                row = TelosOnboardingState(user_id=user_id, started_at=timestamp)
                session.add(row)
            elif row.started_at is None:
                row.started_at = timestamp
            session.flush()
        self._execute(operation)

    def mark_completed(self, user_id: str, *, when: datetime | None = None) -> None:
        timestamp = when or now_utc()
        def operation(session: Session) -> None:
            row = session.scalar(
                select(TelosOnboardingState).where(TelosOnboardingState.user_id == user_id)
            )
            if row is not None:
                row.completed_at = timestamp
                row.cancelled_at = None  # completing supersedes any prior cancel
                session.flush()
        self._execute(operation)

    def mark_cancelled(self, user_id: str, *, when: datetime | None = None) -> None:
        """Sets `cancelled_at` but PRESERVES `current_section` and
        `answers_so_far` so a later start_telos_onboarding call can
        resume from the same place. V3.8 halt condition: cancel must
        not lose state."""
        timestamp = when or now_utc()
        def operation(session: Session) -> None:
            row = session.scalar(
                select(TelosOnboardingState).where(TelosOnboardingState.user_id == user_id)
            )
            if row is not None:
                row.cancelled_at = timestamp
                session.flush()
        self._execute(operation)

    def clear_cancelled(self, user_id: str) -> None:
        """Clears `cancelled_at` when the user resumes onboarding.
        Called by start_telos_onboarding when picking up a paused flow.
        """
        def operation(session: Session) -> None:
            row = session.scalar(
                select(TelosOnboardingState).where(TelosOnboardingState.user_id == user_id)
            )
            if row is not None:
                row.cancelled_at = None
                session.flush()
        self._execute(operation)

    def mark_nudged(self, user_id: str, *, when: datetime | None = None) -> None:
        """Update `last_nudge_at` after the briefing nudge has actually
        been sent to the user. Caller invokes this AFTER notify_user
        succeeds — never inside morning_briefing itself, so test
        harnesses and debug commands that compute the briefing without
        delivering it do not consume the once-per-7-days budget."""
        timestamp = when or now_utc()
        def operation(session: Session) -> None:
            row = session.scalar(
                select(TelosOnboardingState).where(TelosOnboardingState.user_id == user_id)
            )
            if row is None:
                row = TelosOnboardingState(user_id=user_id, last_nudge_at=timestamp)
                session.add(row)
            else:
                row.last_nudge_at = timestamp
            session.flush()
        self._execute(operation)

    def record_answer(self, user_id: str, *, question_id: str, answer: str) -> dict[str, str]:
        """Merge a single answer into `answers_so_far` JSON, persist,
        return the updated answers dict. Caller (answer_telos_question
        tool) uses the returned dict to decide whether the current
        section is complete and the file should be appended.
        Concurrency: this single-row update is wrapped in
        `_execute`'s retry loop, so a busy DB is handled by the same
        backoff every other repo uses."""
        def operation(session: Session) -> dict[str, str]:
            row = session.scalar(
                select(TelosOnboardingState).where(TelosOnboardingState.user_id == user_id)
            )
            if row is None:
                row = TelosOnboardingState(user_id=user_id)
                session.add(row)
                session.flush()
            try:
                answers = json.loads(row.answers_so_far) if row.answers_so_far else {}
            except json.JSONDecodeError:
                answers = {}
            answers[question_id] = answer
            row.answers_so_far = json.dumps(answers, ensure_ascii=False)
            session.flush()
            return answers
        return self._execute(operation)

    def advance_section(self, user_id: str, *, next_section: str) -> None:
        def operation(session: Session) -> None:
            row = session.scalar(
                select(TelosOnboardingState).where(TelosOnboardingState.user_id == user_id)
            )
            if row is not None:
                row.current_section = next_section
                session.flush()
        self._execute(operation)
