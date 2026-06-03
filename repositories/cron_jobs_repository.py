"""Phase 6 — cron_jobs CRUD repository.

The MCP tool layer in nexus-utils writes rows here; the bot's APScheduler-
backed sync loop polls every N seconds, applies pending_sync rows, and runs
the matching CronTrigger jobs.  See scheduler.dispatch_cron_job for the fire
callback path.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from models import CronJob
from repositories.base import BaseRepository


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class CronJobsRepository(BaseRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory)

    def create(
        self,
        *,
        user_id: str,
        description: str,
        cron_expression: str,
        action_text: str,
    ) -> CronJob:
        def op(session: Session) -> CronJob:
            job = CronJob(
                user_id=user_id,
                description=description,
                cron_expression=cron_expression,
                action_text=action_text,
                status="active",
                pending_sync=1,
            )
            session.add(job)
            session.flush()
            return job
        return self._execute(op)

    def get(self, job_id: str, *, user_id: str | None = None) -> CronJob | None:
        def op(session: Session) -> CronJob | None:
            job = session.get(CronJob, job_id)
            if job is None:
                return None
            if user_id is not None and job.user_id != user_id:
                return None
            return job
        return self._execute(op)

    def list_for_user(self, user_id: str, *, include_deleted: bool = False) -> list[CronJob]:
        def op(session: Session) -> list[CronJob]:
            stmt = select(CronJob).where(CronJob.user_id == user_id)
            if not include_deleted:
                stmt = stmt.where(CronJob.status != "deleted")
            stmt = stmt.order_by(CronJob.created_at)
            return list(session.scalars(stmt))
        return self._execute(op)

    def list_pending_sync(self) -> list[CronJob]:
        """All rows the scheduler hasn't picked up yet (new + status changes)."""
        def op(session: Session) -> list[CronJob]:
            stmt = select(CronJob).where(CronJob.pending_sync == 1)
            return list(session.scalars(stmt))
        return self._execute(op)

    def mark_synced(self, job_id: str) -> None:
        def op(session: Session) -> None:
            session.execute(update(CronJob).where(CronJob.id == job_id).values(pending_sync=0))
        self._execute(op)

    def update_status(self, job_id: str, status: str, *, user_id: str | None = None) -> CronJob | None:
        def op(session: Session) -> CronJob | None:
            job = session.get(CronJob, job_id)
            if job is None:
                return None
            if user_id is not None and job.user_id != user_id:
                return None
            job.status = status
            job.pending_sync = 1
            return job
        return self._execute(op)

    def update_expression(self, job_id: str, *, cron_expression: str | None = None,
                          action_text: str | None = None, description: str | None = None,
                          user_id: str | None = None) -> CronJob | None:
        def op(session: Session) -> CronJob | None:
            job = session.get(CronJob, job_id)
            if job is None:
                return None
            if user_id is not None and job.user_id != user_id:
                return None
            if cron_expression is not None:
                job.cron_expression = cron_expression
            if action_text is not None:
                job.action_text = action_text
            if description is not None:
                job.description = description
            job.pending_sync = 1
            return job
        return self._execute(op)

    def record_fire(self, job_id: str) -> None:
        def op(session: Session) -> None:
            job = session.get(CronJob, job_id)
            if job is None:
                return
            job.last_fired_at = _now_utc()
            job.fire_count = (job.fire_count or 0) + 1
        self._execute(op)

    def all_active(self) -> list[CronJob]:
        def op(session: Session) -> list[CronJob]:
            return list(session.scalars(select(CronJob).where(CronJob.status == "active")))
        return self._execute(op)
