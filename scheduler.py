from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger


_RUNTIME: Any | None = None


async def dispatch_reminder_job(reminder_id: str) -> None:
    if _RUNTIME is None:
        return
    await _RUNTIME.reminder_service.fire_reminder(reminder_id)


async def dispatch_proactive_job(loop_type: str, user_id: str) -> None:
    if _RUNTIME is None:
        return
    if not _RUNTIME.settings.proactive.proactive_enabled:
        return
    user = _RUNTIME.users_repository.get_by_id(user_id)
    if user is None or not user.proactive_enabled:
        return
    translator = _RUNTIME.translator_for(user.language)
    if loop_type == 'morning_briefing':
        response = await _RUNTIME.proactive_service.morning_briefing(user, translator)
    elif loop_type == 'midday_check':
        response = await _RUNTIME.proactive_service.midday_check(user, translator)
    else:
        response = await _RUNTIME.proactive_service.evening_wrap(user, translator)
    if not response.metadata.get('should_send', True):
        return
    await _RUNTIME.notify_user(user.id, response.text)
    # V3.8 nudge bookkeeping: only commit last_nudge_at AFTER notify_user
    # returns, and only when this briefing actually included the nudge.
    # Failure of notify_user above raises and skips this commit, leaving
    # the nudge budget unconsumed for next cron firing — correct.
    if response.metadata.get('nudge_included'):
        _RUNTIME.proactive_service.mark_nudge_committed(user.id)


async def dispatch_opportunity_scan_job(user_id: str) -> None:
    if _RUNTIME is None:
        return
    user = _RUNTIME.users_repository.get_by_id(user_id)
    if user is None:
        return
    translator = _RUNTIME.translator_for(user.language)
    suggestions = await _RUNTIME.opportunity_service.scan_user(user, translator)
    for suggestion in suggestions:
        await _RUNTIME.notify_user(user.id, suggestion)


def dispatch_approval_sweep() -> None:
    if _RUNTIME is None:
        return
    _RUNTIME.approval_service.sweep_expired(_RUNTIME.notify_user_sync, _RUNTIME.translator_for('en'))


async def dispatch_cron_job(cron_id: str) -> None:
    """Phase 6 fire callback: re-inject the cron's action_text into the pipeline
    as if the owning user just typed it, then send the response back via Telegram.
    """
    if _RUNTIME is None:
        return
    if _RUNTIME.cron_jobs_repository is None or _RUNTIME.pipeline is None:
        return
    job = _RUNTIME.cron_jobs_repository.get(cron_id)
    if job is None or job.status != 'active':
        return
    user = _RUNTIME.users_repository.get_by_id(job.user_id)
    if user is None:
        return
    from pipeline.types import PipelineInput
    try:
        output = await _RUNTIME.pipeline.handle(PipelineInput(
            kind='text',
            telegram_id=user.telegram_id,
            text=job.action_text,
            username=user.username,
            full_name=user.full_name,
        ))
        text = (output.text or '').strip() if output is not None else ''
        if text:
            await _RUNTIME.notify_user(user.id, f'[⏰ cron] {text}')
    except Exception:
        import logging
        logging.getLogger(__name__).exception('dispatch_cron_job_failed', extra={'cron_id': cron_id})
        return
    finally:
        _RUNTIME.cron_jobs_repository.record_fire(cron_id)


def dispatch_cron_sync() -> None:
    """Periodic sync: pull rows with pending_sync=1 and reconcile APScheduler."""
    if _RUNTIME is None or _RUNTIME.cron_jobs_repository is None:
        return
    repo = _RUNTIME.cron_jobs_repository
    sched = _RUNTIME.nexus_scheduler
    if sched is None:
        return
    for job in repo.list_pending_sync():
        sched.apply_cron_sync(job)
        repo.mark_synced(job.id)


def dispatch_reminder_sync() -> None:
    """H2-049: bridge MCP-created reminders into APScheduler.

    MCP servers run as subprocesses and write reminder rows directly to
    the DB — they have no access to the bot's running APScheduler
    instance. Without this sweep an MCP-created reminder sits in the DB
    with status='active' but no scheduled job, so it never fires (the
    user pattern: "Remind me in 60 seconds" → NEXUS confirms → silence).

    The cron-tools MCP solved the same cross-process problem with a
    pending_sync flag. Reminders pre-date that pattern; we reconcile
    idempotently instead: every active future reminder must have a
    corresponding apscheduler_jobs row.

    `schedule_reminder` uses replace_existing=True so re-queueing a job
    that's already present is a no-op. Reminders that have already fired
    (last_fired_at NOT NULL OR status='fired') are skipped so we never
    resurrect a delivered one.
    """
    if _RUNTIME is None:
        return
    reminder_service = getattr(_RUNTIME, 'reminder_service', None)
    sched = _RUNTIME.nexus_scheduler
    if reminder_service is None or sched is None:
        return
    try:
        reminder_service.reconcile_with_scheduler()
        asyncio.run(reminder_service.redeliver_failed())
    except Exception:
        # Sync failures must never propagate — they'd crash the
        # periodic job and APScheduler would stop running it.
        pass


@dataclass(slots=True)
class NexusScheduler:
    database_url: str
    timezone: str
    scheduler: AsyncIOScheduler = field(init=False)

    def __post_init__(self) -> None:
        self.scheduler = AsyncIOScheduler(jobstores={'default': SQLAlchemyJobStore(url=self.database_url)}, timezone=self.timezone)

    def register_runtime(self, runtime: Any) -> None:
        global _RUNTIME
        _RUNTIME = runtime

    def start(self) -> None:
        self.scheduler.start()

    def shutdown(self) -> None:
        self.scheduler.shutdown(wait=False)

    def schedule_reminder(self, reminder_id: str, run_date: datetime) -> None:
        # H2-048: naive datetimes passed to APScheduler are interpreted in
        # the scheduler's configured timezone (America/New_York for this
        # bot). The reminder pipeline produces UTC-correct values but
        # SQLAlchemy strips the tzinfo before persistence; once a naive
        # UTC value re-enters APScheduler, it's mis-read as local time and
        # fires `app_timezone_offset` hours late (4 hours in EDT). Re-anchor
        # to UTC at this boundary so wall-clock firing matches the value
        # the user agreed to when they created the reminder.
        if run_date.tzinfo is None:
            run_date = run_date.replace(tzinfo=timezone.utc)
        self.scheduler.add_job(
            dispatch_reminder_job, 'date', run_date=run_date,
            args=[reminder_id], id=f'reminder-{reminder_id}',
            replace_existing=True,
        )

    def remove_reminder(self, reminder_id: str) -> None:
        job_id = f'reminder-{reminder_id}'
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)

    def schedule_housekeeping(self, *, approval_interval_sec: int) -> None:
        self.scheduler.add_job(dispatch_approval_sweep, 'interval', seconds=approval_interval_sec, id='approval-sweep', replace_existing=True)

    def schedule_daily_loops(self, users: list[Any], *, morning_hour: int, midday_hour: int, evening_hour: int) -> None:
        for user in users:
            self.scheduler.add_job(dispatch_proactive_job, CronTrigger(hour=morning_hour, minute=0), args=['morning_briefing', user.id], id=f'morning-{user.id}', replace_existing=True)
            self.scheduler.add_job(dispatch_proactive_job, CronTrigger(hour=midday_hour, minute=0), args=['midday_check', user.id], id=f'midday-{user.id}', replace_existing=True)
            self.scheduler.add_job(dispatch_proactive_job, CronTrigger(hour=evening_hour, minute=0), args=['evening_wrap', user.id], id=f'evening-{user.id}', replace_existing=True)

    def schedule_opportunity_scans(self, users: list[Any], *, interval_hours: int) -> None:
        if interval_hours <= 0:
            return
        for user in users:
            self.scheduler.add_job(dispatch_opportunity_scan_job, 'interval', hours=interval_hours, args=[user.id], id=f'opportunity-{user.id}', replace_existing=True)

    # ---------- Phase 6: cron jobs ----------

    def schedule_reminder_sync(self, *, interval_sec: int = 30) -> None:
        """H2-049: periodic reminder reconciliation. MCP create_reminder
        writes the DB only — this sweep queues the apscheduler_jobs row."""
        self.scheduler.add_job(
            dispatch_reminder_sync, 'interval', seconds=interval_sec,
            id='reminder-sync', replace_existing=True,
        )

    def schedule_cron_sync(self, *, interval_sec: int = 30) -> None:
        """Periodic re-sync of cron rows that have pending_sync=1."""
        self.scheduler.add_job(dispatch_cron_sync, 'interval', seconds=interval_sec,
                               id='cron-sync', replace_existing=True)

    def apply_cron_sync(self, job: Any) -> None:
        """Add / update / remove an APScheduler job to match a CronJob DB row."""
        job_id = f'cron-{job.id}'
        if job.status != 'active':
            existing = self.scheduler.get_job(job_id)
            if existing is not None:
                self.scheduler.remove_job(job_id)
            return
        try:
            trigger = CronTrigger.from_crontab(job.cron_expression, timezone=self.timezone)
        except ValueError:
            return
        self.scheduler.add_job(
            dispatch_cron_job, trigger=trigger, args=[job.id],
            id=job_id, replace_existing=True,
        )

    def restore_cron_jobs(self, jobs: list[Any]) -> None:
        """Boot-time recovery: re-register every active CronJob with APScheduler."""
        for job in jobs:
            self.apply_cron_sync(job)
