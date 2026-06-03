"""H2-049 — reminder reconciliation sweep bridges MCP-created reminders
into APScheduler.

The MCP `nexus-reminders-tasks.create_reminder` tool writes the
Reminder row to the DB and stops — it has no access to the bot's
running APScheduler (the MCP server runs as a separate subprocess).
Without a reconciliation sweep, the row sits at status='active' but
the apscheduler_jobs table has no entry, so the reminder NEVER fires
until the next bot restart (when boot_recovery_sweep picks it up).

These tests pin the contract that reconcile_with_scheduler:
  - Queues active+future reminders that aren't yet in APScheduler
  - Skips already-fired reminders (last_fired_at NOT NULL)
  - Skips past-due reminders (handled by boot_recovery_sweep instead)
  - Re-queueing an already-queued reminder is idempotent
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from models import Base, Reminder, User
from repositories.reminders_repository import RemindersRepository
from services.reminder_service import ReminderService


def _make_reminder(*, id: str = 'rem-1', body: str = 'body',
                   next_fire_at: datetime, last_fired_at=None,
                   user_id: str = 'u1') -> MagicMock:
    r = MagicMock()
    r.id = id
    r.body = body
    r.user_id = user_id
    r.next_fire_at = next_fire_at
    r.last_fired_at = last_fired_at
    r.recurrence = None
    return r


def _make_service(*, reminders: list, scheduler) -> ReminderService:
    repo = MagicMock()
    repo.list_all_active.return_value = reminders
    parser = MagicMock()
    conv = MagicMock()
    svc = ReminderService(
        reminders_repository=repo,
        parser=parser,
        conversation_service=conv,
        app_timezone='America/New_York',
        scheduler=scheduler,
    )
    return svc


def test_reconcile_queues_active_future_reminder():
    future = datetime.now(timezone.utc) + timedelta(minutes=5)
    rem = _make_reminder(next_fire_at=future)
    sched = MagicMock()
    svc = _make_service(reminders=[rem], scheduler=sched)

    queued = svc.reconcile_with_scheduler()

    assert queued == 1
    sched.schedule_reminder.assert_called_once_with(rem.id, future)


def test_reconcile_skips_already_fired_reminder():
    """last_fired_at NOT NULL → reminder already delivered."""
    future = datetime.now(timezone.utc) + timedelta(minutes=5)
    past_fire = datetime.now(timezone.utc) - timedelta(minutes=10)
    rem = _make_reminder(next_fire_at=future, last_fired_at=past_fire)
    sched = MagicMock()
    svc = _make_service(reminders=[rem], scheduler=sched)

    queued = svc.reconcile_with_scheduler()

    assert queued == 0
    sched.schedule_reminder.assert_not_called()


def test_reconcile_skips_past_due_reminder():
    past = datetime.now(timezone.utc) - timedelta(minutes=5)
    rem = _make_reminder(next_fire_at=past)
    sched = MagicMock()
    svc = _make_service(reminders=[rem], scheduler=sched)

    queued = svc.reconcile_with_scheduler()

    assert queued == 0
    sched.schedule_reminder.assert_not_called()


def test_reconcile_handles_mixed_set_correctly():
    now = datetime.now(timezone.utc)
    a = _make_reminder(id='a', next_fire_at=now + timedelta(minutes=5))
    b = _make_reminder(id='b', next_fire_at=now + timedelta(minutes=10),
                       last_fired_at=now - timedelta(seconds=1))
    c = _make_reminder(id='c', next_fire_at=now - timedelta(minutes=2))
    d = _make_reminder(id='d', next_fire_at=now + timedelta(hours=1))

    sched = MagicMock()
    svc = _make_service(reminders=[a, b, c, d], scheduler=sched)

    queued = svc.reconcile_with_scheduler()

    assert queued == 2
    called_ids = [call.args[0] for call in sched.schedule_reminder.call_args_list]
    assert set(called_ids) == {'a', 'd'}


def test_reconcile_is_idempotent():
    future = datetime.now(timezone.utc) + timedelta(minutes=5)
    rem = _make_reminder(next_fire_at=future)
    sched = MagicMock()
    svc = _make_service(reminders=[rem], scheduler=sched)

    first = svc.reconcile_with_scheduler()
    second = svc.reconcile_with_scheduler()

    assert first == 1
    assert second == 1
    assert sched.schedule_reminder.call_count == 2


def test_reconcile_no_scheduler_short_circuits():
    future = datetime.now(timezone.utc) + timedelta(minutes=5)
    rem = _make_reminder(next_fire_at=future)
    svc = _make_service(reminders=[rem], scheduler=None)

    assert svc.reconcile_with_scheduler() == 0


def test_reconcile_handles_naive_next_fire_at():
    """DB persists next_fire_at as naive UTC. Must coerce before compare."""
    naive_future = datetime.utcnow() + timedelta(minutes=5)
    rem = _make_reminder(next_fire_at=naive_future)
    sched = MagicMock()
    svc = _make_service(reminders=[rem], scheduler=sched)

    queued = svc.reconcile_with_scheduler()

    assert queued == 1


def test_schedule_reminder_sync_registers_interval_job():
    from scheduler import NexusScheduler
    sched = NexusScheduler(database_url='sqlite:///:memory:',
                            timezone='America/New_York')
    fake = MagicMock()
    sched.scheduler = fake

    sched.schedule_reminder_sync(interval_sec=30)

    fake.add_job.assert_called_once()
    kwargs = fake.add_job.call_args.kwargs
    assert kwargs['id'] == 'reminder-sync'
    assert kwargs['seconds'] == 30
    assert kwargs['replace_existing'] is True

