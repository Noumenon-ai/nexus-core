"""H2-048 — fix the scheduler timezone bug surfaced by the post-restart
60-second reminder test (10:14 EDT, fired 4 hours late at 14:14 EDT).

APScheduler is configured with timezone=America/New_York. Naive datetimes
passed to add_job get interpreted in THAT timezone, but the reminder
pipeline produces UTC values that SQLAlchemy strips the tzinfo off
before persistence. The fix re-anchors to UTC at the scheduler
boundary.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from scheduler import NexusScheduler


def test_schedule_reminder_promotes_naive_to_utc():
    """Naive datetime input → UTC-aware run_date passed to add_job."""
    sched = NexusScheduler(database_url='sqlite:///:memory:', timezone='America/New_York')
    fake_scheduler = MagicMock()
    sched.scheduler = fake_scheduler

    naive_value = datetime(2026, 5, 13, 14, 14, 44)
    sched.schedule_reminder('rem-1', naive_value)

    fake_scheduler.add_job.assert_called_once()
    run_date = fake_scheduler.add_job.call_args.kwargs['run_date']
    assert run_date.tzinfo is timezone.utc, (
        f'naive datetime was not promoted: {run_date!r}'
    )
    assert run_date == naive_value.replace(tzinfo=timezone.utc)


def test_schedule_reminder_preserves_aware_value():
    """A tz-aware datetime passes through unchanged — we only promote on
    the naive path so callers that already do the right thing keep their
    semantics."""
    sched = NexusScheduler(database_url='sqlite:///:memory:', timezone='America/New_York')
    fake_scheduler = MagicMock()
    sched.scheduler = fake_scheduler

    aware_value = datetime(2026, 5, 13, 14, 14, 44, tzinfo=timezone.utc)
    sched.schedule_reminder('rem-2', aware_value)

    run_date = fake_scheduler.add_job.call_args.kwargs['run_date']
    assert run_date == aware_value
    assert run_date.tzinfo is timezone.utc


def test_schedule_reminder_passes_id_and_replace_existing():
    """Sanity — the job id pattern and replace_existing flag survive the
    UTC-promotion wrapping."""
    sched = NexusScheduler(database_url='sqlite:///:memory:', timezone='America/New_York')
    fake_scheduler = MagicMock()
    sched.scheduler = fake_scheduler

    sched.schedule_reminder('rem-abc', datetime(2030, 1, 1))
    kwargs = fake_scheduler.add_job.call_args.kwargs
    assert kwargs['id'] == 'reminder-rem-abc'
    assert kwargs['replace_existing'] is True
    assert kwargs['args'] == ['rem-abc']
