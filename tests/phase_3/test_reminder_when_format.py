from datetime import datetime, timedelta, timezone

from utils.dates import format_reminder_when


TZ = 'America/New_York'


def _utc(year, month, day, hour, minute):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def test_same_day_reminder_shows_clock_time_and_relative_minutes():
    # 8:42 PM ET fires 45 minutes after a 7:57 PM ET reference.
    reference = _utc(2026, 6, 3, 23, 57)  # 7:57 PM ET
    fire = reference + timedelta(minutes=45)
    text = format_reminder_when(fire, TZ, reference=reference)
    assert text == 'at 8:42 PM (in 45 min)'


def test_same_day_reminder_over_an_hour_uses_hours():
    reference = _utc(2026, 6, 3, 23, 57)  # 7:57 PM ET
    fire = reference + timedelta(minutes=62)
    text = format_reminder_when(fire, TZ, reference=reference)
    assert text == 'at 8:59 PM (in 1 hr)'


def test_future_day_reminder_shows_weekday_and_date_not_relative():
    reference = _utc(2026, 6, 2, 16, 0)  # Jun 2, 12:00 PM ET
    fire = _utc(2026, 6, 4, 15, 30)      # Thu Jun 4, 11:30 AM ET
    text = format_reminder_when(fire, TZ, reference=reference)
    assert text == 'Thu Jun 4, 11:30 AM'
    assert 'in ' not in text  # no relative phrase across days


def test_no_full_weekday_year_for_same_day():
    reference = _utc(2026, 6, 3, 23, 57)
    fire = reference + timedelta(minutes=30)
    text = format_reminder_when(fire, TZ, reference=reference)
    assert '2026' not in text
    assert 'Tuesday' not in text and 'Wednesday' not in text
