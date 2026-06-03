from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from babel.dates import format_datetime



def utc_now() -> datetime:
    return datetime.now(timezone.utc)



def app_now(app_timezone: str) -> datetime:
    return utc_now().astimezone(ZoneInfo(app_timezone))



def to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError('Datetime must be timezone aware')
    return value.astimezone(timezone.utc)



def from_utc(value: datetime, app_timezone: str) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(ZoneInfo(app_timezone))



def format_local_datetime(value: datetime, app_timezone: str, locale: str = 'en_US') -> str:
    return format_datetime(from_utc(value, app_timezone), 'EEEE, MMMM d, yyyy h:mm a', locale=locale)



def _relative_phrase(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    if total_seconds < 60:
        return 'in under a minute'
    minutes = round(total_seconds / 60)
    if minutes < 60:
        return f'in {minutes} min'
    hours = round(minutes / 60)
    return f'in {hours} hr'



def format_reminder_when(
    value: datetime,
    app_timezone: str,
    *,
    reference: datetime | None = None,
    locale: str = 'en_US',
) -> str:
    """Compact, human reminder timestamp.

    Same local day  -> 'at 8:42 PM (in 45 min)' (relative only when future).
    A different day  -> 'Thu Jun 4, 11:30 AM' (no relative phrase).

    Avoids the verbose 'Tuesday, June 2, 2026 8:42 PM' for reminders that
    fire today, which reads as noise when something is minutes away.
    """
    local_value = from_utc(value, app_timezone)
    reference = reference if reference is not None else utc_now()
    local_reference = from_utc(reference, app_timezone)
    time_text = format_datetime(local_value, 'h:mm a', locale=locale)
    if local_value.date() == local_reference.date():
        delta = local_value - local_reference
        if delta.total_seconds() > 0:
            return f'at {time_text} ({_relative_phrase(delta)})'
        return f'at {time_text}'
    return format_datetime(local_value, 'EEE MMM d, h:mm a', locale=locale)



def within_hours(reference: datetime, hours: int, *, compared_to: datetime | None = None) -> bool:
    compared = compared_to or utc_now()
    delta = abs(reference - compared)
    return delta <= timedelta(hours=hours)



def humanize_timedelta(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    if total_seconds < 60:
        return f'{total_seconds} seconds'
    minutes, _seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f'{minutes} minutes'
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f'{hours} hours'
    days, _hours = divmod(hours, 24)
    return f'{days} days'
