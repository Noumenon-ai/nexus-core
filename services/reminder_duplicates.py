from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import re
from typing import Any
from zoneinfo import ZoneInfo

from utils.dates import format_local_datetime


_BODY_STOPWORDS = {
    'a',
    'an',
    'please',
    'just',
    'that',
    'the',
    'my',
    'your',
}
_GENERIC_FOLLOW_UP_RE = re.compile(r'^follow up(?:\s+with\s+.+)?$', re.IGNORECASE)


@dataclass(slots=True)
class ReminderDuplicateCluster:
    reminders: list[Any]
    next_fire_at: datetime
    app_timezone: str

    @property
    def duplicate_count(self) -> int:
        return max(0, len(self.reminders) - 1)

    @property
    def reminder_ids(self) -> list[str]:
        return [str(_item_value(item, 'id') or '').strip() for item in self.reminders if str(_item_value(item, 'id') or '').strip()]

    @property
    def display_body(self) -> str:
        best = max(
            self.reminders,
            key=lambda item: (
                _body_specificity_score(_item_value(item, 'body')),
                _item_timestamp(item, 'updated_at') or self.next_fire_at,
                _item_timestamp(item, 'created_at') or self.next_fire_at,
                str(_item_value(item, 'id') or ''),
            ),
        )
        return str(_item_value(best, 'body') or '').strip()

    @property
    def display_bodies(self) -> list[str]:
        return [str(_item_value(item, 'body') or '').strip() for item in self.reminders]

    @property
    def scheduled_label(self) -> str:
        return format_local_datetime(self.next_fire_at, self.app_timezone)

    @property
    def newest_reminder(self) -> Any:
        return max(
            self.reminders,
            key=lambda item: (
                _item_timestamp(item, 'updated_at') or self.next_fire_at,
                _item_timestamp(item, 'created_at') or self.next_fire_at,
                str(_item_value(item, 'id') or ''),
            ),
        )

    @property
    def newest_reminder_id(self) -> str:
        return str(_item_value(self.newest_reminder, 'id') or '').strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            'reminder_ids': self.reminder_ids,
            'scheduled_label': self.scheduled_label,
            'display_body': self.display_body,
            'display_bodies': self.display_bodies,
            'newest_reminder_id': self.newest_reminder_id,
            'duplicate_count': self.duplicate_count,
        }


def normalize_reminder_body(body: str) -> str:
    lowered = ' '.join((body or '').strip().lower().split())
    lowered = re.sub(r'[^a-z0-9 ]+', ' ', lowered)
    lowered = ' '.join(
        token for token in lowered.split()
        if token not in _BODY_STOPWORDS
    )
    return lowered.strip()


def reminders_are_similar(left: Any, right: Any, *, app_timezone: str) -> bool:
    left_time = _item_timestamp(left, 'next_fire_at')
    right_time = _item_timestamp(right, 'next_fire_at')
    if left_time is None or right_time is None:
        return False
    if _local_hour_bucket(left_time, app_timezone) != _local_hour_bucket(right_time, app_timezone):
        return False

    left_body = normalize_reminder_body(str(_item_value(left, 'body') or ''))
    right_body = normalize_reminder_body(str(_item_value(right, 'body') or ''))
    if left_body == right_body:
        return True
    if not left_body or not right_body:
        return False

    if _GENERIC_FOLLOW_UP_RE.fullmatch(left_body) and right_body.startswith('follow up'):
        return True
    if _GENERIC_FOLLOW_UP_RE.fullmatch(right_body) and left_body.startswith('follow up'):
        return True

    left_recipient = _extract_recipient_hint(left_body)
    right_recipient = _extract_recipient_hint(right_body)
    if left_recipient and right_recipient and left_recipient == right_recipient:
        return True
    if left_body.startswith('follow up') and right_body.startswith('follow up'):
        return True
    if left_body in right_body or right_body in left_body:
        shorter = min(len(left_body), len(right_body))
        longer = max(len(left_body), len(right_body))
        if shorter >= max(6, longer // 2):
            return True
    return SequenceMatcher(None, left_body, right_body).ratio() >= 0.84


def cluster_duplicate_reminders(
    reminders: list[Any],
    *,
    app_timezone: str,
    minimum_size: int = 2,
) -> list[ReminderDuplicateCluster]:
    ordered = sorted(
        reminders,
        key=lambda item: (
            _item_timestamp(item, 'next_fire_at') or datetime.max.replace(tzinfo=timezone.utc),
            str(_item_value(item, 'id') or ''),
        ),
    )
    groups: list[list[Any]] = []
    for reminder in ordered:
        matched: list[Any] | None = None
        for group in groups:
            if reminders_are_similar(group[0], reminder, app_timezone=app_timezone):
                matched = group
                break
        if matched is None:
            groups.append([reminder])
        else:
            matched.append(reminder)
    clusters = [
        ReminderDuplicateCluster(
            reminders=group,
            next_fire_at=_item_timestamp(group[0], 'next_fire_at') or datetime.now(timezone.utc),
            app_timezone=app_timezone,
        )
        for group in groups
        if len(group) >= minimum_size
    ]
    clusters.sort(key=lambda item: item.next_fire_at)
    return clusters


def compress_duplicate_reminder_lines(
    reminders: list[Any],
    *,
    app_timezone: str,
    line_formatter,
    limit: int | None = None,
) -> tuple[list[str], int]:
    clusters = cluster_duplicate_reminders(reminders, app_timezone=app_timezone)
    clustered_ids = {
        reminder_id
        for cluster in clusters
        for reminder_id in cluster.reminder_ids
    }
    lines: list[str] = []
    duplicate_count = sum(cluster.duplicate_count for cluster in clusters)
    for cluster in clusters:
        lines.append(line_formatter(cluster))
    for reminder in reminders:
        reminder_id = str(_item_value(reminder, 'id') or '').strip()
        if reminder_id and reminder_id in clustered_ids:
            continue
        lines.append(line_formatter(reminder))
    if limit is not None:
        lines = lines[:limit]
    return lines, duplicate_count


def render_duplicate_audit_text(
    clusters: list[ReminderDuplicateCluster],
    *,
    app_timezone: str,
) -> str:
    if not clusters:
        return 'No duplicate reminders found.'
    lines = [f'I found {len(clusters)} duplicate reminder cluster{"s" if len(clusters) != 1 else ""}:']
    for cluster in clusters:
        lines.append(f'{cluster.scheduled_label}:')
        for index, body in enumerate(cluster.display_bodies, start=1):
            lines.append(f'{index}. {body}')
    lines.append('Want me to keep the newest and cancel the others?')
    return '\n'.join(lines)


def _item_value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _item_timestamp(item: Any, key: str) -> datetime | None:
    value = _item_value(item, key)
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    candidate = str(value).strip()
    if not candidate:
        return None
    if candidate.endswith('Z'):
        candidate = candidate[:-1] + '+00:00'
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _local_hour_bucket(value: datetime, app_timezone: str) -> tuple[int, int, int, int]:
    localized = value.astimezone(ZoneInfo(app_timezone))
    return (
        localized.year,
        localized.month,
        localized.day,
        localized.hour,
    )


def _extract_recipient_hint(normalized_body: str) -> str:
    patterns = (
        r'^follow up with\s+(.+?)(?:\s+about|\s+at|\s+on|\s+tomorrow|\s+today|\s+tonight|$)',
        r'^send (?:a )?message to\s+(.+?)(?:\s+about|\s+at|\s+on|\s+tomorrow|\s+today|\s+tonight|$)',
        r'^tell\s+(.+?)(?:\s+about|\s+at|\s+on|\s+tomorrow|\s+today|\s+tonight|$)',
    )
    for pattern in patterns:
        match = re.match(pattern, normalized_body, flags=re.IGNORECASE)
        if match is not None:
            return ' '.join(match.group(1).split()).strip().casefold()
    return ''


def _body_specificity_score(body: Any) -> tuple[int, int]:
    normalized = normalize_reminder_body(str(body or ''))
    has_recipient = 1 if _extract_recipient_hint(normalized) else 0
    return (has_recipient, len(normalized))
