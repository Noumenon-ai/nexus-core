from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class CalendarEvent:
    id: str
    summary: str
    start: datetime
    end: datetime
    description: str | None = None
    location: str | None = None
    html_link: str | None = None

    def __post_init__(self) -> None:
        for field_name in ('start', 'end'):
            value = getattr(self, field_name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f'{field_name} must be timezone-aware')
        if self.end < self.start:
            raise ValueError('end must be after start')

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'summary': self.summary,
            'start': self.start.isoformat(),
            'end': self.end.isoformat(),
            'description': self.description,
            'location': self.location,
            'html_link': self.html_link,
        }


@dataclass(slots=True)
class BusyBlock:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        for field_name in ('start', 'end'):
            value = getattr(self, field_name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f'{field_name} must be timezone-aware')
        if self.end < self.start:
            raise ValueError('end must be after start')

    def to_dict(self) -> dict[str, Any]:
        return {
            'start': self.start.isoformat(),
            'end': self.end.isoformat(),
        }


@dataclass(slots=True)
class CalendarEventCreate:
    summary: str
    start: datetime
    end: datetime
    description: str | None = None
    location: str | None = None

    def __post_init__(self) -> None:
        if not self.summary or not self.summary.strip():
            raise ValueError('summary must be non-empty')
        for field_name in ('start', 'end'):
            value = getattr(self, field_name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f'{field_name} must be timezone-aware')
        if self.end < self.start:
            raise ValueError('end must be after start')

    def to_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            'summary': self.summary,
            'start': {'dateTime': self.start.isoformat()},
            'end': {'dateTime': self.end.isoformat()},
        }
        if self.description is not None:
            body['description'] = self.description
        if self.location is not None:
            body['location'] = self.location
        return body


@dataclass(slots=True)
class CalendarEventUpdate:
    summary: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    description: str | None = None
    location: str | None = None

    def __post_init__(self) -> None:
        for field_name in ('start', 'end'):
            value = getattr(self, field_name)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f'{field_name} must be timezone-aware')
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError('end must be after start')
        if all(getattr(self, f) is None for f in ('summary', 'start', 'end', 'description', 'location')):
            raise ValueError('at least one field must be set for update')

    def to_patch_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if self.summary is not None:
            body['summary'] = self.summary
        if self.start is not None:
            body['start'] = {'dateTime': self.start.isoformat()}
        if self.end is not None:
            body['end'] = {'dateTime': self.end.isoformat()}
        if self.description is not None:
            body['description'] = self.description
        if self.location is not None:
            body['location'] = self.location
        return body


@dataclass(slots=True)
class GoogleTask:
    id: str
    title: str
    status: str  # 'needsAction' | 'completed'
    due: datetime | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'status': self.status,
            'due': self.due.isoformat() if self.due is not None else None,
            'notes': self.notes,
        }


@dataclass(slots=True)
class GoogleContact:
    resource_name: str  # 'people/c12345' — Google's contact identifier
    display_name: str
    emails: list[str]
    phones: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            'resource_name': self.resource_name,
            'display_name': self.display_name,
            'emails': self.emails,
            'phones': self.phones,
        }
