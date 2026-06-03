"""nexus-calendar MCP server — Phase 2.5a Server 4.

Full Google Calendar access via stored OAuth credentials.

Configuration via environment variables:
  NEXUS_GOOGLE_TOKEN_DIR    where {user_id}.json credential files live
                             (default: /opt/nexus-core/.data/google_tokens)
  NEXUS_MCP_DEFAULT_USER_ID user UUID for tools called without explicit user_id

Every tool returns a dict; failures return {'ok': False, 'reason': str, 'detail': str}.
"""
from __future__ import annotations

import os
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("nexus-calendar")

# SHARED Google OAuth scope set across every MCP server that reads
# .data/google_tokens/{user_id}.json.  Each server passes the SAME superset to
# Credentials.from_authorized_user_file() so that when refresh() rewrites the
# token file, it never narrows the scope set (which would break other servers
# sharing the same token).  Keep this list in sync with mcp_servers.nexus_contacts
# and scripts/google_oauth_flow.py.
_GOOGLE_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/contacts",
    "https://www.googleapis.com/auth/tasks",
]
_DEFAULT_TOKEN_DIR = Path("/opt/nexus-core/.data/google_tokens")


def _token_dir() -> Path:
    raw = os.environ.get("NEXUS_GOOGLE_TOKEN_DIR")
    if raw:
        return Path(raw)
    return _DEFAULT_TOKEN_DIR


def _resolve_user_id(user_id: str | None) -> str | None:
    if user_id and user_id.strip():
        return user_id.strip()
    env_uid = os.environ.get("NEXUS_MCP_DEFAULT_USER_ID")
    return env_uid.strip() if env_uid else None


def _parse_iso(value: str, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 string")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not valid ISO-8601: {exc}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _to_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_calendar(user_id: str | None) -> Any | dict:
    """Build a Google Calendar service for user. Returns service or error dict."""
    uid = _resolve_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id", "detail": "pass user_id or set NEXUS_MCP_DEFAULT_USER_ID"}
    token_path = _token_dir() / f"{uid}.json"
    if not token_path.exists():
        return {"ok": False, "reason": "no_credentials", "detail": f"token file missing: {token_path}"}
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError as exc:
        return {"ok": False, "reason": "google_libs_not_installed", "detail": str(exc)}
    try:
        # scopes=None → use whatever scopes are persisted in the file.  Passing a
        # subset here would cause refresh()+to_json() to write back the subset,
        # narrowing the shared token for other MCP servers that need wider
        # scopes (this exact bug bit nexus-contacts at first live verify of
        # Phase 2.5c).
        creds = Credentials.from_authorized_user_file(str(token_path))
    except Exception as exc:
        return {"ok": False, "reason": "credentials_load_failed", "detail": str(exc)[:300]}
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")
        except Exception as exc:
            return {"ok": False, "reason": "token_refresh_failed", "detail": str(exc)[:300]}
    try:
        return build("calendar", "v3", credentials=creds, cache_discovery=False)
    except Exception as exc:
        return {"ok": False, "reason": "client_build_failed", "detail": str(exc)[:300]}


def _serialize_event(ev: dict) -> dict:
    start = ev.get("start") or {}
    end = ev.get("end") or {}
    return {
        "id": ev.get("id"),
        "summary": ev.get("summary"),
        "description": ev.get("description"),
        "location": ev.get("location"),
        "start": start.get("dateTime") or start.get("date"),
        "end": end.get("dateTime") or end.get("date"),
        "all_day": "date" in start,
        "attendees": [a.get("email") for a in (ev.get("attendees") or []) if a.get("email")],
        "status": ev.get("status"),
        "recurrence": ev.get("recurrence"),
        "html_link": ev.get("htmlLink"),
        "organizer_email": (ev.get("organizer") or {}).get("email"),
    }


# ---------- read tools ----------

@mcp.tool()
def list_events(date_range: list[str], calendar_id: str = "primary", max_results: int = 100, user_id: str | None = None) -> dict:
    """Events in [start, end] (ISO-8601). Default calendar = primary."""
    if not isinstance(date_range, list) or len(date_range) != 2:
        return {"ok": False, "reason": "invalid_date_range", "detail": "must be [start, end]"}
    try:
        start = _parse_iso(date_range[0], "date_range.start")
        end = _parse_iso(date_range[1], "date_range.end")
    except ValueError as exc:
        return {"ok": False, "reason": "invalid_date_range", "detail": str(exc)}
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_max_results", "detail": ""}
    max_results = max(1, min(max_results, 500))
    svc = _get_calendar(user_id)
    if isinstance(svc, dict):
        return svc
    try:
        resp = svc.events().list(
            calendarId=calendar_id,
            timeMin=_to_rfc3339(start),
            timeMax=_to_rfc3339(end),
            singleEvents=True,
            orderBy="startTime",
            maxResults=max_results,
        ).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    items = resp.get("items") or []
    return {"ok": True, "count": len(items), "events": [_serialize_event(e) for e in items]}


@mcp.tool()
def today_schedule(calendar_id: str = "primary", user_id: str | None = None) -> dict:
    """All events for today (UTC day)."""
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return list_events(date_range=[start.isoformat(), end.isoformat()], calendar_id=calendar_id, max_results=200, user_id=user_id)


@mcp.tool()
def week_schedule(calendar_id: str = "primary", user_id: str | None = None) -> dict:
    """All events for the next 7 days."""
    now = datetime.now(timezone.utc)
    return list_events(date_range=[now.isoformat(), (now + timedelta(days=7)).isoformat()],
                       calendar_id=calendar_id, max_results=500, user_id=user_id)


@mcp.tool()
def whats_next(within_hours: int = 24, calendar_id: str = "primary", user_id: str | None = None) -> dict:
    """Return the next upcoming event within N hours."""
    try:
        within_hours = int(within_hours)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_within_hours", "detail": ""}
    if not 1 <= within_hours <= 168:
        return {"ok": False, "reason": "within_hours_out_of_range", "detail": "1..168"}
    now = datetime.now(timezone.utc)
    listing = list_events(
        date_range=[now.isoformat(), (now + timedelta(hours=within_hours)).isoformat()],
        calendar_id=calendar_id, max_results=10, user_id=user_id,
    )
    if not listing["ok"]:
        return listing
    if not listing["events"]:
        return {"ok": True, "next_event": None, "within_hours": within_hours}
    return {"ok": True, "next_event": listing["events"][0], "within_hours": within_hours}


@mcp.tool()
def check_conflicts(proposed_start: str, proposed_end: str, calendar_id: str = "primary", user_id: str | None = None) -> dict:
    """Return events overlapping the proposed window."""
    try:
        start = _parse_iso(proposed_start, "proposed_start")
        end = _parse_iso(proposed_end, "proposed_end")
    except ValueError as exc:
        return {"ok": False, "reason": "invalid_range", "detail": str(exc)}
    if end <= start:
        return {"ok": False, "reason": "end_before_start", "detail": ""}
    listing = list_events(
        date_range=[start.isoformat(), end.isoformat()],
        calendar_id=calendar_id, max_results=50, user_id=user_id,
    )
    if not listing["ok"]:
        return listing
    return {
        "ok": True,
        "proposed_start": start.isoformat(),
        "proposed_end": end.isoformat(),
        "conflict_count": listing["count"],
        "conflicts": listing["events"],
    }


@mcp.tool()
def find_free_time(duration_minutes: int, between_dates: list[str], working_hours: list[int] | None = None,
                   calendar_id: str = "primary", user_id: str | None = None) -> dict:
    """Find slots of `duration_minutes` between [start, end]. working_hours = [start_hour, end_hour], default [9,18]."""
    try:
        duration_minutes = int(duration_minutes)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_duration", "detail": ""}
    if duration_minutes < 5 or duration_minutes > 480:
        return {"ok": False, "reason": "duration_out_of_range", "detail": "5..480 minutes"}
    if not isinstance(between_dates, list) or len(between_dates) != 2:
        return {"ok": False, "reason": "invalid_between_dates", "detail": "must be [start, end]"}
    try:
        win_start = _parse_iso(between_dates[0], "between_dates.start")
        win_end = _parse_iso(between_dates[1], "between_dates.end")
    except ValueError as exc:
        return {"ok": False, "reason": "invalid_between_dates", "detail": str(exc)}
    wh = working_hours or [9, 18]
    if not isinstance(wh, list) or len(wh) != 2 or not all(isinstance(x, int) for x in wh) or not (0 <= wh[0] < wh[1] <= 24):
        return {"ok": False, "reason": "invalid_working_hours", "detail": "must be [start_hour, end_hour]"}

    listing = list_events(
        date_range=[win_start.isoformat(), win_end.isoformat()],
        calendar_id=calendar_id, max_results=500, user_id=user_id,
    )
    if not listing["ok"]:
        return listing
    busy_intervals: list[tuple[datetime, datetime]] = []
    for ev in listing["events"]:
        if ev.get("all_day"):
            continue
        try:
            s = _parse_iso(ev["start"], "event.start")
            e = _parse_iso(ev["end"], "event.end")
        except (KeyError, ValueError):
            continue
        busy_intervals.append((s, e))
    busy_intervals.sort()

    free_slots: list[dict] = []
    day = win_start.date()
    last_day = win_end.date()
    while day <= last_day:
        day_start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).replace(hour=wh[0])
        day_end = day_start.replace(hour=wh[1])
        cursor = max(day_start, win_start)
        end_of_day = min(day_end, win_end)
        for bs, be in busy_intervals:
            if be <= cursor or bs >= end_of_day:
                continue
            if bs > cursor:
                gap = (bs - cursor).total_seconds() / 60
                if gap >= duration_minutes:
                    free_slots.append({"start": cursor.isoformat(), "end": bs.isoformat(), "minutes": int(gap)})
            cursor = max(cursor, be)
        if end_of_day > cursor:
            gap = (end_of_day - cursor).total_seconds() / 60
            if gap >= duration_minutes:
                free_slots.append({"start": cursor.isoformat(), "end": end_of_day.isoformat(), "minutes": int(gap)})
        day = day + timedelta(days=1)
    return {"ok": True, "duration_minutes": duration_minutes, "working_hours": wh, "free_slots": free_slots, "free_slot_count": len(free_slots)}


@mcp.tool()
def find_recurring_meetings(calendar_id: str = "primary", days_ahead: int = 30, user_id: str | None = None) -> dict:
    """List recurring events in the next N days."""
    try:
        days_ahead = int(days_ahead)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_days_ahead", "detail": ""}
    if not 1 <= days_ahead <= 365:
        return {"ok": False, "reason": "days_ahead_out_of_range", "detail": "1..365"}
    svc = _get_calendar(user_id)
    if isinstance(svc, dict):
        return svc
    now = datetime.now(timezone.utc)
    try:
        resp = svc.events().list(
            calendarId=calendar_id,
            timeMin=_to_rfc3339(now),
            timeMax=_to_rfc3339(now + timedelta(days=days_ahead)),
            singleEvents=False,
            maxResults=500,
        ).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    items = resp.get("items") or []
    recurring = [_serialize_event(e) for e in items if e.get("recurrence")]
    return {"ok": True, "count": len(recurring), "events": recurring}


@mcp.tool()
def upcoming_birthdays(days_ahead: int = 30, user_id: str | None = None) -> dict:
    """Upcoming birthdays from Google Contacts birthday field. Requires People API integration."""
    return {
        "ok": False,
        "reason": "not_configured",
        "detail": "upcoming_birthdays requires Google People API integration (Phase 2.5b — Server 5 contacts)",
        "echo": {"days_ahead": days_ahead, "user_id": user_id},
    }


# ---------- write tools ----------

@mcp.tool(meta={"approval_required": True, "approval_template": "Create calendar event?"})
def create_event(title: str, start: str, end: str, attendees: list[str] | None = None,
                 location: str | None = None, description: str | None = None,
                 recurring: str | None = None, calendar_id: str = "primary",
                 user_id: str | None = None) -> dict:
    """Create a calendar event. start/end ISO-8601. recurring = RRULE string (e.g. 'FREQ=WEEKLY;COUNT=10')."""
    if not isinstance(title, str) or not title.strip():
        return {"ok": False, "reason": "empty_title", "detail": ""}
    try:
        start_dt = _parse_iso(start, "start")
        end_dt = _parse_iso(end, "end")
    except ValueError as exc:
        return {"ok": False, "reason": "invalid_datetime", "detail": str(exc)}
    if end_dt <= start_dt:
        return {"ok": False, "reason": "end_before_start", "detail": ""}
    svc = _get_calendar(user_id)
    if isinstance(svc, dict):
        return svc
    body: dict[str, Any] = {
        "summary": title.strip(),
        "start": {"dateTime": _to_rfc3339(start_dt)},
        "end": {"dateTime": _to_rfc3339(end_dt)},
    }
    if location:
        body["location"] = location
    if description:
        body["description"] = description
    if attendees:
        if not isinstance(attendees, list) or not all(isinstance(a, str) and "@" in a for a in attendees):
            return {"ok": False, "reason": "invalid_attendees", "detail": "must be list of email strings"}
        body["attendees"] = [{"email": a} for a in attendees]
    if recurring:
        body["recurrence"] = [f"RRULE:{recurring}"]
    try:
        ev = svc.events().insert(calendarId=calendar_id, body=body).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "event": _serialize_event(ev)}


@mcp.tool(meta={"approval_required": True, "approval_template": "Update calendar event?"})
def update_event(event_id: str, changes: dict, calendar_id: str = "primary", user_id: str | None = None) -> dict:
    """Modify event fields. Allowed: title, start, end, location, description, attendees."""
    if not isinstance(event_id, str) or not event_id.strip():
        return {"ok": False, "reason": "empty_event_id", "detail": ""}
    if not isinstance(changes, dict) or not changes:
        return {"ok": False, "reason": "empty_changes", "detail": ""}
    allowed = {"title", "start", "end", "location", "description", "attendees"}
    extra = set(changes) - allowed
    if extra:
        return {"ok": False, "reason": "unknown_change_keys", "detail": f"unknown: {sorted(extra)}"}
    svc = _get_calendar(user_id)
    if isinstance(svc, dict):
        return svc
    try:
        existing = svc.events().get(calendarId=calendar_id, eventId=event_id).execute()
    except Exception as exc:
        return {"ok": False, "reason": "fetch_failed", "detail": str(exc)[:300]}
    if "title" in changes:
        existing["summary"] = changes["title"]
    if "start" in changes:
        try:
            existing["start"] = {"dateTime": _to_rfc3339(_parse_iso(changes["start"], "start"))}
        except ValueError as exc:
            return {"ok": False, "reason": "invalid_start", "detail": str(exc)}
    if "end" in changes:
        try:
            existing["end"] = {"dateTime": _to_rfc3339(_parse_iso(changes["end"], "end"))}
        except ValueError as exc:
            return {"ok": False, "reason": "invalid_end", "detail": str(exc)}
    if "location" in changes:
        existing["location"] = changes["location"]
    if "description" in changes:
        existing["description"] = changes["description"]
    if "attendees" in changes:
        if not isinstance(changes["attendees"], list):
            return {"ok": False, "reason": "invalid_attendees", "detail": "must be list of emails"}
        existing["attendees"] = [{"email": a} for a in changes["attendees"] if isinstance(a, str)]
    try:
        updated = svc.events().update(calendarId=calendar_id, eventId=event_id, body=existing).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "event": _serialize_event(updated)}


@mcp.tool(meta={"approval_required": True, "approval_template": "Cancel calendar event?"})
def cancel_event(event_id: str, notify_attendees: bool = True, calendar_id: str = "primary", user_id: str | None = None) -> dict:
    """Delete a calendar event. notify_attendees controls whether cancellation emails are sent."""
    if not isinstance(event_id, str) or not event_id.strip():
        return {"ok": False, "reason": "empty_event_id", "detail": ""}
    svc = _get_calendar(user_id)
    if isinstance(svc, dict):
        return svc
    send = "all" if notify_attendees else "none"
    try:
        svc.events().delete(calendarId=calendar_id, eventId=event_id, sendUpdates=send).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "event_id": event_id, "notify_attendees": notify_attendees}


def _respond_to_event(event_id: str, response_status: str, calendar_id: str, user_id: str | None, note: str | None = None) -> dict:
    svc = _get_calendar(user_id)
    if isinstance(svc, dict):
        return svc
    try:
        existing = svc.events().get(calendarId=calendar_id, eventId=event_id).execute()
    except Exception as exc:
        return {"ok": False, "reason": "fetch_failed", "detail": str(exc)[:300]}
    # Find self attendee entry (best-effort).
    attendees = existing.get("attendees") or []
    self_email = None
    for a in attendees:
        if a.get("self"):
            self_email = a.get("email")
            break
    if self_email is None:
        # Try to discover via calendar settings (primary calendar id == user's email).
        try:
            cal = svc.calendars().get(calendarId="primary").execute()
            self_email = cal.get("id")
        except Exception:
            self_email = None
    found = False
    for a in attendees:
        if a.get("email") == self_email:
            a["responseStatus"] = response_status
            if note:
                a["comment"] = note
            found = True
            break
    if not found and self_email:
        attendees.append({"email": self_email, "responseStatus": response_status, **({"comment": note} if note else {})})
    existing["attendees"] = attendees
    try:
        updated = svc.events().update(calendarId=calendar_id, eventId=event_id, body=existing, sendUpdates="all").execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "event_id": event_id, "response": response_status, "event": _serialize_event(updated)}


@mcp.tool(meta={"approval_required": True, "approval_template": "Accept calendar invite?"})
def accept_invite(event_id: str, calendar_id: str = "primary", user_id: str | None = None) -> dict:
    """Respond 'accepted' to an event invitation."""
    if not isinstance(event_id, str) or not event_id.strip():
        return {"ok": False, "reason": "empty_event_id", "detail": ""}
    return _respond_to_event(event_id, "accepted", calendar_id, user_id)


@mcp.tool(meta={"approval_required": True, "approval_template": "Decline calendar invite?"})
def decline_invite(event_id: str, reason: str | None = None, calendar_id: str = "primary", user_id: str | None = None) -> dict:
    """Respond 'declined' with optional note."""
    if not isinstance(event_id, str) or not event_id.strip():
        return {"ok": False, "reason": "empty_event_id", "detail": ""}
    return _respond_to_event(event_id, "declined", calendar_id, user_id, note=reason)


@mcp.tool(meta={"approval_required": True, "approval_template": "Mark invite tentative?"})
def tentative_invite(event_id: str, calendar_id: str = "primary", user_id: str | None = None) -> dict:
    """Respond 'tentative' to an event invitation."""
    if not isinstance(event_id, str) or not event_id.strip():
        return {"ok": False, "reason": "empty_event_id", "detail": ""}
    return _respond_to_event(event_id, "tentative", calendar_id, user_id)


@mcp.tool(meta={"approval_required": True, "approval_template": "Block focus time on calendar?"})
def block_focus_time(start: str, end: str, label: str = "Focus", calendar_id: str = "primary", user_id: str | None = None) -> dict:
    """Create a calendar event titled `label` (default 'Focus') to protect the window."""
    return create_event(title=label, start=start, end=end, calendar_id=calendar_id, user_id=user_id)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
