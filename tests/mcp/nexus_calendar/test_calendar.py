"""Adversarial tests for nexus-calendar MCP server (Phase 2.5a Server 4).

Calendar tools talk to Google's API in production. Tests mock _get_calendar()
to return a stub service whose method calls record args and return canned payloads.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest


_USER_ID = "calendar-user-9999"


@pytest.fixture(autouse=True)
def default_user(monkeypatch):
    monkeypatch.setenv("NEXUS_MCP_DEFAULT_USER_ID", _USER_ID)
    yield


from mcp_servers.nexus_calendar import (
    accept_invite,
    block_focus_time,
    cancel_event,
    check_conflicts,
    create_event,
    decline_invite,
    find_free_time,
    find_recurring_meetings,
    list_events,
    tentative_invite,
    today_schedule,
    upcoming_birthdays,
    update_event,
    week_schedule,
    whats_next,
)


def _future_iso(hours: int = 1) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _make_stub_service(list_items: list[dict] | None = None, insert_return: dict | None = None,
                      get_return: dict | None = None, update_return: dict | None = None) -> MagicMock:
    svc = MagicMock(name="calendar_service")
    events_obj = MagicMock(name="events")
    svc.events.return_value = events_obj
    if list_items is not None:
        events_obj.list.return_value.execute.return_value = {"items": list_items}
    if insert_return is not None:
        events_obj.insert.return_value.execute.return_value = insert_return
    if get_return is not None:
        events_obj.get.return_value.execute.return_value = get_return
    if update_return is not None:
        events_obj.update.return_value.execute.return_value = update_return
    events_obj.delete.return_value.execute.return_value = {}
    cals = MagicMock(name="calendars")
    svc.calendars.return_value = cals
    cals.get.return_value.execute.return_value = {"id": "self@example.com"}
    return svc


# ---------- credential paths ----------

def test_no_user_id_returns_no_user_id(monkeypatch):
    monkeypatch.delenv("NEXUS_MCP_DEFAULT_USER_ID", raising=False)
    r = list_events(date_range=[_future_iso(), _future_iso(2)])
    assert r["ok"] is False
    assert r["reason"] == "no_user_id"


def test_missing_credentials_returns_no_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("NEXUS_GOOGLE_TOKEN_DIR", str(tmp_path))
    r = list_events(date_range=[_future_iso(), _future_iso(2)])
    assert r["ok"] is False
    assert r["reason"] == "no_credentials"


# ---------- list_events ----------

def test_list_events_invalid_date_range_rejected():
    r = list_events(date_range="not-a-list")
    assert r["ok"] is False
    assert r["reason"] == "invalid_date_range"


def test_list_events_bad_iso_rejected():
    r = list_events(date_range=["bad", "also-bad"])
    assert r["ok"] is False


def test_list_events_happy():
    stub = _make_stub_service(list_items=[
        {"id": "ev1", "summary": "Meeting 1", "start": {"dateTime": "2026-05-12T10:00:00Z"},
         "end": {"dateTime": "2026-05-12T11:00:00Z"}, "attendees": [{"email": "a@b.com"}]},
        {"id": "ev2", "summary": "Meeting 2", "start": {"dateTime": "2026-05-12T14:00:00Z"},
         "end": {"dateTime": "2026-05-12T15:00:00Z"}},
    ])
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = list_events(date_range=[_future_iso(), _future_iso(24)])
    assert r["ok"] is True
    assert r["count"] == 2
    assert r["events"][0]["summary"] == "Meeting 1"
    assert r["events"][0]["attendees"] == ["a@b.com"]


# ---------- today / week / whats_next ----------

def test_today_schedule_delegates():
    stub = _make_stub_service(list_items=[])
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = today_schedule()
    assert r["ok"] is True
    assert r["count"] == 0


def test_week_schedule_delegates():
    stub = _make_stub_service(list_items=[])
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = week_schedule()
    assert r["ok"] is True


def test_whats_next_within_hours_out_of_range_rejected():
    r = whats_next(within_hours=500)
    assert r["ok"] is False


def test_whats_next_returns_first_event():
    stub = _make_stub_service(list_items=[
        {"id": "ev1", "summary": "first", "start": {"dateTime": "2026-05-12T10:00:00Z"},
         "end": {"dateTime": "2026-05-12T11:00:00Z"}},
    ])
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = whats_next(within_hours=12)
    assert r["ok"] is True
    assert r["next_event"]["summary"] == "first"


def test_whats_next_empty_window():
    stub = _make_stub_service(list_items=[])
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = whats_next(within_hours=1)
    assert r["ok"] is True
    assert r["next_event"] is None


# ---------- check_conflicts / find_free_time ----------

def test_check_conflicts_end_before_start_rejected():
    later = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    earlier = datetime.now(timezone.utc).isoformat()
    r = check_conflicts(proposed_start=later, proposed_end=earlier)
    assert r["ok"] is False
    assert r["reason"] == "end_before_start"


def test_check_conflicts_returns_overlaps():
    stub = _make_stub_service(list_items=[
        {"id": "ev1", "summary": "overlap", "start": {"dateTime": "2026-05-12T10:00:00Z"},
         "end": {"dateTime": "2026-05-12T11:00:00Z"}},
    ])
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = check_conflicts(proposed_start="2026-05-12T10:30:00Z", proposed_end="2026-05-12T11:30:00Z")
    assert r["ok"] is True
    assert r["conflict_count"] == 1


def test_find_free_time_invalid_duration_rejected():
    r = find_free_time(duration_minutes=1, between_dates=[_future_iso(), _future_iso(8)])
    assert r["ok"] is False
    assert r["reason"] == "duration_out_of_range"


def test_find_free_time_invalid_working_hours_rejected():
    r = find_free_time(duration_minutes=30, between_dates=[_future_iso(), _future_iso(8)], working_hours=[20, 5])
    assert r["ok"] is False
    assert r["reason"] == "invalid_working_hours"


def test_find_free_time_no_events_returns_full_day_slots():
    stub = _make_stub_service(list_items=[])
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = find_free_time(
            duration_minutes=30,
            between_dates=[today.isoformat(), (today + timedelta(days=1)).isoformat()],
            working_hours=[9, 18],
        )
    assert r["ok"] is True
    assert r["free_slot_count"] >= 1


# ---------- create_event ----------

def test_create_event_empty_title_rejected():
    r = create_event(title="", start=_future_iso(), end=_future_iso(2))
    assert r["ok"] is False
    assert r["reason"] == "empty_title"


def test_create_event_end_before_start_rejected():
    r = create_event(title="t", start=_future_iso(3), end=_future_iso(1))
    assert r["ok"] is False
    assert r["reason"] == "end_before_start"


def test_create_event_invalid_attendees_rejected():
    stub = _make_stub_service(insert_return={"id": "new", "summary": "t"})
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = create_event(title="t", start=_future_iso(), end=_future_iso(2), attendees=["not-an-email"])
    assert r["ok"] is False
    assert r["reason"] == "invalid_attendees"


def test_create_event_happy():
    stub = _make_stub_service(insert_return={
        "id": "new123", "summary": "Demo", "start": {"dateTime": "2026-05-12T10:00:00Z"},
        "end": {"dateTime": "2026-05-12T11:00:00Z"},
    })
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = create_event(title="Demo", start=_future_iso(), end=_future_iso(2), attendees=["a@b.com"])
    assert r["ok"] is True
    assert r["event"]["summary"] == "Demo"
    sent_body = stub.events.return_value.insert.call_args.kwargs["body"]
    assert sent_body["summary"] == "Demo"
    assert sent_body["attendees"] == [{"email": "a@b.com"}]


def test_create_event_with_recurring_rule():
    stub = _make_stub_service(insert_return={"id": "rec", "summary": "weekly"})
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = create_event(title="weekly", start=_future_iso(), end=_future_iso(2), recurring="FREQ=WEEKLY;COUNT=4")
    assert r["ok"] is True
    body = stub.events.return_value.insert.call_args.kwargs["body"]
    assert body["recurrence"] == ["RRULE:FREQ=WEEKLY;COUNT=4"]


# ---------- update_event ----------

def test_update_event_unknown_keys_rejected():
    r = update_event(event_id="x", changes={"yolo": 1})
    assert r["ok"] is False
    assert r["reason"] == "unknown_change_keys"


def test_update_event_happy():
    existing = {"id": "x", "summary": "old", "start": {"dateTime": "2026-05-12T10:00:00Z"}, "end": {"dateTime": "2026-05-12T11:00:00Z"}}
    stub = _make_stub_service(get_return=dict(existing), update_return={"id": "x", "summary": "new", **existing})
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = update_event(event_id="x", changes={"title": "new"})
    assert r["ok"] is True


# ---------- cancel_event ----------

def test_cancel_event_empty_id_rejected():
    r = cancel_event(event_id="")
    assert r["ok"] is False


def test_cancel_event_happy():
    stub = _make_stub_service()
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = cancel_event(event_id="x", notify_attendees=False)
    assert r["ok"] is True
    kwargs = stub.events.return_value.delete.call_args.kwargs
    assert kwargs["sendUpdates"] == "none"


# ---------- invite responses ----------

def test_accept_invite_empty_id_rejected():
    r = accept_invite(event_id="")
    assert r["ok"] is False


def test_decline_invite_includes_note():
    existing = {"id": "x", "attendees": [{"email": "self@example.com", "self": True, "responseStatus": "needsAction"}],
                "start": {"dateTime": "2026-05-12T10:00:00Z"}, "end": {"dateTime": "2026-05-12T11:00:00Z"}}
    stub = _make_stub_service(get_return=dict(existing), update_return=dict(existing))
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = decline_invite(event_id="x", reason="conflict")
    assert r["ok"] is True
    body = stub.events.return_value.update.call_args.kwargs["body"]
    me = next(a for a in body["attendees"] if a.get("self"))
    assert me["responseStatus"] == "declined"
    assert me["comment"] == "conflict"


def test_tentative_invite_sets_status():
    existing = {"id": "x", "attendees": [{"email": "self@example.com", "self": True}],
                "start": {"dateTime": "2026-05-12T10:00:00Z"}, "end": {"dateTime": "2026-05-12T11:00:00Z"}}
    stub = _make_stub_service(get_return=dict(existing), update_return=dict(existing))
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = tentative_invite(event_id="x")
    assert r["ok"] is True
    body = stub.events.return_value.update.call_args.kwargs["body"]
    me = next(a for a in body["attendees"] if a.get("self"))
    assert me["responseStatus"] == "tentative"


# ---------- recurring / block focus ----------

def test_find_recurring_meetings_filters_recurrence():
    stub = _make_stub_service(list_items=[
        {"id": "r1", "summary": "weekly", "recurrence": ["RRULE:FREQ=WEEKLY"],
         "start": {"dateTime": "2026-05-12T10:00:00Z"}, "end": {"dateTime": "2026-05-12T11:00:00Z"}},
        {"id": "o1", "summary": "one-off",
         "start": {"dateTime": "2026-05-12T10:00:00Z"}, "end": {"dateTime": "2026-05-12T11:00:00Z"}},
    ])
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = find_recurring_meetings()
    assert r["ok"] is True
    assert r["count"] == 1
    assert r["events"][0]["summary"] == "weekly"


def test_block_focus_time_delegates_to_create():
    stub = _make_stub_service(insert_return={"id": "focus1", "summary": "Focus"})
    with patch("mcp_servers.nexus_calendar._get_calendar", return_value=stub):
        r = block_focus_time(start=_future_iso(), end=_future_iso(2))
    assert r["ok"] is True
    body = stub.events.return_value.insert.call_args.kwargs["body"]
    assert body["summary"] == "Focus"


# ---------- upcoming_birthdays (stub) ----------

def test_upcoming_birthdays_returns_not_configured():
    r = upcoming_birthdays()
    assert r["ok"] is False
    assert r["reason"] == "not_configured"
