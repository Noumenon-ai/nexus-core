"""Adversarial tests for nexus-reminders-tasks MCP server (Phase 2.5a Server 6)."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


_USER_ID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point DATABASE_URL + NEXUS_MCP_DEFAULT_USER_ID at a fresh sqlite per test."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("NEXUS_MCP_DEFAULT_USER_ID", _USER_ID)

    # Build the schema + insert the default user record.
    import mcp_servers.nexus_reminders_tasks as srv
    srv._engine = None
    srv._SessionFactory = None
    from models import Base, User
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, future=True)() as s:
        s.add(User(id=_USER_ID, telegram_id=900000, full_name="MCP Test", language="en", role="user"))
        s.commit()
    yield
    # Reset module globals so the next test rebuilds.
    srv._engine = None
    srv._SessionFactory = None


from mcp_servers.nexus_reminders_tasks import (
    cancel_reminder,
    complete_task,
    create_project,
    create_reminder,
    create_task,
    daily_briefing,
    delete_task,
    list_projects,
    list_reminders,
    list_tasks,
    overdue_items,
    snooze_reminder,
    update_reminder,
    update_task,
    weekly_review,
)


def _future_iso(seconds: int = 3600) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _past_iso(seconds: int = 3600) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


# ---------- create_reminder ----------

def test_create_reminder_empty_body_rejected():
    r = create_reminder(body="", fire_at=_future_iso())
    assert r["ok"] is False
    assert r["reason"] == "empty_body"


def test_create_reminder_invalid_fire_at_rejected():
    r = create_reminder(body="b", fire_at="not-a-date")
    assert r["ok"] is False
    assert r["reason"] == "invalid_fire_at"


def test_create_reminder_body_too_long_rejected():
    r = create_reminder(body="x" * 3000, fire_at=_future_iso())
    assert r["ok"] is False
    assert r["reason"] == "body_too_long"


def test_create_reminder_no_user_rejected(monkeypatch):
    monkeypatch.delenv("NEXUS_MCP_DEFAULT_USER_ID", raising=False)
    r = create_reminder(body="b", fire_at=_future_iso())
    assert r["ok"] is False
    assert r["reason"] == "no_user_id"


def test_create_reminder_happy():
    r = create_reminder(body="Call bank", fire_at=_future_iso(), recurring="FREQ=DAILY")
    assert r["ok"] is True
    assert r["reminder"]["body"] == "Call bank"
    assert r["reminder"]["status"] == "active"
    assert r["reminder"]["recurrence"] == "FREQ=DAILY"


# ---------- list_reminders ----------

def test_list_reminders_active_only_filter():
    create_reminder(body="r1", fire_at=_future_iso())
    r = create_reminder(body="r2", fire_at=_future_iso())
    cancel_reminder(reminder_id=r["reminder"]["id"])
    r_active = list_reminders(active_only=True)
    assert r_active["ok"] is True
    assert all(rem["status"] == "active" for rem in r_active["reminders"])
    r_all = list_reminders(active_only=False)
    assert r_all["count"] >= 2


def test_list_reminders_invalid_time_range_rejected():
    r = list_reminders(time_range=["bad", _future_iso()])
    assert r["ok"] is False
    assert r["reason"] == "invalid_time_range"


# ---------- update_reminder ----------

def test_update_reminder_unknown_change_keys_rejected():
    r = create_reminder(body="x", fire_at=_future_iso())
    r2 = update_reminder(reminder_id=r["reminder"]["id"], changes={"foo": "bar"})
    assert r2["ok"] is False
    assert r2["reason"] == "unknown_change_keys"


def test_update_reminder_not_found_rejected():
    r = update_reminder(reminder_id="nonexistent-id", changes={"body": "x"})
    assert r["ok"] is False
    assert r["reason"] == "not_found"


def test_update_reminder_happy():
    r = create_reminder(body="old", fire_at=_future_iso())
    rid = r["reminder"]["id"]
    r2 = update_reminder(reminder_id=rid, changes={"body": "new", "status": "cancelled"})
    assert r2["ok"] is True
    assert r2["reminder"]["body"] == "new"
    assert r2["reminder"]["status"] == "cancelled"


def test_update_reminder_invalid_status_rejected():
    r = create_reminder(body="x", fire_at=_future_iso())
    r2 = update_reminder(reminder_id=r["reminder"]["id"], changes={"status": "yolo"})
    assert r2["ok"] is False
    assert r2["reason"] == "invalid_status"


# ---------- cancel_reminder / snooze_reminder ----------

def test_cancel_reminder_happy():
    r = create_reminder(body="x", fire_at=_future_iso())
    r2 = cancel_reminder(reminder_id=r["reminder"]["id"])
    assert r2["ok"] is True
    again = list_reminders(active_only=True)
    assert all(rem["id"] != r["reminder"]["id"] for rem in again["reminders"])


def test_snooze_reminder_invalid_until_rejected():
    r = create_reminder(body="x", fire_at=_future_iso())
    r2 = snooze_reminder(reminder_id=r["reminder"]["id"], until="not-a-date")
    assert r2["ok"] is False
    assert r2["reason"] == "invalid_until"


def test_snooze_reminder_reactivates_cancelled():
    r = create_reminder(body="x", fire_at=_past_iso())
    cancel_reminder(reminder_id=r["reminder"]["id"])
    r2 = snooze_reminder(reminder_id=r["reminder"]["id"], until=_future_iso(7200))
    assert r2["ok"] is True
    assert r2["reminder"]["status"] == "active"


# ---------- tasks ----------

def test_create_task_empty_title_rejected():
    r = create_task(title="")
    assert r["ok"] is False
    assert r["reason"] == "empty_title"


def test_create_task_priority_out_of_range_rejected():
    r = create_task(title="t", priority=99)
    assert r["ok"] is False
    assert r["reason"] == "priority_out_of_range"


def test_create_task_invalid_due_rejected():
    r = create_task(title="t", due="bad")
    assert r["ok"] is False
    assert r["reason"] == "invalid_due"


def test_create_task_happy_with_project():
    r = create_task(title="Renew lease for Anna", project="rentals", priority=3, due=_future_iso())
    assert r["ok"] is True
    assert r["task"]["project"] == "rentals"
    assert r["task"]["priority"] == 3


def test_list_tasks_filter_by_status():
    r1 = create_task(title="A")
    create_task(title="B")
    complete_task(task_id=r1["task"]["id"])
    pending = list_tasks(filter={"status": "pending"})
    done = list_tasks(filter={"status": "done"})
    assert all(t["status"] == "pending" for t in pending["tasks"])
    assert all(t["status"] == "done" for t in done["tasks"])
    assert any(t["id"] == r1["task"]["id"] for t in done["tasks"])


def test_list_tasks_unknown_filter_key_rejected():
    r = list_tasks(filter={"banana": 1})
    assert r["ok"] is False
    assert r["reason"] == "unknown_filter_keys"


def test_list_tasks_invalid_status_filter_rejected():
    r = list_tasks(filter={"status": "in_progress"})
    assert r["ok"] is False
    assert r["reason"] == "invalid_status_filter"


def test_list_tasks_min_priority_filter():
    create_task(title="low", priority=0)
    create_task(title="hi", priority=5)
    r = list_tasks(filter={"min_priority": 3})
    assert all(t["priority"] >= 3 for t in r["tasks"])


def test_update_task_unknown_keys_rejected():
    r = create_task(title="t")
    r2 = update_task(task_id=r["task"]["id"], changes={"weird_field": "x"})
    assert r2["ok"] is False
    assert r2["reason"] == "unknown_change_keys"


def test_update_task_clear_due():
    r = create_task(title="t", due=_future_iso())
    r2 = update_task(task_id=r["task"]["id"], changes={"due": None})
    assert r2["ok"] is True
    assert r2["task"]["due_at"] is None


def test_complete_task_sets_completed_at():
    r = create_task(title="x")
    r2 = complete_task(task_id=r["task"]["id"])
    assert r2["ok"] is True
    assert r2["task"]["status"] == "done"
    assert r2["task"]["completed_at"] is not None


def test_delete_task_not_found_rejected():
    r = delete_task(task_id="missing")
    assert r["ok"] is False
    assert r["reason"] == "not_found"


def test_delete_task_happy():
    r = create_task(title="doomed")
    r2 = delete_task(task_id=r["task"]["id"])
    assert r2["ok"] is True
    after = list_tasks()
    assert all(t["id"] != r["task"]["id"] for t in after["tasks"])


# ---------- aggregates ----------

def test_daily_briefing_combines_reminders_and_tasks():
    create_reminder(body="r-today", fire_at=_future_iso(60))
    create_task(title="t-today", due=_future_iso(120))
    create_task(title="t-overdue", due=_past_iso(7200))
    r = daily_briefing()
    assert r["ok"] is True
    assert any(t["title"] == "t-overdue" for t in r["overdue_tasks"])
    assert any(t["title"] == "t-today" for t in r["today_tasks"])


def test_weekly_review_lists_completed_and_incoming():
    r = create_task(title="done_one")
    complete_task(task_id=r["task"]["id"])
    create_task(title="incoming_one", due=_future_iso(86400))
    rev = weekly_review()
    assert rev["ok"] is True
    titles_done = {t["title"] for t in rev["completed_last_7_days"]}
    titles_incoming = {t["title"] for t in rev["incoming_tasks_next_7_days"]}
    assert "done_one" in titles_done
    assert "incoming_one" in titles_incoming


def test_overdue_items_filters_only_past_due():
    create_task(title="future_task", due=_future_iso())
    create_task(title="past_task",   due=_past_iso())
    create_reminder(body="past_reminder", fire_at=_past_iso())
    create_reminder(body="future_reminder", fire_at=_future_iso())
    r = overdue_items()
    assert r["ok"] is True
    assert any(t["title"] == "past_task" for t in r["overdue_tasks"])
    assert not any(t["title"] == "future_task" for t in r["overdue_tasks"])
    assert any(rm["body"] == "past_reminder" for rm in r["overdue_reminders"])


# ---------- projects ----------

def test_list_projects_returns_distinct_sources():
    create_task(title="a", project="rentals")
    create_task(title="b", project="rentals")
    create_task(title="c", project="business")
    r = list_projects()
    assert r["ok"] is True
    assert set(r["projects"]) == {"rentals", "business"}


def test_create_project_empty_rejected():
    r = create_project(name="")
    assert r["ok"] is False


def test_create_project_noop_response():
    r = create_project(name="new-project")
    assert r["ok"] is True
    assert r["project"] == "new-project"


# ---------- no_user_id fallback ----------

def test_no_user_id_for_aggregate_endpoints(monkeypatch):
    monkeypatch.delenv("NEXUS_MCP_DEFAULT_USER_ID", raising=False)
    for fn in (daily_briefing, weekly_review, overdue_items, list_projects):
        r = fn()
        assert r["ok"] is False
        assert r["reason"] == "no_user_id"
