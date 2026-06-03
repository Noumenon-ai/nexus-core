"""nexus-reminders-tasks MCP server — Phase 2.5a Server 6.

Reminders + todo lists. Reads/writes the same SQLite tables the NEXUS daemon uses
so the two systems stay in sync (single source of truth).

Configuration via environment variables:
  DATABASE_URL                  same value the bot uses (default: sqlite:///./.data/nexus.db)
  NEXUS_MCP_DEFAULT_USER_ID    UUID of the user record for tools called without explicit user_id

Tools that mutate state still execute immediately — Claude CLI is expected to gate
them at the user-permission layer. Google Tasks sync deferred to Phase 2.5b (this
server reads/writes the local `tasks` table only).
"""
from __future__ import annotations

import os
import uuid as _uuid
from datetime import date as _date, datetime, timedelta, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from models import Reminder, Task


mcp = FastMCP("nexus-reminders-tasks")


_engine = None
_SessionFactory: sessionmaker | None = None


def _get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "sqlite:///./.data/nexus.db")
    return url


def _get_session_factory() -> sessionmaker:
    global _engine, _SessionFactory
    cur_url = _get_database_url()
    if _engine is None or str(_engine.url) != cur_url:
        _engine = create_engine(cur_url, future=True)
        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, class_=Session, future=True)
    return _SessionFactory  # type: ignore[return-value]


def _resolve_user_id(user_id: str | None) -> str | None:
    if user_id and user_id.strip():
        return user_id.strip()
    env_uid = os.environ.get("NEXUS_MCP_DEFAULT_USER_ID")
    if env_uid:
        return env_uid.strip()
    return None


def _parse_iso(value: str, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 string")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not valid ISO-8601: {exc}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite drops tzinfo on store; coerce loaded datetimes back to UTC-aware."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _serialize_reminder(r: Reminder) -> dict:
    return {
        "id": r.id,
        "user_id": r.user_id,
        "body": r.body,
        "next_fire_at": r.next_fire_at.isoformat() if r.next_fire_at else None,
        "recurrence": r.recurrence,
        "status": r.status,
        "last_fired_at": r.last_fired_at.isoformat() if r.last_fired_at else None,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def _serialize_task(t: Task) -> dict:
    return {
        "id": t.id,
        "user_id": t.user_id,
        "title": t.title,
        "status": t.status,
        "due_at": t.due_at.isoformat() if t.due_at else None,
        "priority": t.priority,
        "project": t.source,
        "completed_at": t.completed_at.isoformat() if t.completed_at else None,
        "created_at": t.created_at.isoformat() if t.created_at else None,
    }


# ---------- reminders ----------

@mcp.tool()
def create_reminder(body: str, fire_at: str, recurring: str | None = None, user_id: str | None = None) -> dict:
    """Create a reminder. fire_at = ISO-8601 datetime. recurring optional (e.g. 'FREQ=DAILY')."""
    if not isinstance(body, str) or not body.strip():
        return {"ok": False, "reason": "empty_body", "detail": ""}
    if len(body) > 2000:
        return {"ok": False, "reason": "body_too_long", "detail": f"{len(body)} chars"}
    try:
        when = _parse_iso(fire_at, "fire_at")
    except ValueError as exc:
        return {"ok": False, "reason": "invalid_fire_at", "detail": str(exc)}
    uid = _resolve_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id", "detail": "pass user_id or set NEXUS_MCP_DEFAULT_USER_ID"}
    session_factory = _get_session_factory()
    with session_factory() as session:
        r = Reminder(user_id=uid, body=body.strip(), next_fire_at=when, recurrence=recurring, status="active")
        session.add(r)
        session.commit()
        session.refresh(r)
        return {"ok": True, "reminder": _serialize_reminder(r)}


@mcp.tool()
def list_reminders(active_only: bool = True, time_range: list[str] | None = None, user_id: str | None = None) -> dict:
    """List reminders. active_only filters out fired/cancelled. time_range = [start_iso, end_iso] on next_fire_at."""
    uid = _resolve_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id", "detail": "pass user_id or set NEXUS_MCP_DEFAULT_USER_ID"}
    start = end = None
    if time_range is not None:
        if not isinstance(time_range, list) or len(time_range) != 2:
            return {"ok": False, "reason": "invalid_time_range", "detail": "must be [start, end]"}
        try:
            start = _parse_iso(time_range[0], "time_range.start") if time_range[0] else None
            end = _parse_iso(time_range[1], "time_range.end") if time_range[1] else None
        except ValueError as exc:
            return {"ok": False, "reason": "invalid_time_range", "detail": str(exc)}
    session_factory = _get_session_factory()
    with session_factory() as session:
        stmt = select(Reminder).where(Reminder.user_id == uid)
        if active_only:
            stmt = stmt.where(Reminder.status == "active")
        if start is not None:
            stmt = stmt.where(Reminder.next_fire_at >= start)
        if end is not None:
            stmt = stmt.where(Reminder.next_fire_at <= end)
        stmt = stmt.order_by(Reminder.next_fire_at)
        rows = list(session.scalars(stmt))
    return {"ok": True, "count": len(rows), "reminders": [_serialize_reminder(r) for r in rows]}


# Reminders are ungated per owner directive 2026-06-02 ("No approval gates on
# reminders"). Editing/cancelling a reminder is low-risk and user-owned; the
# friction of an Approve/Cancel prompt is exactly what made reminders feel
# broken. The destructive-intent classifier no longer registers these tools
# either, so the bot acts immediately.
@mcp.tool()
def update_reminder(reminder_id: str, changes: dict) -> dict:
    """Modify a reminder. Allowed change keys: body, fire_at (ISO), recurrence, status."""
    if not isinstance(reminder_id, str) or not reminder_id.strip():
        return {"ok": False, "reason": "empty_reminder_id", "detail": ""}
    if not isinstance(changes, dict) or not changes:
        return {"ok": False, "reason": "empty_changes", "detail": ""}
    allowed = {"body", "fire_at", "recurrence", "status"}
    extra = set(changes) - allowed
    if extra:
        return {"ok": False, "reason": "unknown_change_keys", "detail": f"unknown: {sorted(extra)}"}
    session_factory = _get_session_factory()
    with session_factory() as session:
        r = session.get(Reminder, reminder_id)
        if r is None:
            return {"ok": False, "reason": "not_found", "detail": reminder_id}
        if "body" in changes:
            if not isinstance(changes["body"], str) or not changes["body"].strip():
                return {"ok": False, "reason": "invalid_body", "detail": ""}
            r.body = changes["body"].strip()
        if "fire_at" in changes:
            try:
                r.next_fire_at = _parse_iso(changes["fire_at"], "fire_at")
            except ValueError as exc:
                return {"ok": False, "reason": "invalid_fire_at", "detail": str(exc)}
        if "recurrence" in changes:
            r.recurrence = changes["recurrence"]
        if "status" in changes:
            if changes["status"] not in {"active", "fired", "cancelled"}:
                return {"ok": False, "reason": "invalid_status", "detail": repr(changes["status"])}
            r.status = changes["status"]
        session.commit()
        session.refresh(r)
        return {"ok": True, "reminder": _serialize_reminder(r)}


@mcp.tool()
def cancel_reminder(reminder_id: str) -> dict:
    """Set reminder status to 'cancelled'."""
    if not isinstance(reminder_id, str) or not reminder_id.strip():
        return {"ok": False, "reason": "empty_reminder_id", "detail": ""}
    session_factory = _get_session_factory()
    with session_factory() as session:
        r = session.get(Reminder, reminder_id)
        if r is None:
            return {"ok": False, "reason": "not_found", "detail": reminder_id}
        r.status = "cancelled"
        session.commit()
        return {"ok": True, "reminder_id": reminder_id}


@mcp.tool()
def snooze_reminder(reminder_id: str, until: str) -> dict:
    """Push next_fire_at later. until = ISO-8601 datetime."""
    if not isinstance(reminder_id, str) or not reminder_id.strip():
        return {"ok": False, "reason": "empty_reminder_id", "detail": ""}
    try:
        new_when = _parse_iso(until, "until")
    except ValueError as exc:
        return {"ok": False, "reason": "invalid_until", "detail": str(exc)}
    session_factory = _get_session_factory()
    with session_factory() as session:
        r = session.get(Reminder, reminder_id)
        if r is None:
            return {"ok": False, "reason": "not_found", "detail": reminder_id}
        r.next_fire_at = new_when
        if r.status != "active":
            r.status = "active"
        session.commit()
        session.refresh(r)
        return {"ok": True, "reminder": _serialize_reminder(r)}


# ---------- tasks ----------

@mcp.tool()
def create_task(title: str, description: str | None = None, due: str | None = None,
                project: str | None = None, priority: int = 0, user_id: str | None = None) -> dict:
    """Create a task. due = ISO-8601 datetime. project stored in `source` column."""
    if not isinstance(title, str) or not title.strip():
        return {"ok": False, "reason": "empty_title", "detail": ""}
    if len(title) > 500:
        return {"ok": False, "reason": "title_too_long", "detail": f"{len(title)} chars"}
    try:
        priority = int(priority)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_priority", "detail": repr(priority)}
    if not -10 <= priority <= 10:
        return {"ok": False, "reason": "priority_out_of_range", "detail": "priority must be -10..10"}
    due_at = None
    if due:
        try:
            due_at = _parse_iso(due, "due")
        except ValueError as exc:
            return {"ok": False, "reason": "invalid_due", "detail": str(exc)}
    uid = _resolve_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id", "detail": "pass user_id or set NEXUS_MCP_DEFAULT_USER_ID"}
    if description:
        title = title + ("\n\n" + description.strip() if description.strip() else "")
    session_factory = _get_session_factory()
    with session_factory() as session:
        t = Task(user_id=uid, title=title, status="pending", due_at=due_at, priority=priority, source=(project or None))
        session.add(t)
        session.commit()
        session.refresh(t)
        return {"ok": True, "task": _serialize_task(t)}


@mcp.tool()
def list_tasks(filter: dict | None = None, user_id: str | None = None, limit: int = 100) -> dict:
    """List tasks. filter keys: status ('pending'|'done'), project (str), min_priority (int), due_before (ISO)."""
    uid = _resolve_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id", "detail": "pass user_id or set NEXUS_MCP_DEFAULT_USER_ID"}
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_limit", "detail": repr(limit)}
    limit = max(1, min(limit, 500))
    flt = filter or {}
    if not isinstance(flt, dict):
        return {"ok": False, "reason": "invalid_filter", "detail": "filter must be a dict"}
    allowed = {"status", "project", "min_priority", "due_before"}
    extra = set(flt) - allowed
    if extra:
        return {"ok": False, "reason": "unknown_filter_keys", "detail": f"unknown: {sorted(extra)}"}

    session_factory = _get_session_factory()
    with session_factory() as session:
        stmt = select(Task).where(Task.user_id == uid)
        if "status" in flt:
            if flt["status"] not in {"pending", "done"}:
                return {"ok": False, "reason": "invalid_status_filter", "detail": repr(flt["status"])}
            stmt = stmt.where(Task.status == flt["status"])
        if "project" in flt:
            stmt = stmt.where(Task.source == flt["project"])
        if "min_priority" in flt:
            try:
                p = int(flt["min_priority"])
            except (TypeError, ValueError):
                return {"ok": False, "reason": "invalid_min_priority", "detail": repr(flt["min_priority"])}
            stmt = stmt.where(Task.priority >= p)
        if "due_before" in flt:
            try:
                cutoff = _parse_iso(flt["due_before"], "due_before")
            except ValueError as exc:
                return {"ok": False, "reason": "invalid_due_before", "detail": str(exc)}
            stmt = stmt.where(Task.due_at.is_not(None)).where(Task.due_at < cutoff)
        stmt = stmt.order_by(Task.priority.desc(), Task.due_at.is_(None), Task.due_at, Task.created_at).limit(limit)
        rows = list(session.scalars(stmt))
    return {"ok": True, "count": len(rows), "tasks": [_serialize_task(t) for t in rows]}


@mcp.tool()
def update_task(task_id: str, changes: dict) -> dict:
    """Modify a task. Allowed change keys: title, due (ISO|null), project, priority (int), status."""
    if not isinstance(task_id, str) or not task_id.strip():
        return {"ok": False, "reason": "empty_task_id", "detail": ""}
    if not isinstance(changes, dict) or not changes:
        return {"ok": False, "reason": "empty_changes", "detail": ""}
    allowed = {"title", "due", "project", "priority", "status"}
    extra = set(changes) - allowed
    if extra:
        return {"ok": False, "reason": "unknown_change_keys", "detail": f"unknown: {sorted(extra)}"}
    session_factory = _get_session_factory()
    with session_factory() as session:
        t = session.get(Task, task_id)
        if t is None:
            return {"ok": False, "reason": "not_found", "detail": task_id}
        if "title" in changes:
            if not isinstance(changes["title"], str) or not changes["title"].strip():
                return {"ok": False, "reason": "invalid_title", "detail": ""}
            t.title = changes["title"].strip()
        if "due" in changes:
            if changes["due"] is None:
                t.due_at = None
            else:
                try:
                    t.due_at = _parse_iso(changes["due"], "due")
                except ValueError as exc:
                    return {"ok": False, "reason": "invalid_due", "detail": str(exc)}
        if "project" in changes:
            t.source = changes["project"] or None
        if "priority" in changes:
            try:
                p = int(changes["priority"])
            except (TypeError, ValueError):
                return {"ok": False, "reason": "invalid_priority", "detail": repr(changes["priority"])}
            if not -10 <= p <= 10:
                return {"ok": False, "reason": "priority_out_of_range", "detail": "priority must be -10..10"}
            t.priority = p
        if "status" in changes:
            if changes["status"] not in {"pending", "done"}:
                return {"ok": False, "reason": "invalid_status", "detail": repr(changes["status"])}
            t.status = changes["status"]
            if changes["status"] == "done" and t.completed_at is None:
                t.completed_at = _now_utc()
            if changes["status"] == "pending":
                t.completed_at = None
        session.commit()
        session.refresh(t)
        return {"ok": True, "task": _serialize_task(t)}


@mcp.tool()
def complete_task(task_id: str) -> dict:
    """Mark a task done."""
    return update_task(task_id=task_id, changes={"status": "done"})


@mcp.tool(meta={"approval_required": True, "approval_template": "Delete task?"})
def delete_task(task_id: str) -> dict:
    """DELETE a task permanently."""
    if not isinstance(task_id, str) or not task_id.strip():
        return {"ok": False, "reason": "empty_task_id", "detail": ""}
    session_factory = _get_session_factory()
    with session_factory() as session:
        t = session.get(Task, task_id)
        if t is None:
            return {"ok": False, "reason": "not_found", "detail": task_id}
        session.delete(t)
        session.commit()
    return {"ok": True, "task_id": task_id}


# ---------- aggregates ----------

@mcp.tool()
def daily_briefing(user_id: str | None = None) -> dict:
    """Today's active reminders + pending tasks due today + overdue items."""
    uid = _resolve_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id", "detail": ""}
    now = _now_utc()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    session_factory = _get_session_factory()
    with session_factory() as session:
        today_reminders = list(session.scalars(
            select(Reminder).where(
                Reminder.user_id == uid, Reminder.status == "active",
                Reminder.next_fire_at >= today_start, Reminder.next_fire_at < today_end,
            ).order_by(Reminder.next_fire_at)
        ))
        today_tasks = list(session.scalars(
            select(Task).where(
                Task.user_id == uid, Task.status == "pending",
                Task.due_at.is_not(None), Task.due_at < today_end,
            ).order_by(Task.priority.desc(), Task.due_at)
        ))
    overdue_reminders = [r for r in today_reminders if _as_utc(r.next_fire_at) is not None and _as_utc(r.next_fire_at) < now]
    overdue_tasks = [t for t in today_tasks if t.due_at and _as_utc(t.due_at) < now]
    return {
        "ok": True,
        "as_of": now.isoformat(),
        "today_reminders": [_serialize_reminder(r) for r in today_reminders],
        "today_tasks": [_serialize_task(t) for t in today_tasks],
        "overdue_reminders": [_serialize_reminder(r) for r in overdue_reminders],
        "overdue_tasks": [_serialize_task(t) for t in overdue_tasks],
    }


@mcp.tool()
def weekly_review(user_id: str | None = None) -> dict:
    """Last 7 days completed + next 7 days incoming."""
    uid = _resolve_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id", "detail": ""}
    now = _now_utc()
    week_ago = now - timedelta(days=7)
    week_ahead = now + timedelta(days=7)
    session_factory = _get_session_factory()
    with session_factory() as session:
        completed = list(session.scalars(
            select(Task).where(
                Task.user_id == uid, Task.status == "done",
                Task.completed_at.is_not(None), Task.completed_at >= week_ago,
            ).order_by(Task.completed_at.desc())
        ))
        incoming_tasks = list(session.scalars(
            select(Task).where(
                Task.user_id == uid, Task.status == "pending",
                Task.due_at.is_not(None), Task.due_at >= now, Task.due_at <= week_ahead,
            ).order_by(Task.due_at)
        ))
        incoming_reminders = list(session.scalars(
            select(Reminder).where(
                Reminder.user_id == uid, Reminder.status == "active",
                Reminder.next_fire_at >= now, Reminder.next_fire_at <= week_ahead,
            ).order_by(Reminder.next_fire_at)
        ))
    return {
        "ok": True,
        "as_of": now.isoformat(),
        "completed_last_7_days": [_serialize_task(t) for t in completed],
        "incoming_tasks_next_7_days": [_serialize_task(t) for t in incoming_tasks],
        "incoming_reminders_next_7_days": [_serialize_reminder(r) for r in incoming_reminders],
    }


@mcp.tool()
def overdue_items(user_id: str | None = None) -> dict:
    """All overdue active reminders + pending tasks."""
    uid = _resolve_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id", "detail": ""}
    now = _now_utc()
    session_factory = _get_session_factory()
    with session_factory() as session:
        overdue_r = list(session.scalars(
            select(Reminder).where(
                Reminder.user_id == uid, Reminder.status == "active", Reminder.next_fire_at < now,
            ).order_by(Reminder.next_fire_at)
        ))
        overdue_t = list(session.scalars(
            select(Task).where(
                Task.user_id == uid, Task.status == "pending",
                Task.due_at.is_not(None), Task.due_at < now,
            ).order_by(Task.due_at)
        ))
    return {
        "ok": True,
        "as_of": now.isoformat(),
        "overdue_reminder_count": len(overdue_r),
        "overdue_task_count": len(overdue_t),
        "overdue_reminders": [_serialize_reminder(r) for r in overdue_r],
        "overdue_tasks": [_serialize_task(t) for t in overdue_t],
    }


# ---------- projects ----------

@mcp.tool()
def list_projects(user_id: str | None = None) -> dict:
    """All project names currently used as task.source for this user."""
    uid = _resolve_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id", "detail": ""}
    session_factory = _get_session_factory()
    with session_factory() as session:
        stmt = select(Task.source).where(Task.user_id == uid, Task.source.is_not(None)).distinct()
        names = sorted({n for n in session.scalars(stmt) if n})
    return {"ok": True, "count": len(names), "projects": names}


@mcp.tool()
def create_project(name: str) -> dict:
    """Register a project name. (Projects emerge from tasks with that source; this is a no-op confirmation.)"""
    if not isinstance(name, str) or not name.strip():
        return {"ok": False, "reason": "empty_name", "detail": ""}
    return {"ok": True, "project": name.strip(), "note": "projects are derived from task.source; create a task with this project name to populate"}


# ---------------------------------------------------------------------------
