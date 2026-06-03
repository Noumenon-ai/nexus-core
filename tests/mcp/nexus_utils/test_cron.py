"""Phase 6 — adversarial tests for cron tools added to nexus-utils MCP."""
from __future__ import annotations

import os
import uuid as _uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, User


_USER_ID = "cron-user-" + str(_uuid.uuid4())


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "cron_test.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("NEXUS_MCP_DEFAULT_USER_ID", _USER_ID)
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, future=True)() as s:
        s.add(User(id=_USER_ID, telegram_id=42, full_name="Cron Test", language="en", role="user"))
        s.commit()
    yield


from mcp_servers.nexus_utils import (
    create_cron_job,
    delete_cron_job,
    list_cron_jobs,
    pause_cron_job,
    resume_cron_job,
)


# ---------- create_cron_job ----------

def test_create_cron_empty_description_rejected():
    r = create_cron_job(description="", schedule_expression="* * * * *", action="ping")
    assert r["ok"] is False
    assert r["reason"] == "empty_description"


def test_create_cron_invalid_expression_rejected():
    r = create_cron_job(description="x", schedule_expression="every 5 min", action="ping")
    assert r["ok"] is False
    assert r["reason"] == "invalid_cron_expression"


def test_create_cron_wrong_field_count_rejected():
    r = create_cron_job(description="x", schedule_expression="* * *", action="ping")
    assert r["ok"] is False
    assert r["reason"] == "invalid_cron_expression"


def test_create_cron_empty_action_rejected():
    r = create_cron_job(description="x", schedule_expression="* * * * *", action="")
    assert r["ok"] is False
    assert r["reason"] == "empty_action"


def test_create_cron_action_too_long_rejected():
    r = create_cron_job(description="x", schedule_expression="* * * * *", action="y" * 3000)
    assert r["ok"] is False
    assert r["reason"] == "action_too_long"


def test_create_cron_no_user_rejected(monkeypatch):
    monkeypatch.delenv("NEXUS_MCP_DEFAULT_USER_ID", raising=False)
    r = create_cron_job(description="x", schedule_expression="* * * * *", action="ping")
    assert r["ok"] is False
    assert r["reason"] == "no_user_id"


def test_create_cron_happy():
    r = create_cron_job(description="every minute tick",
                        schedule_expression="* * * * *", action="say tick")
    assert r["ok"] is True
    assert r["status"] == "active"
    assert r["cron_expression"] == "* * * * *"


# ---------- list / pause / resume / delete ----------

def test_list_cron_jobs_includes_created():
    create_cron_job(description="d", schedule_expression="*/5 * * * *", action="a")
    r = list_cron_jobs()
    assert r["ok"] is True
    assert r["count"] >= 1
    assert any(j["description"] == "d" for j in r["jobs"])


def test_pause_resume_cycle():
    created = create_cron_job(description="d", schedule_expression="* * * * *", action="a")
    jid = created["job_id"]
    p = pause_cron_job(job_id=jid)
    assert p["ok"] is True and p["status"] == "paused"
    r = resume_cron_job(job_id=jid)
    assert r["ok"] is True and r["status"] == "active"


def test_delete_cron_job_marks_deleted():
    created = create_cron_job(description="d", schedule_expression="* * * * *", action="a")
    jid = created["job_id"]
    d = delete_cron_job(job_id=jid)
    assert d["ok"] is True
    # list excludes deleted by default
    listing = list_cron_jobs()
    assert all(j["id"] != jid for j in listing["jobs"])


def test_delete_nonexistent_rejected():
    r = delete_cron_job(job_id="does-not-exist")
    assert r["ok"] is False
    assert r["reason"] == "not_found"


def test_pause_nonexistent_rejected():
    r = pause_cron_job(job_id="ghost")
    assert r["ok"] is False
    assert r["reason"] == "not_found"


def test_resume_nonexistent_rejected():
    r = resume_cron_job(job_id="ghost")
    assert r["ok"] is False
    assert r["reason"] == "not_found"


def test_empty_job_id_rejected():
    for fn in (delete_cron_job, pause_cron_job, resume_cron_job):
        r = fn(job_id="")
        assert r["ok"] is False
        assert r["reason"] == "empty_job_id"


# ---------- cross-user isolation (RLS gate) ----------

def test_delete_cron_job_cannot_touch_another_users_job():
    """A user must not delete/pause/resume a cron job they do not own.
    Owner creates the job; a different user_id gets 'not_found' and the
    job stays active for the owner."""
    created = create_cron_job(description="owner job",
                              schedule_expression="* * * * *", action="a")
    jid = created["job_id"]

    for fn, _status in ((delete_cron_job, "deleted"),
                        (pause_cron_job, "paused"),
                        (resume_cron_job, "active")):
        r = fn(job_id=jid, user_id="someone-else-uuid")
        assert r["ok"] is False, f"{fn.__name__} leaked across users"
        assert r["reason"] == "not_found"

    # Owner's job is untouched and still active.
    listing = list_cron_jobs()
    assert any(j["id"] == jid and j["status"] == "active" for j in listing["jobs"])


def test_owner_can_still_mutate_own_job():
    created = create_cron_job(description="owner job 2",
                              schedule_expression="* * * * *", action="a")
    jid = created["job_id"]
    assert pause_cron_job(job_id=jid)["status"] == "paused"
    assert resume_cron_job(job_id=jid)["status"] == "active"
    assert delete_cron_job(job_id=jid)["ok"] is True
