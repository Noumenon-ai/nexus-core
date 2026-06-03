"""Adversarial tests for nexus-knowledge MCP server (Phase 2.5a Server 11)."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point NEXUS_KNOWLEDGE_DB at a fresh tmp file per test."""
    monkeypatch.setenv("NEXUS_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    yield


# Imports happen AFTER the autouse fixture has set the env var.
from mcp_servers.nexus_knowledge import (
    find_journal_entries,
    forget,
    journal_entry,
    list_memories,
    lookup_decision,
    memory_stats,
    recall,
    recall_by_key,
    recall_by_tag,
    remember,
    save_decision,
    update_memory,
)


# ---------- remember ----------

def test_remember_empty_key_rejected():
    r = remember(key="", value="x")
    assert r["ok"] is False
    assert r["reason"] == "empty_key"


def test_remember_non_string_value_rejected():
    r = remember(key="k", value=123)
    assert r["ok"] is False
    assert r["reason"] == "non_string_value"


def test_remember_key_too_long_rejected():
    r = remember(key="x" * 500, value="y")
    assert r["ok"] is False
    assert r["reason"] == "key_too_long"


def test_remember_value_too_long_rejected():
    r = remember(key="k", value="x" * 200_000)
    assert r["ok"] is False
    assert r["reason"] == "value_too_long"


def test_remember_invalid_tags_rejected():
    r = remember(key="k", value="v", tags=["", " "])
    assert r["ok"] is False
    assert r["reason"] == "invalid_tags"


def test_remember_invalid_expires_rejected():
    r = remember(key="k", value="v", expires="next-tuesday")
    assert r["ok"] is False
    assert r["reason"] == "invalid_expires"


def test_remember_creates_then_updates():
    r1 = remember(key="favorite_color", value="blue")
    assert r1["ok"] is True
    assert r1["action"] == "created"
    r2 = remember(key="favorite_color", value="green", tags=["preferences"])
    assert r2["ok"] is True
    assert r2["action"] == "updated"
    got = recall_by_key(key="favorite_color")
    assert got["value"] == "green"
    assert got["tags"] == ["preferences"]


# ---------- recall ----------

def test_recall_empty_query_rejected():
    r = recall(query="")
    assert r["ok"] is False


def test_recall_substring_match():
    remember(key="meeting_with_sam", value="Discuss Q4 plan")
    remember(key="meeting_with_bob",   value="Tech review")
    r = recall(query="sam")
    assert r["ok"] is True
    assert r["count"] >= 1
    assert any(m["key"] == "meeting_with_sam" for m in r["memories"])


def test_recall_excludes_expired():
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    remember(key="stale_fact", value="expired info", expires=past)
    r = recall(query="expired")
    assert r["ok"] is True
    assert not any(m["key"] == "stale_fact" for m in r["memories"])


def test_recall_by_tag_matches_only_tag_holders():
    remember(key="task_a", value="x", tags=["sprint5"])
    remember(key="task_b", value="y", tags=["sprint5"])
    remember(key="random", value="z", tags=["misc"])
    r = recall_by_tag(tag="SPRINT5")  # case-insensitive
    assert r["ok"] is True
    keys = {m["key"] for m in r["memories"]}
    assert keys >= {"task_a", "task_b"}
    assert "random" not in keys


def test_recall_by_key_not_found():
    r = recall_by_key(key="does_not_exist")
    assert r["ok"] is False
    assert r["reason"] == "not_found"


def test_recall_by_key_expired():
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    remember(key="old", value="v", expires=past)
    r = recall_by_key(key="old")
    assert r["ok"] is False
    assert r["reason"] == "expired"


# ---------- forget ----------

def test_forget_returns_false_for_missing():
    r = forget(key="never_there")
    assert r["ok"] is True
    assert r["deleted"] is False


def test_forget_removes_existing():
    remember(key="bye", value="x")
    r = forget(key="bye")
    assert r["ok"] is True
    assert r["deleted"] is True
    assert recall_by_key(key="bye")["ok"] is False


# ---------- list_memories ----------

def test_list_memories_filter_by_tag():
    remember(key="a", value="1", tags=["alpha"])
    remember(key="b", value="2", tags=["beta"])
    r = list_memories(tag="alpha")
    assert r["ok"] is True
    keys = {m["key"] for m in r["memories"]}
    assert "a" in keys
    assert "b" not in keys


def test_list_memories_limit_clamped():
    for i in range(10):
        remember(key=f"k{i}", value=str(i))
    r = list_memories(limit=5)
    assert r["ok"] is True
    assert r["count"] == 5


# ---------- update_memory ----------

def test_update_memory_not_found_rejected():
    r = update_memory(key="missing", value="x")
    assert r["ok"] is False
    assert r["reason"] == "not_found"


def test_update_memory_happy():
    remember(key="mood", value="ok")
    r = update_memory(key="mood", value="great")
    assert r["ok"] is True
    assert recall_by_key(key="mood")["value"] == "great"


# ---------- decisions ----------

def test_save_decision_empty_topic_rejected():
    r = save_decision(topic="", decision="x")
    assert r["ok"] is False


def test_save_decision_invalid_date_rejected():
    r = save_decision(topic="t", decision="d", date="not-a-date")
    assert r["ok"] is False


def test_save_decision_and_lookup():
    save_decision(topic="NEXUS routing architecture", decision="Path B: tool turns via Gemini")
    save_decision(topic="NEXUS persistence", decision="SQLite for knowledge db")
    r = lookup_decision(topic="routing")
    assert r["ok"] is True
    assert any("Path B" in d["decision"] for d in r["decisions"])


def test_lookup_decision_no_match_returns_empty():
    r = lookup_decision(topic="totally_unrelated")
    assert r["ok"] is True
    assert r["count"] == 0


# ---------- journal ----------

def test_journal_entry_empty_rejected():
    r = journal_entry(content="   ")
    assert r["ok"] is False


def test_journal_entry_invalid_tags_rejected():
    r = journal_entry(content="hi", tags=[""])
    assert r["ok"] is False


def test_journal_entry_happy():
    r = journal_entry(content="Good day. Shipped Phase 2.5a utils server.", mood="energized", tags=["work", "win"])
    assert r["ok"] is True
    found = find_journal_entries(query="utils server")
    assert found["count"] >= 1


def test_find_journal_invalid_date_range_rejected():
    r = find_journal_entries(date_range=["nope", "also nope"])
    assert r["ok"] is False


def test_find_journal_date_range_filter():
    journal_entry(content="old entry text")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
    r = find_journal_entries(date_range=[yesterday, tomorrow])
    assert r["ok"] is True
    assert r["count"] >= 1


# ---------- stats ----------

def test_memory_stats_counts():
    remember(key="a", value="x", tags=["alpha"])
    remember(key="b", value="y", tags=["alpha", "beta"])
    save_decision(topic="t", decision="d")
    journal_entry(content="hi")
    r = memory_stats()
    assert r["ok"] is True
    assert r["memory_count"] == 2
    assert r["decision_count"] == 1
    assert r["journal_count"] == 1
    tag_names = {t["tag"] for t in r["top_tags"]}
    assert "alpha" in tag_names
