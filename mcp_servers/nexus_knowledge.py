"""nexus-knowledge MCP server — Phase 2.5a Server 11.

Long-term memory beyond conversation context. SQLite-backed key/value store plus
decisions log + daily journal. Optional semantic search hook left for Phase 2.5b
(mem0 integration); current `recall` uses substring + tag matching.

Database file: ~/.nexus/knowledge.db (override via NEXUS_KNOWLEDGE_DB env var).
Every tool returns a dict; on failure {'ok': False, 'reason': str, 'detail': str}.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("nexus-knowledge")

_DEFAULT_DB_PATH = Path(os.environ.get("NEXUS_KNOWLEDGE_DB", str(Path.home() / ".nexus" / "knowledge.db")))
_db_lock = threading.Lock()


def _get_db_path() -> Path:
    """Resolve DB path at call time so tests can override NEXUS_KNOWLEDGE_DB."""
    return Path(os.environ.get("NEXUS_KNOWLEDGE_DB", str(_DEFAULT_DB_PATH)))


def _connect() -> sqlite3.Connection:
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            mem_key     TEXT NOT NULL UNIQUE,
            mem_value   TEXT NOT NULL,
            tags_json   TEXT NOT NULL DEFAULT '[]',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            expires_at  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);

        CREATE TABLE IF NOT EXISTS decisions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            topic       TEXT NOT NULL,
            decision    TEXT NOT NULL,
            reasoning   TEXT,
            decided_on  TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_decisions_topic ON decisions(topic);

        CREATE TABLE IF NOT EXISTS journal_entries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            content     TEXT NOT NULL,
            mood        TEXT,
            tags_json   TEXT NOT NULL DEFAULT '[]',
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_journal_created ON journal_entries(created_at);
        """
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_tags(raw: list | None) -> list[str]:
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError("tags must be a list of strings")
    out: list[str] = []
    for t in raw:
        if not isinstance(t, str) or not t.strip():
            raise ValueError(f"tag must be a non-empty string, got {t!r}")
        out.append(t.strip().lower())
    return out


def _is_expired(row: sqlite3.Row) -> bool:
    expires = row["expires_at"]
    if not expires:
        return False
    try:
        return datetime.fromisoformat(expires) < datetime.now(timezone.utc)
    except ValueError:
        return False


# ---------- memories: key/value ----------

@mcp.tool()
def remember(key: str, value: str, tags: list[str] | None = None, expires: str | None = None) -> dict:
    """Save a fact under `key`. Overwrites existing entry. Optional ISO-8601 `expires`."""
    if not isinstance(key, str) or not key.strip():
        return {"ok": False, "reason": "empty_key", "detail": ""}
    if not isinstance(value, str):
        return {"ok": False, "reason": "non_string_value", "detail": repr(type(value).__name__)}
    if len(key) > 200:
        return {"ok": False, "reason": "key_too_long", "detail": f"{len(key)} chars (max 200)"}
    if len(value) > 100_000:
        return {"ok": False, "reason": "value_too_long", "detail": f"{len(value)} chars (max 100_000)"}
    try:
        tag_list = _parse_tags(tags)
    except ValueError as exc:
        return {"ok": False, "reason": "invalid_tags", "detail": str(exc)}
    if expires is not None:
        if not isinstance(expires, str):
            return {"ok": False, "reason": "invalid_expires", "detail": "must be ISO-8601 string"}
        try:
            datetime.fromisoformat(expires)
        except ValueError as exc:
            return {"ok": False, "reason": "invalid_expires", "detail": str(exc)}

    now = _now_iso()
    with _db_lock, _connect() as conn:
        existing = conn.execute("SELECT id, created_at FROM memories WHERE mem_key = ?", (key,)).fetchone()
        if existing:
            created_at = existing["created_at"]
            conn.execute(
                "UPDATE memories SET mem_value = ?, tags_json = ?, updated_at = ?, expires_at = ? WHERE id = ?",
                (value, json.dumps(tag_list), now, expires, existing["id"]),
            )
            action = "updated"
        else:
            conn.execute(
                "INSERT INTO memories (mem_key, mem_value, tags_json, created_at, updated_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                (key, value, json.dumps(tag_list), now, now, expires),
            )
            created_at = now
            action = "created"
    return {"ok": True, "action": action, "key": key, "created_at": created_at, "updated_at": now, "tags": tag_list, "expires_at": expires}


@mcp.tool()
def recall(query: str, limit: int = 20) -> dict:
    """Substring search across keys + values. Tag matches also returned. Excludes expired entries."""
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "reason": "empty_query", "detail": ""}
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_limit", "detail": repr(limit)}
    limit = max(1, min(limit, 200))
    q_lower = query.lower()

    with _db_lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM memories ORDER BY updated_at DESC LIMIT 1000"
        ).fetchall()

    results: list[dict] = []
    for row in rows:
        if _is_expired(row):
            continue
        haystack = f"{row['mem_key']}\n{row['mem_value']}\n{row['tags_json']}".lower()
        if q_lower in haystack:
            results.append({
                "key": row["mem_key"],
                "value": row["mem_value"],
                "tags": json.loads(row["tags_json"] or "[]"),
                "updated_at": row["updated_at"],
            })
            if len(results) >= limit:
                break
    return {"ok": True, "query": query, "count": len(results), "memories": results}


@mcp.tool()
def recall_by_key(key: str) -> dict:
    """Exact key lookup. Returns the stored value or not_found."""
    if not isinstance(key, str) or not key.strip():
        return {"ok": False, "reason": "empty_key", "detail": ""}
    with _db_lock, _connect() as conn:
        row = conn.execute("SELECT * FROM memories WHERE mem_key = ?", (key,)).fetchone()
    if row is None:
        return {"ok": False, "reason": "not_found", "detail": key}
    if _is_expired(row):
        return {"ok": False, "reason": "expired", "detail": f"expires_at={row['expires_at']}"}
    return {
        "ok": True,
        "key": row["mem_key"],
        "value": row["mem_value"],
        "tags": json.loads(row["tags_json"] or "[]"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "expires_at": row["expires_at"],
    }


@mcp.tool()
def recall_by_tag(tag: str, limit: int = 50) -> dict:
    """All memories tagged with the given tag (case-insensitive)."""
    if not isinstance(tag, str) or not tag.strip():
        return {"ok": False, "reason": "empty_tag", "detail": ""}
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_limit", "detail": repr(limit)}
    limit = max(1, min(limit, 500))
    target = tag.strip().lower()
    with _db_lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM memories ORDER BY updated_at DESC").fetchall()
    results: list[dict] = []
    for row in rows:
        if _is_expired(row):
            continue
        tag_list = json.loads(row["tags_json"] or "[]")
        if target in tag_list:
            results.append({
                "key": row["mem_key"],
                "value": row["mem_value"],
                "tags": tag_list,
                "updated_at": row["updated_at"],
            })
            if len(results) >= limit:
                break
    return {"ok": True, "tag": target, "count": len(results), "memories": results}


@mcp.tool(meta={"approval_required": True, "approval_template": "Forget memory entry?"})
def forget(key: str) -> dict:
    """DELETE a memory by key. Returns whether it existed."""
    if not isinstance(key, str) or not key.strip():
        return {"ok": False, "reason": "empty_key", "detail": ""}
    with _db_lock, _connect() as conn:
        cur = conn.execute("DELETE FROM memories WHERE mem_key = ?", (key,))
        deleted = cur.rowcount
    return {"ok": True, "key": key, "deleted": bool(deleted)}


@mcp.tool()
def list_memories(tag: str | None = None, limit: int = 100) -> dict:
    """Browse stored memories, optionally filtered by tag. Excludes expired."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_limit", "detail": repr(limit)}
    limit = max(1, min(limit, 500))
    target = (tag or "").strip().lower()
    with _db_lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM memories ORDER BY updated_at DESC").fetchall()
    out: list[dict] = []
    for row in rows:
        if _is_expired(row):
            continue
        tag_list = json.loads(row["tags_json"] or "[]")
        if target and target not in tag_list:
            continue
        out.append({
            "key": row["mem_key"],
            "value": row["mem_value"][:500],
            "tags": tag_list,
            "updated_at": row["updated_at"],
        })
        if len(out) >= limit:
            break
    return {"ok": True, "count": len(out), "filter_tag": target or None, "memories": out}


@mcp.tool(meta={"approval_required": True, "approval_template": "Update memory entry?"})
def update_memory(key: str, value: str) -> dict:
    """Modify the value of an existing memory. Returns not_found if key absent."""
    if not isinstance(key, str) or not key.strip():
        return {"ok": False, "reason": "empty_key", "detail": ""}
    if not isinstance(value, str):
        return {"ok": False, "reason": "non_string_value", "detail": ""}
    if len(value) > 100_000:
        return {"ok": False, "reason": "value_too_long", "detail": f"{len(value)} chars"}
    now = _now_iso()
    with _db_lock, _connect() as conn:
        cur = conn.execute(
            "UPDATE memories SET mem_value = ?, updated_at = ? WHERE mem_key = ?",
            (value, now, key),
        )
        if cur.rowcount == 0:
            return {"ok": False, "reason": "not_found", "detail": key}
    return {"ok": True, "key": key, "updated_at": now}


# ---------- decisions ----------

@mcp.tool()
def save_decision(topic: str, decision: str, reasoning: str | None = None, date: str | None = None) -> dict:
    """Log an architectural / life decision. Date defaults to today (UTC)."""
    if not isinstance(topic, str) or not topic.strip():
        return {"ok": False, "reason": "empty_topic", "detail": ""}
    if not isinstance(decision, str) or not decision.strip():
        return {"ok": False, "reason": "empty_decision", "detail": ""}
    decided_on = date or datetime.now(timezone.utc).date().isoformat()
    try:
        datetime.fromisoformat(decided_on)
    except ValueError as exc:
        return {"ok": False, "reason": "invalid_date", "detail": str(exc)}
    now = _now_iso()
    with _db_lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO decisions (topic, decision, reasoning, decided_on, created_at) VALUES (?, ?, ?, ?, ?)",
            (topic.strip(), decision.strip(), (reasoning or None), decided_on, now),
        )
        decision_id = cur.lastrowid
    return {"ok": True, "id": decision_id, "topic": topic.strip(), "decided_on": decided_on}


@mcp.tool()
def lookup_decision(topic: str, limit: int = 10) -> dict:
    """Find decisions whose topic contains the query (case-insensitive)."""
    if not isinstance(topic, str) or not topic.strip():
        return {"ok": False, "reason": "empty_topic", "detail": ""}
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_limit", "detail": repr(limit)}
    limit = max(1, min(limit, 200))
    pattern = f"%{topic.strip()}%"
    with _db_lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions WHERE topic LIKE ? COLLATE NOCASE ORDER BY decided_on DESC LIMIT ?",
            (pattern, limit),
        ).fetchall()
    decisions = [dict(r) for r in rows]
    return {"ok": True, "topic": topic, "count": len(decisions), "decisions": decisions}


# ---------- journal ----------

@mcp.tool()
def journal_entry(content: str, mood: str | None = None, tags: list[str] | None = None) -> dict:
    """Append a journal entry. Optional mood + tags."""
    if not isinstance(content, str) or not content.strip():
        return {"ok": False, "reason": "empty_content", "detail": ""}
    if len(content) > 100_000:
        return {"ok": False, "reason": "content_too_long", "detail": f"{len(content)} chars"}
    try:
        tag_list = _parse_tags(tags)
    except ValueError as exc:
        return {"ok": False, "reason": "invalid_tags", "detail": str(exc)}
    now = _now_iso()
    with _db_lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO journal_entries (content, mood, tags_json, created_at) VALUES (?, ?, ?, ?)",
            (content.strip(), (mood or None), json.dumps(tag_list), now),
        )
        entry_id = cur.lastrowid
    return {"ok": True, "id": entry_id, "created_at": now, "mood": mood, "tags": tag_list}


@mcp.tool()
def find_journal_entries(query: str | None = None, date_range: list[str] | None = None, limit: int = 50) -> dict:
    """Search journal entries by content substring and/or date range [start, end] ISO dates."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_limit", "detail": repr(limit)}
    limit = max(1, min(limit, 500))

    start = end = None
    if date_range is not None:
        if not isinstance(date_range, list) or len(date_range) != 2:
            return {"ok": False, "reason": "invalid_date_range", "detail": "must be [start, end]"}
        start, end = date_range
        try:
            if start:
                datetime.fromisoformat(start)
            if end:
                datetime.fromisoformat(end)
        except ValueError as exc:
            return {"ok": False, "reason": "invalid_date_range", "detail": str(exc)}

    sql = "SELECT * FROM journal_entries WHERE 1=1"
    params: list[Any] = []
    if query and query.strip():
        sql += " AND content LIKE ? COLLATE NOCASE"
        params.append(f"%{query.strip()}%")
    if start:
        sql += " AND created_at >= ?"
        params.append(start)
    if end:
        sql += " AND created_at <= ?"
        params.append(end)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with _db_lock, _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    entries = []
    for r in rows:
        entries.append({
            "id": r["id"],
            "content": r["content"],
            "mood": r["mood"],
            "tags": json.loads(r["tags_json"] or "[]"),
            "created_at": r["created_at"],
        })
    return {"ok": True, "count": len(entries), "entries": entries}


# ---------- stats ----------

@mcp.tool()
def memory_stats() -> dict:
    """Counts: total memories, total decisions, total journal entries, top tags."""
    with _db_lock, _connect() as conn:
        m_total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        d_total = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        j_total = conn.execute("SELECT COUNT(*) FROM journal_entries").fetchone()[0]
        mem_rows = conn.execute("SELECT tags_json FROM memories").fetchall()
    tag_counts: dict[str, int] = {}
    for row in mem_rows:
        for t in json.loads(row["tags_json"] or "[]"):
            tag_counts[t] = tag_counts.get(t, 0) + 1
    top_tags = sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "ok": True,
        "memory_count": m_total,
        "decision_count": d_total,
        "journal_count": j_total,
        "tag_count": len(tag_counts),
        "top_tags": [{"tag": k, "count": v} for k, v in top_tags],
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
