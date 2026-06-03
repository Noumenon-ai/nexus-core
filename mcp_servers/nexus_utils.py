"""nexus-utils MCP server — Phase 2.5a Server 12.

Small-stuff utilities that don't fit elsewhere. All read-only, no approval gates.
Stdio transport. Every tool returns a structured dict; on failure returns
{'ok': False, 'reason': str, 'detail': str} rather than raising.

External APIs used (all free, no auth):
- Open-Meteo  (geocoding + forecast)
- ER-API      (currency rates)
- Hebcal      (Jewish calendar + shabbat times + omer)
"""
from __future__ import annotations

import ast
import base64 as _b64
import hashlib
import operator
import os
import random
import re
import secrets
import string
import urllib.parse
import uuid as _uuid
from datetime import date as _date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import qrcode

from services.file_system_tools import (
    FileSystemSecurityError,
    _validate_path,
)
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("nexus-utils")

_HTTP_TIMEOUT = 10.0
_DEFAULT_USER_TIMEZONE = "America/New_York"


# ---------- time / dates ----------

@mcp.tool()
def current_time(timezone: str | None = None) -> dict:
    """Current time in specified IANA timezone (e.g. 'America/New_York'). UTC if omitted."""
    tz_name = timezone or "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as exc:
        return {"ok": False, "reason": "unknown_timezone", "detail": str(exc)}
    now = datetime.now(tz)
    return {
        "ok": True,
        "iso": now.isoformat(),
        "timezone": tz_name,
        "human": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "weekday": now.strftime("%A"),
    }


# ---------- unit conversions ----------

_UNIT_TO_BASE: dict[str, tuple[str, float, float]] = {
    # unit -> (base_dimension, factor, offset)  base_value = value * factor + offset
    "km":   ("distance", 1000.0, 0.0),
    "m":    ("distance", 1.0, 0.0),
    "mi":   ("distance", 1609.344, 0.0),
    "ft":   ("distance", 0.3048, 0.0),
    "in":   ("distance", 0.0254, 0.0),
    "cm":   ("distance", 0.01, 0.0),

    "kg":   ("mass", 1.0, 0.0),
    "g":    ("mass", 0.001, 0.0),
    "lb":   ("mass", 0.45359237, 0.0),
    "oz":   ("mass", 0.028349523125, 0.0),

    "c":    ("temperature", 1.0, 273.15),
    "f":    ("temperature", 5/9, 273.15 - 32*5/9),
    "k":    ("temperature", 1.0, 0.0),

    "l":    ("volume", 1.0, 0.0),
    "ml":   ("volume", 0.001, 0.0),
    "gal":  ("volume", 3.785411784, 0.0),    # US gallon

    "s":    ("time", 1.0, 0.0),
    "min":  ("time", 60.0, 0.0),
    "h":    ("time", 3600.0, 0.0),
    "day":  ("time", 86400.0, 0.0),
}


@mcp.tool()
def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert a value between units. Supports distance, mass, temperature, volume, time."""
    src = (from_unit or "").lower().strip()
    dst = (to_unit or "").lower().strip()
    if src not in _UNIT_TO_BASE:
        return {"ok": False, "reason": "unknown_from_unit", "detail": from_unit}
    if dst not in _UNIT_TO_BASE:
        return {"ok": False, "reason": "unknown_to_unit", "detail": to_unit}
    src_dim, src_f, src_o = _UNIT_TO_BASE[src]
    dst_dim, dst_f, dst_o = _UNIT_TO_BASE[dst]
    if src_dim != dst_dim:
        return {"ok": False, "reason": "dimension_mismatch", "detail": f"{src_dim} vs {dst_dim}"}
    try:
        v = float(value)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_numeric_value", "detail": repr(value)}
    base = v * src_f + src_o
    out = (base - dst_o) / dst_f
    return {"ok": True, "value": out, "from": src, "to": dst, "dimension": src_dim}


# ---------- currency ----------

@mcp.tool()
def currency_convert(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert currency at live exchange rates via open.er-api.com (free, no auth)."""
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_numeric_amount", "detail": repr(amount)}
    src = (from_currency or "").upper().strip()
    dst = (to_currency or "").upper().strip()
    if not re.fullmatch(r"[A-Z]{3}", src) or not re.fullmatch(r"[A-Z]{3}", dst):
        return {"ok": False, "reason": "invalid_currency_code", "detail": f"{src!r} or {dst!r}"}
    try:
        r = httpx.get(f"https://open.er-api.com/v6/latest/{src}", timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as exc:
        return {"ok": False, "reason": "http_error", "detail": str(exc)[:200]}
    rates = data.get("rates") or {}
    if dst not in rates:
        return {"ok": False, "reason": "unknown_target_currency", "detail": dst}
    rate = float(rates[dst])
    return {
        "ok": True,
        "amount": amt,
        "from": src,
        "to": dst,
        "rate": rate,
        "converted": amt * rate,
        "as_of": data.get("time_last_update_utc"),
    }


# ---------- safe math ----------

_SAFE_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"non-numeric constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        return _SAFE_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
        return _SAFE_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"disallowed node: {type(node).__name__}")


@mcp.tool()
def do_math(expression: str) -> dict:
    """Evaluate an arithmetic expression with + - * / // % ** and parentheses. No functions, no names."""
    if not isinstance(expression, str) or not expression.strip():
        return {"ok": False, "reason": "empty_expression", "detail": repr(expression)}
    if len(expression) > 500:
        return {"ok": False, "reason": "expression_too_long", "detail": f"{len(expression)} chars"}
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return {"ok": False, "reason": "parse_error", "detail": str(exc)}
    try:
        result = _safe_eval(tree)
    except ZeroDivisionError:
        return {"ok": False, "reason": "division_by_zero", "detail": expression}
    except (ValueError, OverflowError) as exc:
        return {"ok": False, "reason": "eval_error", "detail": str(exc)}
    return {"ok": True, "expression": expression, "result": result}


# ---------- weather ----------

@mcp.tool()
def weather(location: str | None = None, days_ahead: int = 0) -> dict:
    """Current + multi-day forecast from Open-Meteo. days_ahead 0=today only (clamped 0-7)."""
    if not location or not str(location).strip():
        return {"ok": False, "reason": "no_location", "detail": "location is required"}
    days = max(0, min(int(days_ahead or 0), 7))
    try:
        geo = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en"},
            timeout=_HTTP_TIMEOUT,
        )
        geo.raise_for_status()
        gdata = geo.json().get("results") or []
    except httpx.HTTPError as exc:
        return {"ok": False, "reason": "geocoding_failed", "detail": str(exc)[:200]}
    if not gdata:
        return {"ok": False, "reason": "location_not_found", "detail": location}
    lat = gdata[0]["latitude"]
    lon = gdata[0]["longitude"]
    label = f"{gdata[0].get('name','')}, {gdata[0].get('country_code','')}"
    try:
        fc = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "forecast_days": days + 1,
                "timezone": "auto",
            },
            timeout=_HTTP_TIMEOUT,
        )
        fc.raise_for_status()
    except httpx.HTTPError as exc:
        return {"ok": False, "reason": "forecast_failed", "detail": str(exc)[:200]}
    return {
        "ok": True,
        "location": label,
        "latitude": lat,
        "longitude": lon,
        "data": fc.json(),
    }


# ---------- translate (placeholder — no provider wired) ----------

@mcp.tool()
def translate(text: str, target_language: str, source_language: str | None = None) -> dict:
    """Translate text. NOTE: no translation provider currently configured."""
    return {
        "ok": False,
        "reason": "not_configured",
        "detail": "translate has no provider wired yet. Add provider in Phase 2.5b.",
        "echo": {"text": text, "target": target_language, "source": source_language},
    }


# ---------- passwords / random ----------

@mcp.tool()
def generate_password(length: int = 20, complexity: str = "strong", count: int = 1) -> dict:
    """Generate one or more cryptographically random passwords."""
    try:
        length = int(length)
        count = int(count)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_argument", "detail": "length/count must be ints"}
    if not 4 <= length <= 256:
        return {"ok": False, "reason": "length_out_of_range", "detail": "length must be 4..256"}
    if not 1 <= count <= 50:
        return {"ok": False, "reason": "count_out_of_range", "detail": "count must be 1..50"}
    if complexity == "alphanum":
        alphabet = string.ascii_letters + string.digits
    elif complexity == "letters":
        alphabet = string.ascii_letters
    else:
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*-_=+[]{}<>?"
    passwords = ["".join(secrets.choice(alphabet) for _ in range(length)) for _ in range(count)]
    return {"ok": True, "count": count, "length": length, "complexity": complexity, "passwords": passwords}


@mcp.tool()
def random_pick(options: list) -> dict:
    """Pick one item uniformly at random from a list."""
    if not isinstance(options, list) or not options:
        return {"ok": False, "reason": "empty_or_invalid_options", "detail": repr(options)[:100]}
    pick = random.choice(options)
    return {"ok": True, "pick": pick, "from_count": len(options)}


@mcp.tool()
def roll_dice(sides: int = 6, count: int = 1) -> dict:
    """Roll N dice with S sides each."""
    try:
        sides = int(sides)
        count = int(count)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_argument", "detail": "sides/count must be ints"}
    if sides < 2 or sides > 1000:
        return {"ok": False, "reason": "sides_out_of_range", "detail": "sides must be 2..1000"}
    if count < 1 or count > 100:
        return {"ok": False, "reason": "count_out_of_range", "detail": "count must be 1..100"}
    rolls = [random.randint(1, sides) for _ in range(count)]
    return {"ok": True, "sides": sides, "count": count, "rolls": rolls, "total": sum(rolls)}


@mcp.tool()
def flip_coin(count: int = 1) -> dict:
    """Flip N coins."""
    try:
        count = int(count)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_argument", "detail": repr(count)}
    if count < 1 or count > 1000:
        return {"ok": False, "reason": "count_out_of_range", "detail": "count must be 1..1000"}
    flips = [random.choice(["heads", "tails"]) for _ in range(count)]
    return {"ok": True, "count": count, "flips": flips, "heads": flips.count("heads"), "tails": flips.count("tails")}


# ---------- ids / hashes / encoding ----------

@mcp.tool()
def uuid() -> dict:
    """Generate a random UUIDv4."""
    return {"ok": True, "uuid": str(_uuid.uuid4())}


_HASH_ALGOS = {"md5", "sha1", "sha224", "sha256", "sha384", "sha512"}


@mcp.tool()
def hash_text(text: str, algorithm: str = "sha256") -> dict:
    """Hash text with one of: md5, sha1, sha224, sha256, sha384, sha512."""
    algo = (algorithm or "sha256").lower().strip()
    if algo not in _HASH_ALGOS:
        return {"ok": False, "reason": "unknown_algorithm", "detail": algo, "supported": sorted(_HASH_ALGOS)}
    if not isinstance(text, str):
        return {"ok": False, "reason": "non_string_input", "detail": repr(type(text).__name__)}
    h = hashlib.new(algo)
    h.update(text.encode("utf-8"))
    return {"ok": True, "algorithm": algo, "hex": h.hexdigest()}


@mcp.tool()
def base64_encode(text: str) -> dict:
    """Encode string as base64 (UTF-8)."""
    if not isinstance(text, str):
        return {"ok": False, "reason": "non_string_input", "detail": repr(type(text).__name__)}
    return {"ok": True, "base64": _b64.b64encode(text.encode("utf-8")).decode("ascii")}


@mcp.tool()
def base64_decode(text: str) -> dict:
    """Decode base64 to UTF-8 string."""
    if not isinstance(text, str):
        return {"ok": False, "reason": "non_string_input", "detail": repr(type(text).__name__)}
    try:
        raw = _b64.b64decode(text, validate=True)
    except Exception as exc:
        return {"ok": False, "reason": "invalid_base64", "detail": str(exc)[:200]}
    try:
        return {"ok": True, "text": raw.decode("utf-8")}
    except UnicodeDecodeError as exc:
        return {"ok": False, "reason": "not_utf8", "detail": str(exc)[:200]}


@mcp.tool()
def url_encode(text: str) -> dict:
    """Percent-encode a string for use in URLs."""
    if not isinstance(text, str):
        return {"ok": False, "reason": "non_string_input", "detail": repr(type(text).__name__)}
    return {"ok": True, "encoded": urllib.parse.quote(text, safe="")}


@mcp.tool()
def url_decode(text: str) -> dict:
    """Decode a percent-encoded string."""
    if not isinstance(text, str):
        return {"ok": False, "reason": "non_string_input", "detail": repr(type(text).__name__)}
    return {"ok": True, "decoded": urllib.parse.unquote(text)}


@mcp.tool()
def regex_test(pattern: str, text: str) -> dict:
    """Test whether a regex pattern matches text. Returns match groups if any."""
    if not isinstance(pattern, str) or not isinstance(text, str):
        return {"ok": False, "reason": "non_string_input", "detail": "pattern and text must be strings"}
    if len(pattern) > 1000:
        return {"ok": False, "reason": "pattern_too_long", "detail": f"{len(pattern)} chars"}
    try:
        m = re.search(pattern, text)
    except re.error as exc:
        return {"ok": False, "reason": "invalid_regex", "detail": str(exc)[:200]}
    if not m:
        return {"ok": True, "match": False}
    return {
        "ok": True,
        "match": True,
        "matched_text": m.group(0),
        "groups": list(m.groups()),
        "span": list(m.span()),
    }


# ---------- QR code (writes file, path-validated) ----------

@mcp.tool()
def qr_code(content: str, output_path: str) -> dict:
    """Generate a QR code PNG. output_path must resolve inside ALLOWED_ROOTS."""
    if not isinstance(content, str) or not content:
        return {"ok": False, "reason": "empty_content", "detail": "content must be a non-empty string"}
    if len(content) > 2000:
        return {"ok": False, "reason": "content_too_long", "detail": f"{len(content)} chars (max 2000)"}
    try:
        resolved = _validate_path(output_path, must_exist=False)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "path_denied", "detail": str(exc)}
    if resolved.exists() and not resolved.is_file():
        return {"ok": False, "reason": "path_not_a_file", "detail": str(resolved)}
    parent = resolved.parent
    if not parent.exists():
        return {"ok": False, "reason": "parent_does_not_exist", "detail": str(parent)}
    img = qrcode.make(content)
    img.save(str(resolved))
    size = resolved.stat().st_size
    return {"ok": True, "path": str(resolved), "size_bytes": size, "content_chars": len(content)}


# ---------- Jewish calendar ----------

def _today_or(date_str: str | None) -> _date:
    if date_str is None:
        return datetime.now(ZoneInfo(_DEFAULT_USER_TIMEZONE)).date()
    try:
        return _date.fromisoformat(date_str)
    except ValueError as exc:
        raise ValueError(str(exc))


@mcp.tool()
def hebcal(date: str | None = None) -> dict:
    """Hebrew date + parsha + holidays for the given (YYYY-MM-DD) or today's date."""
    try:
        d = _today_or(date)
    except ValueError as exc:
        return {"ok": False, "reason": "invalid_date", "detail": str(exc)}
    try:
        r = httpx.get(
            "https://www.hebcal.com/converter",
            params={"cfg": "json", "gy": d.year, "gm": d.month, "gd": d.day, "g2h": 1},
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        conv = r.json()
    except httpx.HTTPError as exc:
        return {"ok": False, "reason": "hebcal_http_error", "detail": str(exc)[:200]}
    return {"ok": True, "date": d.isoformat(), "hebrew": conv}


@mcp.tool()
def shabbos_times(location: str | None = None, date: str | None = None) -> dict:
    """Shabbat candle-lighting + havdalah for the given location and date (defaults to today)."""
    if not location or not str(location).strip():
        return {"ok": False, "reason": "no_location", "detail": "location (city name) is required"}
    try:
        d = _today_or(date)
    except ValueError as exc:
        return {"ok": False, "reason": "invalid_date", "detail": str(exc)}
    try:
        r = httpx.get(
            "https://www.hebcal.com/shabbat",
            params={
                "cfg": "json",
                "geonameid": "",
                "city": location,
                "gy": d.year, "gm": d.month, "gd": d.day,
                "leyning": "off",
            },
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as exc:
        return {"ok": False, "reason": "hebcal_http_error", "detail": str(exc)[:200]}
    return {"ok": True, "location": location, "date": d.isoformat(), "shabbat": data}


@mcp.tool()
def omer_count(date: str | None = None) -> dict:
    """Count of the Omer for the given date if applicable (returns None outside Omer window)."""
    try:
        d = _today_or(date)
    except ValueError as exc:
        return {"ok": False, "reason": "invalid_date", "detail": str(exc)}
    try:
        r = httpx.get(
            "https://www.hebcal.com/hebcal",
            params={
                "cfg": "json", "v": 1, "maj": "off", "min": "off",
                "mod": "off", "nx": "off", "year": d.year, "month": d.month,
                "ss": "off", "mf": "off", "c": "off", "geo": "none",
                "o": "on",
            },
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as exc:
        return {"ok": False, "reason": "hebcal_http_error", "detail": str(exc)[:200]}
    items = data.get("items") or []
    iso = d.isoformat()
    for item in items:
        if item.get("date") == iso and item.get("category") == "omer":
            return {"ok": True, "date": iso, "in_omer": True, "title": item.get("title"), "memo": item.get("memo")}
    return {"ok": True, "date": iso, "in_omer": False}


# ---------- Phase 6: cron jobs ----------
#
# These are CORE NEXUS scheduling primitives (per H2-037 spec note).  The
# MCP layer just writes to the shared cron_jobs table; the bot's APScheduler
# sync loop (every 30s) picks them up and registers / removes the actual
# triggers.  When a cron fires the bot re-injects the action_text into the
# pipeline as if the user just typed it.

_DEFAULT_DATABASE_URL = "sqlite:///./.data/nexus.db"


def _get_cron_session_factory():
    import os
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, Session
    url = os.environ.get("DATABASE_URL", _DEFAULT_DATABASE_URL)
    engine = create_engine(url, future=True)
    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session, future=True)


def _resolve_default_user_id(user_id: str | None) -> str | None:
    if user_id and user_id.strip():
        return user_id.strip()
    return (os.environ.get("NEXUS_MCP_DEFAULT_USER_ID") or "").strip() or None


def _validate_cron_expr(expr: str) -> str | None:
    """Return None if expression parses as a valid 5-field cron string, else error msg."""
    if not isinstance(expr, str) or not expr.strip():
        return "cron_expression must be a non-empty string"
    parts = expr.strip().split()
    if len(parts) != 5:
        return f"cron_expression must have exactly 5 fields (min hour dom mon dow), got {len(parts)}"
    try:
        from apscheduler.triggers.cron import CronTrigger
        CronTrigger.from_crontab(expr.strip())
    except Exception as exc:
        return f"invalid cron expression: {str(exc)[:200]}"
    return None


@mcp.tool(meta={"approval_required": True, "approval_template": "Create recurring cron job?"})
def create_cron_job(description: str, schedule_expression: str, action: str,
                    user_id: str | None = None) -> dict:
    """Register a recurring action. schedule_expression is a standard 5-field cron string
    (e.g. '*/5 * * * *' = every 5 min, '0 9 * * *' = 9am daily). The `action` text is
    re-injected into NEXUS as if the user typed it when the cron fires.
    """
    if not isinstance(description, str) or not description.strip():
        return {"ok": False, "reason": "empty_description", "detail": ""}
    err = _validate_cron_expr(schedule_expression)
    if err:
        return {"ok": False, "reason": "invalid_cron_expression", "detail": err}
    if not isinstance(action, str) or not action.strip():
        return {"ok": False, "reason": "empty_action", "detail": ""}
    if len(action) > 2000:
        return {"ok": False, "reason": "action_too_long", "detail": f"{len(action)} chars (max 2000)"}
    uid = _resolve_default_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id",
                "detail": "pass user_id or set NEXUS_MCP_DEFAULT_USER_ID"}
    from repositories.cron_jobs_repository import CronJobsRepository
    repo = CronJobsRepository(_get_cron_session_factory())
    job = repo.create(
        user_id=uid,
        description=description.strip(),
        cron_expression=schedule_expression.strip(),
        action_text=action.strip(),
    )
    return {
        "ok": True,
        "job_id": job.id,
        "description": job.description,
        "cron_expression": job.cron_expression,
        "action": job.action_text,
        "status": job.status,
        "note": "Scheduler picks this up within 30 seconds.",
    }


@mcp.tool()
def list_cron_jobs(user_id: str | None = None, include_paused: bool = True) -> dict:
    """All active (and optionally paused) cron jobs for this user."""
    uid = _resolve_default_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id", "detail": ""}
    from repositories.cron_jobs_repository import CronJobsRepository
    repo = CronJobsRepository(_get_cron_session_factory())
    rows = repo.list_for_user(uid, include_deleted=False)
    if not include_paused:
        rows = [r for r in rows if r.status == 'active']
    return {
        "ok": True,
        "count": len(rows),
        "jobs": [
            {
                "id": r.id,
                "description": r.description,
                "cron_expression": r.cron_expression,
                "action": r.action_text,
                "status": r.status,
                "fire_count": r.fire_count,
                "last_fired_at": r.last_fired_at.isoformat() if r.last_fired_at else None,
            }
            for r in rows
        ],
    }


@mcp.tool(meta={"approval_required": True, "approval_template": "Delete cron job?"})
def delete_cron_job(job_id: str, user_id: str | None = None) -> dict:
    """Permanently remove a cron job. APPROVAL-EQUIVALENT — host should gate.
    Scoped to the calling user: a job owned by another user is not_found."""
    if not isinstance(job_id, str) or not job_id.strip():
        return {"ok": False, "reason": "empty_job_id", "detail": ""}
    uid = _resolve_default_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id", "detail": ""}
    from repositories.cron_jobs_repository import CronJobsRepository
    repo = CronJobsRepository(_get_cron_session_factory())
    job = repo.update_status(job_id.strip(), "deleted", user_id=uid)
    if job is None:
        return {"ok": False, "reason": "not_found", "detail": job_id}
    return {"ok": True, "job_id": job.id, "status": "deleted"}


@mcp.tool()
def pause_cron_job(job_id: str, user_id: str | None = None) -> dict:
    """Temporarily stop a cron job firing. Resume with resume_cron_job.
    Scoped to the calling user."""
    if not isinstance(job_id, str) or not job_id.strip():
        return {"ok": False, "reason": "empty_job_id", "detail": ""}
    uid = _resolve_default_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id", "detail": ""}
    from repositories.cron_jobs_repository import CronJobsRepository
    repo = CronJobsRepository(_get_cron_session_factory())
    job = repo.update_status(job_id.strip(), "paused", user_id=uid)
    if job is None:
        return {"ok": False, "reason": "not_found", "detail": job_id}
    return {"ok": True, "job_id": job.id, "status": "paused"}


@mcp.tool()
def resume_cron_job(job_id: str, user_id: str | None = None) -> dict:
    """Reactivate a paused cron job. Scoped to the calling user."""
    if not isinstance(job_id, str) or not job_id.strip():
        return {"ok": False, "reason": "empty_job_id", "detail": ""}
    uid = _resolve_default_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id", "detail": ""}
    from repositories.cron_jobs_repository import CronJobsRepository
    repo = CronJobsRepository(_get_cron_session_factory())
    job = repo.update_status(job_id.strip(), "active", user_id=uid)
    if job is None:
        return {"ok": False, "reason": "not_found", "detail": job_id}
    return {"ok": True, "job_id": job.id, "status": "active"}


# No user_id: notes are not per-user, they're "the owner's notes."


def main() -> None:
    """Stdio entrypoint."""
    mcp.run()


if __name__ == "__main__":
    main()
