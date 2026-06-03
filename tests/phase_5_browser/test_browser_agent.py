"""Phase 5 — input-validation tests for BrowserAgent (no real Firefox spawned).

The agent's RUNTIME behavior (Playwright session, real network) is exercised
via the live-verification harness; this file locks down the pre-network safety
boundary — URL validation, timeout clamping, output-path validation.
"""
from __future__ import annotations

import pytest

from services.browser_agent import (
    BrowserAgent,
    BrowserActionResult,
    _normalize_timeout,
    _validate_url,
)


# ---------- _validate_url ----------

@pytest.mark.parametrize("bad", [
    "",
    "   ",
    "not a url",
    "example.com",          # missing scheme
    "ftp://example.com",    # disallowed scheme
    "file:///etc/passwd",   # disallowed scheme
    "javascript:alert(1)",
    "https://" + "x" * 4100,  # too long
])
def test_invalid_urls_rejected(bad):
    assert _validate_url(bad) is not None


@pytest.mark.parametrize("good", [
    "http://example.com",
    "https://example.com",
    "https://example.com/path?q=1",
    "https://sub.example.com:8443/x",
])
def test_valid_urls_accepted(good):
    assert _validate_url(good) is None


# ---------- _normalize_timeout ----------

def test_timeout_none_default():
    assert _normalize_timeout(None) == 30


def test_timeout_clamped_to_cap():
    assert _normalize_timeout(9999) == 120


def test_timeout_floor_at_one():
    assert _normalize_timeout(-5) == 1


def test_timeout_non_integer_defaults():
    assert _normalize_timeout("abc") == 30


# ---------- BrowserActionResult shape ----------

def test_action_result_ok_basic():
    r = BrowserActionResult(ok=True, data={"x": 1})
    assert r.ok is True
    assert r.data == {"x": 1}


def test_action_result_failure_shape():
    r = BrowserActionResult(ok=False, reason="invalid_url", detail="x")
    assert r.ok is False
    assert r.reason == "invalid_url"


# ---------- BrowserAgent enter requires valid user_id ----------

@pytest.mark.asyncio
async def test_unknown_user_raises_profile_error():
    from services.browser_profile_resolver import BrowserProfileError
    agent = BrowserAgent("nonexistent-uuid")
    with pytest.raises(BrowserProfileError, match="unknown user_id"):
        await agent.__aenter__()


# ---------- fetch_url URL validation (no real network) ----------

@pytest.mark.asyncio
async def test_fetch_url_rejects_bad_scheme():
    agent = BrowserAgent("x")
    r = await agent.fetch_url(url="ftp://example.com")
    assert r.ok is False
    assert r.reason == "invalid_url"


@pytest.mark.asyncio
async def test_fetch_url_rejects_empty():
    agent = BrowserAgent("x")
    r = await agent.fetch_url(url="")
    assert r.ok is False
    assert r.reason == "invalid_url"


@pytest.mark.asyncio
async def test_screenshot_rejects_non_png_output():
    agent = BrowserAgent("x")
    r = await agent.screenshot(url="https://example.com",
                                output_path="/home/user/nexus_workspace/shot.txt")
    assert r.ok is False
    # Either output_path_denied (parent-not-exist style) or output_must_be_png
    assert r.reason in {"output_must_be_png", "output_path_denied"}


@pytest.mark.asyncio
async def test_screenshot_rejects_outside_allowed_roots():
    agent = BrowserAgent("x")
    r = await agent.screenshot(url="https://example.com", output_path="/etc/shot.png")
    assert r.ok is False
    assert r.reason == "output_path_denied"


@pytest.mark.asyncio
async def test_fill_form_empty_field_values_rejected():
    agent = BrowserAgent("x")
    r = await agent.fill_form(url="https://example.com", field_values={})
    assert r.ok is False
    assert r.reason == "empty_field_values"


@pytest.mark.asyncio
async def test_scrape_invalid_schema_rejected():
    agent = BrowserAgent("x")
    r = await agent.scrape(url="https://example.com", schema={})
    assert r.ok is False
    assert r.reason == "invalid_schema"


@pytest.mark.asyncio
async def test_search_web_empty_query_rejected():
    agent = BrowserAgent("x")
    r = await agent.search_web(query="")
    assert r.ok is False
    assert r.reason == "empty_query"
