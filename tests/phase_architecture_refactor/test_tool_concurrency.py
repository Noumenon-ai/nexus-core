"""Tests for Step 3 primitives — gather_tools, TTLCache, ThinkingIndicator."""
from __future__ import annotations

import asyncio
import time

import pytest

from services.tool_concurrency import (
    CACHE_TTL_SECONDS,
    ThinkingIndicator,
    ToolResult,
    TTLCache,
    cache_ttl_for,
    gather_tools,
)


# ── gather_tools ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gather_tools_runs_calls_in_parallel():
    async def slow_one():
        await asyncio.sleep(0.05)
        return 'one'

    async def slow_two():
        await asyncio.sleep(0.05)
        return 'two'

    async def slow_three():
        await asyncio.sleep(0.05)
        return 'three'

    start = time.monotonic()
    results = await gather_tools({
        'one': slow_one,
        'two': slow_two,
        'three': slow_three,
    })
    elapsed = time.monotonic() - start
    # Parallel: should finish in ~0.05s, definitely under 3x serial (0.15s)
    assert elapsed < 0.13, f'gather_tools too slow ({elapsed:.3f}s) — likely sequential'
    assert results['one'].value == 'one'
    assert results['two'].value == 'two'
    assert results['three'].value == 'three'
    assert all(r.ok for r in results.values())


@pytest.mark.asyncio
async def test_gather_tools_isolates_failures():
    async def good():
        return 42

    async def boom():
        raise RuntimeError('tool died')

    results = await gather_tools({'good': good, 'bad': boom})
    assert results['good'].ok is True
    assert results['good'].value == 42
    assert results['bad'].ok is False
    assert isinstance(results['bad'].error, RuntimeError)


@pytest.mark.asyncio
async def test_gather_tools_records_duration_per_call():
    async def quick():
        return 'x'

    async def slower():
        await asyncio.sleep(0.02)
        return 'y'

    results = await gather_tools({'quick': quick, 'slower': slower})
    assert results['slower'].duration_ms >= results['quick'].duration_ms


@pytest.mark.asyncio
async def test_gather_tools_empty_returns_empty():
    assert await gather_tools({}) == {}


# ── TTLCache ────────────────────────────────────────────────────────────────


def test_ttl_cache_set_and_get():
    cache = TTLCache(default_ttl_seconds=60)
    cache.set('k', 'v')
    assert cache.get('k') == 'v'


def test_ttl_cache_expires():
    cache = TTLCache(default_ttl_seconds=60)
    cache.set('k', 'v', ttl_seconds=0.01)
    time.sleep(0.02)
    assert cache.get('k') is None


def test_ttl_cache_invalidate():
    cache = TTLCache(default_ttl_seconds=60)
    cache.set('k', 'v')
    cache.invalidate('k')
    assert cache.get('k') is None


def test_ttl_cache_clear():
    cache = TTLCache(default_ttl_seconds=60)
    cache.set('a', 1)
    cache.set('b', 2)
    cache.clear()
    assert len(cache) == 0


def test_ttl_cache_contains():
    cache = TTLCache(default_ttl_seconds=60)
    cache.set('k', 'v')
    assert 'k' in cache
    assert 'missing' not in cache


@pytest.mark.asyncio
async def test_get_or_set_async_caches_first_call():
    cache = TTLCache(default_ttl_seconds=60)
    calls: list[int] = []

    async def fetcher():
        calls.append(1)
        return 'fresh'

    a = await cache.get_or_set_async('k', fetcher)
    b = await cache.get_or_set_async('k', fetcher)
    assert a == 'fresh'
    assert b == 'fresh'
    assert len(calls) == 1, 'second call should hit cache'


@pytest.mark.asyncio
async def test_get_or_set_async_respects_custom_ttl():
    cache = TTLCache(default_ttl_seconds=60)

    async def fetcher():
        return 'value'

    await cache.get_or_set_async('k', fetcher, ttl_seconds=0.01)
    time.sleep(0.02)
    # Should expire and refetch
    calls: list[int] = []

    async def counted_fetch():
        calls.append(1)
        return 'next'

    out = await cache.get_or_set_async('k', counted_fetch)
    assert out == 'next'
    assert len(calls) == 1


def test_cache_ttl_for_known_keys_matches_spec():
    # NEXUS_ARCHITECTURE_REFACTOR.md step 3:
    #   rental_summary       5 min
    #   email_unread_count   5 min
    #   weather              5 min
    #   square_today_sales   2 min
    assert CACHE_TTL_SECONDS['rental_summary'] == 300.0
    assert CACHE_TTL_SECONDS['email_unread_count'] == 300.0
    assert CACHE_TTL_SECONDS['weather'] == 300.0
    assert CACHE_TTL_SECONDS['square_today_sales'] == 120.0
    # Unknown key falls back to default
    assert cache_ttl_for('unknown') == 300.0


# ── ThinkingIndicator ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thinking_indicator_posts_then_edits():
    sent: list[str] = []
    edited: list[tuple] = []

    async def send(text):
        sent.append(text)
        return 'msg-1'

    async def edit(message_id, text):
        edited.append((message_id, text))

    indicator = ThinkingIndicator(send=send, edit=edit)
    await indicator.start()
    await indicator.finish('here is the real reply')

    assert sent == ['⏳']
    assert edited == [('msg-1', 'here is the real reply')]


@pytest.mark.asyncio
async def test_thinking_indicator_falls_back_to_send_when_placeholder_lost():
    sent: list[str] = []

    async def send(text):
        sent.append(text)
        return None  # post failed

    async def edit(message_id, text):  # pragma: no cover - never reached
        raise AssertionError('edit should not run when placeholder is missing')

    indicator = ThinkingIndicator(send=send, edit=edit)
    await indicator.start()
    await indicator.finish('the real reply')

    assert sent == ['⏳', 'the real reply']


@pytest.mark.asyncio
async def test_thinking_indicator_finish_without_start_just_sends_once():
    sent: list[str] = []

    async def send(text):
        sent.append(text)
        return 'msg'

    async def edit(message_id, text):  # pragma: no cover
        raise AssertionError('edit should not run when start() was skipped')

    indicator = ThinkingIndicator(send=send, edit=edit)
    await indicator.finish('reply')
    assert sent == ['reply']


@pytest.mark.asyncio
async def test_thinking_indicator_recovers_when_edit_fails():
    sent: list[str] = []

    async def send(text):
        sent.append(text)
        return 'msg-1'

    async def edit(message_id, text):
        raise RuntimeError('telegram edit rejected (message too old)')

    indicator = ThinkingIndicator(send=send, edit=edit)
    await indicator.start()
    await indicator.finish('the real reply')
    # Recovery: send the final text fresh
    assert sent == ['⏳', 'the real reply']
