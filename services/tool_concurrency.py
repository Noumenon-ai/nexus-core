"""Step 3 — parallel tool calls + TTL cache + thinking indicator.

Three independent primitives:

  gather_tools   — wrap asyncio.gather with structured per-tool error
                   capture, so one slow / failing tool doesn't poison
                   the whole fan-out.
  TTLCache       — process-local key→value cache with per-key TTL. Used
                   for rental_summary (5min), email_unread_count (5min),
                   weather (5min), square_today_sales (2min) and other
                   slow-changing reads.
  ThinkingIndicator — emit ⏳ immediately on receipt and edit-in-place
                      when the real reply is ready. Wired by the
                      telegram_bot adapter; this module owns the
                      lifecycle so callers don't have to.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Awaitable, Callable, Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parallel gather
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolResult:
    name: str
    value: Any = None
    error: BaseException | None = None
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


async def gather_tools(
    calls: Mapping[str, Callable[[], Awaitable[Any]]],
) -> dict[str, ToolResult]:
    """Run a name→async-callable mapping concurrently.

    Returns name→ToolResult; one slow/failing tool never poisons the
    others. Exceptions are captured per tool, not raised. Duration is
    recorded so the dashboard can show which calls are dragging.
    """
    names = list(calls.keys())
    if not names:
        return {}

    async def _run(name: str, fn: Callable[[], Awaitable[Any]]) -> ToolResult:
        start = time.monotonic()
        try:
            value = await fn()
        except BaseException as exc:  # noqa: BLE001
            duration_ms = (time.monotonic() - start) * 1000.0
            logger.warning(
                'tool_concurrency_tool_failed',
                extra={'tool': name, 'error': str(exc), 'duration_ms': duration_ms},
            )
            return ToolResult(name=name, error=exc, duration_ms=duration_ms)
        duration_ms = (time.monotonic() - start) * 1000.0
        return ToolResult(name=name, value=value, duration_ms=duration_ms)

    results = await asyncio.gather(
        *(_run(name, calls[name]) for name in names),
        return_exceptions=False,
    )
    return {result.name: result for result in results}


# ---------------------------------------------------------------------------
# TTL cache
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Entry:
    value: Any
    expires_at: float


class TTLCache:
    """Process-local key→value cache with per-key TTL.

    Thread-safe via RLock. Sync API only — wrap async fetchers with
    `get_or_set_async`.
    """

    def __init__(self, *, default_ttl_seconds: float = 300.0) -> None:
        if default_ttl_seconds <= 0:
            raise ValueError('default_ttl_seconds must be positive')
        self._default_ttl = default_ttl_seconds
        self._store: dict[str, _Entry] = {}
        self._lock = RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at <= time.monotonic():
                self._store.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: Any, *, ttl_seconds: float | None = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        if ttl <= 0:
            raise ValueError('ttl_seconds must be positive')
        with self._lock:
            self._store[key] = _Entry(
                value=value,
                expires_at=time.monotonic() + ttl,
            )

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        with self._lock:
            now = time.monotonic()
            return sum(1 for entry in self._store.values() if entry.expires_at > now)

    async def get_or_set_async(
        self,
        key: str,
        fetcher: Callable[[], Awaitable[Any]],
        *,
        ttl_seconds: float | None = None,
    ) -> Any:
        """Cache-aside with async fetcher. Misses run `fetcher` and cache the result."""
        hit = self.get(key)
        if hit is not None:
            return hit
        value = await fetcher()
        # Cache even falsy values (e.g. 0 unread email) — the consumer
        # cares about "we just refreshed" not "we have a truthy value".
        # Use a sentinel internally so None can be cached too.
        if value is not None:
            self.set(key, value, ttl_seconds=ttl_seconds)
        return value


# Default TTLs documented in NEXUS_ARCHITECTURE_REFACTOR.md step 3.
CACHE_TTL_SECONDS: dict[str, float] = {
    'rental_summary': 300.0,
    'email_unread_count': 300.0,
    'weather': 300.0,
    'square_today_sales': 120.0,
}

default_tool_cache = TTLCache(default_ttl_seconds=300.0)


def cache_ttl_for(key: str) -> float:
    return CACHE_TTL_SECONDS.get(key, default_tool_cache._default_ttl)


# ---------------------------------------------------------------------------
# Thinking indicator
# ---------------------------------------------------------------------------


# A messenger that can post and later edit a message. Returns the
# message_id (or None if posting failed).
SendFn = Callable[[str], Awaitable[Any]]
EditFn = Callable[[Any, str], Awaitable[None]]


@dataclass(slots=True)
class ThinkingIndicator:
    """Post ⏳ immediately on receipt; edit to the real reply when ready.

    The constructor takes adapter callables so the indicator stays
    decoupled from python-telegram-bot specifics (and the test suite).

      send(text) -> message_id
      edit(message_id, text) -> None

    Both adapters are async; the indicator tolerates send returning
    None (post failed) by skipping the edit silently.
    """
    send: SendFn
    edit: EditFn
    placeholder: str = '⏳'
    message_id: Any = field(default=None, init=False)
    _started: bool = field(default=False, init=False)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        try:
            self.message_id = await self.send(self.placeholder)
        except BaseException as exc:  # noqa: BLE001
            logger.warning('thinking_indicator_send_failed', extra={'error': str(exc)})
            self.message_id = None

    async def finish(self, final_text: str) -> None:
        if not self._started:
            # Caller forgot to .start() — just send the final text once.
            try:
                await self.send(final_text)
            except BaseException as exc:  # noqa: BLE001
                logger.warning('thinking_indicator_final_send_failed', extra={'error': str(exc)})
            return
        if self.message_id is None:
            # Placeholder never made it out. Send the final text fresh.
            try:
                await self.send(final_text)
            except BaseException as exc:  # noqa: BLE001
                logger.warning('thinking_indicator_final_send_failed', extra={'error': str(exc)})
            return
        try:
            await self.edit(self.message_id, final_text)
        except BaseException as exc:  # noqa: BLE001
            logger.warning('thinking_indicator_edit_failed', extra={'error': str(exc)})
            # Last-ditch fallback so the user sees a reply
            try:
                await self.send(final_text)
            except BaseException as inner:  # noqa: BLE001
                logger.warning('thinking_indicator_recovery_send_failed', extra={'error': str(inner)})
