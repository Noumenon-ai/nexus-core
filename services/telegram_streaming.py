"""V3.7 Telegram streaming session — turns a multi-second dispatcher
turn into visible progress (Thinking → tool stages → final reply) by
editing one placeholder message and falling back to a fresh send if
the edit-message API fails.

Throttle is hard-capped at 2 events / sec via `edit_throttle_seconds`
(spec section "Locked Decisions"). Long final replies are split at
sentence boundaries with a hard-split fallback for boundary-free
text — break at sentence boundaries comes first; mid-word breaks are
the last-resort fallback only (spec halt condition).

V3.7 narrow scope: streaming + tool-stage visibility ONLY. Compaction
is parked as V3.7.5 — see HARDENING_PASS_V2.md H2-021 for the
spec-audit catch and the V3.7.5 prerequisite (archive-turn loading
into dispatcher contents) that makes compaction meaningful.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Protocol

try:
    from telegram.error import TelegramError
except ImportError:
    class TelegramError(Exception):
        """Fallback when python-telegram-bot is unavailable in the
        test environment. Mirrors the parent of BadRequest /
        NetworkError / RetryAfter / TimedOut so a single except
        clause catches the family."""


logger = logging.getLogger(__name__)


_DEFAULT_EDIT_THROTTLE_SECONDS = 0.5  # 2 events / sec
_DEFAULT_MAX_MESSAGE_LEN = 4000  # Telegram hard cap is 4096; leave headroom
_HARD_SPLIT_LOOKBACK = 500  # if no boundary in last N chars, hard-split


class TelegramBotLike(Protocol):
    """Minimum surface the streaming session needs from the bot.

    `TelegramBot` (telegram_bot.py) implements both methods. Tests
    inject a recorder stub. Decoupling here means future
    python-telegram-bot version bumps are localized to TelegramBot,
    not to streaming logic — per H2-021 user decision #6.
    """

    async def send_text(self, *, chat_id: int, text: str) -> int: ...
    async def edit_text(self, *, chat_id: int, message_id: int, text: str) -> None: ...


def split_at_sentence_boundary(text: str, max_len: int = _DEFAULT_MAX_MESSAGE_LEN) -> list[str]:
    """Split `text` into chunks of at most `max_len` chars, breaking
    at sentence boundaries (`.`, `!`, `?`, paragraph break) when one
    is available within the last `_HARD_SPLIT_LOOKBACK` chars of the
    window. Falls back to a hard split at `max_len` only when no
    boundary is reachable — guards the spec halt condition that
    long-message split MUST prefer sentence boundaries.

    Whitespace at chunk borders is collapsed: each output chunk is
    `.rstrip()`ed at its trailing edge and the remainder gets
    `.lstrip()`ed before the next iteration. This means a leading
    space inherited from `". "` between sentences does not show up
    in the next chunk.
    """
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remainder = text
    while len(remainder) > max_len:
        window = remainder[: max_len + 1]
        boundary_idx = -1

        # Paragraph break wins when present — preserves visual structure.
        para_idx = window.rfind('\n\n')
        if para_idx != -1:
            boundary_idx = para_idx + 2

        if boundary_idx == -1:
            # Rightmost single-char sentence terminator.
            for ch in '.!?':
                idx = window.rfind(ch)
                if idx > boundary_idx:
                    boundary_idx = idx + 1

        # Hard-split fallback: no boundary, OR boundary fell so far back
        # in the window that splitting there would waste most of the
        # max_len budget. Spec halt condition: prefer sentence boundary
        # but accept hard split as last resort.
        if boundary_idx <= 0 or boundary_idx < max_len - _HARD_SPLIT_LOOKBACK:
            boundary_idx = max_len

        chunk = remainder[:boundary_idx].rstrip()
        if chunk:
            chunks.append(chunk)
        remainder = remainder[boundary_idx:].lstrip()

    if remainder:
        chunks.append(remainder)
    return chunks


@dataclass
class StreamingSession:
    """One Telegram chat's in-flight streaming state for a single
    dispatcher turn. Holds the placeholder message id, the throttle
    clock, the finalize flag, and a reference to the TelegramBot
    wrapper so the dispatcher can drive `update(text)` / `finalize(text)`
    without threading a bot ref through every call site.

    Lifecycle:
      1. Caller constructs a session per `dispatcher.handle()` call,
         passing in the chat_id and the bot wrapper.
      2. Each progress beat calls `update("Thinking..."/stage)`. The
         first update sends a new placeholder message; later updates
         edit it in place. Edits faster than `edit_throttle_seconds`
         are silently dropped.
      3. After the LLM produces final text, caller calls
         `finalize(full_text)`. Long text is split at sentence
         boundaries; the placeholder is replaced with chunk[0] and
         remaining chunks land as fresh messages.
      4. `final_sent` flips True after finalize, so a misfiring
         caller cannot finalize twice; the post-dispatcher caller
         reads this flag to decide whether to skip its own text-send.
    """

    chat_id: int
    telegram_bot: TelegramBotLike
    placeholder_message_id: Optional[int] = None
    last_edit_at: float = 0.0
    edit_throttle_seconds: float = _DEFAULT_EDIT_THROTTLE_SECONDS
    final_sent: bool = False
    time_provider: Callable[[], float] = field(default=time.monotonic)

    async def update(self, text: str) -> None:
        if self.final_sent:
            # Caller bug: trying to update after finalize. Silently no-op
            # rather than raising — streaming is a best-effort UX layer
            # and shouldn't crash the dispatcher's main path.
            return
        now = self.time_provider()
        if now - self.last_edit_at < self.edit_throttle_seconds:
            # Throttled. Skip this beat. Hard cap = 2 events/sec.
            return

        if self.placeholder_message_id is None:
            try:
                self.placeholder_message_id = await self.telegram_bot.send_text(
                    chat_id=self.chat_id, text=text,
                )
            except TelegramError as exc:
                logger.warning(
                    'streaming_send_placeholder_failed',
                    extra={'chat_id': self.chat_id, 'error_class': type(exc).__name__},
                )
                # Leave placeholder None so the next update tries again.
        else:
            try:
                await self.telegram_bot.edit_text(
                    chat_id=self.chat_id,
                    message_id=self.placeholder_message_id,
                    text=text,
                )
            except TelegramError as exc:
                # Fall back: clear placeholder so the next call sends
                # fresh. Common cause: user deleted the placeholder
                # mid-stream, or Telegram returned "message not modified."
                logger.warning(
                    'streaming_edit_failed_falling_back',
                    extra={'chat_id': self.chat_id, 'error_class': type(exc).__name__},
                )
                self.placeholder_message_id = None

        self.last_edit_at = now

    async def finalize(self, full_text: str) -> None:
        if self.final_sent:
            return
        chunks = split_at_sentence_boundary(full_text) or ['']

        if self.placeholder_message_id is None:
            for chunk in chunks:
                try:
                    await self.telegram_bot.send_text(chat_id=self.chat_id, text=chunk)
                except TelegramError as exc:
                    logger.warning(
                        'streaming_finalize_send_failed',
                        extra={'chat_id': self.chat_id, 'error_class': type(exc).__name__},
                    )
        else:
            first, *rest = chunks
            try:
                await self.telegram_bot.edit_text(
                    chat_id=self.chat_id,
                    message_id=self.placeholder_message_id,
                    text=first,
                )
            except TelegramError as exc:
                logger.warning(
                    'streaming_finalize_edit_failed_falling_back',
                    extra={'chat_id': self.chat_id, 'error_class': type(exc).__name__},
                )
                try:
                    await self.telegram_bot.send_text(chat_id=self.chat_id, text=first)
                except TelegramError:
                    logger.warning(
                        'streaming_finalize_send_fallback_also_failed',
                        extra={'chat_id': self.chat_id},
                    )
            for chunk in rest:
                try:
                    await self.telegram_bot.send_text(chat_id=self.chat_id, text=chunk)
                except TelegramError:
                    logger.warning(
                        'streaming_finalize_continuation_failed',
                        extra={'chat_id': self.chat_id},
                    )

        self.final_sent = True
