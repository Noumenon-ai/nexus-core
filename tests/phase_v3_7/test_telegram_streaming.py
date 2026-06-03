"""V3.7 streaming session unit tests (step 1 of 2 — module isolation).

Covers:
- update() throttle: hard cap 2 events / sec via edit_throttle_seconds=0.5
- update() fallback: TelegramError on edit clears placeholder so next
  call sends fresh
- finalize() splits long messages at sentence boundaries; chunks
  ≤ max_len; first chunk replaces placeholder, rest sent as new
- finalize() with short text edits placeholder once
- split_at_sentence_boundary() prefers `.`/`!`/`?`
- split_at_sentence_boundary() falls back to hard split when no
  boundary exists in the lookback window

Dispatcher integration tests live in test_dispatcher_streaming_*
(landing in V3.7 step 2).
"""
from __future__ import annotations

from typing import Any

import pytest

from services.telegram_streaming import (
    StreamingSession,
    TelegramError,
    split_at_sentence_boundary,
)


class _FakeTelegramBot:
    """Records every send_text / edit_text call for assertions.
    Each `send_text` returns a deterministic monotonically-increasing
    message_id so the session has something to remember."""

    def __init__(self, *, edit_raises: type[BaseException] | None = None,
                 send_raises_after: int | None = None):
        self.send_calls: list[dict[str, Any]] = []
        self.edit_calls: list[dict[str, Any]] = []
        self._next_id = 1000
        self._edit_raises = edit_raises
        self._send_raises_after = send_raises_after

    async def send_text(self, *, chat_id: int, text: str) -> int:
        self.send_calls.append({'chat_id': chat_id, 'text': text})
        if self._send_raises_after is not None and len(self.send_calls) > self._send_raises_after:
            raise TelegramError('synthetic send failure')
        msg_id = self._next_id
        self._next_id += 1
        return msg_id

    async def edit_text(self, *, chat_id: int, message_id: int, text: str) -> None:
        self.edit_calls.append({'chat_id': chat_id, 'message_id': message_id, 'text': text})
        if self._edit_raises is not None:
            raise self._edit_raises('synthetic edit failure')


# ---- update() throttle ------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_session_throttles_edits():
    """Fire 10 updates over 0.9 simulated seconds with throttle=0.5s.
    Total wrapper events (send placeholder + edits combined) must be
    ≤ 2 — that is the hard 2 events / sec cap from the spec.
    """
    bot = _FakeTelegramBot()
    timestamps = iter([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
    session = StreamingSession(chat_id=42, telegram_bot=bot, time_provider=lambda: next(timestamps))

    for i in range(10):
        await session.update(f'beat {i}')

    total_events = len(bot.send_calls) + len(bot.edit_calls)
    assert total_events <= 2, (
        f'Expected ≤2 wrapper events (throttle = 2/sec), got {total_events}: '
        f'sends={bot.send_calls}, edits={bot.edit_calls}'
    )
    # First event must be the placeholder send.
    assert len(bot.send_calls) == 1
    # Second event (if any) must be an edit, not a duplicate send.
    if total_events == 2:
        assert len(bot.edit_calls) == 1


@pytest.mark.asyncio
async def test_streaming_session_throttle_first_update_fires_through():
    """Sanity check: with `last_edit_at=0.0` default and a non-zero
    time provider, the very first update must NOT be skipped — the
    user has to see the placeholder appear or streaming has no value."""
    bot = _FakeTelegramBot()
    session = StreamingSession(chat_id=42, telegram_bot=bot, time_provider=lambda: 100.0)

    await session.update('Thinking...')

    assert len(bot.send_calls) == 1
    assert bot.send_calls[0]['text'] == 'Thinking...'
    assert session.placeholder_message_id is not None


# ---- update() edit-failure fallback ----------------------------------------

@pytest.mark.asyncio
async def test_streaming_session_falls_back_on_edit_failure():
    """When edit_text raises a TelegramError, the next update must
    send a new placeholder rather than keep trying to edit a
    message that may no longer exist (user deleted it / Telegram
    returned 'message not modified' / etc)."""
    bot = _FakeTelegramBot(edit_raises=TelegramError)
    timestamps = iter([10.0, 10.6, 11.2])  # all past throttle
    session = StreamingSession(chat_id=42, telegram_bot=bot, time_provider=lambda: next(timestamps))

    await session.update('Thinking...')   # send placeholder
    await session.update('Still working...')  # edit -> raises -> fallback
    await session.update('Almost there...')   # placeholder cleared -> send fresh

    assert len(bot.send_calls) == 2, (
        f'Expected 2 sends (initial placeholder + fallback after edit failure), '
        f'got {len(bot.send_calls)}: {bot.send_calls}'
    )
    assert len(bot.edit_calls) == 1  # only the failed edit attempt


# ---- finalize() ------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_session_finalize_edits_placeholder_with_short_message():
    """finalize() with a short reply must edit the existing
    placeholder rather than send a brand-new message."""
    bot = _FakeTelegramBot()
    timestamps = iter([1.0, 100.0])
    session = StreamingSession(chat_id=42, telegram_bot=bot, time_provider=lambda: next(timestamps))
    await session.update('Thinking...')  # set placeholder
    placeholder_id = session.placeholder_message_id

    short_reply = 'Here is your reply.'
    await session.finalize(short_reply)

    assert session.final_sent is True
    # No new send during finalize — only the edit.
    assert len(bot.send_calls) == 1  # the original placeholder
    assert len(bot.edit_calls) == 1
    assert bot.edit_calls[0]['message_id'] == placeholder_id
    assert bot.edit_calls[0]['text'] == short_reply


@pytest.mark.asyncio
async def test_streaming_session_finalize_splits_long_messages():
    """Spec test: finalize with 8000-char text → 2+ messages, each
    ≤ 4000 chars, splits at sentence boundaries (no mid-word).

    Build a long text out of complete sentences so the boundary
    finder always has a `.` available.
    """
    bot = _FakeTelegramBot()
    session = StreamingSession(chat_id=42, telegram_bot=bot, time_provider=lambda: 100.0)
    await session.update('Thinking...')

    sentence = 'This is a sentence with some content. '  # 38 chars
    long_text = sentence * 220  # ~8360 chars, sentence-aligned
    assert len(long_text) > 8000

    await session.finalize(long_text)

    # Edit replaces placeholder with first chunk.
    assert len(bot.edit_calls) == 1
    first_chunk = bot.edit_calls[0]['text']
    assert len(first_chunk) <= 4000
    # Rest sent as fresh messages.
    follow_up_chunks = [c['text'] for c in bot.send_calls[1:]]
    assert len(follow_up_chunks) >= 1
    for chunk in follow_up_chunks:
        assert len(chunk) <= 4000
    # No mid-word break: every chunk ends at a sentence terminator
    # (allowing trailing period since rstrip preserves it).
    for chunk in [first_chunk, *follow_up_chunks]:
        assert chunk.rstrip()[-1] in '.!?', (
            f'Chunk does not end at sentence boundary: ...{chunk[-30:]!r}'
        )


@pytest.mark.asyncio
async def test_streaming_session_finalize_without_placeholder_sends_all_as_new():
    """When update() never fired (or fell back so placeholder is None
    again), finalize() sends every chunk as a fresh message — never
    tries to edit a non-existent message_id."""
    bot = _FakeTelegramBot()
    session = StreamingSession(chat_id=42, telegram_bot=bot, time_provider=lambda: 100.0)
    # Skip update() entirely.
    await session.finalize('Hello world.')

    assert session.final_sent is True
    assert len(bot.edit_calls) == 0
    assert len(bot.send_calls) == 1
    assert bot.send_calls[0]['text'] == 'Hello world.'


# ---- split_at_sentence_boundary -------------------------------------------

def test_split_at_sentence_boundary_breaks_at_period():
    """Spec test: 'First. Second. Third.' with max_len=10 → ["First.", "Second.", "Third."]"""
    result = split_at_sentence_boundary('First. Second. Third.', max_len=10)
    assert result == ['First.', 'Second.', 'Third.']


def test_split_at_sentence_boundary_falls_back_to_hard_split():
    """Spec test: text with no sentence boundary in 4000 chars must
    hard-split rather than crash or balloon to one chunk. This is
    the last-resort fallback halt-condition path."""
    boundary_free = 'a' * 5000  # zero `.!?\n\n`
    result = split_at_sentence_boundary(boundary_free, max_len=4000)

    assert len(result) >= 2
    assert len(result[0]) == 4000  # hit hard-split exactly at max_len
    # Total length preserved across chunks (no character loss).
    assert sum(len(c) for c in result) == 5000


def test_split_at_sentence_boundary_short_text_returns_unchanged():
    """Inputs ≤ max_len pass through as a single-chunk list — no
    artificial chunking when the text fits in one Telegram message."""
    text = 'Short reply.'
    assert split_at_sentence_boundary(text, max_len=4000) == [text]


def test_split_at_sentence_boundary_prefers_paragraph_break_when_present():
    """When a paragraph break (\\n\\n) is available before max_len, it
    wins over a single-char sentence terminator — preserves the
    visual structure of multi-paragraph replies."""
    text = 'Para one. End sentence.\n\nPara two starts here. More content.'
    # max_len chosen so both '.' and '\n\n' fall in the window.
    chunks = split_at_sentence_boundary(text, max_len=30)
    # First chunk should end at the paragraph break, not at the
    # earlier '.' boundary.
    assert chunks[0] == 'Para one. End sentence.'
