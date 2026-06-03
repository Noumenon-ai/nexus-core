"""Step 2 — in-memory per-user thread context.

Last-5-turns ring buffer per user, used by ReasoningAdapter so Claude
can resolve "nah not him" / "wait friday" style corrections against the
recent conversation. In memory only; the canonical conversation
history still lives in conversation_turns_repository — this is a fast
short-lived shadow tuned for reasoning.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Iterable


_MAX_TURNS = 5
_ROLE_USER = 'user'
_ROLE_ASSISTANT = 'assistant'
_VALID_ROLES = frozenset({_ROLE_USER, _ROLE_ASSISTANT, 'system'})


@dataclass(slots=True)
class ThreadTurn:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {'role': self.role, 'content': self.content}


class ThreadContextStore:
    """Process-local, per-user rolling thread.

    Thread-safe via RLock; deque(maxlen=N) drops the oldest turn when
    we append a sixth. Bounded memory per user.
    """

    def __init__(self, *, max_turns: int = _MAX_TURNS) -> None:
        if max_turns < 1:
            raise ValueError('max_turns must be >= 1')
        self._max_turns = max_turns
        self._threads: dict[str, deque[ThreadTurn]] = {}
        self._lock = RLock()

    @property
    def max_turns(self) -> int:
        return self._max_turns

    def append_user(self, user_id: str, content: str) -> None:
        self._append(user_id, role=_ROLE_USER, content=content)

    def append_assistant(self, user_id: str, content: str) -> None:
        self._append(user_id, role=_ROLE_ASSISTANT, content=content)

    def append_turn(self, user_id: str, *, role: str, content: str) -> None:
        if role not in _VALID_ROLES:
            raise ValueError(f'invalid role: {role!r}')
        self._append(user_id, role=role, content=content)

    def get(self, user_id: str) -> list[dict[str, str]]:
        """Return turns oldest → newest in dict shape for the reasoning adapter."""
        with self._lock:
            buf = self._threads.get(user_id)
            if not buf:
                return []
            return [turn.to_dict() for turn in buf]

    def clear(self, user_id: str) -> None:
        with self._lock:
            self._threads.pop(user_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._threads.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._threads)

    def _append(self, user_id: str, *, role: str, content: str) -> None:
        clean = (content or '').strip()
        if not clean:
            return
        with self._lock:
            buf = self._threads.get(user_id)
            if buf is None:
                buf = deque(maxlen=self._max_turns)
                self._threads[user_id] = buf
            buf.append(ThreadTurn(role=role, content=clean))


# Module-level default store. Most callers should use this; tests
# instantiate their own ThreadContextStore for isolation.
default_store = ThreadContextStore()


def remember_turn_pair(
    store: ThreadContextStore,
    *,
    user_id: str,
    user_text: str,
    assistant_text: str | None,
) -> None:
    """Convenience for the common case after each pipeline turn."""
    store.append_user(user_id, user_text)
    if assistant_text:
        store.append_assistant(user_id, assistant_text)


def thread_for_reasoning(
    store: ThreadContextStore,
    *,
    user_id: str,
    extra_turns: Iterable[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Return the user's thread plus any caller-supplied extras (oldest first).

    `extra_turns` lets a caller graft turns from the canonical
    conversation-turns repository when the in-memory store is empty
    (post-restart). Extras are appended AFTER the store's contents so
    the most-recent turn stays last.
    """
    base = store.get(user_id)
    if extra_turns:
        for turn in extra_turns:
            role = str(turn.get('role') or '').strip()
            content = str(turn.get('content') or '').strip()
            if role not in _VALID_ROLES or not content:
                continue
            base.append({'role': role, 'content': content})
    # Trim to the configured window in case extras overflowed
    if len(base) > store.max_turns:
        base = base[-store.max_turns:]
    return base
