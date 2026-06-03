"""V3.6 ConversationTurnsRepository tests.

Covers:
- insert returns turn_id and persists row
- insert rejects role values outside ('user', 'assistant')
- resolve_conversation_id returns fresh UUID for first turn
- 2-hour silence rule: continues conversation_id within window
- 2-hour silence rule: advances conversation_id after gap
- per-user isolation: User A's turns never bridge to User B
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session

from models import ConversationTurn, User
from repositories.conversation_turns_repository import (
    CONVERSATION_SILENCE_GAP,
    ConversationTurnsRepository,
)


@pytest.fixture
def repo(container):
    return ConversationTurnsRepository(container.database.session_factory)


def _make_user(container, telegram_id: int) -> User:
    return container.users_repository.get_or_create(telegram_id)


def test_insert_user_turn_returns_turn_id_and_persists(container, repo):
    user = _make_user(container, 111)
    now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)

    turn_id = repo.insert(
        user_id=user.id,
        role='user',
        content='hello',
        conversation_id='conv-1',
        created_at=now,
    )

    assert turn_id
    with Session(container.database.engine) as session:
        row = session.get(ConversationTurn, turn_id)
        assert row is not None
        assert row.user_id == user.id
        assert row.role == 'user'
        assert row.content == 'hello'
        assert row.conversation_id == 'conv-1'
        assert row.mem0_persisted_at is None
        assert row.mem0_memory_id is None


def test_insert_rejects_disallowed_role(container, repo):
    user = _make_user(container, 111)
    with pytest.raises(ValueError):
        repo.insert(
            user_id=user.id,
            role='system',
            content='nope',
            conversation_id='conv-1',
        )


def test_resolve_returns_fresh_id_when_no_history(container, repo):
    user = _make_user(container, 111)
    now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
    conv_id = repo.resolve_conversation_id(user_id=user.id, now=now)
    assert conv_id  # any non-empty UUID

    # Verify it's different on a second call when no turn was written.
    conv_id_2 = repo.resolve_conversation_id(user_id=user.id, now=now)
    assert conv_id != conv_id_2


def test_resolve_continues_id_within_silence_gap(container, repo):
    user = _make_user(container, 111)
    base = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
    repo.insert(
        user_id=user.id,
        role='user',
        content='hi',
        conversation_id='conv-A',
        created_at=base,
    )

    # 1h59m later — still within the 2h gap
    later = base + timedelta(hours=1, minutes=59)
    resolved = repo.resolve_conversation_id(user_id=user.id, now=later)
    assert resolved == 'conv-A'


def test_resolve_advances_id_after_silence_gap(container, repo):
    user = _make_user(container, 111)
    base = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
    repo.insert(
        user_id=user.id,
        role='user',
        content='hi',
        conversation_id='conv-A',
        created_at=base,
    )

    # Just past 2h silence
    later = base + CONVERSATION_SILENCE_GAP + timedelta(seconds=1)
    resolved = repo.resolve_conversation_id(user_id=user.id, now=later)
    assert resolved != 'conv-A'


def test_resolve_ignores_assistant_turns_when_measuring_silence(container, repo):
    """Silence gap is measured between user-turns only — an assistant turn
    inserted shortly before the gap-edge must NOT reset the clock."""
    user = _make_user(container, 111)
    base = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
    repo.insert(
        user_id=user.id,
        role='user',
        content='hi',
        conversation_id='conv-A',
        created_at=base,
    )
    # Assistant turn inserted 1 hour later — would reset gap if buggy
    repo.insert(
        user_id=user.id,
        role='assistant',
        content='hello back',
        conversation_id='conv-A',
        created_at=base + timedelta(hours=1),
    )

    # Now query at base + 2h05m — past gap measured from user-turn,
    # but only 1h05m past the assistant turn.
    later = base + timedelta(hours=2, minutes=5)
    resolved = repo.resolve_conversation_id(user_id=user.id, now=later)
    assert resolved != 'conv-A'


def test_resolve_is_isolated_per_user(container, repo):
    """User A's recent turn must not affect User B's resolution."""
    user_a = _make_user(container, 111)
    user_b = _make_user(container, 222)
    now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)

    # User A starts conv-A
    repo.insert(
        user_id=user_a.id,
        role='user',
        content='A speaking',
        conversation_id='conv-A',
        created_at=now,
    )

    # User B's first turn — must NOT inherit conv-A
    resolved_b = repo.resolve_conversation_id(user_id=user_b.id, now=now)
    assert resolved_b != 'conv-A'
