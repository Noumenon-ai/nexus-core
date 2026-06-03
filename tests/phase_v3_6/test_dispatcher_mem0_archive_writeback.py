"""V3.6 checkpoint 2: mem0 persistence updates archive rows on success.

Three cases:

1. test_mem0_persistence_updates_archive_row — mem0.add returns a
   memory_id; both turn rows must have mem0_persisted_at + mem0_memory_id
   set, and they drop out of the mem0_pending partial index.

2. test_mem0_persistence_failure_leaves_pending_marker — mem0.add raises;
   archive rows must remain with mem0_persisted_at IS NULL so a future
   recovery script can find them via the partial index.

3. test_archive_index_makes_pending_query_fast — the
   `idx_conversation_turns_mem0_pending` partial index must be eligible
   for the planner's choice on the canonical pending-query shape; we
   assert via `EXPLAIN QUERY PLAN` that the index is used.

Halt-condition coverage:
- Per-user isolation regression on assistant turns: covered by
  test_mem0_persistence_per_user_isolation_on_assistant_writeback.
- mark_mem0_persisted writes wrong memory_id / wrong turn_id: covered by
  test_mem0_persistence_updates_archive_row's content-and-id assertion.
"""
from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from models import ConversationTurn
from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.read_tools import register_read_tools
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry


class _ScriptedLLM:
    def __init__(self, replies: list[dict[str, Any]]):
        self._replies = list(replies)

    async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
        return self._replies.pop(0)


class _SuccessMem0:
    """Returns a mem0-shaped success envelope so _extract_memory_id finds an id."""

    def __init__(self, memory_id: str = 'mem0-id-success-1'):
        self.memory_id = memory_id
        self.add_calls: list[dict[str, Any]] = []

    def search(self, query, *, user_id, limit=5):
        return []

    def add(self, messages, *, user_id):
        self.add_calls.append({'messages': messages, 'user_id': user_id})
        return {'results': [{'id': self.memory_id, 'memory': 'snip', 'event': 'ADD'}]}


class _FailingMem0:
    def search(self, query, *, user_id, limit=5):
        return []

    def add(self, messages, *, user_id):
        raise RuntimeError('mem0 down')


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


@pytest.fixture
def registry(container, telos_service):
    reg = ToolRegistry()
    register_read_tools(
        reg,
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        emails_repository=container.emails_repository,
        approvals_repository=container.approvals_repository,
        telos_service=telos_service,
        app_timezone='UTC',
    )
    return reg


def _make_dispatcher(container, registry, telos_service, llm, mem0) -> ToolDispatcher:
    return ToolDispatcher(
        llm=llm,
        registry=registry,
        telos_service=telos_service,
        mem0=mem0,
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )


def _all_turns(container) -> list[ConversationTurn]:
    with Session(container.database.engine) as session:
        return list(session.scalars(select(ConversationTurn).order_by(ConversationTurn.created_at)).all())


@pytest.mark.asyncio
async def test_mem0_persistence_updates_archive_row(container, registry, telos_service):
    """Successful mem0.add must mark BOTH archive rows with the returned
    memory_id and a non-null persisted_at timestamp. After this, the
    pending partial index must return zero pending rows for that user."""
    user = container.users_repository.get_or_create(111)
    mem0 = _SuccessMem0(memory_id='mem0-id-abc')
    dispatcher = _make_dispatcher(
        container, registry, telos_service,
        llm=_ScriptedLLM([{'text': 'hi back'}]),
        mem0=mem0,
    )

    await dispatcher.handle(DispatcherInput(user=user, text='hello'))
    # H2-046 Part 0: archival is fire-and-forget — block until the
    # in-flight task finishes before asserting on the post-archival state.
    await dispatcher.wait_for_archival_idle()

    turns = _all_turns(container)
    assert len(turns) == 2
    for turn in turns:
        assert turn.mem0_persisted_at is not None, f'{turn.role} row not marked persisted'
        assert turn.mem0_memory_id == 'mem0-id-abc', (
            f'{turn.role} row got memory_id {turn.mem0_memory_id!r}, '
            f'expected mem0-id-abc — wrong-id bug surfaces here'
        )

    with Session(container.database.engine) as session:
        pending = session.scalar(text(
            'SELECT COUNT(*) FROM conversation_turns WHERE mem0_persisted_at IS NULL'
        ))
        assert pending == 0


@pytest.mark.asyncio
async def test_mem0_persistence_failure_leaves_pending_marker(container, registry, telos_service):
    """mem0.add raising must NOT propagate (assistant reply still
    delivered) AND must leave both archive rows with mem0_persisted_at
    IS NULL so the partial-index recovery query finds them later."""
    user = container.users_repository.get_or_create(111)
    dispatcher = _make_dispatcher(
        container, registry, telos_service,
        llm=_ScriptedLLM([{'text': 'reply text'}]),
        mem0=_FailingMem0(),
    )

    out = await dispatcher.handle(DispatcherInput(user=user, text='hello'))
    assert out.text == 'reply text'  # mem0 failure does not break user-facing path
    await dispatcher.wait_for_archival_idle()  # H2-046 Part 0: drain background task

    turns = _all_turns(container)
    assert len(turns) == 2
    for turn in turns:
        assert turn.mem0_persisted_at is None, f'{turn.role} unexpectedly marked persisted'
        assert turn.mem0_memory_id is None

    with Session(container.database.engine) as session:
        pending = session.scalar(text(
            'SELECT COUNT(*) FROM conversation_turns WHERE mem0_persisted_at IS NULL'
        ))
        assert pending == 2


def test_archive_index_makes_pending_query_fast(container):
    """EXPLAIN QUERY PLAN over the canonical pending-recovery query
    must show SQLite picking `idx_conversation_turns_mem0_pending`.

    To make the partial index meaningfully more selective than the
    full-user index, we seed many persisted rows and few pending
    rows, then `ANALYZE` so the optimizer has stats to compare.
    With very few rows or no stats, SQLite picks either index
    arbitrarily — that does not prove the partial index is
    *useful*, only that it's present. The seeded mix below mirrors
    the production shape (most rows persisted, a small tail
    awaiting recovery) and is the regime in which the partial
    index actually pays off.
    """
    from datetime import datetime, timedelta, timezone

    user = container.users_repository.get_or_create(111)
    base = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)

    # 200 persisted rows — these are EXCLUDED from the partial index.
    repo = container.conversation_turns_repository
    persisted_ids = []
    for i in range(200):
        tid = repo.insert(
            user_id=user.id, role='assistant',
            content=f'msg-{i}', conversation_id='conv-bulk',
            created_at=base - timedelta(minutes=i),
        )
        persisted_ids.append(tid)
    repo.mark_mem0_persisted(turn_ids=persisted_ids, memory_id='mem0-bulk')

    # 3 pending rows — these are the recovery query's target set.
    for i in range(3):
        repo.insert(
            user_id=user.id, role='user',
            content=f'pending-{i}', conversation_id='conv-pending',
            created_at=base + timedelta(minutes=i),
        )

    with Session(container.database.engine) as session:
        session.execute(text('ANALYZE'))
        session.commit()

    with Session(container.database.engine) as session:
        rows = session.execute(text(
            "EXPLAIN QUERY PLAN "
            "SELECT turn_id FROM conversation_turns "
            "WHERE user_id = :uid AND mem0_persisted_at IS NULL "
            "ORDER BY created_at"
        ), {'uid': user.id}).all()

    plan_blob = ' | '.join(str(row) for row in rows)
    assert 'idx_conversation_turns_mem0_pending' in plan_blob, (
        f'Pending-query plan does not use the partial index. Plan: {plan_blob}'
    )


@pytest.mark.asyncio
async def test_mem0_persistence_per_user_isolation_on_assistant_writeback(container, registry, telos_service):
    """Halt-condition coverage: User A's mem0 success must not mark
    User B's pending turns persisted. Both users hit the same dispatcher
    instance with the same SuccessMem0 — the writeback must be scoped
    to A's turn_ids only, leaving B's turns pending."""
    a = container.users_repository.get_or_create(111)
    b = container.users_repository.get_or_create(222)

    # User B turn lands first via FailingMem0 (mem0 down), then User A
    # via SuccessMem0 (mem0 recovers). B's pending rows must remain
    # pending after A's writeback.
    dispatcher_failing = _make_dispatcher(
        container, registry, telos_service,
        llm=_ScriptedLLM([{'text': 'B reply'}]),
        mem0=_FailingMem0(),
    )
    await dispatcher_failing.handle(DispatcherInput(user=b, text='B speaking'))
    await dispatcher_failing.wait_for_archival_idle()

    dispatcher_success = _make_dispatcher(
        container, registry, telos_service,
        llm=_ScriptedLLM([{'text': 'A reply'}]),
        mem0=_SuccessMem0(memory_id='mem0-id-only-A'),
    )
    await dispatcher_success.handle(DispatcherInput(user=a, text='A speaking'))
    await dispatcher_success.wait_for_archival_idle()

    turns = _all_turns(container)
    by_user = {t.user_id: [] for t in turns}
    for t in turns:
        by_user[t.user_id].append(t)

    # User B: both pending
    for t in by_user[b.id]:
        assert t.mem0_persisted_at is None, f'B {t.role} bridged to A writeback'
        assert t.mem0_memory_id is None

    # User A: both persisted with correct id
    for t in by_user[a.id]:
        assert t.mem0_persisted_at is not None
        assert t.mem0_memory_id == 'mem0-id-only-A'
