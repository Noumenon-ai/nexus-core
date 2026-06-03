"""Step 5 — audit_events table + repository + logger."""
from __future__ import annotations

import pytest

from repositories.audit_events_repository import AuditEventsRepository
from services.audit_logger import AuditLogger, StopWatch


def _repo(container) -> AuditEventsRepository:
    return AuditEventsRepository(container.database.session_factory)


def test_table_exists_after_bootstrap(container):
    # The bootstrap migration ran in conftest; table should be queryable.
    repo = _repo(container)
    assert repo.count() == 0


def test_write_inserts_row_and_returns_id(container):
    repo = _repo(container)
    user = container.users_repository.get_or_create(123)
    row_id = repo.write(
        user_id=user.id,
        intent='create_reminder',
        parameters={'message': 'take vitamins', 'time': '9am'},
        result='ok',
        response_time_ms=42,
        provider='claude',
    )
    assert row_id is not None
    assert repo.count() == 1
    rows = repo.list_recent(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row.user_id == user.id
    assert row.intent == 'create_reminder'
    assert row.result == 'ok'
    assert row.response_time_ms == 42
    assert row.provider == 'claude'
    assert row.parameters == {'message': 'take vitamins', 'time': '9am'}


def test_write_never_raises_on_bad_parameters(container):
    repo = _repo(container)
    # Lambda is not directly JSON-serializable; the repository uses
    # default=str so the row succeeds with a string repr in place of
    # the non-serializable value. The contract is "write never raises
    # and always inserts a row" — exact value shape is best-effort.
    row_id = repo.write(
        user_id=None,
        intent='odd',
        parameters={'callback': lambda: 1},  # type: ignore[arg-type]
        result='fail',
    )
    assert row_id is not None
    rows = repo.list_recent(limit=10)
    # The row exists. Whatever ended up in parameters is the stringified
    # fallback, not a crash.
    assert len(rows) == 1
    assert 'callback' in rows[0].parameters


def test_write_truncates_long_intent_and_result(container):
    repo = _repo(container)
    long_intent = 'x' * 200
    long_result = 'y' * 200
    repo.write(user_id=None, intent=long_intent, result=long_result)
    rows = repo.list_recent(limit=1)
    assert len(rows[0].intent) == 64
    assert len(rows[0].result) == 64


def test_list_recent_orders_newest_first(container):
    repo = _repo(container)
    user = container.users_repository.get_or_create(124)
    for i in range(5):
        repo.write(user_id=user.id, intent=f'intent_{i}', result='ok')
    rows = repo.list_recent(limit=3)
    assert len(rows) == 3
    assert rows[0].intent == 'intent_4'
    assert rows[1].intent == 'intent_3'
    assert rows[2].intent == 'intent_2'


def test_list_recent_filters_by_user(container):
    repo = _repo(container)
    user_a = container.users_repository.get_or_create(125)
    user_b = container.users_repository.get_or_create(126)
    repo.write(user_id=user_a.id, intent='a1', result='ok')
    repo.write(user_id=user_b.id, intent='b1', result='ok')
    repo.write(user_id=user_a.id, intent='a2', result='ok')
    a_rows = repo.list_recent(user_id=user_a.id)
    assert [r.intent for r in a_rows] == ['a2', 'a1']
    b_rows = repo.list_recent(user_id=user_b.id)
    assert [r.intent for r in b_rows] == ['b1']


# ── AuditLogger ─────────────────────────────────────────────────────────────


def test_logger_write_event_delegates_to_repository(container):
    repo = _repo(container)
    logger = AuditLogger(repository=repo)
    user = container.users_repository.get_or_create(127)
    row_id = logger.write_event(
        user_id=user.id,
        intent='list_reminders',
        parameters={},
        result='ok',
        response_time_ms=15,
        provider='claude',
    )
    assert row_id is not None
    assert repo.count(user_id=user.id) == 1


@pytest.mark.asyncio
async def test_logger_executor_callback_maps_payload(container):
    repo = _repo(container)
    audit = AuditLogger(repository=repo)
    user = container.users_repository.get_or_create(128)
    await audit.write_executor_event({
        'user_id': user.id,
        'intent': 'create_reminder',
        'parameters': {'message': 'water plants'},
        'provider': 'claude',
        'result': 'ok',
    })
    rows = repo.list_recent(user_id=user.id)
    assert rows[0].intent == 'create_reminder'
    assert rows[0].provider == 'claude'
    assert rows[0].parameters == {'message': 'water plants'}


@pytest.mark.asyncio
async def test_logger_executor_callback_never_raises(container):
    # Pass a bogus repository to force the inner exception path.
    class _BadRepo:
        def write(self, **kw):
            raise RuntimeError('database disappeared')

    audit = AuditLogger(repository=_BadRepo())  # type: ignore[arg-type]
    # Must not raise even when the underlying repo blows up.
    await audit.write_executor_event({
        'user_id': 'u', 'intent': 'x', 'result': 'fail',
    })


def test_stopwatch_measures_elapsed_ms(container):
    import time
    with StopWatch() as sw:
        time.sleep(0.015)
    assert sw.elapsed_ms >= 10
