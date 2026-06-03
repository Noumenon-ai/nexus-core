from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry


class ScriptedLLM:
    def __init__(self, script: list[dict[str, Any]]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
        self.calls.append({
            'user_id': user_id,
            'system_prompt': system_prompt,
            'contents': [item for item in contents],
            'tool_catalog': tool_catalog,
        })
        return self.script.pop(0)


class RecordingMem0:
    def __init__(self, *, search_result: list[Any] | None = None):
        self.search_calls: list[dict[str, Any]] = []
        self._search_result = search_result or []

    def search(self, query, *, user_id, limit: int = 5):
        self.search_calls.append({'query': query, 'user_id': user_id, 'limit': limit})
        return self._search_result

    def add(self, messages, *, user_id):
        return None


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


def _user(container, telegram_id: int):
    return container.users_repository.get_or_create(telegram_id)


def _insert_turn(repo, *, user_id: str, role: str, content: str, conversation_id: str, created_at: datetime) -> None:
    repo.insert(
        user_id=user_id,
        role=role,
        content=content,
        conversation_id=conversation_id,
        created_at=created_at,
    )


def _make_dispatcher(container, telos_service, llm, mem0) -> ToolDispatcher:
    return ToolDispatcher(
        llm=llm,
        registry=ToolRegistry(),
        telos_service=telos_service,
        mem0=mem0,
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
        app_timezone='America/New_York',
    )


def test_list_recent_for_user_returns_last_n_in_chronological_order(container):
    repo = container.conversation_turns_repository
    user = _user(container, 111)
    base = datetime.now(timezone.utc) - timedelta(minutes=40)

    for idx in range(25):
        _insert_turn(
            repo,
            user_id=user.id,
            role='user' if idx % 2 == 0 else 'assistant',
            content=f'prior-{idx + 1:02d}',
            conversation_id='conv-active',
            created_at=base + timedelta(minutes=idx),
        )

    turns = repo.list_recent_for_user(user_id=user.id, limit=20, conversation_id='conv-active')

    assert [turn.content for turn in turns] == [f'prior-{idx:02d}' for idx in range(6, 26)]


def test_list_recent_for_user_scopes_to_user_and_conversation(container):
    repo = container.conversation_turns_repository
    user_a = _user(container, 111)
    user_b = _user(container, 222)
    base = datetime.now(timezone.utc) - timedelta(hours=4)

    _insert_turn(repo, user_id=user_a.id, role='user', content='old-a-1', conversation_id='conv-old', created_at=base)
    _insert_turn(repo, user_id=user_a.id, role='assistant', content='old-a-2', conversation_id='conv-old', created_at=base + timedelta(minutes=1))
    _insert_turn(repo, user_id=user_a.id, role='user', content='new-a-1', conversation_id='conv-new', created_at=base + timedelta(hours=3))
    _insert_turn(repo, user_id=user_a.id, role='assistant', content='new-a-2', conversation_id='conv-new', created_at=base + timedelta(hours=3, minutes=1))
    _insert_turn(repo, user_id=user_b.id, role='user', content='b-1', conversation_id='conv-new', created_at=base + timedelta(hours=3, minutes=2))

    turns = repo.list_recent_for_user(user_id=user_a.id, limit=20, conversation_id='conv-new')

    assert [turn.content for turn in turns] == ['new-a-1', 'new-a-2']


@pytest.mark.asyncio
async def test_dispatcher_loads_last_20_archived_turns_into_first_llm_call(container, telos_service):
    repo = container.conversation_turns_repository
    user = _user(container, 111)
    base = datetime.now(timezone.utc) - timedelta(minutes=30)

    for idx in range(25):
        _insert_turn(
            repo,
            user_id=user.id,
            role='user' if idx % 2 == 0 else 'assistant',
            content=f'prior-{idx + 1:02d}',
            conversation_id='conv-active',
            created_at=base + timedelta(minutes=idx),
        )

    llm = ScriptedLLM([{'text': 'reply text'}])
    dispatcher = _make_dispatcher(container, telos_service, llm, RecordingMem0())

    await dispatcher.handle(DispatcherInput(user=user, text='what was that reminder for?'))

    first_call_contents = llm.calls[0]['contents']
    texts = [item['parts'][0]['text'] for item in first_call_contents]

    assert len(first_call_contents) == 21
    assert texts[0] == 'prior-06'
    assert texts[-2] == 'prior-25'
    assert texts[-1] == 'what was that reminder for?'
    assert 'prior-05' not in texts
    assert {item['role'] for item in first_call_contents} == {'user', 'model'}


@pytest.mark.asyncio
async def test_dispatcher_preserves_all_20_prior_turns_when_current_turn_is_archived_first(container, telos_service):
    repo = container.conversation_turns_repository
    user = _user(container, 333)
    base = datetime.now(timezone.utc) - timedelta(minutes=22)

    for idx in range(20):
        _insert_turn(
            repo,
            user_id=user.id,
            role='user' if idx % 2 == 0 else 'assistant',
            content=f'prior-{idx + 1:02d}',
            conversation_id='conv-boundary',
            created_at=base + timedelta(minutes=idx),
        )

    llm = ScriptedLLM([{'text': 'reply text'}])
    dispatcher = _make_dispatcher(container, telos_service, llm, RecordingMem0())

    await dispatcher.handle(DispatcherInput(user=user, text='what was that reminder for?'))

    texts = [item['parts'][0]['text'] for item in llm.calls[0]['contents']]

    assert len(texts) == 21
    assert texts[:-1] == [f'prior-{idx:02d}' for idx in range(1, 21)]
    assert texts[-1] == 'what was that reminder for?'
    assert texts.count('what was that reminder for?') == 1


@pytest.mark.asyncio
async def test_dispatcher_injects_mem0_results_and_recent_turns_for_followup(container, telos_service):
    repo = container.conversation_turns_repository
    user = _user(container, 111)
    base = datetime.now(timezone.utc) - timedelta(minutes=3)

    _insert_turn(repo, user_id=user.id, role='user', content='set a reminder for noom', conversation_id='conv-reminder', created_at=base)
    _insert_turn(repo, user_id=user.id, role='assistant', content='I set a reminder for noom.', conversation_id='conv-reminder', created_at=base + timedelta(seconds=30))

    mem0 = RecordingMem0(search_result=[
        {'memory': 'Reminder context: the reminder was for noom.'},
        {'memory': 'User asked for a follow-up about that reminder.'},
    ])
    llm = ScriptedLLM([{'text': 'It was for noom.'}])
    dispatcher = _make_dispatcher(container, telos_service, llm, mem0)

    out = await dispatcher.handle(DispatcherInput(user=user, text='what was that reminder for?'))

    assert out.text == 'It was for noom.'
    assert mem0.search_calls == [{'query': 'what was that reminder for?', 'user_id': user.id, 'limit': 5}]

    first_call = llm.calls[0]
    assert 'Reminder context: the reminder was for noom.' in first_call['system_prompt']
    assert 'User asked for a follow-up about that reminder.' in first_call['system_prompt']
    assert [item['parts'][0]['text'] for item in first_call['contents']] == [
        'set a reminder for noom',
        'I set a reminder for noom.',
        'what was that reminder for?',
    ]
    assert [item['role'] for item in first_call['contents']] == ['user', 'model', 'user']
