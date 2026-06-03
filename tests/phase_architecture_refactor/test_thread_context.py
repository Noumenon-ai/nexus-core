"""Tests for the per-user thread context store (Step 2)."""
from __future__ import annotations

import pytest
from threading import Thread

from services.thread_context import (
    ThreadContextStore,
    remember_turn_pair,
    thread_for_reasoning,
)


def test_empty_store_returns_empty_thread():
    store = ThreadContextStore()
    assert store.get('owner') == []


def test_append_user_and_assistant_roundtrip():
    store = ThreadContextStore()
    store.append_user('owner', 'remind sam tomorrow morning about vitamins')
    store.append_assistant('owner', 'got it, 9am tomorrow.')
    out = store.get('owner')
    assert out == [
        {'role': 'user', 'content': 'remind sam tomorrow morning about vitamins'},
        {'role': 'assistant', 'content': 'got it, 9am tomorrow.'},
    ]


def test_per_user_isolation():
    store = ThreadContextStore()
    store.append_user('owner', 'hi from owner')
    store.append_user('sam', 'hi from sam')
    assert store.get('owner') == [{'role': 'user', 'content': 'hi from owner'}]
    assert store.get('sam') == [{'role': 'user', 'content': 'hi from sam'}]


def test_ring_buffer_drops_oldest_at_five():
    store = ThreadContextStore()
    for i in range(7):
        store.append_user('owner', f'turn {i}')
    out = store.get('owner')
    assert len(out) == 5
    # Oldest two dropped; newest five preserved (oldest → newest)
    assert out[0]['content'] == 'turn 2'
    assert out[-1]['content'] == 'turn 6'


def test_blank_content_is_ignored():
    store = ThreadContextStore()
    store.append_user('owner', '   ')
    store.append_assistant('owner', '')
    assert store.get('owner') == []


def test_invalid_role_raises():
    store = ThreadContextStore()
    with pytest.raises(ValueError):
        store.append_turn('owner', role='tool', content='hi')


def test_clear_removes_thread_for_one_user_only():
    store = ThreadContextStore()
    store.append_user('owner', 'a')
    store.append_user('sam', 'b')
    store.clear('owner')
    assert store.get('owner') == []
    assert store.get('sam') == [{'role': 'user', 'content': 'b'}]


def test_clear_all_resets_every_user():
    store = ThreadContextStore()
    store.append_user('owner', 'a')
    store.append_user('sam', 'b')
    store.clear_all()
    assert store.get('owner') == []
    assert store.get('sam') == []


def test_remember_turn_pair_appends_both_when_assistant_text_given():
    store = ThreadContextStore()
    remember_turn_pair(
        store, user_id='owner',
        user_text='turn',
        assistant_text='ok',
    )
    assert store.get('owner') == [
        {'role': 'user', 'content': 'turn'},
        {'role': 'assistant', 'content': 'ok'},
    ]


def test_remember_turn_pair_skips_assistant_when_blank():
    store = ThreadContextStore()
    remember_turn_pair(store, user_id='owner', user_text='turn', assistant_text=None)
    assert store.get('owner') == [{'role': 'user', 'content': 'turn'}]


def test_thread_for_reasoning_grafts_extras_after_store():
    store = ThreadContextStore()
    store.append_user('owner', 'recent in-memory turn')
    out = thread_for_reasoning(
        store,
        user_id='owner',
        extra_turns=[
            {'role': 'user', 'content': 'newer turn from DB'},
            {'role': 'assistant', 'content': 'newer reply from DB'},
        ],
    )
    # Store turns first (oldest), then grafted extras (newest)
    assert out[0]['content'] == 'recent in-memory turn'
    assert out[-1]['content'] == 'newer reply from DB'
    assert len(out) == 3


def test_thread_for_reasoning_trims_to_window():
    store = ThreadContextStore()
    for i in range(3):
        store.append_user('owner', f'store turn {i}')
    extras = [
        {'role': 'user', 'content': f'extra turn {i}'} for i in range(5)
    ]
    out = thread_for_reasoning(store, user_id='owner', extra_turns=extras)
    # 3 store + 5 extras = 8, must trim to 5 newest
    assert len(out) == 5
    assert out[-1]['content'] == 'extra turn 4'


def test_thread_for_reasoning_ignores_invalid_extras():
    store = ThreadContextStore()
    out = thread_for_reasoning(
        store, user_id='owner',
        extra_turns=[
            {'role': 'invalid', 'content': 'nope'},
            {'role': 'user', 'content': '   '},
            {'role': 'user', 'content': 'kept'},
        ],
    )
    assert out == [{'role': 'user', 'content': 'kept'}]


def test_concurrent_appends_do_not_drop_data():
    store = ThreadContextStore(max_turns=200)

    def worker(label: str):
        for i in range(20):
            store.append_user(label, f'{label}-{i}')

    threads = [Thread(target=worker, args=(f'user{n}',)) for n in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for n in range(5):
        assert len(store.get(f'user{n}')) == 20
