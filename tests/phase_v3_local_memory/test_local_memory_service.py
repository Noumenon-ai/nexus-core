"""H2-046 — LocalMemoryService unit + curated-recall tests.

Two layers of coverage:

  1. **Storage/API contract** — round-trip semantics, user isolation, schema
     stability, mem0 envelope shape. These are deterministic and fast.

  2. **Curated recall** — store a small set of known facts ("user lives in
     Brooklyn", "daughter Sam, age 8") and assert that semantically related
     queries surface them in top-N. Catches gross retrieval quality
     regressions without trying to assert bit-identical scores against any
     other system.

Note on model loading cost: sentence-transformers takes ~10s to initialize
cold and downloads ~80MB on first call. Subsequent runs hit the local cache
and reload in ~1s. We accept that cost once per test session via a module-
scoped service fixture rather than a fresh service per test.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_memory_db(tmp_path, monkeypatch):
    """Each test gets its own SQLite file so concurrent runs / leftover state
    can't bleed in."""
    db = tmp_path / 'memory.db'
    monkeypatch.setenv('NEXUS_LOCAL_MEMORY_DB', str(db))
    return db


@pytest.fixture(scope='module')
def shared_model_service():
    """Module-scoped service so we pay the sentence-transformers cold-start
    once rather than per test. DB is throwaway."""
    import tempfile
    from services.local_memory_service import LocalMemoryService
    with tempfile.TemporaryDirectory() as td:
        os.environ['NEXUS_LOCAL_MEMORY_DB'] = str(Path(td) / 'mem.db')
        yield LocalMemoryService()


def _fresh_service(db_path: Path):
    """Per-test instance pointing at the per-test DB but reusing the
    module-shared model under the hood (sentence-transformers caches the
    weights internally so a fresh instance is cheap once warm)."""
    from services.local_memory_service import LocalMemoryService
    return LocalMemoryService(db_path=db_path)


# ---------------------------------------------------------------------------
# Schema / storage contract
# ---------------------------------------------------------------------------


def test_constructor_creates_schema_on_fresh_db(isolated_memory_db):
    """The db file + memories table must exist after construction."""
    _fresh_service(isolated_memory_db)
    assert isolated_memory_db.exists()
    conn = sqlite3.connect(isolated_memory_db)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = [r[0] for r in rows]
        assert 'memories' in names
        # Index exists for user_id + created_at recency queries
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        index_names = [r[0] for r in rows]
        assert any('user' in n for n in index_names)
    finally:
        conn.close()


def test_add_returns_mem0_shaped_envelope(isolated_memory_db, shared_model_service):
    """`add` must return {'results': [{id, memory, event}, ...]} so existing
    callers reach for `result['results'][0]['id']` unchanged."""
    svc = _fresh_service(isolated_memory_db)
    out = svc.add([
        {'role': 'user', 'content': 'hello world'},
        {'role': 'assistant', 'content': 'hi'},
    ], user_id='u1')
    assert 'results' in out
    assert len(out['results']) == 2
    for item in out['results']:
        assert 'id' in item and item['id'].isdigit()
        assert 'memory' in item
        assert item['event'] == 'ADD'


def test_add_accepts_string_input(isolated_memory_db):
    """mem0 accepts a string as well as a message list. Parity."""
    svc = _fresh_service(isolated_memory_db)
    out = svc.add('just a plain string', user_id='u1')
    assert len(out['results']) == 1
    assert out['results'][0]['memory'] == 'just a plain string'


def test_add_skips_empty_content(isolated_memory_db):
    """Empty / whitespace-only content is dropped, not stored as a blank row."""
    svc = _fresh_service(isolated_memory_db)
    out = svc.add([
        {'role': 'user', 'content': ''},
        {'role': 'user', 'content': '   '},
        {'role': 'user', 'content': 'real content'},
    ], user_id='u1')
    assert len(out['results']) == 1
    assert out['results'][0]['memory'] == 'real content'


def test_count_isolates_per_user(isolated_memory_db):
    """`count(user_id)` returns only that user's rows."""
    svc = _fresh_service(isolated_memory_db)
    svc.add('user-1 row', user_id='u1')
    svc.add('user-2 row', user_id='u2')
    svc.add('another user-1 row', user_id='u1')
    assert svc.count('u1') == 2
    assert svc.count('u2') == 1
    assert svc.count() == 3


def test_delete_removes_single_row(isolated_memory_db):
    svc = _fresh_service(isolated_memory_db)
    out = svc.add('about to be deleted', user_id='u1')
    row_id = out['results'][0]['id']
    assert svc.delete(row_id) is True
    assert svc.count('u1') == 0
    assert svc.delete(row_id) is False  # idempotent


def test_delete_rejects_non_integer_ids(isolated_memory_db):
    svc = _fresh_service(isolated_memory_db)
    assert svc.delete('not-a-number') is False
    assert svc.delete(None) is False


def test_reset_wipes_user_only(isolated_memory_db):
    svc = _fresh_service(isolated_memory_db)
    svc.add('u1 row', user_id='u1')
    svc.add('u2 row', user_id='u2')
    deleted = svc.reset('u1')
    assert deleted == 1
    assert svc.count('u1') == 0
    assert svc.count('u2') == 1


def test_search_empty_query_returns_empty(isolated_memory_db):
    svc = _fresh_service(isolated_memory_db)
    svc.add('something', user_id='u1')
    assert svc.search('', user_id='u1') == []
    assert svc.search('   ', user_id='u1') == []


def test_search_isolates_by_user(isolated_memory_db):
    """Hard requirement: User A's search must never return User B's memories."""
    svc = _fresh_service(isolated_memory_db)
    svc.add('User A has a dog named Rex', user_id='userA')
    svc.add('User B has a cat named Mittens', user_id='userB')
    a_hits = svc.search('what pet do I have', user_id='userA')
    b_hits = svc.search('what pet do I have', user_id='userB')
    assert any('Rex' in h['memory'] for h in a_hits)
    assert all('Mittens' not in h['memory'] for h in a_hits)
    assert any('Mittens' in h['memory'] for h in b_hits)
    assert all('Rex' not in h['memory'] for h in b_hits)


def test_metadata_round_trips(isolated_memory_db):
    """Metadata supplied at add() time must come back through search()."""
    svc = _fresh_service(isolated_memory_db)
    svc.add([{'role': 'user', 'content': 'about projects'}],
            user_id='u1',
            metadata={'tag': 'project_management', 'priority': 5})
    hits = svc.search('what about projects', user_id='u1', limit=1)
    assert hits
    md = hits[0]['metadata']
    assert md.get('tag') == 'project_management'
    assert md.get('priority') == 5
    assert md.get('role') == 'user'  # role from message merged in


# ---------------------------------------------------------------------------
# Curated recall — the substance of the swap
# ---------------------------------------------------------------------------


# 12 known facts spanning identity, family, preferences, work, calendar
# context, and locations. Crafted so each one has a clear natural-language
# query partner. If the embedding model's recall regresses meaningfully,
# at least one assertion will fail.
KNOWN_FACTS = [
    ('I live in Brooklyn, New York.', 'Where do I live?'),
    ('My daughter Sam is 8 years old.', 'How old is my daughter?'),
    ('I own a 2022 Tesla Model 3 in white.', 'What car do I drive?'),
    ('My ExampleBrand Gmail account is for the online store.', 'What email do I use for ExampleBrand?'),
    ('Acme Properties manages four rental units in Manhattan.', 'How many rentals does Acme Properties own?'),
    ('The standard lease template lives at /mnt/data/Acme Properties/lease.pdf.',
     'Where is the lease template stored?'),
    ('My birthday is March 14.', 'When is my birthday?'),
    ('I prefer coffee over tea, especially espresso.', 'What do I like to drink?'),
    ('Example LLC renews every May 15.', 'When does my LLC renew?'),
    ('Anna Pavlovich is my rental tenant in unit 3.', 'Who is the tenant in unit 3?'),
    ('I keep my work files under /mnt/data/AI BRAIN/Projects.',
     'Where are my project files?'),
    ('My doctor appointments are usually with Dr. Lerner at NYU Langone.',
     'Who is my doctor?'),
]


@pytest.fixture
def curated_corpus(isolated_memory_db, shared_model_service):
    svc = _fresh_service(isolated_memory_db)
    for fact, _ in KNOWN_FACTS:
        svc.add([{'role': 'user', 'content': fact}], user_id='curated')
    return svc


@pytest.mark.parametrize('fact,query', KNOWN_FACTS)
def test_curated_recall_top5_contains_expected_fact(curated_corpus, fact, query):
    """For each (fact, query) pair, the fact must appear in the top-5
    cosine-similarity hits when we search the query. Top-5 (not top-1)
    because semantic embeddings have legitimate ambiguity; top-1 would
    spuriously fail on edge cases that are still good recall."""
    hits = curated_corpus.search(query, user_id='curated', limit=5)
    memories = [h['memory'] for h in hits]
    assert fact in memories, (
        f'curated recall regression: query={query!r}\n'
        f'  expected fact: {fact!r}\n'
        f'  got top-5    : {memories}'
    )


def test_curated_recall_quality_floor(curated_corpus):
    """Aggregate floor: at least 10 of the 12 (fact, query) pairs must put
    the right fact in the top-5. Gives one or two pairs the freedom to be
    edge-case-bad without failing the whole suite, while still catching
    gross retrieval regressions (e.g. wrong model loaded, vectors all
    near-orthogonal). Adjust if model swaps demand it; do NOT relax
    below 10 without explicit reviewer sign-off."""
    score = 0
    misses: list[str] = []
    for fact, query in KNOWN_FACTS:
        hits = curated_corpus.search(query, user_id='curated', limit=5)
        memories = [h['memory'] for h in hits]
        if fact in memories:
            score += 1
        else:
            misses.append(f'{query!r} -> {memories[:2]}')
    assert score >= 10, (
        f'curated recall floor (10/12) failed: only {score}/12 facts in top-5.\n'
        f'Misses:\n  ' + '\n  '.join(misses)
    )


def test_curated_top1_for_unambiguous_queries(curated_corpus):
    """For the most unambiguous identity-style queries, the EXACT fact should
    be rank-1, not just in the top-5. Tighter check on the easy cases so a
    drift in embedding quality surfaces here before the floor test slips."""
    high_confidence_pairs = [
        ('I live in Brooklyn, New York.', 'Where do I live?'),
        ('My daughter Sam is 8 years old.', 'How old is my daughter?'),
        ('My birthday is March 14.', 'When is my birthday?'),
    ]
    for fact, query in high_confidence_pairs:
        hits = curated_corpus.search(query, user_id='curated', limit=5)
        assert hits and hits[0]['memory'] == fact, (
            f'rank-1 expected for unambiguous query={query!r}\n'
            f'  expected: {fact!r}\n'
            f'  got rank-1: {hits[0]["memory"] if hits else "<no hits>"!r}'
        )


def test_search_score_is_in_valid_cosine_range(curated_corpus):
    """Defense in depth: every returned score must be in [-1.0, 1.0]. A
    bug in normalization (or unintentionally feeding L2 distance) would
    surface here before it shows up as bad recall."""
    hits = curated_corpus.search('anything', user_id='curated', limit=10)
    for h in hits:
        assert -1.0 - 1e-5 <= h['score'] <= 1.0 + 1e-5, h


def test_search_respects_limit(curated_corpus):
    hits = curated_corpus.search('what about my LLC', user_id='curated', limit=3)
    assert len(hits) <= 3


def test_search_returns_empty_when_user_has_no_memories(isolated_memory_db, shared_model_service):
    """No rows for the user → empty list, no exception, no model dimension
    mismatch crash."""
    svc = _fresh_service(isolated_memory_db)
    assert svc.search('anything at all', user_id='no-such-user') == []
