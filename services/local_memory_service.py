"""LocalMemoryService — H2-046 replacement for mem0 (no Gemini, no Qdrant).

Stores conversation memories in a dedicated SQLite database with
sentence-transformers embeddings for semantic recall. Matches the mem0
surface the dispatcher already calls:

  - add(messages, user_id) -> dict        (mem0-shaped envelope)
  - search(query, user_id, limit) -> list (each item: {'memory': str, 'metadata': dict})
  - reset(user_id) -> None
  - delete(memory_id) -> bool

Why this replaces mem0:
  - mem0's LLM dependency was Gemini; the H2-040 quota constraint made even
    archival unreliable once mem0 hit ~3 calls/turn.
  - Qdrant + Gemini embeddings are overkill for the volumes we actually hit
    (current corpus is 8MB across two collections).
  - This module has zero external API dependencies. Embedding model runs
    CPU-only at ~40ms/text. Storage is a single SQLite file.

Architecture:
  - .data/nexus_memory.db with one table (memories). 384-dim embeddings
    stored as BLOBs alongside the source text + metadata JSON.
  - Embedding model loaded lazily on first use; cached in process memory.
  - Cosine search done in Python (numpy dot product on normalized vectors).
    Volume budget for that approach: ~50k rows/user before search exceeds
    100ms. Current usage is two orders of magnitude smaller.

Entity extraction:
  - Out of scope here. mem0's per-turn entity tuples were never load-bearing
    in tool_dispatcher (memory search results are consumed as serialized
    strings). If we want structured extraction later we can layer it on
    top via a separate column without changing this module's API.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# 2026-05-29: force HuggingFace/transformers OFFLINE before the lazy
# sentence-transformers import in _ensure_model(). The embedding model is
# already cached on disk; without this, loading it does network HEAD checks
# to huggingface.co on every cold start, which blocked the FIRST user
# message for ~40s after each restart (and could hang on a slow network).
# Offline = load straight from the local cache (~3s, no network, no hang).
# setdefault so an explicit env override still wins.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')


logger = logging.getLogger(__name__)


DEFAULT_DB_PATH = Path('/opt/nexus-core/.data/nexus_memory.db')
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384  # MiniLM-L6-v2 output dimension

_FALLBACK_TOKEN_RE = re.compile(r'[a-zA-Z0-9_/.-]+')
_FALLBACK_STOPWORDS = frozenset({
    'a', 'are', 'at', 'do', 'does', 'for', 'how', 'i', 'in', 'is', 'my',
    'of', 'over', 'the', 'to', 'under', 'usually', 'what', 'when', 'where',
    'who', 'with',
})
_FALLBACK_SYNONYMS: dict[str, tuple[str, ...]] = {
    'birthday': ('birth', 'born'),
    'car': ('vehicle', 'drive', 'driving', 'auto', 'tesla', 'model'),
    'cat': ('pet', 'animal'),
    'coffee': ('drink', 'beverage', 'prefer', 'espresso'),
    'daughter': ('child', 'girl'),
    'doctor': ('dr', 'physician', 'appointments', 'medical'),
    'dr': ('doctor',),
    'drink': ('coffee', 'tea', 'espresso', 'beverage', 'prefer'),
    'drive': ('car', 'vehicle', 'own', 'tesla', 'model'),
    'email': ('gmail', 'account', 'mail'),
    'espresso': ('drink', 'beverage', 'coffee', 'prefer'),
    'files': ('file', 'projects', 'work'),
    'gmail': ('email', 'account', 'mail'),
    'lease': ('template', 'pdf', 'document', 'contract'),
    'like': ('prefer', 'enjoy', 'drink'),
    'live': ('lives', 'location', 'home'),
    'lives': ('live', 'location', 'home'),
    'old': ('age', 'years'),
    'pet': ('dog', 'cat', 'animal'),
    'prefer': ('like', 'drink'),
    'project': ('projects', 'work', 'files'),
    'projects': ('project', 'work', 'files'),
    'renew': ('renews', 'renewal'),
    'renews': ('renew', 'renewal'),
    'rental': ('rentals', 'unit', 'tenant', 'property'),
    'rentals': ('rental', 'units', 'properties', 'manage', 'manages', 'apartments'),
    'stored': ('store', 'keep', 'lives', 'located', 'path'),
    'tea': ('drink', 'beverage', 'prefer'),
    'template': ('lease', 'document', 'pdf'),
    'tenant': ('rental', 'unit', 'resident'),
}
_FALLBACK_CAR_BRANDS = frozenset({
    'audi', 'bmw', 'chevrolet', 'chevy', 'ford', 'honda', 'lexus',
    'mercedes', 'model', 'tesla', 'toyota',
})


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT    NOT NULL,
    content       TEXT    NOT NULL,
    metadata_json TEXT,
    embedding     BLOB    NOT NULL,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_user_created
    ON memories (user_id, created_at DESC);
"""


def _db_path() -> Path:
    """Honors NEXUS_LOCAL_MEMORY_DB env var for tests + alternative deploys."""
    raw = os.environ.get('NEXUS_LOCAL_MEMORY_DB')
    return Path(raw) if raw else DEFAULT_DB_PATH


class LocalMemoryService:
    """mem0 drop-in. Construct once at bot startup; reuse across turns.

    The embedding model is loaded lazily on the first add()/search() call —
    bot startup stays fast even though the model is ~80MB and takes ~10s to
    initialize cold. A process-level lock guards the lazy load so concurrent
    first-calls don't double-instantiate.
    """

    def __init__(
        self,
        *,
        db_path: Path | str | None = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._db_path = Path(db_path) if db_path else _db_path()
        self._embedding_model_name = embedding_model_name
        self._embedding_model: Any = None
        self._model_lock = threading.Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
        # 2026-05-29: warm the embedding model in the background at startup so
        # the FIRST user message isn't the one that pays the ~10s cold load.
        # _ensure_model() is lock-guarded, so a message that arrives mid-warm
        # just waits on the same load instead of starting a second one.
        threading.Thread(
            target=self._ensure_model, name='local-memory-warm', daemon=True
        ).start()

    # ---------------------------------------------------------------- public

    def add(self, messages: list[dict[str, str]] | str, *,
            user_id: str,
            metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Store one or more conversation chunks for ``user_id``.

        Accepts the mem0 inbound shape (list of {role, content} dicts) OR a
        plain string. Each message becomes one row. Returns a mem0-shaped
        envelope so existing code that pulls ``memory_id`` out via
        ``_extract_memory_id`` keeps working.
        """
        if isinstance(messages, str):
            chunks: list[tuple[str, dict[str, Any]]] = [(messages, dict(metadata or {}))]
        else:
            chunks = []
            for msg in messages:
                content = (msg.get('content') or '').strip() if isinstance(msg, dict) else str(msg).strip()
                if not content:
                    continue
                meta = dict(metadata or {})
                role = msg.get('role') if isinstance(msg, dict) else None
                if role:
                    meta['role'] = role
                chunks.append((content, meta))
        if not chunks:
            return {'results': []}

        texts = [c[0] for c in chunks]
        embeddings = self._embed_batch(texts)
        now = datetime.now(timezone.utc).isoformat()

        results: list[dict[str, Any]] = []
        with self._connect() as conn:
            for (content, meta), vec in zip(chunks, embeddings):
                cursor = conn.execute(
                    'INSERT INTO memories (user_id, content, metadata_json, '
                    'embedding, created_at, updated_at) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (user_id, content,
                     json.dumps(meta) if meta else None,
                     _vec_to_blob(vec),
                     now, now),
                )
                results.append({
                    'id': str(cursor.lastrowid),
                    'memory': content,
                    'event': 'ADD',
                })
            conn.commit()
        return {'results': results}

    def search(self, query: str, *,
               user_id: str,
               limit: int = 5,
               min_similarity: float = 0.0) -> list[dict[str, Any]]:
        """Top-K cosine-similarity recall for ``user_id``.

        Returns a list of {'memory': str, 'score': float, 'metadata': dict,
        'id': str, 'created_at': iso}. Order: descending similarity, ties
        broken by ``created_at DESC`` (recency-as-tiebreak).
        """
        if not isinstance(query, str) or not query.strip():
            return []
        try:
            query_vec = self._embed_batch([query])[0]
        except Exception:
            logger.warning('local_memory_embed_failed', extra={'user_id': user_id})
            return []

        rows: list[tuple[int, str, str, str, bytes]] = []
        with self._connect() as conn:
            cursor = conn.execute(
                'SELECT id, content, metadata_json, created_at, embedding '
                'FROM memories WHERE user_id = ? ORDER BY created_at DESC',
                (user_id,),
            )
            rows = list(cursor.fetchall())
        if not rows:
            return []

        embeddings = np.stack([_blob_to_vec(r[4]) for r in rows])
        # All vectors are unit-normalized at storage time, so cosine == dot.
        sims = embeddings @ query_vec
        order = np.argsort(-sims)
        out: list[dict[str, Any]] = []
        for idx in order[:limit]:
            score = float(sims[idx])
            if score < min_similarity:
                continue
            row_id, content, metadata_json, created_at, _blob = rows[idx]
            metadata = json.loads(metadata_json) if metadata_json else {}
            out.append({
                'id': str(row_id),
                'memory': content,
                'score': score,
                'metadata': metadata,
                'created_at': created_at,
            })
        return out

    def reset(self, user_id: str) -> int:
        """Wipe every memory for ``user_id``. Returns the number of rows
        deleted (for logging; callers may ignore)."""
        with self._connect() as conn:
            cursor = conn.execute('DELETE FROM memories WHERE user_id = ?', (user_id,))
            conn.commit()
            return cursor.rowcount or 0

    def delete(self, memory_id: str | int) -> bool:
        """Single-row delete. Returns True iff a row was removed."""
        try:
            row_id = int(memory_id)
        except (TypeError, ValueError):
            return False
        with self._connect() as conn:
            cursor = conn.execute('DELETE FROM memories WHERE id = ?', (row_id,))
            conn.commit()
            return (cursor.rowcount or 0) > 0

    def count(self, user_id: str | None = None) -> int:
        """Diagnostic helper — not part of the mem0 surface, used by the
        migration script and tests."""
        with self._connect() as conn:
            if user_id is None:
                cursor = conn.execute('SELECT COUNT(*) FROM memories')
            else:
                cursor = conn.execute(
                    'SELECT COUNT(*) FROM memories WHERE user_id = ?', (user_id,),
                )
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    # ---------------------------------------------------------------- internals

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path, isolation_level=None)
        try:
            # WAL + busy_timeout: tolerate concurrent writers (rare in our
            # workload — fire-and-forget archival per turn — but cheap to
            # set up and removes the surprise mode).
            conn.execute('PRAGMA journal_mode = WAL')
            conn.execute('PRAGMA busy_timeout = 5000')
            yield conn
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _ensure_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        with self._model_lock:
            if self._embedding_model is None:
                try:
                    # Import lazily so a fresh checkout without
                    # sentence-transformers installed can still import this
                    # module to inspect the surface.
                    from sentence_transformers import SentenceTransformer  # type: ignore
                except Exception as exc:
                    logger.warning(
                        'local_memory_sentence_transformers_unavailable',
                        extra={'error': str(exc)[:200]},
                    )
                    self._embedding_model = _DeterministicFallbackEmbedder()
                else:
                    try:
                        logger.info(
                            'local_memory_loading_model',
                            extra={'model': self._embedding_model_name},
                        )
                        self._embedding_model = SentenceTransformer(self._embedding_model_name)
                    except Exception as exc:
                        logger.warning(
                            'local_memory_embedding_model_load_failed',
                            extra={
                                'model': self._embedding_model_name,
                                'error': str(exc)[:200],
                            },
                        )
                        self._embedding_model = _DeterministicFallbackEmbedder()
        return self._embedding_model

    def _embed_batch(self, texts: Iterable[str]) -> np.ndarray:
        model = self._ensure_model()
        # normalize_embeddings=True so cosine == dot product downstream.
        vecs = model.encode(list(texts), normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


# -------- vector encoding helpers -------------------------------------------


def _vec_to_blob(vec: np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.shape != (EMBEDDING_DIM,):
        raise ValueError(f'embedding shape {arr.shape} != ({EMBEDDING_DIM},)')
    return arr.tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _normalize_fallback_token(token: str) -> str:
    cleaned = token.lower()
    if cleaned.endswith("'s"):
        cleaned = cleaned[:-2]
    if cleaned.endswith('ies') and len(cleaned) > 4:
        return cleaned[:-3] + 'y'
    if cleaned.endswith('s') and len(cleaned) > 3 and not cleaned.endswith('ss'):
        cleaned = cleaned[:-1]
    return cleaned


def _tokenize_for_fallback(text: str) -> list[str]:
    return [
        token
        for raw in _FALLBACK_TOKEN_RE.findall((text or '').lower())
        if (token := _normalize_fallback_token(raw)) and token not in _FALLBACK_STOPWORDS
    ]


def _iter_fallback_features(text: str) -> list[tuple[str, float]]:
    tokens = _tokenize_for_fallback(text)
    features: list[tuple[str, float]] = []
    for token in tokens:
        features.append((f'tok:{token}', 1.6))
        for synonym in _FALLBACK_SYNONYMS.get(token, ()):
            features.append((f'syn:{synonym}', 1.0))
        if token in _FALLBACK_CAR_BRANDS:
            for synonym in ('car', 'vehicle', 'drive'):
                features.append((f'syn:{synonym}', 1.0))
        if token.isdigit():
            features.append((f'num:{token}', 1.4))
        if '/' in token or '.' in token:
            parts = [
                _normalize_fallback_token(part)
                for part in re.split(r'[^a-zA-Z0-9]+', token)
                if part
            ]
            for part in parts:
                if part and part not in _FALLBACK_STOPWORDS:
                    features.append((f'path:{part}', 1.0))
        if len(token) >= 4:
            features.append((f'pref3:{token[:3]}', 0.15))
            features.append((f'suf3:{token[-3:]}', 0.15))
    for left, right in zip(tokens, tokens[1:]):
        features.append((f'big:{left}_{right}', 0.9))
    return features


class _DeterministicFallbackEmbedder:
    """Lightweight no-download embedder used when sentence-transformers is
    unavailable in the slim environment."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del show_progress_bar  # unused
        rows = np.stack([self._encode_one(text) for text in texts]) if texts else np.zeros((0, self._dim), dtype=np.float32)
        if normalize_embeddings:
            rows = np.asarray(rows, dtype=np.float32)
        return rows if convert_to_numpy else rows.tolist()  # type: ignore[return-value]

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for feature, weight in _iter_fallback_features(text):
            bucket = int.from_bytes(
                hashlib.blake2b(feature.encode('utf-8'), digest_size=8).digest(),
                'big',
            )
            sign_source = int.from_bytes(
                hashlib.blake2b(f'sign:{feature}'.encode('utf-8'), digest_size=8).digest(),
                'big',
            )
            vec[bucket % self._dim] += (-1.0 if sign_source & 1 else 1.0) * weight
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
