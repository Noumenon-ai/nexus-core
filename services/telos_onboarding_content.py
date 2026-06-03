"""V3.8 TELOS onboarding question content loader.

Loads `resources/telos_onboarding_questions.json` once (memoized) and
exposes a per-language section accessor with Hebrew-empty fallback.

If `user.language == 'he'` AND the `he` section for the requested
section name is missing or placeholder-only, we fall back to the
English content and return `used_fallback=True` so the calling tool
can include a one-line note in its user-facing reply ("Hebrew
translation pending — using English for now."). This satisfies V3.8
halt condition #6: Hebrew users with empty `he` section must not get
empty replies.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = _PROJECT_ROOT / 'resources' / 'telos_onboarding_questions.json'

_DEFAULT_SECTION_ORDER = (
    'identity', 'building', 'values', 'priorities', 'decision_rules', 'push_back',
)

HEBREW_FALLBACK_NOTE = (
    'Hebrew translation pending — using English for now.'
)


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, Any]:
    """Read the JSON file once. lru_cache makes repeat calls free."""
    return json.loads(QUESTIONS_PATH.read_text(encoding='utf-8'))


def reload_for_tests() -> None:
    """Test hook — clears the lru_cache so a test that rewrites the
    JSON on disk sees fresh content. NOT for production use."""
    _load_raw.cache_clear()


def section_order() -> tuple[str, ...]:
    """Return the canonical section order from the JSON file, falling
    back to a hard-coded default if the file omits `_section_order`."""
    raw = _load_raw()
    order = raw.get('_section_order')
    if isinstance(order, list) and all(isinstance(s, str) for s in order):
        return tuple(order)
    return _DEFAULT_SECTION_ORDER


def _section_is_populated(section_data: Any) -> bool:
    """A section is populated if it has a non-empty `intro` AND at
    least one question with an `id` and `prompt`. Anything less is
    treated as a placeholder."""
    if not isinstance(section_data, dict):
        return False
    intro = section_data.get('intro')
    if not isinstance(intro, str) or not intro.strip():
        return False
    questions = section_data.get('questions')
    if not isinstance(questions, list) or not questions:
        return False
    for q in questions:
        if not isinstance(q, dict):
            return False
        if not isinstance(q.get('id'), str) or not q['id'].strip():
            return False
        if not isinstance(q.get('prompt'), str) or not q['prompt'].strip():
            return False
    return True


def get_section(language: str, section: str) -> tuple[dict[str, Any], bool]:
    """Return (section_data, used_fallback).

    - For language='en' or unknown languages: returns the English
      section directly. used_fallback=False.
    - For language='he' with a populated he section: returns Hebrew.
      used_fallback=False.
    - For language='he' with missing/placeholder he section: returns
      English content. used_fallback=True so the tool can prepend the
      `HEBREW_FALLBACK_NOTE`.

    Raises `KeyError` if the requested section name is unknown in
    English. (English is the source-of-truth language; if it lacks the
    section, that's a content bug, not a runtime fallback case.)
    """
    raw = _load_raw()
    en_sections = raw.get('en', {})
    if section not in en_sections:
        raise KeyError(f'Unknown TELOS onboarding section: {section!r}')
    en_data = en_sections[section]

    if language == 'he':
        he_sections = raw.get('he', {})
        he_data = he_sections.get(section)
        if _section_is_populated(he_data):
            return he_data, False
        # Hebrew section missing/placeholder — fall back to English.
        return en_data, True

    return en_data, False


def find_next_unanswered_question(
    section_data: dict[str, Any], answers_so_far: dict[str, str],
) -> dict[str, Any] | None:
    """Return the first question in `section_data['questions']` whose
    `id` is not yet in `answers_so_far`. None when section is fully
    answered."""
    questions = section_data.get('questions') or []
    for q in questions:
        if not isinstance(q, dict):
            continue
        qid = q.get('id')
        if qid and qid not in answers_so_far:
            return q
    return None


def is_section_complete(section_data: dict[str, Any], answers_so_far: dict[str, str]) -> bool:
    """A section is complete when every question id has an answer."""
    questions = section_data.get('questions') or []
    return all(
        isinstance(q, dict) and q.get('id') in answers_so_far
        for q in questions
    )


def next_section_name(current: str) -> str | None:
    """Return the section that comes after `current` in the canonical
    order, or None when `current` is the last section."""
    order = section_order()
    try:
        idx = order.index(current)
    except ValueError:
        return None
    if idx + 1 >= len(order):
        return None
    return order[idx + 1]
