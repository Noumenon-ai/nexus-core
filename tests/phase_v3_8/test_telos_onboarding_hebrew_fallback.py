"""V3.8 halt condition #6: Hebrew users with empty `he` section in
the onboarding-questions JSON must NOT get empty replies. They get
the English content PLUS a one-line note explaining the fallback.
"""
from __future__ import annotations

import json
import textwrap

import pytest

from services.telos_onboarding_content import (
    HEBREW_FALLBACK_NOTE,
    get_section,
    reload_for_tests,
)
from services.telos_onboarding_tools import make_telos_onboarding_tools
from services.telos_service import TelosService


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


@pytest.fixture
def tools(container, telos_service):
    pairs = make_telos_onboarding_tools(
        telos_service=telos_service,
        onboarding_repository=container.onboarding_repository,
        users_repository=container.users_repository,
    )
    return {meta['name']: fn for fn, meta in pairs}


def _hebrew_user(container, telegram_id):
    user = container.users_repository.get_or_create(telegram_id)
    # Persist language=he via session direct write (Users repo doesn't
    # expose a language setter and the V3.8 tools read User.language
    # via users_repository.get_by_id).
    from sqlalchemy.orm import Session
    with Session(container.database.engine) as session:
        session.merge(type(user)(id=user.id, telegram_id=user.telegram_id, language='he'))
        session.commit()
    return container.users_repository.get_by_id(user.id)


def test_hebrew_fallback_returns_english_with_note_when_he_section_empty(container, tools):
    """Default JSON ships with `he` as a placeholder marker. A Hebrew
    user starting onboarding must receive the English questions PLUS
    the HEBREW_FALLBACK_NOTE prepended — never an empty reply."""
    user = _hebrew_user(container, 222)
    assert user.language == 'he'

    result = tools['start_telos_onboarding'](user_id=user.id)
    assert result.success is True
    assert HEBREW_FALLBACK_NOTE in result.announcement
    # The English `identity` intro must be in the body. (Stable string
    # from the canonical JSON.)
    assert "Let's start with who you are" in result.announcement


def test_hebrew_fallback_unit_get_section_signals_used_fallback():
    """Unit-level: `get_section('he', 'identity')` returns
    (english_data, used_fallback=True) when he section is empty."""
    section_data, used_fallback = get_section('he', 'identity')
    assert used_fallback is True
    # English content surfaces.
    assert 'identity' not in section_data  # not the wrapper key
    assert section_data.get('intro')
    questions = section_data.get('questions') or []
    assert len(questions) >= 1


def test_english_user_never_uses_fallback_path():
    """Sanity: English users never trigger the fallback path even if
    he section happens to be populated for some other reason."""
    section_data, used_fallback = get_section('en', 'identity')
    assert used_fallback is False
    assert section_data.get('intro')


def test_hebrew_fallback_disabled_when_he_section_populated(tmp_path, monkeypatch):
    """When the JSON ships a populated `he` section, Hebrew users get
    Hebrew content and used_fallback is False. Simulate by monkey-
    patching the loader to return a Hebrew-populated copy."""
    populated_he = {
        '_section_order': ['identity', 'building', 'values', 'priorities', 'decision_rules', 'push_back'],
        'en': {'identity': {'intro': 'EN intro', 'questions': [{'id': 'role', 'prompt': 'EN prompt', 'example': 'EN ex'}]}},
        'he': {'identity': {'intro': 'HE_INTRO', 'questions': [{'id': 'role', 'prompt': 'HE_PROMPT', 'example': 'HE_EX'}]}},
    }

    fake_path = tmp_path / 'fake_questions.json'
    fake_path.write_text(json.dumps(populated_he), encoding='utf-8')

    import services.telos_onboarding_content as content_mod
    monkeypatch.setattr(content_mod, 'QUESTIONS_PATH', fake_path)
    reload_for_tests()
    try:
        section_data, used_fallback = get_section('he', 'identity')
        assert used_fallback is False
        assert section_data['intro'] == 'HE_INTRO'
        assert section_data['questions'][0]['prompt'] == 'HE_PROMPT'
    finally:
        # Restore normal loader so other tests see the canonical JSON.
        reload_for_tests()
