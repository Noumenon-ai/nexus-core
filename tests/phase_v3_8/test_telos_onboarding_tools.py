"""V3.8 onboarding tools — start/answer/view/cancel behavior tests.

Coverage map:
- start: creates state for new user, resumes existing, returns
  complete-message when already done
- answer: records answer, advances section when complete, writes
  TELOS file at section close, completes after last section
- view: returns content / no-file message
- cancel: sets cancelled_at, preserves current_section + answers_so_far
- resume after cancel: picks up at correct section
- per-user isolation: User A answers never reach User B file
- mode 600: TELOS file mode preserved through append (intrinsic via
  TelosService.append, this test pins the property)
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from services.telos_onboarding_content import (
    HEBREW_FALLBACK_NOTE,
    section_order,
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


def _user(container, telegram_id):
    return container.users_repository.get_or_create(telegram_id)


# ---- start_telos_onboarding ------------------------------------------------

def test_start_telos_creates_state_for_new_user(container, tools):
    user = _user(container, 111)
    result = tools['start_telos_onboarding'](user_id=user.id)

    assert result.success is True
    state = container.onboarding_repository.get(user.id)
    assert state is not None
    assert state.started_at is not None
    assert state.current_section == 'identity'
    assert state.answers_so_far == '{}'
    assert state.completed_at is None
    # The first identity question's prompt must be in the announcement.
    assert 'Reply with your answer' in result.announcement


def test_start_telos_resumes_existing_state(container, tools):
    """When called twice, the second start picks up at the same section
    without resetting answers."""
    user = _user(container, 111)
    tools['start_telos_onboarding'](user_id=user.id)
    # Simulate prior answer recorded.
    container.onboarding_repository.record_answer(
        user.id, question_id='role', answer='solo dev',
    )
    # Advance to building section to simulate prior completion.
    container.onboarding_repository.advance_section(user.id, next_section='building')

    result = tools['start_telos_onboarding'](user_id=user.id)
    assert result.success is True
    state = container.onboarding_repository.get(user.id)
    assert state.current_section == 'building'
    answers = json.loads(state.answers_so_far)
    assert answers.get('role') == 'solo dev'  # not lost on resume


def test_start_telos_returns_complete_message_when_already_done(container, tools):
    user = _user(container, 111)
    container.onboarding_repository.get_or_create(user.id)
    container.onboarding_repository.mark_started(user.id)
    container.onboarding_repository.mark_completed(user.id)

    result = tools['start_telos_onboarding'](user_id=user.id)
    assert result.success is True
    assert 'already complete' in result.announcement.lower()
    assert result.data['status'] == 'already_complete'


# ---- answer_telos_question -------------------------------------------------

def test_answer_telos_question_records_answer(container, tools):
    user = _user(container, 111)
    tools['start_telos_onboarding'](user_id=user.id)
    result = tools['answer_telos_question'](
        answer='Solo dev building AI tools.', user_id=user.id,
    )
    assert result.success is True
    state = container.onboarding_repository.get(user.id)
    answers = json.loads(state.answers_so_far)
    assert answers.get('role') == 'Solo dev building AI tools.'


def test_answer_telos_question_advances_section_when_complete(container, tools, telos_service):
    """`identity` has one question. Answering it completes the section
    and advances to `building`. The TELOS file must contain the
    identity content."""
    user = _user(container, 111)
    tools['start_telos_onboarding'](user_id=user.id)
    tools['answer_telos_question'](answer='Solo dev.', user_id=user.id)

    state = container.onboarding_repository.get(user.id)
    assert state.current_section == 'building'
    # File written.
    content = telos_service.load(user.id)
    assert content is not None
    assert 'Identity' in content
    assert 'Solo dev.' in content


def test_answer_telos_question_writes_to_file_at_section_close(container, tools, telos_service):
    """Section content lands in the file ONLY when the section
    completes — partial answers (multi-question section like
    `priorities`) do not flush mid-section."""
    user = _user(container, 111)
    tools['start_telos_onboarding'](user_id=user.id)
    # Walk to priorities section by completing identity, building, values.
    tools['answer_telos_question'](answer='dev', user_id=user.id)
    tools['answer_telos_question'](answer='V3', user_id=user.id)
    tools['answer_telos_question'](answer='family, work, health', user_id=user.id)
    # Now in priorities (2 questions).
    state = container.onboarding_repository.get(user.id)
    assert state.current_section == 'priorities'

    # Answer first priorities question — section NOT yet complete, file
    # should NOT yet contain priorities content.
    tools['answer_telos_question'](answer='Family wins.', user_id=user.id)
    content_after_first = telos_service.load(user.id) or ''
    assert 'Priorities' not in content_after_first

    # Answer second priorities question — section completes, file gains
    # the priorities block.
    tools['answer_telos_question'](answer='Health.', user_id=user.id)
    content_after_second = telos_service.load(user.id) or ''
    assert 'Priorities' in content_after_second
    assert 'Family wins.' in content_after_second
    assert 'Health.' in content_after_second


def test_answer_telos_question_completes_onboarding_after_last_section(container, tools):
    """Walking through all 6 sections produces a completed state."""
    user = _user(container, 111)
    tools['start_telos_onboarding'](user_id=user.id)
    for section in section_order():
        # Keep answering until section advances (each section has 1-2 q).
        for _ in range(5):  # safety loop cap
            state = container.onboarding_repository.get(user.id)
            if state.current_section != section or state.completed_at is not None:
                break
            tools['answer_telos_question'](answer=f'answer for {section}', user_id=user.id)

    state = container.onboarding_repository.get(user.id)
    assert state.completed_at is not None


def test_answer_telos_question_rejects_empty_answer(container, tools):
    user = _user(container, 111)
    tools['start_telos_onboarding'](user_id=user.id)
    result = tools['answer_telos_question'](answer='   ', user_id=user.id)
    assert result.data['status'] == 'empty_answer'


def test_answer_telos_question_no_active_flow(container, tools):
    """No state row + no started_at → 'no active onboarding' reply."""
    user = _user(container, 111)
    result = tools['answer_telos_question'](answer='hi', user_id=user.id)
    assert 'no active onboarding' in result.announcement.lower()


# ---- view_my_telos ---------------------------------------------------------

def test_view_my_telos_returns_file_content(container, tools, telos_service):
    user = _user(container, 111)
    telos_service.append(user.id, '\n## Manual\nSome text.\n')
    result = tools['view_my_telos'](user_id=user.id)
    assert result.success is True
    assert 'Some text.' in result.announcement
    assert result.data['present'] is True


def test_view_my_telos_returns_no_file_message(container, tools):
    user = _user(container, 111)
    result = tools['view_my_telos'](user_id=user.id)
    assert result.data['present'] is False
    assert 'no telos file' in result.announcement.lower()


# ---- cancel_telos_onboarding -----------------------------------------------

def test_cancel_telos_onboarding_sets_cancelled_at(container, tools):
    user = _user(container, 111)
    tools['start_telos_onboarding'](user_id=user.id)
    result = tools['cancel_telos_onboarding'](user_id=user.id)
    assert result.success is True
    assert result.data['status'] == 'cancelled'
    state = container.onboarding_repository.get(user.id)
    assert state.cancelled_at is not None


def test_cancel_preserves_current_section_for_resume(container, tools):
    """Halt condition: cancel must not lose state. After 3 sections of
    progress, cancel should leave current_section unchanged."""
    user = _user(container, 111)
    tools['start_telos_onboarding'](user_id=user.id)
    tools['answer_telos_question'](answer='dev', user_id=user.id)  # advance to building
    tools['answer_telos_question'](answer='V3', user_id=user.id)  # advance to values
    state_before = container.onboarding_repository.get(user.id)
    section_before = state_before.current_section
    answers_before = state_before.answers_so_far

    tools['cancel_telos_onboarding'](user_id=user.id)

    state_after = container.onboarding_repository.get(user.id)
    assert state_after.current_section == section_before
    assert state_after.answers_so_far == answers_before
    assert state_after.cancelled_at is not None


def test_resume_after_cancel_picks_up_at_correct_section(container, tools):
    user = _user(container, 111)
    tools['start_telos_onboarding'](user_id=user.id)
    tools['answer_telos_question'](answer='dev', user_id=user.id)  # advance to building
    tools['cancel_telos_onboarding'](user_id=user.id)

    # Resume — start_telos_onboarding should clear cancelled_at AND
    # show the building question (NOT identity).
    result = tools['start_telos_onboarding'](user_id=user.id)
    state = container.onboarding_repository.get(user.id)
    assert state.cancelled_at is None  # cleared on resume
    assert state.current_section == 'building'
    # Result text should reflect building section's intro / question.
    assert 'building' in result.announcement.lower() or 'focus' in result.announcement.lower() or 'working' in result.announcement.lower()


# ---- per-user isolation ----------------------------------------------------

def test_per_user_isolation_onboarding_state(container, tools, telos_service, tmp_path):
    """Halt condition: User A answers never reach User B file."""
    a = _user(container, 111)
    b = _user(container, 222)

    tools['start_telos_onboarding'](user_id=a.id)
    tools['answer_telos_question'](answer='A SECRET answer', user_id=a.id)

    tools['start_telos_onboarding'](user_id=b.id)
    tools['answer_telos_question'](answer='B speaking', user_id=b.id)

    a_content = telos_service.load(a.id) or ''
    b_content = telos_service.load(b.id) or ''

    assert 'A SECRET answer' in a_content
    assert 'A SECRET answer' not in b_content
    assert 'B speaking' in b_content
    assert 'B speaking' not in a_content


# ---- mode 600 preservation -------------------------------------------------

def test_telos_file_mode_600_preserved_after_append(container, tools, telos_service):
    """V3.8 halt condition #1: TELOS file mode 600 preserved on every
    write. TelosService.append chmods 0o600 internally — this test
    pins the property so future refactors of telos_service.py can't
    silently regress it."""
    user = _user(container, 111)
    tools['start_telos_onboarding'](user_id=user.id)
    tools['answer_telos_question'](answer='solo dev', user_id=user.id)

    path = telos_service.path_for(user.id)
    assert path.exists()
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, f'Expected mode 0o600, got 0o{mode:o}. File: {path}'

    # Append again — mode must remain 0o600 after the second write.
    tools['answer_telos_question'](answer='V3 dispatcher', user_id=user.id)
    mode_after = stat.S_IMODE(os.stat(path).st_mode)
    assert mode_after == 0o600, f'Mode regressed after second append: 0o{mode_after:o}'
