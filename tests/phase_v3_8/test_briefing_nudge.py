"""V3.8 briefing nudge — once-per-7-days frequency cap with
post-send commit semantics.

Halt-condition coverage:
- nudge appears for user without TELOS (cron path)
- nudge skipped within 7-day cap
- nudge skipped when user already has TELOS
- explicit-path briefings ('good morning' user request) suppress nudge
  entirely (no commit-after-send wiring needed for that rare path)
- last_nudge_at ONLY commits AFTER notify_user — calling
  morning_briefing alone does NOT consume the budget (test harnesses
  / debug commands stay safe)
- last_nudge_at column persists across the morning_briefing call
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from services.telos_service import TelosService
from utils.dates import utc_now
from utils.i18n import Translator


@pytest.fixture
def translator():
    return Translator('en')


@pytest.fixture
def fresh_user(container):
    return container.users_repository.get_or_create(333)


# ---- nudge appears / skipped scenarios -------------------------------------

@pytest.mark.asyncio
async def test_briefing_nudge_appears_for_user_without_telos(container, translator, fresh_user):
    """Cron path: user has no TELOS file, no prior nudge — briefing
    must include the P.S. and metadata['nudge_included'] = True."""
    response = await container.proactive_service.morning_briefing(fresh_user, translator)
    assert response.metadata.get('nudge_included') is True
    assert "start telos" in response.text.lower()
    assert "p.s." in response.text.lower()


@pytest.mark.asyncio
async def test_briefing_nudge_skipped_within_7_day_cap(container, translator, fresh_user):
    """Last nudge 3 days ago → nudge skipped, metadata['nudge_included']=False."""
    container.onboarding_repository.mark_nudged(
        fresh_user.id, when=utc_now() - timedelta(days=3),
    )
    response = await container.proactive_service.morning_briefing(fresh_user, translator)
    assert response.metadata.get('nudge_included') is False
    assert "start telos" not in response.text.lower()


@pytest.mark.asyncio
async def test_briefing_nudge_appears_after_7_days(container, translator, fresh_user):
    """Last nudge 8 days ago → nudge fires again."""
    container.onboarding_repository.mark_nudged(
        fresh_user.id, when=utc_now() - timedelta(days=8),
    )
    response = await container.proactive_service.morning_briefing(fresh_user, translator)
    assert response.metadata.get('nudge_included') is True


@pytest.mark.asyncio
async def test_briefing_nudge_skipped_for_user_with_telos(container, translator, fresh_user, tmp_path):
    """User already has TELOS → no nudge regardless of timing."""
    # Use the SAME telos_service the proactive_service was wired with
    # in build_test_container (tmp_path / 'telos').
    container.proactive_service.telos_service.append(fresh_user.id, '\n## Manual\nAlready set up.\n')
    response = await container.proactive_service.morning_briefing(fresh_user, translator)
    assert response.metadata.get('nudge_included') is False
    assert "start telos" not in response.text.lower()


@pytest.mark.asyncio
async def test_briefing_nudge_suppressed_on_explicit_path(container, translator, fresh_user):
    """Explicit 'good morning' from user — nudge is suppressed entirely
    so the explicit-path doesn't need post-send commit wiring (per V3.8
    design choice: cron-path is the dominant nudge surface)."""
    response = await container.proactive_service.morning_briefing(
        fresh_user, translator, explicit=True,
    )
    assert response.metadata.get('nudge_included') is False


# ---- post-send commit semantics --------------------------------------------

@pytest.mark.asyncio
async def test_morning_briefing_alone_does_not_commit_last_nudge_at(
    container, translator, fresh_user,
):
    """Critical V3.8 invariant: calling morning_briefing must NOT
    consume the once-per-7-days budget. Only the caller's explicit
    `mark_nudge_committed(user_id)` after notify_user succeeds may
    update last_nudge_at. Otherwise test harnesses and debug commands
    would silently burn nudges."""
    state_before = container.onboarding_repository.get(fresh_user.id)
    last_before = state_before.last_nudge_at if state_before else None

    response = await container.proactive_service.morning_briefing(fresh_user, translator)
    assert response.metadata.get('nudge_included') is True

    state_after = container.onboarding_repository.get(fresh_user.id)
    last_after = state_after.last_nudge_at if state_after else None
    assert last_before == last_after, (
        f'morning_briefing leaked last_nudge_at update: before={last_before}, '
        f'after={last_after}. The V3.8 contract requires post-send caller '
        f'commit only.'
    )


@pytest.mark.asyncio
async def test_mark_nudge_committed_writes_last_nudge_at(container, fresh_user):
    """Pinning the contract: ProactiveService.mark_nudge_committed
    is what callers invoke after a successful notify_user. It writes
    a fresh timestamp to the onboarding state row."""
    before = utc_now()
    container.proactive_service.mark_nudge_committed(fresh_user.id)
    state = container.onboarding_repository.get(fresh_user.id)
    assert state is not None
    assert state.last_nudge_at is not None
    after_committed = state.last_nudge_at
    if after_committed.tzinfo is None:
        after_committed = after_committed.replace(tzinfo=timezone.utc)
    assert after_committed >= before


@pytest.mark.asyncio
async def test_full_cycle_compute_send_commit_then_skipped_on_next_call(
    container, translator, fresh_user,
):
    """End-to-end shape: first cron call includes the nudge, caller
    commits, second cron call (within 7 days) skips the nudge.
    Simulates the dispatch_proactive_job pattern without the live
    Telegram dependency."""
    response_1 = await container.proactive_service.morning_briefing(fresh_user, translator)
    assert response_1.metadata.get('nudge_included') is True
    # Simulate: notify_user succeeded → caller commits.
    container.proactive_service.mark_nudge_committed(fresh_user.id)

    response_2 = await container.proactive_service.morning_briefing(fresh_user, translator)
    assert response_2.metadata.get('nudge_included') is False, (
        'Nudge fired twice within 7 days — frequency cap broken.'
    )
