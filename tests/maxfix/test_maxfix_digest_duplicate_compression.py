from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import services.proactive_service as proactive_service_module
from utils.i18n import Translator


@pytest.mark.asyncio
async def test_morning_digest_collapses_duplicate_followup_reminders(container, monkeypatch):
    fixed_now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(proactive_service_module, 'utc_now', lambda: fixed_now)

    user = container.users_repository.get_or_create(111)
    fire_at = fixed_now + timedelta(hours=21)
    container.reminders_repository.create(
        user_id=user.id,
        body='Follow up with Acme Corp',
        next_fire_at=fire_at,
        recurrence=None,
    )
    container.reminders_repository.create(
        user_id=user.id,
        body='Follow up with Acme Corp',
        next_fire_at=fire_at,
        recurrence=None,
    )
    container.reminders_repository.create(
        user_id=user.id,
        body='Follow up',
        next_fire_at=fire_at,
        recurrence=None,
    )

    out = await container.proactive_service.morning_briefing(
        user,
        Translator('en'),
        explicit=False,
    )

    assert out.text.count('Follow up with Acme Corp') == 1
    assert '(2 duplicates found)' in out.text
    assert 'Want me to clean them up?' in out.text
    assert out.metadata.get('should_send') is True
