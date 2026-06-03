from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

import main as main_module
from models import ProactiveNotification
from utils.dates import utc_now
from utils.i18n import Translator


@pytest.mark.asyncio
async def test_morning_briefing_includes_reminders_tasks_email_and_habit(container):
    user = container.users_repository.get_or_create(111)
    container.reminders_repository.create(user_id=user.id, body='call bank', next_fire_at=utc_now() + timedelta(hours=2), recurrence=None)
    container.tasks_repository.create(user_id=user.id, title='finish report', due_at=utc_now() + timedelta(hours=4), priority=1, source='user')
    container.emails_repository.upsert(user_id=user.id, gmail_message_id='m1', subject='Invoice due', sender='Billing', snippet='Pay today', received_at=utc_now(), category='bill', extracted_json='{}')
    container.habit_service.record_reminder_creation(user.id, utc_now())
    response = await container.proactive_service.morning_briefing(user, Translator(), explicit=True)
    assert 'good morning' in response.text.lower()
    assert 'call bank' in response.text.lower()
    assert 'finish report' in response.text.lower()
    assert 'invoice due' in response.text.lower()


@pytest.mark.asyncio
async def test_scheduled_briefings_are_deduped_per_day(container):
    user = container.users_repository.get_or_create(111)
    first = await container.proactive_service.morning_briefing(user, Translator(), explicit=False)
    second = await container.proactive_service.morning_briefing(user, Translator(), explicit=False)
    with container.database.session_factory() as session:
        count = session.query(ProactiveNotification).count()
    assert first.metadata['should_send'] is True
    assert second.metadata['should_send'] is False
    assert count == 1


@pytest.mark.asyncio
async def test_each_briefing_loop_has_its_own_daily_dedupe_key(container):
    user = container.users_repository.get_or_create(111)
    await container.proactive_service.morning_briefing(user, Translator(), explicit=False)
    response = await container.proactive_service.midday_check(user, Translator())
    with container.database.session_factory() as session:
        count = session.query(ProactiveNotification).count()
    assert response.metadata['should_send'] is True
    assert count == 2


@pytest.mark.asyncio
async def test_late_briefing_recovery_skips_when_too_old(container):
    user = container.users_repository.get_or_create(111)
    response = await container.proactive_service.recover_briefing(user, 'morning_briefing', utc_now() - timedelta(hours=4), Translator())
    assert response is None


@pytest.mark.asyncio
async def test_startup_recovery_sends_recent_missed_briefing(container, monkeypatch):
    user = container.users_repository.get_or_create(111)
    sent: list[tuple[str, str]] = []

    class FakeRuntime:
        def __init__(self):
            self.settings = container.settings
            self.users_repository = container.users_repository
            self.reminder_service = container.reminder_service
            self.proactive_service = container.proactive_service

        def translator_for(self, language: str):
            return Translator(language)

        async def notify_user(self, user_id: str, text: str) -> None:
            sent.append((user_id, text))

    current = utc_now()
    settings = replace(
        container.settings,
        proactive=replace(container.settings.proactive, midday_check_hour=current.hour),
    )

    def fake_app_now(_timezone: str):
        return current.replace(second=0, microsecond=0)

    monkeypatch.setattr(main_module, 'app_now', fake_app_now)
    await main_module.recover_startup_state(FakeRuntime(), settings)

    assert sent
    assert any('midday check' in text.lower() for _user_id, text in sent)
