from __future__ import annotations

from datetime import timedelta

import pytest

from pipeline.types import PipelineInput
from utils.dates import utc_now
from utils.i18n import Translator


@pytest.mark.asyncio
async def test_recurring_reminder_parses_layer_one(container):
    result = await container.reminder_parser.parse(container.users_repository.get_or_create(111).id, 'remind me every Friday at 2pm to do payroll')
    assert result.type == 'recurring'
    assert result.rrule is not None
    assert 'BYDAY=FR' in result.rrule
    assert 'BYHOUR=14' in result.rrule
    assert result.defaulted_time is False


def test_expired_pending_reminder_cannot_be_confirmed(container):
    user = container.users_repository.get_or_create(111)
    container.conversation_service.store_pending_reminder(
        user.id,
        {
            'body': 'call insurance',
            'datetime_utc': utc_now().isoformat(),
            'rrule': None,
            'type': 'one_shot',
            'when_text': 'tomorrow at 9:00 AM',
            'expires_at': (utc_now() - timedelta(seconds=1)).isoformat(),
        },
    )
    response = container.reminder_service.confirm_pending(user, Translator())
    assert 'what would you like me to do with that' in response.text.lower()
    assert container.reminders_repository.list_active(user.id) == []


@pytest.mark.asyncio
async def test_boot_recovery_fires_missed_one_shot_once(container):
    fired: list[tuple[str, str]] = []
    async def notifier(*, user_id: str, text: str):
        fired.append((user_id, text))
    container.reminder_service.notifier = notifier
    user = container.users_repository.get_or_create(111)
    container.reminders_repository.create(user_id=user.id, body='late thing', next_fire_at=utc_now() - timedelta(minutes=1), recurrence=None)
    processed = await container.reminder_service.boot_recovery_sweep()
    assert processed == 1
    assert fired and fired[0][1].startswith('Delayed reminder')


@pytest.mark.asyncio
async def test_recurring_recovery_jumps_to_next_slot(container):
    user = container.users_repository.get_or_create(111)
    reminder = container.reminders_repository.create(user_id=user.id, body='weekly thing', next_fire_at=utc_now() - timedelta(days=3), recurrence='FREQ=WEEKLY;BYDAY=FR;BYHOUR=14;BYMINUTE=0')
    await container.reminder_service.boot_recovery_sweep()
    updated = container.reminders_repository.get_by_id(reminder.id)
    assert updated is not None
    assert updated.next_fire_at > utc_now()
