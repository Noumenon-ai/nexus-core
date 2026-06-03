from __future__ import annotations

from datetime import timedelta

from pipeline.types import PipelineInput
from utils.dates import utc_now
from utils.i18n import Translator


def test_explicit_memory_is_saved_and_listed(container):
    user = container.users_repository.get_or_create(111)
    saved = container.memory_service.remember(user, 'remember I prefer morning reminders', __import__('utils.i18n', fromlist=['Translator']).Translator())
    listed = container.memory_service.list_memories(user, __import__('utils.i18n', fromlist=['Translator']).Translator())
    assert 'remember' in saved.text.lower()
    assert 'reminder_time_preference' in listed.text


def test_habit_learning_tracks_hours(container):
    user = container.users_repository.get_or_create(111)
    now = utc_now()
    container.habit_service.record_reminder_creation(user.id, now)
    container.habit_service.record_reminder_creation(user.id, now)
    suggestion = container.habit_service.suggestion_for_user(user.id)
    assert suggestion is not None
    assert str(now.hour).zfill(2) in suggestion


def test_task_add_and_mark_done(container):
    user = container.users_repository.get_or_create(111)
    translator = __import__('utils.i18n', fromlist=['Translator']).Translator()
    added = container.task_service.add_task(user, 'add task: go to gym tomorrow', translator)
    done = container.task_service.mark_done(user, 'mark gym done', translator)
    assert 'added' in added.text.lower()
    assert 'done' in done.text.lower()


def test_task_due_phrase_is_stripped_from_title(container):
    user = container.users_repository.get_or_create(111)
    translator = __import__('utils.i18n', fromlist=['Translator']).Translator()
    container.task_service.add_task(user, 'add task: go to gym tomorrow at 9am', translator)
    task = container.tasks_repository.list_pending(user.id)[0]
    assert task.title == 'go to gym'


def test_empty_mark_done_does_not_complete_first_task(container):
    user = container.users_repository.get_or_create(111)
    translator = Translator()
    container.task_service.add_task(user, 'add task: go to gym tomorrow', translator)
    response = container.task_service.mark_done(user, 'mark done', translator)
    assert 'which task' in response.text.lower()
    assert len(container.tasks_repository.list_pending(user.id)) == 1


def test_day_organization_prioritizes_nearby_reminders(container):
    user = container.users_repository.get_or_create(111)
    container.reminders_repository.create(user_id=user.id, body='leave now', next_fire_at=utc_now() + timedelta(minutes=30), recurrence=None)
    container.tasks_repository.create(user_id=user.id, title='later task', due_at=utc_now() + timedelta(hours=5), priority=0, source='user')
    plan = container.task_service.organize_day(user, __import__('utils.i18n', fromlist=['Translator']).Translator())
    first_line = plan.text.splitlines()[0]
    assert first_line.startswith('Reminder ')
    assert 'leave now' in first_line
    assert 'in 30 min' in first_line


def test_day_organization_includes_upcoming_cross_midnight_task(container):
    user = container.users_repository.get_or_create(111)
    container.tasks_repository.create(
        user_id=user.id,
        title='finish report',
        due_at=utc_now() + timedelta(hours=4),
        priority=1,
        source='user',
    )
    plan = container.task_service.organize_day(user, Translator())
    assert 'finish report' in plan.text.lower()


def test_cancel_reminder_without_query_does_not_cancel_only_reminder(container):
    user = container.users_repository.get_or_create(111)
    translator = Translator()
    container.reminders_repository.create(user_id=user.id, body='doctor appointment', next_fire_at=utc_now() + timedelta(hours=1), recurrence=None)
    response = container.reminder_service.handle
    result = __import__('asyncio').run(response(user, 'cancel reminder', translator))
    assert 'which reminder' in result.text.lower()
    assert len(container.reminders_repository.list_active(user.id)) == 1


def test_habit_learning_is_wired_into_live_reminders_and_tasks(container):
    user = container.users_repository.get_or_create(111)
    translator = Translator()
    now = utc_now()
    container.conversation_service.store_pending_reminder(
        user.id,
        {
            'body': 'stretch',
            'datetime_utc': (now + timedelta(hours=1)).isoformat(),
            'rrule': None,
            'type': 'one_shot',
            'when_text': 'in one hour',
            'expires_at': (now + timedelta(minutes=10)).isoformat(),
        },
    )
    container.reminder_service.confirm_pending(user, translator)
    container.task_service.add_task(user, 'add task: review notes tomorrow', translator)
    container.task_service.mark_done(user, 'mark review notes done', translator)
    assert container.memories_repository.get(user_id=user.id, memory_type='habit', key='reminder_creation_hours') is not None
    assert container.memories_repository.get(user_id=user.id, memory_type='habit', key='task_completion_hours') is not None
