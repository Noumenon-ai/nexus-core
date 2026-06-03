from __future__ import annotations

import re
from datetime import timedelta

import dateparser

from models import User
from pipeline.types import ServiceResponse
from repositories.memories_repository import MemoriesRepository
from repositories.reminders_repository import RemindersRepository
from services.reminder_duplicates import compress_duplicate_reminder_lines
from repositories.tasks_repository import TasksRepository
from utils.dates import app_now, format_local_datetime, format_reminder_when, to_utc, utc_now
from utils.i18n import Translator


class TaskService:
    def __init__(self, tasks_repository: TasksRepository, reminders_repository: RemindersRepository, memories_repository: MemoriesRepository, app_timezone: str, habit_service=None) -> None:
        self.tasks_repository = tasks_repository
        self.reminders_repository = reminders_repository
        self.memories_repository = memories_repository
        self.app_timezone = app_timezone
        self.habit_service = habit_service

    def handle(self, user: User, text: str, translator: Translator) -> ServiceResponse:
        normalized = text.strip().lower()
        if normalized.startswith('add task'):
            return self.add_task(user, text, translator)
        if normalized.startswith('mark '):
            return self.mark_done(user, text, translator)
        return self.organize_day(user, translator)

    def add_task(self, user: User, text: str, translator: Translator) -> ServiceResponse:
        body = text.split(':', 1)[1].strip() if ':' in text else self._suffix_after_prefix(text, 'add task')
        if not body:
            return ServiceResponse(text=translator.t('task_need_title'))
        due_at = self._extract_due_at(body)
        clean_title = self._strip_due_phrase(body)
        if not clean_title.strip():
            return ServiceResponse(text=translator.t('task_need_title'))
        self.tasks_repository.create(user_id=user.id, title=clean_title, due_at=due_at, priority=0, source='user')
        return ServiceResponse(text=translator.t('task_added'))

    def mark_done(self, user: User, text: str, translator: Translator) -> ServiceResponse:
        cleaned = text.removeprefix('mark').replace('done', '').strip()
        if not cleaned:
            return ServiceResponse(text=translator.t('task_need_title'))
        task = self.tasks_repository.mark_done(user.id, cleaned)
        if task is None:
            return ServiceResponse(text=translator.t('task_not_found'))
        if self.habit_service is not None and task.completed_at is not None:
            self.habit_service.record_task_completion(user.id, task.completed_at)
        return ServiceResponse(text=translator.t('task_done'))

    def organize_day(self, user: User, translator: Translator) -> ServiceResponse:
        reminders = self.reminders_repository.list_due_within(user.id, minutes=60)
        tasks = self.tasks_repository.list_pending(user.id)
        now = utc_now()
        overdue = [task for task in tasks if task.due_at and task.due_at < now]
        due_today = [task for task in tasks if task.due_at and task.due_at.date() == now.date() and task not in overdue]
        upcoming_soon = [
            task for task in tasks
            if task.due_at
            and task not in overdue
            and task not in due_today
            and task.due_at <= now + timedelta(hours=6)
        ]
        unscheduled = [task for task in tasks if task.due_at is None]
        lines = []
        def _format_reminder(item) -> str:
            when_text = format_reminder_when(item.next_fire_at, self.app_timezone, reference=now)
            if hasattr(item, 'display_body'):
                suffix = (
                    f' ({item.duplicate_count} duplicates found)'
                    if item.duplicate_count > 0
                    else ''
                )
                return translator.t(
                    'task_line_reminder_soon',
                    body=f'{item.display_body}{suffix}',
                    when_text=when_text,
                )
            return translator.t('task_line_reminder_soon', body=item.body, when_text=when_text)

        reminder_lines, _duplicate_count = compress_duplicate_reminder_lines(
            reminders,
            app_timezone=self.app_timezone,
            line_formatter=_format_reminder,
        )
        lines.extend(reminder_lines)
        for task in sorted(overdue, key=lambda item: (-(item.priority), item.due_at or now)):
            lines.append(translator.t('task_line_overdue', title=task.title))
        for task in sorted(due_today, key=lambda item: (-(item.priority), item.due_at or now))[:3]:
            when_text = format_local_datetime(task.due_at, self.app_timezone)
            lines.append(translator.t('task_line_today', title=task.title, when_text=when_text))
        for task in sorted(upcoming_soon, key=lambda item: (-(item.priority), item.due_at or now))[:3]:
            when_text = format_local_datetime(task.due_at, self.app_timezone)
            lines.append(translator.t('task_line_upcoming', title=task.title, when_text=when_text))
        for task in unscheduled[:3]:
            lines.append(translator.t('task_line_pending', title=task.title))
        if not lines:
            lines.append(translator.t('task_clear'))
        preference = self.memories_repository.get(user_id=user.id, memory_type='preference', key='reminder_time_preference')
        if preference is not None:
            lines.append(translator.t('task_preference', value=preference.value))
        return ServiceResponse(text='\n'.join(lines), voice_appropriate=False)

    def _extract_due_at(self, body: str):
        parsed = dateparser.parse(body, settings={'TIMEZONE': self.app_timezone, 'RETURN_AS_TIMEZONE_AWARE': True, 'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': app_now(self.app_timezone)})
        return to_utc(parsed) if parsed else None

    def _strip_due_phrase(self, body: str) -> str:
        return re.sub(r'\b(today|tomorrow|tonight|at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b', '', body, flags=re.IGNORECASE).strip(' ,') or body

    def _suffix_after_prefix(self, text: str, prefix: str) -> str:
        lowered = text.lower()
        index = lowered.find(prefix)
        if index == -1:
            return text.strip()
        return text[index + len(prefix):].strip()
