from __future__ import annotations

from datetime import datetime, timedelta, timezone

from models import User
from pipeline.types import ServiceResponse
from repositories.emails_ingested_repository import EmailsIngestedRepository
from repositories.proactive_notifications_repository import ProactiveNotificationsRepository
from repositories.telos_onboarding_state_repository import TelosOnboardingStateRepository
from services.email_service import EmailService
from services.habit_service import HabitService
from services.reminder_duplicates import (
    compress_duplicate_reminder_lines,
)
from services.reminder_service import ReminderService
from services.task_service import TaskService
from services.telos_service import TelosService
from utils.dates import app_now, format_local_datetime, utc_now
from utils.i18n import Translator


_TELOS_NUDGE_INTERVAL_DAYS = 7
_TELOS_NUDGE_TEXT = (
    "P.S. — I work better when I know your priorities. "
    "Reply 'start telos' (5 minutes) to set that up."
)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)

def _compress_reminders_for_briefing(reminders, *, app_timezone: str) -> tuple[list[str], int]:
    def _format(item) -> str:
        if hasattr(item, 'display_body'):
            suffix = (
                f' ({item.duplicate_count} duplicates found)'
                if item.duplicate_count > 0
                else ''
            )
            return f'- {item.scheduled_label}: {item.display_body}{suffix}'
        return (
            f"- {format_local_datetime(item.next_fire_at, app_timezone)}: "
            f'{item.body}'
        )

    return compress_duplicate_reminder_lines(
        reminders,
        app_timezone=app_timezone,
        line_formatter=_format,
        limit=3,
    )


class ProactiveService:
    def __init__(self, proactive_repository: ProactiveNotificationsRepository, reminder_service: ReminderService, task_service: TaskService, email_service: EmailService, emails_repository: EmailsIngestedRepository, habit_service: HabitService, app_timezone: str, skip_if_late_hours: int, *, telos_service: TelosService | None = None, onboarding_repository: TelosOnboardingStateRepository | None = None) -> None:
        self.proactive_repository = proactive_repository
        self.reminder_service = reminder_service
        self.task_service = task_service
        self.email_service = email_service
        self.emails_repository = emails_repository
        self.habit_service = habit_service
        self.app_timezone = app_timezone
        self.skip_if_late_hours = skip_if_late_hours
        # V3.8 nudge integration deps. Both optional so existing test
        # fixtures that construct ProactiveService without TELOS deps
        # keep compiling. When either is None, the nudge logic is a
        # silent no-op — briefings simply ship without a P.S. line.
        self.telos_service = telos_service
        self.onboarding_repository = onboarding_repository

    async def morning_briefing(self, user: User, translator: Translator, *, explicit: bool = False) -> ServiceResponse:
        reminders = self.reminder_service.reminders_repository.list_active(user.id)[:5]
        tasks_response = self.task_service.organize_day(user, translator)
        emails = self.emails_repository.list_recent(user.id, hours=24)[:3]
        habit = self.habit_service.suggestion_for_user(user.id, translator)
        lines = [translator.t('good_morning_intro')]
        if reminders:
            lines.append(translator.t('proactive_reminders_today'))
            reminder_lines, duplicate_count = _compress_reminders_for_briefing(
                reminders[:5],
                app_timezone=self.app_timezone,
            )
            lines.extend(reminder_lines)
            if duplicate_count:
                lines.append(
                    f'I found {duplicate_count} similar reminder'
                    f'{"s" if duplicate_count != 1 else ""}. Want me to clean them up?'
                )
        if tasks_response.text:
            lines.append(translator.t('proactive_top_tasks'))
            lines.extend(tasks_response.text.splitlines()[:3])
        if emails:
            lines.append(translator.t('proactive_email_signals'))
            for email in emails:
                lines.append(f"- [{email.category}] {email.subject or '(no subject)'}")
        if habit:
            lines.append(habit)
        # V3.8 nudge: append the TELOS-onboarding P.S. line ONLY when the
        # user has no TELOS file yet AND last nudge was >7 days ago AND
        # this is a SCHEDULED (cron) briefing — not an explicit user
        # request via "good morning". Explicit-path is suppressed because
        # commit-after-send wiring through the explicit reply path would
        # require new contract surface for what is a rare case (the user
        # already engaged the bot manually). We do NOT update last_nudge_at
        # here regardless — caller commits via mark_nudge_committed AFTER
        # notify_user succeeds, so test harnesses / debug commands that
        # compute the briefing without delivering it never burn the
        # once-per-7-days budget.
        nudge_included = (not explicit) and self._should_include_telos_nudge(user.id)
        if nudge_included:
            lines.append('')
            lines.append(_TELOS_NUDGE_TEXT)
        message = '\n'.join(lines)
        should_send = True
        if not explicit:
            should_send = self._record_if_new(user.id, 'morning_briefing', self._daily_dedupe_key('morning_briefing'), message)
        metadata = {'should_send': should_send, 'nudge_included': nudge_included}
        return ServiceResponse(text=message, voice_appropriate=True, metadata=metadata)

    def _should_include_telos_nudge(self, user_id: str) -> bool:
        """V3.8 nudge gate. Returns True iff:
          (a) telos_service + onboarding_repository are both wired
          (b) the user has no TELOS file yet
          (c) last_nudge_at is NULL or >= 7 days ago

        All three must hold. Any False short-circuits to False.
        """
        if self.telos_service is None or self.onboarding_repository is None:
            return False
        if self.telos_service.has_telos(user_id):
            return False
        state = self.onboarding_repository.get(user_id)
        if state is None or state.last_nudge_at is None:
            return True
        last_at = _ensure_utc(state.last_nudge_at)
        return (utc_now() - last_at) >= timedelta(days=_TELOS_NUDGE_INTERVAL_DAYS)

    def mark_nudge_committed(self, user_id: str) -> None:
        """V3.8: callers invoke this AFTER notify_user succeeds AND
        morning_briefing's metadata['nudge_included'] was True. Updates
        last_nudge_at to now. No-op when onboarding_repository is None
        (test fixtures without TELOS deps)."""
        if self.onboarding_repository is None:
            return
        self.onboarding_repository.mark_nudged(user_id)

    async def midday_check(self, user: User, translator: Translator) -> ServiceResponse:
        plan = self.task_service.organize_day(user, translator)
        emails = self.emails_repository.list_recent(user.id, hours=6)[:3]
        lines = [translator.t('proactive_midday_intro')]
        lines.extend(plan.text.splitlines()[:4])
        if emails:
            lines.append(translator.t('proactive_midday_email'))
            lines.extend(f"- [{email.category}] {email.subject or '(no subject)'}" for email in emails)
        message = '\n'.join(lines)
        should_send = self._record_if_new(user.id, 'midday_check', self._daily_dedupe_key('midday_check'), message)
        return ServiceResponse(text=message, voice_appropriate=True, metadata={'should_send': should_send})

    async def evening_wrap(self, user: User, translator: Translator) -> ServiceResponse:
        plan = self.task_service.organize_day(user, translator)
        deviation = self.habit_service.detect_deviation(user.id, app_now(self.app_timezone).hour, translator)
        lines = [translator.t('proactive_evening_intro')]
        lines.extend(plan.text.splitlines()[:4])
        if deviation:
            lines.append(deviation)
        message = '\n'.join(lines)
        should_send = self._record_if_new(user.id, 'evening_wrap', self._daily_dedupe_key('evening_wrap'), message)
        return ServiceResponse(text=message, voice_appropriate=True, metadata={'should_send': should_send})

    async def recover_briefing(self, user: User, loop_type: str, scheduled_for: datetime, translator: Translator) -> ServiceResponse | None:
        if utc_now() - scheduled_for > timedelta(hours=self.skip_if_late_hours):
            return None
        if loop_type == 'morning_briefing':
            return await self.morning_briefing(user, translator)
        if loop_type == 'midday_check':
            return await self.midday_check(user, translator)
        return await self.evening_wrap(user, translator)

    def _daily_dedupe_key(self, notification_type: str) -> str:
        return f"{notification_type}:{app_now(self.app_timezone).date().isoformat()}"

    def _record_if_new(self, user_id: str, notification_type: str, dedupe_key: str, message: str) -> bool:
        if self.proactive_repository.exists_recent(user_id=user_id, dedupe_key=dedupe_key, hours=24):
            return False
        self.proactive_repository.record(user_id=user_id, notification_type=notification_type, message=message, dedupe_key=dedupe_key)
        return True
