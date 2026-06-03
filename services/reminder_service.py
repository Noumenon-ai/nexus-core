from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from models import Reminder, User, now_utc
from pipeline.types import InlineButton, ServiceResponse
from repositories.reminders_repository import RemindersRepository
from services.conversation_service import ConversationService
from services.delivery_truth import (
    DELIVERY_FAILED,
    DELIVERY_REQUESTED,
    normalize_dispatch_delivery,
)
from services.reminder_duplicates import cluster_duplicate_reminders
from services.reminder_parser import ReminderParseResult, ReminderParser
from utils.dates import format_local_datetime, utc_now
from utils.i18n import Translator


logger = logging.getLogger(__name__)

_MAX_DELIVERY_ATTEMPTS = 3
_REDELIVERY_WINDOW = timedelta(hours=24)


class ReminderService:
    def __init__(self, reminders_repository: RemindersRepository,
                 parser: ReminderParser | None = None,
                 conversation_service: ConversationService | None = None,
                 app_timezone: str | None = None,
                 scheduler=None, notifier=None, habit_service=None,
                 session_factory=None) -> None:
        self.reminders_repository = reminders_repository
        self.parser = parser
        self.conversation_service = conversation_service
        self.app_timezone = app_timezone
        self.scheduler = scheduler
        self.notifier = notifier
        self.habit_service = habit_service
        # session_factory falls back to the repo's session_factory so existing
        # call sites work without passing it explicitly.
        self._session_factory = session_factory or getattr(
            reminders_repository, 'session_factory', None,
        )

    async def handle(self, user: User, text: str, translator: Translator) -> ServiceResponse:
        normalized = text.strip().lower()
        if normalized in {'yes', 'correct', 'confirm'}:
            return self.confirm_pending(user, translator)
        if normalized in {'no', 'edit', 'change'}:
            self.conversation_service.clear_pending_reminder(user.id)
            return ServiceResponse(text=translator.t('clarify_generic'))
        if normalized.startswith('show reminders') or normalized.startswith('list my reminders'):
            return self.list_reminders(user, translator)
        if normalized.startswith('cancel reminder') or normalized.startswith('cancel my'):
            return self.cancel_reminder(user, self._cancel_query_from_text(text), translator)
        context = self.conversation_service.load_context(user.id)
        context_subject = context.last_entity_value if 'about that' in normalized else None
        parse = await self.parser.parse(user.id, text, context_subject=context_subject)
        if parse.type == 'ambiguous' or parse.datetime_utc is None:
            return ServiceResponse(text=parse.clarifying_question or translator.t('clarify_generic'))
        when_text = self._render_confirmation(parse, translator)
        payload = {
            'body': parse.body,
            'datetime_utc': parse.datetime_utc.isoformat(),
            'rrule': parse.rrule,
            'type': parse.type,
            'when_text': when_text,
            'expires_at': (utc_now() + timedelta(seconds=60)).isoformat(),
        }
        self.conversation_service.store_pending_reminder(user.id, payload)
        return ServiceResponse(
            text=translator.t('reminder_confirm', when_text=when_text, body=parse.body),
            buttons=[
                InlineButton(text=translator.t('reminder_button_confirm'), callback_data='reminder:confirm'),
                InlineButton(text=translator.t('reminder_button_edit'), callback_data='reminder:edit'),
            ],
        )

    def confirm_pending(self, user: User, translator: Translator) -> ServiceResponse:
        pending = self.conversation_service.get_pending_reminder(user.id)
        if pending is None:
            return ServiceResponse(text=translator.t('clarify_generic'))
        reminder = self.reminders_repository.create(
            user_id=user.id,
            body=pending['body'],
            next_fire_at=datetime.fromisoformat(pending['datetime_utc']),
            recurrence=pending.get('rrule'),
        )
        if self.scheduler is not None:
            self.scheduler.schedule_reminder(reminder.id, reminder.next_fire_at)
        if self.habit_service is not None:
            self.habit_service.record_reminder_creation(user.id, reminder.created_at)
        self.conversation_service.clear_pending_reminder(user.id)
        return ServiceResponse(text=translator.t('reminder_created', when_text=pending['when_text']))

    def list_reminders(self, user: User, translator: Translator) -> ServiceResponse:
        reminders = self.reminders_repository.list_active(user.id)
        if not reminders:
            return ServiceResponse(text=translator.t('reminder_list_empty'))
        lines = [f"- {format_local_datetime(reminder.next_fire_at, self.app_timezone)}: {reminder.body}" for reminder in reminders]
        return ServiceResponse(text='\n'.join(lines), voice_appropriate=False)

    def cancel_reminder(self, user: User, query: str, translator: Translator) -> ServiceResponse:
        if not query.strip():
            return ServiceResponse(text=translator.t('reminder_need_match'))
        reminder = self.reminders_repository.find_for_cancel(user.id, query)
        if reminder is None:
            return ServiceResponse(text=translator.t('reminder_not_found'))
        self.reminders_repository.cancel(reminder.id)
        if self.scheduler is not None:
            self.scheduler.remove_reminder(reminder.id)
        return ServiceResponse(text=translator.t('reminder_cancelled'))

    def _cancel_query_from_text(self, text: str) -> str:
        lowered = text.lower().strip()
        for prefix in ('cancel reminder', 'cancel my'):
            if lowered.startswith(prefix):
                return text[len(prefix):].strip()
        return text.strip()

    async def fire_reminder(self, reminder_id: str, *, delayed: bool = False) -> bool:
        reminder = self.reminders_repository.get_by_id(reminder_id)
        if reminder is None or reminder.status != 'active':
            return False
        duplicate_decision = self._duplicate_fire_decision(reminder)
        if duplicate_decision == 'suppressed':
            logger.info(
                'reminder_duplicate_fire_suppressed',
                extra={'reminder_id': reminder.id, 'user_id': reminder.user_id},
            )
            self._mark_duplicate_suppressed(reminder)
            return True
        expected_next_fire = self._as_utc(reminder.next_fire_at)
        last_fired_at = self._as_utc(reminder.last_fired_at) if reminder.last_fired_at is not None else None
        if last_fired_at is not None and last_fired_at >= expected_next_fire:
            return False
        fired_at = utc_now()
        if reminder.recurrence:
            next_fire_at = self.parser.compute_next_slot(reminder.recurrence, after=fired_at)
            claimed = self.reminders_repository.claim_fire(reminder.id, expected_next_fire_at=expected_next_fire, fired_at=fired_at, next_fire_at=next_fire_at)
            if claimed is None:
                return False
            if self.scheduler is not None:
                self.scheduler.schedule_reminder(reminder.id, next_fire_at)
        else:
            claimed = self.reminders_repository.claim_fire(reminder.id, expected_next_fire_at=expected_next_fire, fired_at=fired_at, status='fired')
            if claimed is None:
                return False
        if self.notifier is not None:
            translator = Translator('en')
            text = reminder.body if not delayed else translator.t('reminder_delayed', body=reminder.body)
            await self.notifier(user_id=reminder.user_id, text=text)
        return True


    async def redeliver_failed(self) -> int:
        # External-contact delivery removed in nexus-core; nothing to redeliver.
        return 0

    async def boot_recovery_sweep(self) -> int:
        reminders = self.reminders_repository.list_all_active()
        processed = 0
        now = utc_now()
        for reminder in reminders:
            next_fire_at = self._as_utc(reminder.next_fire_at)
            if next_fire_at > now:
                if self.scheduler is not None:
                    self.scheduler.schedule_reminder(reminder.id, next_fire_at)
                processed += 1
                continue
            await self.fire_reminder(reminder.id, delayed=True)
            processed += 1
        return processed

    def reconcile_with_scheduler(self) -> int:
        """H2-049: periodic reconciliation called by dispatch_reminder_sync.

        Idempotent — re-queueing an already-queued reminder is a no-op
        thanks to replace_existing=True. The aim is to catch MCP-created
        reminders that wrote to the DB but never queued with APScheduler
        (the MCP server runs as a subprocess and can't reach the bot's
        running scheduler instance).

        Returns the count of reminders re-queued (for diagnostic logging,
        if a caller wants it). Skips reminders that have already fired
        (status != 'active' OR last_fired_at IS NOT NULL OR next_fire_at
        in the past) so we never resurrect a delivered reminder.
        """
        if self.scheduler is None:
            return 0
        reminders = self.reminders_repository.list_all_active()
        now = utc_now()
        queued = 0
        for reminder in reminders:
            if getattr(reminder, 'last_fired_at', None) is not None:
                # One-shot reminder already fired (its status would also be
                # 'fired'; double-check via last_fired_at to handle any
                # races between repo writes).
                continue
            next_fire_at = self._as_utc(reminder.next_fire_at)
            if next_fire_at <= now:
                # Past-due: leave for boot_recovery_sweep on next restart
                # rather than firing late from inside an interval job
                # (recovery handles the "fire delayed" semantics).
                continue
            self.scheduler.schedule_reminder(reminder.id, next_fire_at)
            queued += 1
        return queued


    def _render_confirmation(self, parse: ReminderParseResult, translator: Translator) -> str:
        assert parse.datetime_utc is not None
        display = format_local_datetime(parse.datetime_utc, self.app_timezone)
        if parse.type == 'recurring' and parse.natural_text:
            if not parse.defaulted_time:
                return parse.natural_text
            return translator.t('reminder_default_time_recurring', natural_text=parse.natural_text)
        if parse.defaulted_time:
            return translator.t('reminder_default_time', display=display)
        return display

    def _as_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _duplicate_fire_decision(self, reminder) -> str:
        reminders = self.reminders_repository.list_active(reminder.user_id)
        due_candidates = [
            item for item in reminders
            if self._as_utc(item.next_fire_at) <= self._as_utc(reminder.next_fire_at)
            and item.status == 'active'
        ]
        clusters = cluster_duplicate_reminders(
            due_candidates,
            app_timezone=self.app_timezone or 'UTC',
        )
        for cluster in clusters:
            if reminder.id not in cluster.reminder_ids:
                continue
            if cluster.newest_reminder_id == reminder.id:
                return 'primary'
            return 'suppressed'
        return 'primary'

    def _mark_duplicate_suppressed(self, reminder) -> None:
        expected_next_fire = self._as_utc(reminder.next_fire_at)
        fired_at = utc_now()
        if reminder.recurrence:
            next_fire_at = self.parser.compute_next_slot(
                reminder.recurrence,
                after=fired_at,
            )
            claimed = self.reminders_repository.claim_fire(
                reminder.id,
                expected_next_fire_at=expected_next_fire,
                fired_at=fired_at,
                next_fire_at=next_fire_at,
            )
            if claimed is not None and self.scheduler is not None:
                self.scheduler.schedule_reminder(reminder.id, next_fire_at)
            return
        self.reminders_repository.claim_fire(
            reminder.id,
            expected_next_fire_at=expected_next_fire,
            fired_at=fired_at,
            status='fired',
        )
