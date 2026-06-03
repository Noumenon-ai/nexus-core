from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from bootstrap import StartupValidator
from config import Settings, get_settings
from db import create_database
from migrations.conversation_turns_migration import run_conversation_turns_migration
from migrations.telos_onboarding_state_migration import run_telos_onboarding_state_migration
from migrations.google_oauth_migration import run_google_oauth_migration
from migrations.reminder_delivery_attempts_migration import (
    run_reminder_delivery_attempts_migration,
)
from migrations.voice_credentials_migration import run_voice_credentials_migration
from pipeline.google_intent import GoogleIntentHandler
from pipeline.tool_dispatcher import ToolDispatcher
from pipeline.types import PipelineOutput
from pipeline.unified import UnifiedPipeline
from repositories.approvals_repository import ApprovalsRepository
from repositories.conversation_context_repository import ConversationContextRepository
from repositories.conversation_turns_repository import ConversationTurnsRepository
from repositories.telos_onboarding_state_repository import TelosOnboardingStateRepository
from repositories.elevenlabs_usage_repository import ElevenLabsUsageRepository
from repositories.emails_ingested_repository import EmailsIngestedRepository
from repositories.memories_repository import MemoriesRepository
from repositories.opportunity_signals_repository import OpportunitySignalsRepository
from repositories.proactive_notifications_repository import ProactiveNotificationsRepository
from repositories.reminders_repository import RemindersRepository
from repositories.tasks_repository import TasksRepository
from repositories.users_repository import UsersRepository
from scheduler import NexusScheduler
from services.action_service import ActionService
from services.brain_router import BrainRouter
from services.approval_service import ApprovalService
from services.conversation_service import ConversationService
from services.dispatcher_registry import build_dispatcher_registry
from services.email_service import EmailService
from services.google_auth_service import GoogleAuthService
from services.habit_service import HabitService
from services.local_memory_service import LocalMemoryService
from services.memory_service import MemoryService
from services.opportunity_service import OpportunityService
from services.proactive_service import ProactiveService
from services.capability_registry import CapabilityRegistry
from services.reminder_parser import ReminderParser
from services.reminder_service import ReminderService
from services.runtime_identity import log_runtime_identity
from services.search_service import SearchService
from services.task_service import TaskService
from services.telos_service import TelosService
from services.voice_input_service import VoiceInputService
from services.voice_output_service import VoiceOutputService
from telegram_bot import TelegramBot
from utils.dates import app_now
from utils.i18n import Translator
from utils.web_search import WebSearchClient


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Runtime:
    settings: Settings
    users_repository: UsersRepository
    reminder_service: ReminderService
    proactive_service: ProactiveService
    approval_service: ApprovalService
    opportunity_service: OpportunityService
    telegram_bot: TelegramBot | None = None
    # Phase 6: cron tools + scheduler reconciliation
    cron_jobs_repository: 'CronJobsRepository | None' = None
    pipeline: 'UnifiedPipeline | None' = None
    nexus_scheduler: 'NexusScheduler | None' = None

    def translator_for(self, language: str) -> Translator:
        return Translator(language)

    async def notify_user(self, user_id: str, text: str) -> None:
        user = self.users_repository.get_by_id(user_id)
        if user is None or self.telegram_bot is None:
            return
        try:
            await self.telegram_bot._send_output(user.telegram_id, PipelineOutput(text=text))
        except Exception as exc:
            logger.warning('notify_user_failed: telegram_id=%s err=%s: %s', user.telegram_id, type(exc).__name__, exc)

    def notify_user_sync(self, user_id: str, text: str) -> None:
        asyncio.create_task(self.notify_user(user_id, text))


def build_runtime() -> tuple[Runtime, TelegramBot, NexusScheduler]:
    settings = get_settings()
    database = create_database(settings)
    StartupValidator(settings).run(database.engine)

    users_repository = UsersRepository(database.session_factory)
    for telegram_id in settings.core.allowed_telegram_ids:
        users_repository.get_or_create(telegram_id)
    run_voice_credentials_migration(database.engine, settings)
    run_google_oauth_migration(database.engine, settings)
    run_reminder_delivery_attempts_migration(database.engine)
    run_conversation_turns_migration(database.engine)
    run_telos_onboarding_state_migration(database.engine)
    reminders_repository = RemindersRepository(database.session_factory)
    tasks_repository = TasksRepository(database.session_factory)
    memories_repository = MemoriesRepository(database.session_factory)
    context_repository = ConversationContextRepository(database.session_factory)
    conversation_turns_repository = ConversationTurnsRepository(database.session_factory)
    onboarding_repository = TelosOnboardingStateRepository(database.session_factory)
    emails_repository = EmailsIngestedRepository(database.session_factory)
    approvals_repository = ApprovalsRepository(database.session_factory)
    proactive_repository = ProactiveNotificationsRepository(database.session_factory)
    signals_repository = OpportunitySignalsRepository(database.session_factory)
    # H2-047 Wave 2: GeminiUsageRepository instantiation removed. The ORM
    # model + table stay in DB for historical-data forensics; nothing
    # writes to it now that AIService is gone.
    elevenlabs_usage_repository = ElevenLabsUsageRepository(database.session_factory)

    telos_service = TelosService(settings.data_dir / 'telos')
    conversation_service = ConversationService(context_repository)
    # H2-047 Wave 2: brain_router is the single LLM entry point. The
    # reminder parser + voice rewrite previously held an AIService
    # instance for direct Gemini calls; both now route through this
    # same brain_router instance.
    brain_router = BrainRouter(ollama_url=None)
    reminder_parser = ReminderParser(brain_router, settings.core.app_timezone)
    scheduler = NexusScheduler(settings.core.database_url, settings.core.app_timezone)
    memory_service = MemoryService(memories_repository)
    habit_service = HabitService(memories_repository)
    reminder_service = ReminderService(reminders_repository, reminder_parser, conversation_service, settings.core.app_timezone, scheduler=scheduler, habit_service=habit_service)
    task_service = TaskService(tasks_repository, reminders_repository, memories_repository, settings.core.app_timezone, habit_service=habit_service)
    approval_service = ApprovalService(approvals_repository, message_ttl_minutes=settings.approval.approval_ttl_message_minutes, system_ttl_minutes=settings.approval.approval_ttl_system_minutes)
    search_service = SearchService(WebSearchClient(backend=settings.search.web_search_backend, serpapi_key=settings.search.serpapi_key), enabled=settings.search.web_search_enabled, max_query_chars=settings.search.web_search_max_query_chars)
    email_service = EmailService(settings, emails_repository)
    opportunity_service = OpportunityService(settings, memories_repository, signals_repository, search_service)
    proactive_service = ProactiveService(proactive_repository, reminder_service, task_service, email_service, emails_repository, habit_service, settings.core.app_timezone, settings.proactive.briefing_skip_if_late_by_hours, telos_service=telos_service, onboarding_repository=onboarding_repository)
    action_service = ActionService(memory_service, task_service)
    voice_input_service = VoiceInputService(backend=settings.voice.voice_input_backend, model_size=settings.voice.voice_model_size, groq_api_key=settings.voice.groq_api_key, groq_model=settings.voice.groq_transcription_model) if settings.voice.voice_input_enabled else None
    voice_output_service = VoiceOutputService(settings, users_repository=users_repository, usage_repository=elevenlabs_usage_repository, brain_router=brain_router) if settings.voice.voice_output_enabled else None

    google_auth_service = GoogleAuthService(settings, users_repository) if settings.google.enabled else None
    google_intent_handler = GoogleIntentHandler(google_auth_service) if google_auth_service is not None else None

    # V3.9: dispatcher is the only message-handling path. Bootstrap cost:
    # Memory.from_config() loads BM25 + spaCy NER + spaCy lemma eagerly
    # at startup (~3.6 s, paid once at process start). The
    # NEXUS_DISPATCHER_ENABLED env var is read here only as a no-op
    # tombstone: deploy environments may still set it; we ignore the
    # value. Removal of the var itself is a future cleanup.
    _ = os.getenv('NEXUS_DISPATCHER_ENABLED')  # tombstone read; value ignored
    tool_dispatcher: ToolDispatcher | None = None
    try:
        # H2-047 Wave 2: LocalMemoryService is the only archive backend.
        # The H2-046 NEXUS_MEMORY_BACKEND=mem0 rollback is retired now
        # that Gemini is fully removed — mem0_service.py + mem0ai +
        # qdrant-client are deleted in this commit. .data/qdrant/ stays
        # on disk as historical data per the standing-no-delete rule.
        archive_memory = LocalMemoryService()
        logger.info('memory_backend_selected', extra={'backend': 'local'})
        if google_auth_service is not None:
            async def _google_disconnect(uid: str) -> None:
                await google_auth_service.disconnect(uid)
            dispatcher_registry = build_dispatcher_registry(
                reminders_repository=reminders_repository,
                tasks_repository=tasks_repository,
                memories_repository=memories_repository,
                emails_repository=emails_repository,
                approvals_repository=approvals_repository,
                telos_service=telos_service,
                scheduler=scheduler,
                habit_service=habit_service,
                google_disconnect=_google_disconnect,
                app_timezone=settings.core.app_timezone,
                onboarding_repository=onboarding_repository,
                users_repository=users_repository,
            )
            tool_dispatcher = ToolDispatcher(
                llm=brain_router,  # H2-047: single LLM entry, no AIService
                registry=dispatcher_registry,
                telos_service=telos_service,
                mem0=archive_memory,  # LocalMemoryService — same surface as mem0 had
                approval_service=approval_service,
                conversation_turns_repository=conversation_turns_repository,
                conversation_service=conversation_service,
                approvals_repository=approvals_repository,
                proactive_notifications_repository=proactive_repository,
                app_timezone=settings.core.app_timezone,
                capability_registry=CapabilityRegistry(
                    settings=settings,
                    google_auth_service=google_auth_service,
                ),
            )
    except Exception:
        logger.exception('dispatcher_bootstrap_failed')
        tool_dispatcher = None

    pipeline = UnifiedPipeline(
        users_repository=users_repository,
        conversation_service=conversation_service,
        voice_input_service=voice_input_service,
        voice_output_service=voice_output_service,
        allowed_telegram_ids=settings.core.allowed_telegram_ids,
        tool_dispatcher=tool_dispatcher,
    )
    telegram_bot = TelegramBot(
        token=settings.core.telegram_bot_token,
        pipeline=pipeline,
        voice_in_dir=settings.voice_in_dir,
        max_voice_file_bytes=settings.voice.voice_input_max_file_bytes,
    )
    from repositories.cron_jobs_repository import CronJobsRepository
    cron_jobs_repository = CronJobsRepository(database.session_factory)

    runtime = Runtime(
        settings=settings, users_repository=users_repository,
        reminder_service=reminder_service, proactive_service=proactive_service,
        approval_service=approval_service, opportunity_service=opportunity_service,
        telegram_bot=telegram_bot,
        cron_jobs_repository=cron_jobs_repository,
        pipeline=pipeline,
        nexus_scheduler=scheduler,
    )
    reminder_service.notifier = runtime.notify_user
    action_service.messenger = runtime.notify_user_sync
    if google_intent_handler is not None:
        google_intent_handler.notifier = runtime.notify_user
    scheduler.register_runtime(runtime)
    return runtime, telegram_bot, scheduler


async def recover_startup_state(runtime: Runtime, settings: Settings) -> None:
    await runtime.reminder_service.boot_recovery_sweep()
    if not settings.proactive.proactive_enabled:
        return
    now = app_now(settings.core.app_timezone)
    loops = (
        ('morning_briefing', settings.proactive.morning_briefing_hour),
        ('midday_check', settings.proactive.midday_check_hour),
        ('evening_wrap', settings.proactive.evening_wrap_hour),
    )
    for user in runtime.users_repository.list_all():
        if not user.proactive_enabled:
            continue
        translator = runtime.translator_for(user.language)
        for loop_type, hour in loops:
            scheduled_for = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if scheduled_for > now:
                continue
            response = await runtime.proactive_service.recover_briefing(user, loop_type, scheduled_for, translator)
            if response is None or not response.metadata.get('should_send', True):
                continue
            await runtime.notify_user(user.id, response.text)
            if response.metadata.get('nudge_included'):
                runtime.proactive_service.mark_nudge_committed(user.id)


def main() -> None:
    settings = get_settings()
    runtime, telegram_bot, scheduler = build_runtime()
    log_runtime_identity(logger)
    scheduler.schedule_housekeeping(approval_interval_sec=settings.approval.approval_sweep_interval_sec)
    users = runtime.users_repository.list_all()
    if settings.proactive.proactive_enabled:
        scheduler.schedule_daily_loops(users, morning_hour=settings.proactive.morning_briefing_hour, midday_hour=settings.proactive.midday_check_hour, evening_hour=settings.proactive.evening_wrap_hour)
    if settings.search.web_search_enabled:
        scheduler.schedule_opportunity_scans(users, interval_hours=settings.search.radar_run_interval_hours)
    # Phase 6: pick up cron rows + register them with APScheduler. Sync loop runs
    # every 30s so MCP-created jobs start firing within at most that window.
    if runtime.cron_jobs_repository is not None:
        scheduler.schedule_cron_sync(interval_sec=30)
        scheduler.restore_cron_jobs(runtime.cron_jobs_repository.all_active())
    # H2-049: same pattern for reminders. MCP create_reminder writes the
    # DB but can't reach this process's APScheduler instance directly, so
    # the sync sweep bridges the gap every 30s.
    scheduler.schedule_reminder_sync(interval_sec=30)

    async def _post_init(_application) -> None:
        await recover_startup_state(runtime, settings)
        scheduler.start()

    async def _post_stop(_application) -> None:
        # H2-040: graceful drain. Wait up to 10s for any in-flight Claude/Codex
        # CLI subprocess (tracked by BrainRouter._active_subprocesses) to
        # finish before the process exits. Only effective when the systemd
        # unit uses KillMode=mixed; the default KillMode=control-group sends
        # SIGTERM to the entire cgroup, killing children before we can drain.
        drained_cleanly = await BrainRouter.drain(timeout_seconds=10.0)
        if drained_cleanly:
            logger.info('shutdown_drain_complete')
        else:
            logger.warning('shutdown_drain_timeout')

    telegram_bot.application.post_init = _post_init
    telegram_bot.application.post_stop = _post_stop
    telegram_bot.run()


if __name__ == '__main__':
    main()
