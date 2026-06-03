from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bootstrap import StartupValidator
from config import get_settings
from db import create_all, create_database
from pipeline.unified import UnifiedPipeline
from repositories.approvals_repository import ApprovalsRepository
from repositories.conversation_context_repository import ConversationContextRepository
from repositories.conversation_turns_repository import ConversationTurnsRepository
from repositories.elevenlabs_usage_repository import ElevenLabsUsageRepository
from repositories.emails_ingested_repository import EmailsIngestedRepository
from repositories.memories_repository import MemoriesRepository
from repositories.opportunity_signals_repository import OpportunitySignalsRepository
from repositories.proactive_notifications_repository import ProactiveNotificationsRepository
from repositories.reminders_repository import RemindersRepository
from repositories.tasks_repository import TasksRepository
from repositories.telos_onboarding_state_repository import TelosOnboardingStateRepository
from repositories.users_repository import UsersRepository
from services.action_service import ActionService
from services.approval_service import ApprovalService
from services.brain_router import BrainRouter
from services.conversation_service import ConversationService
from services.email_service import EmailService
from services.habit_service import HabitService
from services.memory_service import MemoryService
from services.opportunity_service import OpportunityService
from services.proactive_service import ProactiveService
from services.reminder_parser import ReminderParser
from services.reminder_service import ReminderService
from services.search_service import SearchService
from services.task_service import TaskService
from services.telos_service import TelosService
from services.voice_output_service import VoiceOutputService, VoiceSynthesisResult
from utils.web_search import WebSearchClient


class DummyScheduler:
    def __init__(self) -> None:
        self.scheduled: list[tuple[str, object]] = []
        self.removed: list[str] = []

    def schedule_reminder(self, reminder_id: str, run_date) -> None:
        self.scheduled.append((reminder_id, run_date))

    def remove_reminder(self, reminder_id: str) -> None:
        self.removed.append(reminder_id)


@dataclass(slots=True)
class TestContainer:
    settings: Any
    database: Any
    users_repository: UsersRepository
    reminders_repository: RemindersRepository
    tasks_repository: TasksRepository
    memories_repository: MemoriesRepository
    context_repository: ConversationContextRepository
    conversation_turns_repository: ConversationTurnsRepository
    onboarding_repository: TelosOnboardingStateRepository
    emails_repository: EmailsIngestedRepository
    approvals_repository: ApprovalsRepository
    proactive_repository: ProactiveNotificationsRepository
    signals_repository: OpportunitySignalsRepository
    elevenlabs_usage_repository: ElevenLabsUsageRepository
    conversation_service: ConversationService
    brain_router: BrainRouter
    reminder_parser: ReminderParser
    reminder_service: ReminderService
    memory_service: MemoryService
    habit_service: HabitService
    task_service: TaskService
    approval_service: ApprovalService
    search_service: SearchService
    email_service: EmailService
    opportunity_service: OpportunityService
    proactive_service: ProactiveService
    action_service: ActionService
    voice_output_service: VoiceOutputService
    pipeline: UnifiedPipeline
    scheduler: DummyScheduler


class StubVoiceInputService:
    def __init__(self, text: str, confidence: float) -> None:
        self.text = text
        self.confidence = confidence

    async def transcribe(self, file_path: str):
        from services.voice_input_service import TranscriptionResult
        return TranscriptionResult(text=self.text, confidence=self.confidence)


class StubVoiceOutputService:
    def __init__(self, audio_bytes: bytes = b'OggSstub', should_fail: bool = False, notice: str | None = None) -> None:
        self.audio_bytes = audio_bytes
        self.should_fail = should_fail
        self.notice = notice

    async def synthesize(self, text: str, language: str = 'en', **kwargs) -> VoiceSynthesisResult:
        if self.should_fail:
            raise RuntimeError('tts failed')
        return VoiceSynthesisResult(audio_bytes=self.audio_bytes, notice=self.notice, synthesized_text=text)


def configure_test_env(monkeypatch, tmp_path: Path, *, extra: dict[str, str] | None = None):
    db_path = tmp_path / 'nexus_test.db'
    values = {
        'TELEGRAM_BOT_TOKEN': 'test-token',
        'ALLOWED_TELEGRAM_IDS': '111,222',
        'APP_TIMEZONE': 'UTC',
        'DATABASE_URL': f'sqlite:///{db_path}',
        'GEMINI_ENABLED': 'false',
        'GMAIL_ENABLED': 'false',
        'WEB_SEARCH_ENABLED': 'true',
        'WEB_SEARCH_BACKEND': 'ddg',
        'PROACTIVE_ENABLED': 'true',
        'DESTRUCTIVE_APPROVAL_ENABLED': 'true',
        'VOICE_INPUT_ENABLED': 'true',
        'VOICE_OUTPUT_ENABLED': 'true',
        'VOICE_OUTPUT_BACKEND': 'piper',
        'VOICE_INPUT_MAX_FILE_BYTES': '10485760',
        'GEMINI_API_KEY_USER_1': '',
        'GEMINI_API_KEY_USER_2': '',
        'ELEVENLABS_API_KEY_USER_1': '',
        'ELEVENLABS_VOICE_ID_USER_1': '',
        'ELEVENLABS_DAILY_CHAR_QUOTA_USER_1': '5000',
        'ELEVENLABS_API_KEY_USER_2': '',
        'ELEVENLABS_VOICE_ID_USER_2': '',
        'ELEVENLABS_DAILY_CHAR_QUOTA_USER_2': '5000',
        'ELEVENLABS_MODEL': 'eleven_turbo_v2_5',
        'GMAIL_TOKEN_DIR': str(tmp_path / 'gmail_tokens'),
        'GMAIL_CREDENTIALS_PATH': str(tmp_path / 'secrets' / 'gmail_credentials.json'),
        'NEXUS_RENTALS_DB': str(tmp_path / 'rentals.db'),
    }
    if extra:
        values.update(extra)
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def build_test_container(
    monkeypatch,
    tmp_path: Path,
    *,
    extra_env: dict[str, str] | None = None,
    validate_startup: bool = True,
) -> TestContainer:
    settings = configure_test_env(monkeypatch, tmp_path, extra=extra_env)
    database = create_database(settings)
    if validate_startup:
        StartupValidator(settings).run(database.engine)
    else:
        create_all(database.engine, settings)

    users_repository = UsersRepository(database.session_factory)
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
    elevenlabs_usage_repository = ElevenLabsUsageRepository(database.session_factory)

    conversation_service = ConversationService(context_repository)
    # H2-047 Wave 2: AIService is gone. Tests that need an LLM-shaped
    # stub use BrainRouter (or a scripted stub passed via fixtures).
    brain_router = BrainRouter(ollama_url=None)
    reminder_parser = ReminderParser(brain_router, settings.core.app_timezone)
    scheduler = DummyScheduler()
    memory_service = MemoryService(memories_repository)
    habit_service = HabitService(memories_repository)
    reminder_service = ReminderService(reminders_repository, reminder_parser, conversation_service, settings.core.app_timezone, scheduler=scheduler, habit_service=habit_service)
    task_service = TaskService(tasks_repository, reminders_repository, memories_repository, settings.core.app_timezone, habit_service=habit_service)
    approval_service = ApprovalService(approvals_repository, message_ttl_minutes=settings.approval.approval_ttl_message_minutes, system_ttl_minutes=settings.approval.approval_ttl_system_minutes)
    search_service = SearchService(WebSearchClient(backend=settings.search.web_search_backend, serpapi_key=settings.search.serpapi_key), enabled=settings.search.web_search_enabled, max_query_chars=settings.search.web_search_max_query_chars)
    email_service = EmailService(settings, emails_repository)
    opportunity_service = OpportunityService(settings, memories_repository, signals_repository, search_service)
    telos_service_for_proactive = TelosService(tmp_path / 'telos')
    proactive_service = ProactiveService(proactive_repository, reminder_service, task_service, email_service, emails_repository, habit_service, settings.core.app_timezone, settings.proactive.briefing_skip_if_late_by_hours, telos_service=telos_service_for_proactive, onboarding_repository=onboarding_repository)
    action_service = ActionService(memory_service, task_service, messenger=lambda user_id, text: None)
    # H2-047 Wave 2: don't pass a real BrainRouter to VoiceOutputService in
    # tests — `_rewrite_text` would otherwise spawn a real `claude -p`
    # subprocess every test run. Voice tests that need to verify rewrite
    # behavior (test_content_gating) inject a stub via container.voice_output_service.brain_router=...
    voice_output_service = VoiceOutputService(settings, users_repository=users_repository, usage_repository=elevenlabs_usage_repository, brain_router=None)
    pipeline = UnifiedPipeline(users_repository=users_repository, conversation_service=conversation_service, voice_input_service=None, voice_output_service=None, allowed_telegram_ids=settings.core.allowed_telegram_ids)
    return TestContainer(settings=settings, database=database, users_repository=users_repository, reminders_repository=reminders_repository, tasks_repository=tasks_repository, memories_repository=memories_repository, context_repository=context_repository, conversation_turns_repository=conversation_turns_repository, onboarding_repository=onboarding_repository, emails_repository=emails_repository, approvals_repository=approvals_repository, proactive_repository=proactive_repository, signals_repository=signals_repository, elevenlabs_usage_repository=elevenlabs_usage_repository, conversation_service=conversation_service, brain_router=brain_router, reminder_parser=reminder_parser, reminder_service=reminder_service, memory_service=memory_service, habit_service=habit_service, task_service=task_service, approval_service=approval_service, search_service=search_service, email_service=email_service, opportunity_service=opportunity_service, proactive_service=proactive_service, action_service=action_service, voice_output_service=voice_output_service, pipeline=pipeline, scheduler=scheduler)
