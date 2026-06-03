from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent


class ConfigError(ValueError):
    """Raised when environment configuration is invalid."""


def _load_environment() -> None:
    load_dotenv(BASE_DIR / '.env', override=False)


def _get_required(name: str) -> str:
    value = os.getenv(name, '').strip()
    if not value:
        raise ConfigError(f'Missing required environment variable: {name}')
    return value


def _get_str(name: str, default: str = '') -> str:
    return os.getenv(name, default).strip()


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {'1', 'true', 'yes', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off'}:
        return False
    raise ConfigError(f'Invalid boolean for {name}: {raw}')


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f'Invalid integer for {name}: {raw}') from exc


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f'Invalid float for {name}: {raw}') from exc


def _get_list(name: str) -> list[str]:
    raw = _get_str(name)
    if not raw:
        return []
    return [part.strip() for part in raw.split(',') if part.strip()]


@dataclass(frozen=True, slots=True)
class CoreSettings:
    telegram_bot_token: str
    allowed_telegram_ids: tuple[int, ...]
    app_timezone: str
    database_url: str


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """H2-047 Wave 2: Gemini fields removed. Pre-Wave-2 this dataclass held
    seven gemini_* config options driving AIService rate limits + circuit
    breaker; with AIService deleted, none of those values had a downstream
    consumer. brain_router is the only LLM entry point now and runs
    Claude CLI as a local subprocess — no per-day / per-minute quotas to
    configure here."""
    pass


@dataclass(frozen=True, slots=True)
class ElevenLabsUserSettings:
    slot: int
    api_key: str
    voice_id: str
    daily_char_quota: int


@dataclass(frozen=True, slots=True)
class VoiceSettings:
    voice_input_enabled: bool
    voice_input_backend: str
    voice_model_size: str
    voice_input_max_file_bytes: int
    groq_api_key: str
    groq_transcription_model: str
    voice_output_enabled: bool
    voice_output_backend: str
    voice_openai_api_key: str
    openai_tts_model: str
    openai_tts_voice: str
    piper_model_path: str
    elevenlabs_users: tuple[ElevenLabsUserSettings, ...]
    elevenlabs_model: str

    def elevenlabs_user(self, slot: int) -> ElevenLabsUserSettings | None:
        for config in self.elevenlabs_users:
            if config.slot == slot:
                return config
        return None


@dataclass(frozen=True, slots=True)
class GmailSettings:
    gmail_enabled: bool
    gmail_credentials_path: Path
    gmail_token_dir: Path


@dataclass(frozen=True, slots=True)
class GoogleSettings:
    enabled: bool
    credentials_path: Path
    token_dir: Path
    scopes: tuple[str, ...]
    calendar_cache_ttl_sec: int
    calendar_lookahead_days: int
    calendar_lookback_days: int
    tasks_cache_ttl_sec: int
    tasks_include_in_briefing: bool
    people_cache_ttl_sec: int
    people_max_results_per_lookup: int


@dataclass(frozen=True, slots=True)
class SearchSettings:
    web_search_enabled: bool
    web_search_backend: str
    serpapi_key: str
    web_search_max_query_chars: int
    radar_run_interval_hours: int
    radar_max_signals_per_category_per_week: int
    radar_min_confidence: float


@dataclass(frozen=True, slots=True)
class ProactiveSettings:
    proactive_enabled: bool
    morning_briefing_hour: int
    midday_check_hour: int
    evening_wrap_hour: int
    briefing_skip_if_late_by_hours: int


@dataclass(frozen=True, slots=True)
class ApprovalSettings:
    approval_ttl_message_minutes: int
    approval_ttl_system_minutes: int
    approval_sweep_interval_sec: int
    destructive_approval_enabled: bool


@dataclass(frozen=True, slots=True)
class SecuritySettings:
    threat_model: str
    sqlcipher_key: str


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    log_level: str
    redact_patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Settings:
    core: CoreSettings
    llm: LLMSettings
    voice: VoiceSettings
    gmail: GmailSettings
    google: GoogleSettings
    search: SearchSettings
    proactive: ProactiveSettings
    approval: ApprovalSettings
    security: SecuritySettings
    logging: LoggingSettings
    data_dir: Path
    voice_in_dir: Path
    voice_out_dir: Path

    @property
    def database_path(self) -> Path | None:
        for prefix in ('sqlite:///', 'sqlite+pysqlcipher:///'):
            if self.core.database_url.startswith(prefix):
                return (BASE_DIR / self.core.database_url.removeprefix(prefix)).resolve()
        return None


# H2-047 Wave 2: _load_gemini_keys was the only consumer of GEMINI_API_KEY_USER_*
# env vars. Removed along with the rest of the Gemini stack.


_ELEVENLABS_API_KEY_PATTERN = re.compile(r'^ELEVENLABS_API_KEY_USER_(\d+)$')
_ELEVENLABS_VOICE_ID_PATTERN = re.compile(r'^ELEVENLABS_VOICE_ID_USER_(\d+)$')
_ELEVENLABS_QUOTA_PATTERN = re.compile(r'^ELEVENLABS_DAILY_CHAR_QUOTA_USER_(\d+)$')


def _load_elevenlabs_users(values: Iterable[tuple[str, str]]) -> tuple[ElevenLabsUserSettings, ...]:
    api_keys: dict[int, str] = {}
    voice_ids: dict[int, str] = {}
    quotas: dict[int, int] = {}
    slots: set[int] = set()
    for key, value in values:
        api_match = _ELEVENLABS_API_KEY_PATTERN.match(key)
        if api_match is not None:
            slot = int(api_match.group(1))
            slots.add(slot)
            if value.strip():
                api_keys[slot] = value.strip()
            continue
        voice_match = _ELEVENLABS_VOICE_ID_PATTERN.match(key)
        if voice_match is not None:
            slot = int(voice_match.group(1))
            slots.add(slot)
            if value.strip():
                voice_ids[slot] = value.strip()
            continue
        quota_match = _ELEVENLABS_QUOTA_PATTERN.match(key)
        if quota_match is not None:
            slot = int(quota_match.group(1))
            slots.add(slot)
            if value.strip():
                try:
                    quotas[slot] = int(value.strip())
                except ValueError as exc:
                    raise ConfigError(f'Invalid integer for {key}: {value}') from exc
    return tuple(
        ElevenLabsUserSettings(
            slot=slot,
            api_key=api_keys.get(slot, ''),
            voice_id=voice_ids.get(slot, ''),
            daily_char_quota=quotas.get(slot, 5000),
        )
        for slot in sorted(slots)
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_environment()

    telegram_token = _get_required('TELEGRAM_BOT_TOKEN')
    try:
        allowed_ids = tuple(int(value) for value in _get_list('ALLOWED_TELEGRAM_IDS'))
    except ValueError as exc:
        raise ConfigError('ALLOWED_TELEGRAM_IDS must contain only integer Telegram user ids') from exc
    if not allowed_ids:
        raise ConfigError('ALLOWED_TELEGRAM_IDS must contain at least one Telegram user id')

    data_dir = (BASE_DIR / '.data').resolve()
    gmail_token_dir = (BASE_DIR / _get_str('GMAIL_TOKEN_DIR', '.data/gmail_tokens')).resolve()

    settings = Settings(
        core=CoreSettings(
            telegram_bot_token=telegram_token,
            allowed_telegram_ids=allowed_ids,
            app_timezone=_get_str('APP_TIMEZONE', 'UTC'),
            database_url=_get_str('DATABASE_URL', 'sqlite:///./.data/nexus.db'),
        ),
        llm=LLMSettings(),  # H2-047 Wave 2: fields retired with Gemini stack
        voice=VoiceSettings(
            voice_input_enabled=_get_bool('VOICE_INPUT_ENABLED', True),
            voice_input_backend=_get_str('VOICE_INPUT_BACKEND', 'whisper_local'),
            voice_model_size=_get_str('VOICE_MODEL_SIZE', 'base'),
            voice_input_max_file_bytes=_get_int('VOICE_INPUT_MAX_FILE_BYTES', 10 * 1024 * 1024),
            groq_api_key=_get_str('GROQ_API_KEY'),
            groq_transcription_model=_get_str('GROQ_TRANSCRIPTION_MODEL', 'whisper-large-v3-turbo'),
            voice_output_enabled=_get_bool('VOICE_OUTPUT_ENABLED', True),
            voice_output_backend=_get_str('VOICE_OUTPUT_BACKEND', 'piper'),
            voice_openai_api_key=_get_str('VOICE_OPENAI_API_KEY'),
            openai_tts_model=_get_str('OPENAI_TTS_MODEL', 'gpt-4o-mini-tts'),
            openai_tts_voice=_get_str('OPENAI_TTS_VOICE', 'alloy'),
            piper_model_path=_get_str('PIPER_MODEL_PATH'),
            elevenlabs_users=_load_elevenlabs_users(os.environ.items()),
            elevenlabs_model=_get_str('ELEVENLABS_MODEL', 'eleven_turbo_v2_5'),
        ),
        gmail=GmailSettings(
            gmail_enabled=_get_bool('GMAIL_ENABLED', False),
            gmail_credentials_path=(BASE_DIR / _get_str('GMAIL_CREDENTIALS_PATH', 'secrets/gmail_credentials.json')).resolve(),
            gmail_token_dir=gmail_token_dir,
        ),
        google=GoogleSettings(
            enabled=_get_bool('GOOGLE_INTEGRATION_ENABLED', False),
            credentials_path=(BASE_DIR / _get_str('GOOGLE_CREDENTIALS_PATH', 'credentials.json')).resolve(),
            token_dir=(BASE_DIR / _get_str('GOOGLE_TOKEN_DIR', '.data/google_tokens')).resolve(),
            scopes=tuple(_get_list('GOOGLE_SCOPES') or [
                'openid',
                'https://www.googleapis.com/auth/userinfo.email',
                'https://www.googleapis.com/auth/userinfo.profile',
                'https://www.googleapis.com/auth/calendar',
                'https://www.googleapis.com/auth/tasks.readonly',
                'https://www.googleapis.com/auth/contacts',
            ]),
            calendar_cache_ttl_sec=_get_int('GOOGLE_CALENDAR_CACHE_TTL_SEC', 60),
            calendar_lookahead_days=_get_int('GOOGLE_CALENDAR_LOOKAHEAD_DAYS', 90),
            calendar_lookback_days=_get_int('GOOGLE_CALENDAR_LOOKBACK_DAYS', 7),
            tasks_cache_ttl_sec=_get_int('GOOGLE_TASKS_CACHE_TTL_SEC', 120),
            tasks_include_in_briefing=_get_bool('GOOGLE_TASKS_INCLUDE_IN_BRIEFING', True),
            people_cache_ttl_sec=_get_int('GOOGLE_PEOPLE_CACHE_TTL_SEC', 300),
            people_max_results_per_lookup=_get_int('GOOGLE_PEOPLE_MAX_RESULTS_PER_LOOKUP', 10),
        ),
        search=SearchSettings(
            web_search_enabled=_get_bool('WEB_SEARCH_ENABLED', True),
            web_search_backend=_get_str('WEB_SEARCH_BACKEND', 'ddg'),
            serpapi_key=_get_str('SERPAPI_KEY'),
            web_search_max_query_chars=_get_int('WEB_SEARCH_MAX_QUERY_CHARS', 256),
            radar_run_interval_hours=_get_int('RADAR_RUN_INTERVAL_HOURS', 24),
            radar_max_signals_per_category_per_week=_get_int('RADAR_MAX_SIGNALS_PER_CATEGORY_PER_WEEK', 3),
            radar_min_confidence=_get_float('RADAR_MIN_CONFIDENCE', 0.6),
        ),
        proactive=ProactiveSettings(
            proactive_enabled=_get_bool('PROACTIVE_ENABLED', True),
            morning_briefing_hour=_get_int('MORNING_BRIEFING_HOUR', 8),
            midday_check_hour=_get_int('MIDDAY_CHECK_HOUR', 13),
            evening_wrap_hour=_get_int('EVENING_WRAP_HOUR', 20),
            briefing_skip_if_late_by_hours=_get_int('BRIEFING_SKIP_IF_LATE_BY_HOURS', 2),
        ),
        approval=ApprovalSettings(
            approval_ttl_message_minutes=_get_int('APPROVAL_TTL_MESSAGE_MINUTES', 15),
            approval_ttl_system_minutes=_get_int('APPROVAL_TTL_SYSTEM_MINUTES', 60),
            approval_sweep_interval_sec=_get_int('APPROVAL_SWEEP_INTERVAL_SEC', 60),
            destructive_approval_enabled=_get_bool('DESTRUCTIVE_APPROVAL_ENABLED', True),
        ),
        security=SecuritySettings(
            threat_model=_get_str('THREAT_MODEL', 'single_user_local'),
            sqlcipher_key=_get_str('SQLCIPHER_KEY'),
        ),
        logging=LoggingSettings(
            log_level=_get_str('LOG_LEVEL', 'INFO').upper(),
            redact_patterns=tuple(_get_list('LOG_REDACT_PATTERNS') or ['api_key', 'token', 'password', 'Bearer']),
        ),
        data_dir=data_dir,
        voice_in_dir=(data_dir / 'voice_in').resolve(),
        voice_out_dir=(data_dir / 'voice_out').resolve(),
    )
    _validate_settings(settings)
    return settings


def _validate_settings(settings: Settings) -> None:
    try:
        ZoneInfo(settings.core.app_timezone)
    except ZoneInfoNotFoundError as exc:
        raise ConfigError(f'APP_TIMEZONE must be a valid IANA timezone: {settings.core.app_timezone}') from exc
    if settings.voice.voice_input_backend not in {'whisper_local', 'groq'}:
        raise ConfigError('VOICE_INPUT_BACKEND must be whisper_local or groq')
    if settings.voice.voice_output_backend not in {'piper', 'openai_tts', 'elevenlabs'}:
        raise ConfigError('VOICE_OUTPUT_BACKEND must be piper, openai_tts, or elevenlabs')
    if settings.search.web_search_backend not in {'ddg', 'serpapi'}:
        raise ConfigError('WEB_SEARCH_BACKEND must be ddg or serpapi')
    if settings.security.threat_model not in {'single_user_local', 'hosted'}:
        raise ConfigError('THREAT_MODEL must be single_user_local or hosted')
    if settings.security.threat_model == 'hosted' and not settings.security.sqlcipher_key:
        raise ConfigError('SQLCIPHER_KEY is required when THREAT_MODEL=hosted')
    if settings.security.threat_model == 'hosted' and not settings.core.database_url.startswith('sqlite+pysqlcipher:///'):
        raise ConfigError('Hosted mode requires a SQLCipher database URL such as sqlite+pysqlcipher:///./.data/nexus.db')
    if settings.search.web_search_backend == 'serpapi' and not settings.search.serpapi_key:
        raise ConfigError('SERPAPI_KEY is required when WEB_SEARCH_BACKEND=serpapi')
    if settings.voice.voice_input_max_file_bytes <= 0:
        raise ConfigError('VOICE_INPUT_MAX_FILE_BYTES must be greater than 0')
    if settings.voice.voice_input_enabled and settings.voice.voice_input_backend == 'groq' and not settings.voice.groq_api_key:
        raise ConfigError('GROQ_API_KEY is required when VOICE_INPUT_BACKEND=groq')
    if settings.voice.voice_output_enabled and settings.voice.voice_output_backend == 'openai_tts' and not settings.voice.voice_openai_api_key:
        raise ConfigError('VOICE_OPENAI_API_KEY is required when VOICE_OUTPUT_BACKEND=openai_tts')
    if settings.voice.voice_output_enabled and settings.voice.voice_output_backend == 'elevenlabs':
        if not settings.voice.elevenlabs_model:
            raise ConfigError('ELEVENLABS_MODEL must not be empty')
        for config in settings.voice.elevenlabs_users:
            if config.daily_char_quota <= 0:
                raise ConfigError(f'ELEVENLABS_DAILY_CHAR_QUOTA_USER_{config.slot} must be greater than 0')
    if settings.gmail.gmail_enabled and not settings.gmail.gmail_credentials_path.exists():
        raise ConfigError('GMAIL_CREDENTIALS_PATH must point to an existing file when GMAIL_ENABLED=true')
    if not settings.voice.groq_transcription_model:
        raise ConfigError('GROQ_TRANSCRIPTION_MODEL must not be empty')
    if not settings.voice.openai_tts_model:
        raise ConfigError('OPENAI_TTS_MODEL must not be empty')
    if not settings.voice.openai_tts_voice:
        raise ConfigError('OPENAI_TTS_VOICE must not be empty')
    if settings.search.web_search_max_query_chars <= 0:
        raise ConfigError('WEB_SEARCH_MAX_QUERY_CHARS must be greater than 0')
    if settings.search.radar_run_interval_hours <= 0:
        raise ConfigError('RADAR_RUN_INTERVAL_HOURS must be greater than 0')
    if settings.search.radar_max_signals_per_category_per_week <= 0:
        raise ConfigError('RADAR_MAX_SIGNALS_PER_CATEGORY_PER_WEEK must be greater than 0')
    if not 0 <= settings.search.radar_min_confidence <= 1:
        raise ConfigError('RADAR_MIN_CONFIDENCE must be between 0 and 1')
    for name, value in (
        ('MORNING_BRIEFING_HOUR', settings.proactive.morning_briefing_hour),
        ('MIDDAY_CHECK_HOUR', settings.proactive.midday_check_hour),
        ('EVENING_WRAP_HOUR', settings.proactive.evening_wrap_hour),
    ):
        if not 0 <= value <= 23:
            raise ConfigError(f'{name} must be between 0 and 23')
    if settings.proactive.briefing_skip_if_late_by_hours < 0:
        raise ConfigError('BRIEFING_SKIP_IF_LATE_BY_HOURS must be 0 or greater')
