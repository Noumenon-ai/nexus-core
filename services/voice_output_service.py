from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from config import Settings
from models import User
from repositories.elevenlabs_usage_repository import ElevenLabsUsageRepository
from repositories.users_repository import UsersRepository
from services.brain_router import BrainRouter
from services.voice_preprocessing import (
    chunk_synthesis_text,
    is_confirmation_text,
    is_parsed_event_preview,
    preprocess_for_language,
    should_rewrite_for_voice,
    should_skip_voice,
)
from utils.language import detect_language


logger = logging.getLogger(__name__)


ADAM_VOICE_ID = 'pNInz6obpgDQGcFmaJgB'
DEFAULT_VOICE_SETTINGS = {'stability': 0.50, 'similarity_boost': 0.75, 'style': 0.0, 'use_speaker_boost': True}
BRIEFING_VOICE_SETTINGS = {'stability': 0.40, 'similarity_boost': 0.75, 'style': 0.10, 'use_speaker_boost': True}
REMINDER_VOICE_SETTINGS = {'stability': 0.60, 'similarity_boost': 0.80, 'style': 0.0, 'use_speaker_boost': True}
CONFIRMATION_VOICE_SETTINGS = {'stability': 0.55, 'similarity_boost': 0.75, 'style': 0.0, 'use_speaker_boost': True}
VOICE_SETTINGS = {
    'default': DEFAULT_VOICE_SETTINGS,
    'briefing': BRIEFING_VOICE_SETTINGS,
    'reminder': REMINDER_VOICE_SETTINGS,
    'confirmation': CONFIRMATION_VOICE_SETTINGS,
}
ELEVENLABS_RETRY_DELAYS = (0.5, 1.0, 2.0)
_ELEVENLABS_VOICE_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,128}$')


@dataclass(slots=True)
class VoiceSynthesisResult:
    audio_bytes: bytes | None = None
    notice: str | None = None
    synthesized_text: str = ''


@dataclass(slots=True)
class _ResolvedVoiceRoute:
    api_key: str
    voice_id: str
    daily_quota: int


class _ElevenLabsAuthError(RuntimeError):
    pass


class _ElevenLabsRateLimitError(RuntimeError):
    pass


class VoiceOutputService:
    def __init__(
        self,
        settings: Settings,
        *,
        users_repository: UsersRepository | None = None,
        usage_repository: ElevenLabsUsageRepository | None = None,
        brain_router: BrainRouter | None = None,
        output_dir: Path | None = None,
    ) -> None:
        # H2-047 Wave 2: voice rewrite now routes through brain_router
        # (Claude CLI subprocess) instead of ai_service (Gemini). The
        # ai_service parameter is gone — main.py + tests must pass a
        # BrainRouter instance instead.
        self.settings = settings
        self.users_repository = users_repository
        self.usage_repository = usage_repository
        self.brain_router = brain_router
        self.output_dir = output_dir or settings.voice_out_dir
        self.backend = settings.voice.voice_output_backend
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize(
        self,
        text: str,
        language: str = 'en',
        *,
        user: User | None = None,
        context_name: str = 'default',
        has_inline_buttons: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSynthesisResult:
        metadata = metadata or {}
        effective_language = self._effective_language(text, language, user)
        gating_text = text
        if has_inline_buttons:
            gating_text = f'{text}\n[telegram_inline_keyboard]'
        if should_skip_voice(gating_text):
            return VoiceSynthesisResult()

        prepared_text = preprocess_for_language(text, effective_language)
        if self._should_rewrite(text, prepared_text, has_inline_buttons):
            rewritten = await self._rewrite_text(prepared_text, effective_language, user)
            prepared_text = preprocess_for_language(rewritten, effective_language)
        chunks = chunk_synthesis_text(prepared_text, max_chars=500)
        synthesized_text = ' '.join(chunks)
        if not chunks:
            return VoiceSynthesisResult(synthesized_text=synthesized_text)

        if self.backend == 'piper' and effective_language != 'en':
            return VoiceSynthesisResult(synthesized_text=synthesized_text)
        if self.backend == 'openai_tts':
            audio = await self._synthesize_openai(synthesized_text)
            return VoiceSynthesisResult(audio_bytes=audio, synthesized_text=synthesized_text)
        if self.backend == 'elevenlabs':
            return await self._synthesize_elevenlabs(
                chunks=chunks,
                synthesized_text=synthesized_text,
                language=effective_language,
                user=user,
                context_name=context_name,
            )
        audio = await self._synthesize_piper(synthesized_text)
        return VoiceSynthesisResult(audio_bytes=audio, synthesized_text=synthesized_text)

    async def _rewrite_text(self, text: str, language: str, user: User | None) -> str:
        """H2-047 Wave 2: route voice rewrite through brain_router instead of
        Gemini. The system prompt below mirrors the pre-Wave-2 ai_service
        VOICE_REWRITE_PROMPT_EN/HE so the rewritten text retains the same
        shape (strip markdown, prefer short spoken sentences, keep names /
        dates / commitments). On any failure the original text passes
        through unchanged so the user still gets the original reply text
        spoken — losing only the spoken-cleanup pass."""
        if self.brain_router is None:
            return text
        system_prompt = (
            "אתה ממיר תשובה טקסטואלית לטקסט מתאים לקריינות קולית. "
            "החזר רק את הנוסח הקולי המשוכתב. שמור על המשמעות. "
            "הסר Markdown, טבלאות וסימני עיצוב. "
            "המר רשימות למשפטים קצרים. "
            "שמור על שמות, תאריכים ומספרים. "
            "אל תוסיף עובדות חדשות."
            if language == 'he' else
            "You are converting an assistant reply into voice-appropriate "
            "spoken text. Return only the rewritten spoken reply. Keep the "
            "meaning exact. Remove markdown, tables, code fences, and "
            "formatting artifacts. Rewrite list-heavy text into short "
            "natural spoken sentences. Preserve names, dates, times, "
            "numbers, and commitments. Do not add new facts or commentary. "
            "If the input is already concise and speakable, make only "
            "minimal cleanup."
        )
        try:
            result = await self.brain_router.generate(
                text,
                system_prompt=system_prompt,
                user_id=user.id if user is not None else None,
            )
            # H2-047 Wave 2: if brain_router fell through to its canned-text
            # fallback (no CLI available, breaker open, etc.), preserve the
            # ORIGINAL text rather than spoken-form the canned message. This
            # also keeps unit tests deterministic: a test env without a
            # working `claude` binary should get the original text back, not
            # have the user "hear" the canned fallback prose.
            if getattr(result, 'fallback_used', False):
                return text
            rewritten = (result.text or '').strip()
            return rewritten or text
        except Exception:
            logger.warning('voice_rewrite_failed')
            return text

    def _should_rewrite(self, original_text: str, prepared_text: str, has_inline_buttons: bool) -> bool:
        if is_confirmation_text(prepared_text):
            return False
        if has_inline_buttons and is_parsed_event_preview(original_text):
            return False
        return should_rewrite_for_voice(original_text)

    async def _synthesize_piper(self, text: str) -> bytes | None:
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path is None:
            logger.warning('tts_ffmpeg_missing')
            return None
        wav_path = self.output_dir / f'{uuid.uuid4()}.wav'
        ogg_path = self.output_dir / f'{uuid.uuid4()}.ogg'
        command = ['piper']
        if self.settings.voice.piper_model_path:
            command.extend(['--model', self.settings.voice.piper_model_path])
        command.extend(['--output_file', str(wav_path)])
        try:
            process = subprocess.run(command, input=text.encode('utf-8'), capture_output=True, timeout=30)
        except FileNotFoundError:
            logger.warning('tts_piper_missing')
            wav_path.unlink(missing_ok=True)
            return None
        except subprocess.TimeoutExpired:
            logger.warning('tts_synthesis_timeout')
            wav_path.unlink(missing_ok=True)
            return None
        if process.returncode != 0:
            wav_path.unlink(missing_ok=True)
            return None
        try:
            _run_process(
                [ffmpeg_path, '-y', '-i', str(wav_path), '-f', 'ogg', '-c:a', 'libopus', '-b:a', '32k', str(ogg_path)],
                timeout=10,
            )
            return ogg_path.read_bytes()
        except subprocess.TimeoutExpired:
            logger.warning('tts_transcode_timeout')
            return None
        except Exception:
            logger.warning('tts_transcode_failed')
            return None
        finally:
            wav_path.unlink(missing_ok=True)
            ogg_path.unlink(missing_ok=True)

    async def _synthesize_openai(self, text: str) -> bytes | None:
        if not self.settings.voice.voice_openai_api_key:
            return None
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                'https://api.openai.com/v1/audio/speech',
                headers={'Authorization': f'Bearer {self.settings.voice.voice_openai_api_key}'},
                json={'model': self.settings.voice.openai_tts_model, 'voice': self.settings.voice.openai_tts_voice, 'input': text},
            )
            response.raise_for_status()
        try:
            return mp3_to_ogg_opus_bytes(response.content)
        except FileNotFoundError:
            logger.warning('tts_ffmpeg_missing')
            return None
        except subprocess.TimeoutExpired:
            logger.warning('tts_transcode_timeout')
            return None
        except Exception:
            logger.warning('tts_transcode_failed')
            return None

    async def _synthesize_elevenlabs(
        self,
        *,
        chunks: list[str],
        synthesized_text: str,
        language: str,
        user: User | None,
        context_name: str,
    ) -> VoiceSynthesisResult:
        route = self._resolve_elevenlabs_route(user)
        if route is None:
            return VoiceSynthesisResult(synthesized_text=synthesized_text)

        usage_day = self._local_day() if user is not None and self.usage_repository is not None else None
        reserved_chars = len(synthesized_text)
        quota_reserved = False
        success = False
        if user is not None and self.usage_repository is not None and usage_day is not None:
            try:
                usage = self.usage_repository.reserve_chars_with_quota(
                    user_id=user.id,
                    day=usage_day,
                    chars=reserved_chars,
                    quota=route.daily_quota,
                )
            except Exception:
                logger.warning('elevenlabs_quota_reservation_failed')
                return VoiceSynthesisResult(synthesized_text=synthesized_text)
            if usage is None:
                return VoiceSynthesisResult(notice=self._quota_reached_notice(language), synthesized_text=synthesized_text)
            quota_reserved = True
            if usage.chars_count >= int(route.daily_quota * 0.8):
                logger.warning('elevenlabs_quota_soft_alert', extra={'user_id': user.id, 'projected': usage.chars_count, 'quota': route.daily_quota})

        voice_settings = self._voice_settings_for_context(context_name, user)
        mp3_payload = bytearray()
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                for chunk in chunks:
                    mp3_payload.extend(
                        await self._synthesize_elevenlabs_chunk(
                            client=client,
                            route=route,
                            chunk=chunk,
                            voice_settings=voice_settings,
                            language=language,
                        )
                    )
            ogg_bytes = mp3_to_ogg_opus_bytes(bytes(mp3_payload))
            if user is not None and self.usage_repository is not None and usage_day is not None:
                try:
                    self.usage_repository.increment_api_calls(user_id=user.id, day=usage_day, api_calls=len(chunks))
                except Exception:
                    logger.warning('elevenlabs_usage_record_failed')
                    return VoiceSynthesisResult(synthesized_text=synthesized_text)
            success = True
            return VoiceSynthesisResult(audio_bytes=ogg_bytes, synthesized_text=synthesized_text)
        except _ElevenLabsAuthError:
            logger.warning('elevenlabs_auth_failed')
            return VoiceSynthesisResult(notice=self._voice_unavailable_notice(language), synthesized_text=synthesized_text)
        except _ElevenLabsRateLimitError:
            logger.warning('elevenlabs_rate_limited_exhausted')
            return VoiceSynthesisResult(synthesized_text=synthesized_text)
        except FileNotFoundError:
            logger.warning('tts_ffmpeg_missing')
            return VoiceSynthesisResult(notice=self._voice_unavailable_notice(language), synthesized_text=synthesized_text)
        except subprocess.TimeoutExpired:
            logger.warning('tts_transcode_timeout')
            return VoiceSynthesisResult(synthesized_text=synthesized_text)
        except httpx.TimeoutException:
            logger.warning('elevenlabs_timeout')
            return VoiceSynthesisResult(synthesized_text=synthesized_text)
        except httpx.HTTPStatusError as exc:
            logger.warning('elevenlabs_http_error: status=%s', exc.response.status_code)
            return VoiceSynthesisResult(synthesized_text=synthesized_text)
        except Exception:
            logger.warning('elevenlabs_synthesis_failed')
            return VoiceSynthesisResult(synthesized_text=synthesized_text)
        finally:
            if quota_reserved and not success and user is not None and self.usage_repository is not None and usage_day is not None:
                try:
                    self.usage_repository.release_chars(user_id=user.id, day=usage_day, chars=reserved_chars)
                except Exception:
                    logger.warning('elevenlabs_quota_release_failed')

    async def _synthesize_elevenlabs_chunk(
        self,
        *,
        client: httpx.AsyncClient,
        route: _ResolvedVoiceRoute,
        chunk: str,
        voice_settings: dict[str, Any],
        language: str,
    ) -> bytes:
        for attempt in range(len(ELEVENLABS_RETRY_DELAYS) + 1):
            response = await client.post(
                f'https://api.elevenlabs.io/v1/text-to-speech/{route.voice_id}',
                headers={'xi-api-key': route.api_key, 'accept': 'audio/mpeg'},
                json={
                    'text': chunk,
                    'model_id': self.settings.voice.elevenlabs_model,
                    'language_code': language,
                    'voice_settings': voice_settings,
                },
            )
            if response.status_code == 401:
                raise _ElevenLabsAuthError
            if response.status_code == 429:
                if attempt >= len(ELEVENLABS_RETRY_DELAYS):
                    raise _ElevenLabsRateLimitError
                delay = ELEVENLABS_RETRY_DELAYS[attempt]
                logger.warning('elevenlabs_rate_limited', extra={'attempt': attempt + 1, 'delay': delay})
                await asyncio.sleep(delay)
                continue
            response.raise_for_status()
            return response.content
        raise _ElevenLabsRateLimitError

    def _resolve_elevenlabs_route(self, user: User | None) -> _ResolvedVoiceRoute | None:
        slot = self.users_repository.get_voice_env_slot(user.id) if user is not None and self.users_repository is not None else None
        quota = self._quota_for_slot(slot)
        user_voice_id = self._validated_voice_id(user.elevenlabs_voice_id) if user is not None else None
        if user is not None and user.elevenlabs_api_key and user_voice_id:
            return _ResolvedVoiceRoute(api_key=user.elevenlabs_api_key, voice_id=user_voice_id, daily_quota=quota)
        if user is not None and user.elevenlabs_api_key and not user_voice_id:
            return _ResolvedVoiceRoute(api_key=user.elevenlabs_api_key, voice_id=ADAM_VOICE_ID, daily_quota=quota)
        if user is not None and user.elevenlabs_migrated_at is not None:
            return None
        env_config = self.settings.voice.elevenlabs_user(slot or 0)
        if env_config is None or not env_config.api_key:
            return None
        voice_id = self._validated_voice_id(env_config.voice_id) or ADAM_VOICE_ID
        return _ResolvedVoiceRoute(api_key=env_config.api_key, voice_id=voice_id, daily_quota=env_config.daily_char_quota)

    def _quota_for_slot(self, slot: int | None) -> int:
        if slot is None:
            return 5000
        config = self.settings.voice.elevenlabs_user(slot)
        if config is None:
            return 5000
        return config.daily_char_quota

    def _quota_reached_notice(self, language: str) -> str:
        return 'מכסת הקול להיום הסתיימה' if language == 'he' else 'voice quota for today reached'

    def _voice_settings_for_context(self, context_name: str, user: User | None) -> dict[str, Any]:
        settings = dict(VOICE_SETTINGS.get(context_name, VOICE_SETTINGS['default']))
        if user is None or not user.elevenlabs_voice_settings:
            return settings
        try:
            overrides = json.loads(user.elevenlabs_voice_settings)
        except json.JSONDecodeError:
            logger.warning('invalid_elevenlabs_voice_settings_json')
            return settings
        if not isinstance(overrides, dict):
            return settings
        for key in ('stability', 'similarity_boost', 'style'):
            value = self._coerce_voice_float(overrides.get(key))
            if value is not None:
                settings[key] = value
        speaker_boost = self._coerce_voice_bool(overrides.get('use_speaker_boost'))
        if speaker_boost is not None:
            settings['use_speaker_boost'] = speaker_boost
        return settings

    def _validated_voice_id(self, voice_id: str | None) -> str | None:
        if not voice_id:
            return None
        candidate = voice_id.strip()
        if _ELEVENLABS_VOICE_ID_RE.fullmatch(candidate):
            return candidate
        logger.warning('invalid_elevenlabs_voice_id')
        return None

    def _coerce_voice_float(self, value: Any) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, numeric))

    def _coerce_voice_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {'1', 'true', 'yes', 'on'}:
                return True
            if normalized in {'0', 'false', 'no', 'off'}:
                return False
        return None

    def _effective_language(self, text: str, language: str, user: User | None) -> str:
        detected = detect_language(text)
        if detected == 'he':
            return 'he'
        if user is not None and user.language in {'en', 'he'}:
            return user.language
        return 'he' if language == 'he' else 'en'

    def _local_day(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(self.settings.core.app_timezone)).date()

    def _voice_unavailable_notice(self, language: str) -> str:
        return 'תשובת הקול לא זמינה כרגע.' if language == 'he' else 'Voice reply unavailable right now.'


def _run_process(command: list[str], *, timeout: float, input_bytes: bytes | None = None) -> bytes:
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_bytes is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(input=input_bytes, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
    return stdout


def mp3_to_ogg_opus_bytes(mp3_bytes: bytes) -> bytes:
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        raise FileNotFoundError('ffmpeg not found on PATH')
    return _run_process(
        [
            ffmpeg_path,
            '-hide_banner',
            '-loglevel',
            'error',
            '-i',
            'pipe:0',
            '-f',
            'ogg',
            '-c:a',
            'libopus',
            '-b:a',
            '32k',
            'pipe:1',
        ],
        timeout=10,
        input_bytes=mp3_bytes,
    )