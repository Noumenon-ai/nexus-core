from __future__ import annotations

import pytest

from pipeline.types import PipelineInput
from services.voice_output_service import VoiceOutputService
from tests.helpers import StubVoiceInputService, StubVoiceOutputService, build_test_container, configure_test_env


@pytest.mark.asyncio
async def test_low_confidence_voice_returns_audio_unclear(container):
    container.pipeline.voice_input_service = StubVoiceInputService('', 0.1)
    output = await container.pipeline.handle(PipelineInput(kind='voice', telegram_id=111, voice_file_path='noise.ogg'))
    assert "couldn't make that out" in output.text.lower()


@pytest.mark.asyncio
async def test_voice_transcription_backend_failure_returns_audio_unclear(container):
    class BrokenVoiceInputService:
        async def transcribe(self, file_path: str):
            raise RuntimeError('backend failed')

    container.pipeline.voice_input_service = BrokenVoiceInputService()
    output = await container.pipeline.handle(PipelineInput(kind='voice', telegram_id=111, voice_file_path='noise.ogg'))
    assert "couldn't make that out" in output.text.lower()


@pytest.mark.asyncio
async def test_voice_output_skips_responses_marked_not_voice_appropriate(container):
    user = container.users_repository.get_or_create(111)
    container.users_repository.set_voice_output_enabled(user.id, True)
    container.memory_service.remember(user, 'remember I like green tea', __import__('utils.i18n', fromlist=['Translator']).Translator())
    container.pipeline.voice_output_service = StubVoiceOutputService(b'OggSfake-voice')
    output = await container.pipeline.handle(PipelineInput(kind='text', telegram_id=111, text='what do you remember'))
    assert output.voice_audio is None


@pytest.mark.asyncio
async def test_tts_failure_falls_back_to_text_only(container):
    user = container.users_repository.get_or_create(111)
    container.users_repository.set_voice_output_enabled(user.id, True)
    container.pipeline.voice_output_service = StubVoiceOutputService(should_fail=True)
    output = await container.pipeline.handle(PipelineInput(kind='text', telegram_id=111, text='how are you'))
    assert output.voice_audio is None
    assert output.text


@pytest.mark.asyncio
async def test_openai_tts_transcodes_to_ogg(monkeypatch, tmp_path):
    class Response:
        content = b'mp3-bytes'

        def raise_for_status(self):
            return None

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            return Response()

    import services.voice_output_service as voice_output_module

    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=30.0: Client())
    monkeypatch.setattr(voice_output_module, 'mp3_to_ogg_opus_bytes', lambda data: b'OggS' + data)
    settings = configure_test_env(monkeypatch, tmp_path, extra={'VOICE_OUTPUT_BACKEND': 'openai_tts', 'VOICE_OPENAI_API_KEY': 'key'})
    service = VoiceOutputService(settings, output_dir=tmp_path)
    result = await service.synthesize('hello')
    assert result.audio_bytes == b'OggSmp3-bytes'


@pytest.mark.asyncio
async def test_elevenlabs_synthesizes_to_ogg(monkeypatch, tmp_path):
    captured = {}

    class Response:
        content = b'fake-mp3-bytes'
        status_code = 200

        def raise_for_status(self):
            return None

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            captured['url'] = url
            captured['headers'] = kwargs.get('headers', {})
            captured['json'] = kwargs.get('json', {})
            return Response()

    import services.voice_output_service as voice_output_module

    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: Client())
    monkeypatch.setattr(voice_output_module, 'mp3_to_ogg_opus_bytes', lambda data: b'OggS' + data)
    container = build_test_container(monkeypatch, tmp_path, extra_env={'VOICE_OUTPUT_BACKEND': 'elevenlabs', 'ELEVENLABS_API_KEY_USER_1': 'sk_test', 'ELEVENLABS_VOICE_ID_USER_1': 'voice_xyz'}, validate_startup=False)
    user = container.users_repository.get_or_create(111)
    result = await container.voice_output_service.synthesize('test', user=user)
    assert result.audio_bytes == b'OggSfake-mp3-bytes'
    assert captured['url'] == 'https://api.elevenlabs.io/v1/text-to-speech/voice_xyz'
    assert captured['headers']['xi-api-key'] == 'sk_test'
    assert captured['json']['text'] == 'test'
    assert captured['json']['model_id'] == 'eleven_turbo_v2_5'
