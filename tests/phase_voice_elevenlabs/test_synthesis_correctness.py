from __future__ import annotations

import pytest

import services.voice_output_service as voice_output_module
from tests.phase_voice_elevenlabs.helpers import build_elevenlabs_container, create_user


class _Response:
    def __init__(self, content: bytes = b'fake-mp3', status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        return None


class _Client:
    def __init__(self, captured: list[dict]) -> None:
        self.captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, **kwargs):
        self.captured.append({'url': url, 'headers': kwargs.get('headers', {}), 'json': kwargs.get('json', {})})
        return _Response()


def _patch_success(monkeypatch, captured: list[dict]) -> None:
    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: _Client(captured))
    monkeypatch.setattr(voice_output_module, 'mp3_to_ogg_opus_bytes', lambda data: b'OggS' + data)


@pytest.mark.asyncio
async def test_synthesize_english_returns_ogg_bytes(monkeypatch, tmp_path):
    captured: list[dict] = []
    _patch_success(monkeypatch, captured)
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container, language='en')

    result = await container.voice_output_service.synthesize('Hello world', user=user, language='en')

    assert result.audio_bytes == b'OggSfake-mp3'
    assert captured[0]['url'].endswith('/voice_user_1')
    assert captured[0]['json']['text'] == 'Hello world'


@pytest.mark.asyncio
async def test_synthesize_hebrew_returns_ogg_bytes(monkeypatch, tmp_path):
    captured: list[dict] = []
    _patch_success(monkeypatch, captured)
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container, language='he')

    result = await container.voice_output_service.synthesize('שלום עולם', user=user, language='he')

    assert result.audio_bytes == b'OggSfake-mp3'
    assert captured[0]['json']['text'] == 'שלום עולם'


@pytest.mark.asyncio
async def test_synthesize_codeswitched_text(monkeypatch, tmp_path):
    captured: list[dict] = []
    _patch_success(monkeypatch, captured)
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container, language='he')

    result = await container.voice_output_service.synthesize('תבדוק ב Google Calendar את meeting של מחר', user=user, language='he')

    assert result.audio_bytes == b'OggSfake-mp3'
    assert 'Google Calendar' in captured[0]['json']['text']


@pytest.mark.asyncio
async def test_synthesize_500_char_max(monkeypatch, tmp_path):
    captured: list[dict] = []
    _patch_success(monkeypatch, captured)
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container)
    text = 'a' * 500

    result = await container.voice_output_service.synthesize(text, user=user, language='en')

    assert result.audio_bytes == b'OggSfake-mp3'
    assert len(captured) == 1
    assert len(captured[0]['json']['text']) == 500


@pytest.mark.asyncio
async def test_synthesize_strips_emoji_before_send(monkeypatch, tmp_path):
    captured: list[dict] = []
    _patch_success(monkeypatch, captured)
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container)

    result = await container.voice_output_service.synthesize('Hello 😊 world', user=user, language='en')

    assert result.audio_bytes == b'OggSfake-mp3'
    assert '😊' not in captured[0]['json']['text']


@pytest.mark.asyncio
async def test_synthesize_sends_language_code_en(monkeypatch, tmp_path):
    captured: list[dict] = []
    _patch_success(monkeypatch, captured)
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container, language='en')

    await container.voice_output_service.synthesize('Hello world', user=user, language='en')

    assert captured[0]['json'].get('language_code') == 'en'


@pytest.mark.asyncio
async def test_synthesize_sends_language_code_he(monkeypatch, tmp_path):
    captured: list[dict] = []
    _patch_success(monkeypatch, captured)
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container, language='he')

    await container.voice_output_service.synthesize('שלום עולם', user=user, language='he')

    assert captured[0]['json'].get('language_code') == 'he'
