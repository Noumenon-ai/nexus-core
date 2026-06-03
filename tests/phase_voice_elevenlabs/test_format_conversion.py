from __future__ import annotations

from pathlib import Path

import httpx
import pytest

import services.voice_output_service as voice_output_module
from tests.phase_voice_elevenlabs.helpers import build_elevenlabs_container, create_user


FIXTURE_MP3 = Path(__file__).resolve().parent / 'fixtures' / 'sample.mp3'


class _Response:
    def __init__(self, *, content: bytes = b'fake-mp3', status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request('POST', 'https://api.elevenlabs.io')
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError('error', request=request, response=response)
        return None


@pytest.mark.asyncio
async def test_mp3_to_ogg_produces_valid_opus():
    sample_mp3 = FIXTURE_MP3.read_bytes()
    ogg_bytes = voice_output_module.mp3_to_ogg_opus_bytes(sample_mp3)
    assert ogg_bytes.startswith(b'OggS')
    assert len(ogg_bytes) > 128


@pytest.mark.asyncio
async def test_ffmpeg_missing_falls_back_to_text(monkeypatch, tmp_path):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            return _Response()

    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: Client())
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    monkeypatch.setattr(voice_output_module.shutil, 'which', lambda name: None)
    user = create_user(container)

    result = await container.voice_output_service.synthesize('hello there', user=user, language='en')

    assert result.audio_bytes is None
    assert result.notice == 'Voice reply unavailable right now.'


@pytest.mark.asyncio
async def test_elevenlabs_timeout_falls_back(monkeypatch, tmp_path):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            raise httpx.TimeoutException('timed out')

    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: Client())
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container)

    result = await container.voice_output_service.synthesize('hello there', user=user, language='en')

    assert result.audio_bytes is None
    assert result.notice is None


@pytest.mark.asyncio
async def test_elevenlabs_401_user_visible_error(monkeypatch, tmp_path):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            return _Response(status_code=401)

    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: Client())
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container)

    result = await container.voice_output_service.synthesize('hello there', user=user, language='en')

    assert result.audio_bytes is None
    assert result.notice == 'Voice reply unavailable right now.'


@pytest.mark.asyncio
async def test_elevenlabs_429_retries_then_falls_back(monkeypatch, tmp_path):
    attempts: list[int] = []
    sleeps: list[float] = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            attempts.append(1)
            return _Response(status_code=429)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(voice_output_module.asyncio, 'sleep', fake_sleep)
    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: Client())
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container)

    result = await container.voice_output_service.synthesize('hello there', user=user, language='en')

    assert result.audio_bytes is None
    assert result.notice is None
    assert sleeps == [0.5, 1.0, 2.0]
    assert len(attempts) == 4
