from __future__ import annotations

import pytest

import services.voice_output_service as voice_output_module
from tests.phase_voice_elevenlabs.helpers import build_elevenlabs_container, create_user


class _ForbiddenClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, **kwargs):
        raise AssertionError('HTTP should not be called when voice is gated')


class _Response:
    def __init__(self) -> None:
        self.content = b'fake-mp3'
        self.status_code = 200

    def raise_for_status(self):
        return None


class _Client:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, **kwargs):
        return _Response()


class _FakeBrainRouter:
    """H2-047 Wave 2: voice rewrite now flows through brain_router.generate
    (instead of ai_service.voice_rewrite). This stub records each call and
    echoes the input prompt as the rewritten text so existing assertions
    about call shape + count carry over with minimal change."""
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def generate(self, prompt: str, *, system_prompt: str | None = None,
                       tool_catalog: list | None = None,
                       user_id: str | None = None):
        self.calls.append((prompt, user_id))
        from services.brain_router import BrainResult
        return BrainResult(text=prompt, provider_used='fake')


@pytest.mark.asyncio
async def test_url_in_reply_suppresses_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: _ForbiddenClient())
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container)

    result = await container.voice_output_service.synthesize('Read this https://example.com link', user=user, language='en')

    assert result.audio_bytes is None


@pytest.mark.asyncio
async def test_approval_buttons_suppress_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: _ForbiddenClient())
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container)

    result = await container.voice_output_service.synthesize('Approval needed: send the message?', user=user, language='en', has_inline_buttons=True)

    assert result.audio_bytes is None


@pytest.mark.asyncio
async def test_long_reply_triggers_rewrite(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: _Client())
    monkeypatch.setattr(voice_output_module, 'mp3_to_ogg_opus_bytes', lambda data: b'OggS' + data)
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    fake_ai = _FakeBrainRouter()
    container.voice_output_service.brain_router = fake_ai
    user = create_user(container)
    text = 'This is a longer assistant reply that should be rewritten for voice because it is clearly longer than eighty characters and keeps going.'

    result = await container.voice_output_service.synthesize(text, user=user, language='en')

    assert result.audio_bytes == b'OggSfake-mp3'
    assert len(fake_ai.calls) == 1


@pytest.mark.asyncio
async def test_short_reply_bypasses_rewrite(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: _Client())
    monkeypatch.setattr(voice_output_module, 'mp3_to_ogg_opus_bytes', lambda data: b'OggS' + data)
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    fake_ai = _FakeBrainRouter()
    container.voice_output_service.brain_router = fake_ai
    user = create_user(container)

    result = await container.voice_output_service.synthesize('Done', user=user, language='en')

    assert result.audio_bytes == b'OggSfake-mp3'
    assert fake_ai.calls == []


@pytest.mark.asyncio
async def test_code_block_suppresses_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: _ForbiddenClient())
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container)

    result = await container.voice_output_service.synthesize('```python\nprint(1)\n```', user=user, language='en')

    assert result.audio_bytes is None


@pytest.mark.asyncio
async def test_stack_trace_suppresses_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: _ForbiddenClient())
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container)

    result = await container.voice_output_service.synthesize('Traceback (most recent call last):\n  File "app.py", line 10, in <module>', user=user, language='en')

    assert result.audio_bytes is None


@pytest.mark.asyncio
async def test_plain_line_number_is_not_treated_as_stack_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: _Client())
    monkeypatch.setattr(voice_output_module, 'mp3_to_ogg_opus_bytes', lambda data: b'OggS' + data)
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container)

    result = await container.voice_output_service.synthesize('Please review line 42 in the budget doc.', user=user, language='en')

    assert result.audio_bytes == b'OggSfake-mp3'
