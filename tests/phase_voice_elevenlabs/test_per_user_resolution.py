from __future__ import annotations

from datetime import datetime, timezone

import pytest

import services.voice_output_service as voice_output_module
from tests.phase_voice_elevenlabs.helpers import build_elevenlabs_container, create_user


class _Response:
    def __init__(self) -> None:
        self.content = b'fake-mp3'
        self.status_code = 200

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
        self.captured.append({'url': url, 'headers': kwargs.get('headers', {})})
        return _Response()


def _patch_success(monkeypatch, captured: list[dict]) -> None:
    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: _Client(captured))
    monkeypatch.setattr(voice_output_module, 'mp3_to_ogg_opus_bytes', lambda data: b'OggS' + data)


@pytest.mark.asyncio
async def test_user_with_voice_id_uses_their_voice(monkeypatch, tmp_path):
    captured: list[dict] = []
    _patch_success(monkeypatch, captured)
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container)
    container.users_repository.set_elevenlabs_profile(user.id, voice_id='db_voice', api_key='db_key', migrated_at=datetime.now(timezone.utc))
    user = container.users_repository.get_by_id(user.id)

    result = await container.voice_output_service.synthesize('hello', user=user, language='en')

    assert result.audio_bytes == b'OggSfake-mp3'
    assert captured[0]['url'].endswith('/db_voice')
    assert captured[0]['headers']['xi-api-key'] == 'db_key'


@pytest.mark.asyncio
async def test_user_without_voice_id_falls_back_to_env_pre_migration(monkeypatch, tmp_path):
    captured: list[dict] = []
    _patch_success(monkeypatch, captured)
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container)

    result = await container.voice_output_service.synthesize('hello', user=user, language='en')

    assert result.audio_bytes == b'OggSfake-mp3'
    assert captured[0]['url'].endswith('/voice_user_1')
    assert captured[0]['headers']['xi-api-key'] == 'sk_user_1'


@pytest.mark.asyncio
async def test_user_without_voice_id_post_migration_fails_gracefully(monkeypatch, tmp_path):
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container)
    container.users_repository.set_elevenlabs_profile(user.id, voice_id=None, api_key=None, migrated_at=datetime.now(timezone.utc))
    user = container.users_repository.get_by_id(user.id)

    result = await container.voice_output_service.synthesize('hello', user=user, language='en')

    assert result.audio_bytes is None
    assert result.notice is None


@pytest.mark.asyncio
async def test_both_unset_hard_fallback_to_adam(monkeypatch, tmp_path):
    captured: list[dict] = []
    _patch_success(monkeypatch, captured)
    container = build_elevenlabs_container(monkeypatch, tmp_path, extra_env={'ELEVENLABS_VOICE_ID_USER_1': ''})
    user = create_user(container)

    result = await container.voice_output_service.synthesize('hello', user=user, language='en')

    assert result.audio_bytes == b'OggSfake-mp3'
    assert captured[0]['url'].endswith('/pNInz6obpgDQGcFmaJgB')
    assert captured[0]['headers']['xi-api-key'] == 'sk_user_1'


@pytest.mark.asyncio
async def test_invalid_user_voice_id_falls_back_to_adam(monkeypatch, tmp_path):
    captured: list[dict] = []
    _patch_success(monkeypatch, captured)
    container = build_elevenlabs_container(monkeypatch, tmp_path)
    user = create_user(container)
    container.users_repository.set_elevenlabs_profile(user.id, voice_id='../bad-voice', api_key='db_key', migrated_at=datetime.now(timezone.utc))
    user = container.users_repository.get_by_id(user.id)

    result = await container.voice_output_service.synthesize('hello', user=user, language='en')

    assert result.audio_bytes == b'OggSfake-mp3'
    assert captured[0]['url'].endswith('/pNInz6obpgDQGcFmaJgB')
    assert captured[0]['headers']['xi-api-key'] == 'db_key'
