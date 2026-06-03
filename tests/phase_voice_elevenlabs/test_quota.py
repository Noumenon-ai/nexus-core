from __future__ import annotations

import asyncio
import logging

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
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, **kwargs):
        return _Response()


def _patch_success(monkeypatch) -> None:
    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: _Client())
    monkeypatch.setattr(voice_output_module, 'mp3_to_ogg_opus_bytes', lambda data: b'OggS' + data)


@pytest.mark.asyncio
async def test_user_under_quota_synthesizes(monkeypatch, tmp_path):
    _patch_success(monkeypatch)
    container = build_elevenlabs_container(monkeypatch, tmp_path, extra_env={'ELEVENLABS_DAILY_CHAR_QUOTA_USER_1': '100'})
    user = create_user(container)

    result = await container.voice_output_service.synthesize('hello world', user=user, language='en')

    assert result.audio_bytes == b'OggSfake-mp3'


@pytest.mark.asyncio
async def test_user_at_quota_suppressed_with_notice(monkeypatch, tmp_path):
    _patch_success(monkeypatch)
    container = build_elevenlabs_container(monkeypatch, tmp_path, extra_env={'ELEVENLABS_DAILY_CHAR_QUOTA_USER_1': '20'})
    user = create_user(container)
    container.elevenlabs_usage_repository.increment(user_id=user.id, day=container.voice_output_service._local_day(), chars=20, api_calls=1)

    result = await container.voice_output_service.synthesize('hello', user=user, language='en')

    assert result.audio_bytes is None
    assert result.notice == 'voice quota for today reached'


@pytest.mark.asyncio
async def test_quota_counter_increments_per_call(monkeypatch, tmp_path):
    _patch_success(monkeypatch)
    container = build_elevenlabs_container(monkeypatch, tmp_path, extra_env={'ELEVENLABS_DAILY_CHAR_QUOTA_USER_1': '100'})
    user = create_user(container)

    await container.voice_output_service.synthesize('hello world', user=user, language='en')
    usage = container.elevenlabs_usage_repository.get_or_create(user_id=user.id, day=container.voice_output_service._local_day())

    assert usage.chars_count == len('hello world')
    assert usage.api_calls == 1


@pytest.mark.asyncio
async def test_soft_alert_at_80_percent_logged(monkeypatch, tmp_path, caplog):
    _patch_success(monkeypatch)
    container = build_elevenlabs_container(monkeypatch, tmp_path, extra_env={'ELEVENLABS_DAILY_CHAR_QUOTA_USER_1': '100'})
    user = create_user(container)
    container.elevenlabs_usage_repository.increment(user_id=user.id, day=container.voice_output_service._local_day(), chars=79, api_calls=1)

    with caplog.at_level(logging.WARNING):
        await container.voice_output_service.synthesize('a', user=user, language='en')

    assert any(record.msg == 'elevenlabs_quota_soft_alert' for record in caplog.records)


@pytest.mark.asyncio
async def test_quota_reservation_blocks_concurrent_overflow(monkeypatch, tmp_path):
    gate = asyncio.Event()
    call_count = 0

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            nonlocal call_count
            call_count += 1
            await gate.wait()
            return _Response()

    monkeypatch.setattr(voice_output_module.httpx, 'AsyncClient', lambda timeout=15.0: Client())
    monkeypatch.setattr(voice_output_module, 'mp3_to_ogg_opus_bytes', lambda data: b'OggS' + data)
    container = build_elevenlabs_container(monkeypatch, tmp_path, extra_env={'ELEVENLABS_DAILY_CHAR_QUOTA_USER_1': '10'})
    user = create_user(container)

    first = asyncio.create_task(container.voice_output_service.synthesize('hello!', user=user, language='en'))
    await asyncio.sleep(0)
    second = asyncio.create_task(container.voice_output_service.synthesize('hello!', user=user, language='en'))
    await asyncio.sleep(0)
    gate.set()
    first_result, second_result = await asyncio.gather(first, second)
    usage = container.elevenlabs_usage_repository.get_or_create(user_id=user.id, day=container.voice_output_service._local_day())

    assert len([result for result in (first_result, second_result) if result.audio_bytes is not None]) == 1
    assert len([result for result in (first_result, second_result) if result.notice == 'voice quota for today reached']) == 1
    assert call_count == 1
    assert usage.chars_count == len('hello!')
    assert usage.api_calls == 1
