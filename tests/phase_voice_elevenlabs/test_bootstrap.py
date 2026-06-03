from __future__ import annotations

import pytest

import bootstrap as bootstrap_module
import main as main_module
from bootstrap import StartupValidationError, StartupValidator
from db import create_database
from tests.helpers import configure_test_env


class _DummyTelegramBot:
    def __init__(self, *, token: str, pipeline, voice_in_dir, max_voice_file_bytes):
        self.token = token
        self.pipeline = pipeline
        self.voice_in_dir = voice_in_dir
        self.max_voice_file_bytes = max_voice_file_bytes
        self.application = type('DummyApplication', (), {})()


class _DummyScheduler:
    def __init__(self, *args, **kwargs) -> None:
        self.runtime = None

    def register_runtime(self, runtime) -> None:
        self.runtime = runtime


def test_bootstrap_passes_with_valid_config(monkeypatch, tmp_path):
    settings = configure_test_env(monkeypatch, tmp_path, extra={
        'VOICE_OUTPUT_BACKEND': 'elevenlabs',
        'ELEVENLABS_API_KEY_USER_1': 'key_1',
    })
    database = create_database(settings)
    StartupValidator(settings).run(database.engine)


def test_bootstrap_fails_when_ffmpeg_missing(monkeypatch, tmp_path):
    settings = configure_test_env(monkeypatch, tmp_path, extra={
        'VOICE_OUTPUT_BACKEND': 'elevenlabs',
        'ELEVENLABS_API_KEY_USER_1': 'key_1',
    })
    monkeypatch.setattr(bootstrap_module.shutil, 'which', lambda name: None)
    with pytest.raises(StartupValidationError):
        StartupValidator(settings).validate_environment()


def test_bootstrap_fails_when_no_keys_set_with_elevenlabs_backend(monkeypatch, tmp_path):
    settings = configure_test_env(monkeypatch, tmp_path, extra={'VOICE_OUTPUT_BACKEND': 'elevenlabs'})
    with pytest.raises(StartupValidationError):
        StartupValidator(settings).validate_environment()


def test_bootstrap_skips_when_backend_is_piper(monkeypatch, tmp_path):
    piper_settings = configure_test_env(monkeypatch, tmp_path, extra={'VOICE_OUTPUT_BACKEND': 'piper'})
    StartupValidator(piper_settings).validate_environment()


def test_build_runtime_migrates_seeded_users(monkeypatch, tmp_path):
    configure_test_env(monkeypatch, tmp_path, extra={
        'VOICE_OUTPUT_ENABLED': 'false',
        'VOICE_INPUT_ENABLED': 'false',
        'ELEVENLABS_API_KEY_USER_1': 'key_1',
        'ELEVENLABS_VOICE_ID_USER_1': 'voice_1',
    })
    monkeypatch.setattr(main_module, 'TelegramBot', _DummyTelegramBot)
    monkeypatch.setattr(main_module, 'NexusScheduler', _DummyScheduler)

    runtime, _telegram_bot, _scheduler = main_module.build_runtime()
    users = runtime.users_repository.list_by_created_at()

    assert users[0].elevenlabs_api_key == 'key_1'
    assert users[0].elevenlabs_voice_id == 'voice_1'
