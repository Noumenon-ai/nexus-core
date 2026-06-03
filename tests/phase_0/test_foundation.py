from __future__ import annotations

import pytest

from bootstrap import StartupValidator
from config import ConfigError, get_settings
from db import create_database
from utils.logging import RedactionFilter


def test_settings_loads_expected_defaults(settings):
    assert settings.core.telegram_bot_token == 'test-token'
    assert settings.core.allowed_telegram_ids == (111, 222)
    assert settings.core.app_timezone == 'UTC'
    assert settings.approval.destructive_approval_enabled is True


def test_destructive_approval_enabled_defaults_true(monkeypatch, tmp_path):
    db_path = tmp_path / 'nexus_test.db'
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'test-token')
    monkeypatch.setenv('ALLOWED_TELEGRAM_IDS', '111')
    monkeypatch.setenv('APP_TIMEZONE', 'UTC')
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{db_path}')
    monkeypatch.delenv('DESTRUCTIVE_APPROVAL_ENABLED', raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.approval.destructive_approval_enabled is True
    get_settings.cache_clear()


def test_startup_validator_creates_data_directories(settings):
    database = create_database(settings)
    StartupValidator(settings).run(database.engine)
    assert settings.data_dir.exists()
    assert settings.voice_in_dir.exists()
    assert settings.gmail.gmail_token_dir.exists()


def test_startup_validator_secures_sqlite_database_file(settings):
    database = create_database(settings)
    StartupValidator(settings).run(database.engine)
    assert settings.database_path is not None
    assert settings.database_path.exists()
    assert settings.database_path.stat().st_mode & 0o777 == 0o600


def test_database_schema_contains_core_tables(container):
    engine = container.database.engine
    table_names = set(engine.table_names()) if hasattr(engine, 'table_names') else set(engine.dialect.get_table_names(engine.connect()))
    assert {'users', 'reminders', 'tasks', 'memories', 'approvals'}.issubset(table_names)


def test_redaction_filter_masks_sensitive_terms():
    redaction = RedactionFilter(['token', 'password', 'Bearer'])

    class Record:
        msg = 'token=abc password="xyz" Authorization: Bearer secret-token'
        args = ()

    record = Record()
    assert redaction.filter(record) is True
    assert 'abc' not in record.msg
    assert 'xyz' not in record.msg
    assert 'secret-token' not in record.msg
    assert 'Bearer [REDACTED]' in record.msg


def test_hosted_mode_requires_sqlcipher_database_url(monkeypatch, tmp_path):
    db_path = tmp_path / 'nexus_test.db'
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'test-token')
    monkeypatch.setenv('ALLOWED_TELEGRAM_IDS', '111')
    monkeypatch.setenv('APP_TIMEZONE', 'UTC')
    monkeypatch.setenv('THREAT_MODEL', 'hosted')
    monkeypatch.setenv('SQLCIPHER_KEY', 'secret')
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{db_path}')
    get_settings.cache_clear()
    with pytest.raises(ConfigError):
        get_settings()
