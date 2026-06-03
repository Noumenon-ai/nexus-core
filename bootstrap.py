from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from config import Settings
from db import create_all
from migrations.audit_events_migration import run_audit_events_migration
from migrations.reminder_delivery_attempts_migration import (
    run_reminder_delivery_attempts_migration,
)
from utils.logging import configure_logging


logger = logging.getLogger(__name__)


class StartupValidationError(RuntimeError):
    """Raised when startup validation fails."""


class StartupValidator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, engine) -> None:
        configure_logging(self.settings.logging.log_level, self.settings.logging.redact_patterns)
        self._ensure_data_directories()
        self._validate_threat_model()
        self.validate_environment()
        self._secure_optional_credentials()
        create_all(engine, self.settings)
        run_reminder_delivery_attempts_migration(engine)
        run_audit_events_migration(engine)
        self._secure_database_file()
        logger.info('startup_validation_complete')

    def validate_environment(self) -> None:
        self._validate_voice_environment()
        self._validate_gmail_environment()
        self._validate_google_environment()

    def _validate_voice_environment(self) -> None:
        if not self.settings.voice.voice_output_enabled:
            return
        if self.settings.voice.voice_output_backend != 'elevenlabs':
            return
        if shutil.which('ffmpeg') is None:
            raise StartupValidationError('VOICE_OUTPUT_BACKEND=elevenlabs requires ffmpeg on PATH. Install ffmpeg and restart Nexus.')
        if not any(config.api_key for config in self.settings.voice.elevenlabs_users):
            raise StartupValidationError('VOICE_OUTPUT_BACKEND=elevenlabs requires at least one ELEVENLABS_API_KEY_USER_N environment variable.')
        if not self.settings.voice.elevenlabs_model:
            raise StartupValidationError('VOICE_OUTPUT_BACKEND=elevenlabs requires ELEVENLABS_MODEL to be set.')

    def _validate_gmail_environment(self) -> None:
        creds_path = self.settings.gmail.gmail_credentials_path
        secure_private_file(creds_path)
        if not self.settings.gmail.gmail_enabled:
            return
        if not creds_path.exists():
            raise StartupValidationError(
                f'GMAIL_ENABLED=true requires {creds_path} (OAuth client JSON for InstalledAppFlow).'
            )
        try:
            payload = json.loads(creds_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise StartupValidationError(f'Cannot parse Gmail credentials at {creds_path}: {exc}') from exc
        if not isinstance(payload, dict) or not ({'installed', 'web'} & set(payload.keys())):
            raise StartupValidationError(
                f'Gmail credentials at {creds_path} must be an OAuth client JSON with a top-level '
                "'installed' or 'web' key. Service-account credentials are not supported."
            )
        token_dir = self.settings.gmail.gmail_token_dir
        token_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(token_dir, 0o700)
        except PermissionError as exc:
            raise StartupValidationError(f'Unable to secure {token_dir}: {exc}') from exc

    def _validate_google_environment(self) -> None:
        creds_path = self.settings.google.credentials_path
        if not self.settings.google.enabled:
            secure_private_file(creds_path)
            return
        if not creds_path.exists():
            raise StartupValidationError(
                f'GOOGLE_INTEGRATION_ENABLED=true requires {creds_path} (Desktop OAuth client JSON from Google Cloud Console).'
            )
        try:
            payload = json.loads(creds_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise StartupValidationError(f'Cannot parse Google credentials at {creds_path}: {exc}') from exc
        if 'installed' not in payload:
            raise StartupValidationError(
                f'Google credentials at {creds_path} must be a Desktop OAuth client '
                "(top-level key 'installed'). Service-account credentials are not supported."
            )
        secure_private_file(creds_path)
        token_dir = self.settings.google.token_dir
        token_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(token_dir, 0o700)
        except PermissionError as exc:
            raise StartupValidationError(f'Unable to secure {token_dir}: {exc}') from exc
        if not self.settings.google.scopes:
            raise StartupValidationError('GOOGLE_INTEGRATION_ENABLED=true requires GOOGLE_SCOPES to be non-empty.')

    def _ensure_data_directories(self) -> None:
        protected_paths = (
            self.settings.data_dir,
            self.settings.voice_in_dir,
            self.settings.voice_out_dir,
            self.settings.gmail.gmail_token_dir,
        )
        for path in protected_paths:
            path.mkdir(parents=True, exist_ok=True)
        try:
            for path in protected_paths:
                os.chmod(path, 0o700)
        except PermissionError as exc:
            raise StartupValidationError('Unable to set required directory permissions') from exc

    def _validate_threat_model(self) -> None:
        if self.settings.security.threat_model == 'hosted' and not self.settings.security.sqlcipher_key:
            raise StartupValidationError('Hosted mode requires SQLCIPHER_KEY')

    def _secure_optional_credentials(self) -> None:
        secure_private_file(self.settings.gmail.gmail_credentials_path)
        secure_private_file(self.settings.google.credentials_path)

    def _secure_database_file(self) -> None:
        database_path = self.settings.database_path
        if database_path is None or not database_path.exists():
            return
        try:
            os.chmod(database_path, 0o600)
        except PermissionError as exc:
            raise StartupValidationError('Unable to set required database file permissions') from exc


def secure_private_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        os.chmod(path, 0o600)


def secure_token_file(path: Path) -> None:
    secure_private_file(path)
