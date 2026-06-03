from __future__ import annotations

import pytest

from utils.i18n import Translator


class FakeGmailMessageGetter:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class FakeGmailMessages:
    def list(self, **kwargs):
        return FakeGmailMessageGetter({'messages': [{'id': '1'}]})

    def get(self, **kwargs):
        return FakeGmailMessageGetter({
            'payload': {'headers': [{'name': 'Subject', 'value': 'Invoice due'}, {'name': 'From', 'value': 'Billing <bill@example.com>'}]},
            'snippet': 'Your payment is due tomorrow.',
        })


class FakeGmailUsers:
    def messages(self):
        return FakeGmailMessages()


class FakeGmailService:
    def users(self):
        return FakeGmailUsers()


@pytest.mark.asyncio
async def test_email_scan_categorizes_recent_mail(container, monkeypatch):
    monkeypatch.setattr(container.email_service, '_build_client', lambda user_id: FakeGmailService())
    object.__setattr__(
        container.email_service.settings,
        'gmail',
        container.email_service.settings.gmail.__class__(
            gmail_enabled=True,
            gmail_credentials_path=container.email_service.settings.gmail.gmail_credentials_path,
            gmail_token_dir=container.email_service.settings.gmail.gmail_token_dir,
        ),
    )
    response = await container.email_service.handle(container.users_repository.get_or_create(111), Translator())
    assert '[bill]' in response.text.lower()
    assert 'invoice due' in response.text.lower()


@pytest.mark.asyncio
async def test_email_failure_returns_safe_error(container, monkeypatch):
    monkeypatch.setattr(container.email_service, '_build_client', lambda user_id: (_ for _ in ()).throw(RuntimeError('boom')))
    object.__setattr__(
        container.email_service.settings,
        'gmail',
        container.email_service.settings.gmail.__class__(
            gmail_enabled=True,
            gmail_credentials_path=container.email_service.settings.gmail.gmail_credentials_path,
            gmail_token_dir=container.email_service.settings.gmail.gmail_token_dir,
        ),
    )
    response = await container.email_service.handle(container.users_repository.get_or_create(111), Translator())
    assert response.text == 'Email connection error.'
