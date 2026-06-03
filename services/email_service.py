from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
except ImportError:
    Request = None
    Credentials = None
    InstalledAppFlow = None
    build = None

from bootstrap import secure_token_file
from config import Settings
from models import User
from pipeline.types import ServiceResponse
from repositories.emails_ingested_repository import EmailsIngestedRepository
from utils.i18n import Translator
from utils.text_safety import sanitize_plaintext


logger = logging.getLogger(__name__)
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


async def _run_blocking_without_default_executor(fn, /, *args, **kwargs):
    """Run blocking Gmail work without relying on asyncio's default executor.

    In this environment, `asyncio.to_thread()` can let a worker thread linger
    long enough to make async tests and short-lived processes hang on exit even
    after the blocking work itself has completed. Use a one-off daemon thread
    and await a simple event instead.
    """
    result_box: dict[str, object] = {}
    done = threading.Event()

    def worker():
        try:
            result_box['result'] = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - surfaced via awaiter
            result_box['exception'] = exc
        finally:
            done.set()

    threading.Thread(
        target=worker,
        daemon=True,
        name='email-service-blocking',
    ).start()
    while not done.is_set():
        await asyncio.sleep(0.01)
    if 'exception' in result_box:
        raise result_box['exception']  # type: ignore[misc]
    return result_box.get('result')


class EmailService:
    def __init__(self, settings: Settings, emails_repository: EmailsIngestedRepository) -> None:
        self.settings = settings
        self.emails_repository = emails_repository

    async def handle(self, user: User, translator: Translator) -> ServiceResponse:
        if not self.settings.gmail.gmail_enabled:
            return ServiceResponse(text=translator.t('email_connection_error'))
        try:
            return await _run_blocking_without_default_executor(self._handle_sync, user, translator)
        except Exception:
            logger.warning('email_connection_error')
            return ServiceResponse(text=translator.t('email_connection_error'))

    def _handle_sync(self, user: User, translator: Translator) -> ServiceResponse:
        service = self._build_client(user.id)
        messages = service.users().messages().list(userId='me', q='newer_than:1d', maxResults=25).execute().get('messages', [])
        summaries: list[str] = []
        for item in messages:
            message = service.users().messages().get(userId='me', id=item['id'], format='metadata', metadataHeaders=['Subject', 'From', 'Date']).execute()
            metadata = {header['name']: header['value'] for header in message.get('payload', {}).get('headers', [])}
            subject = metadata.get('Subject')
            sender = metadata.get('From')
            snippet = message.get('snippet')
            category = self._categorize(subject or '', snippet or '')
            received_at = self._received_at(message)
            extracted = {'subject': subject, 'sender': sender, 'snippet': snippet, 'category': category}
            self.emails_repository.upsert(user_id=user.id, gmail_message_id=item['id'], subject=subject, sender=sender, snippet=snippet, received_at=received_at, category=category, extracted_json=json.dumps(extracted, separators=(',', ':'), sort_keys=True))
            summaries.append(self._format_summary(subject, sender, snippet, category, translator))
        if not summaries:
            return ServiceResponse(text=translator.t('email_no_updates'), voice_appropriate=False)
        return ServiceResponse(text='\n'.join(summaries[:8]), voice_appropriate=False)

    def _build_client(self, user_id: str):
        if Credentials is None or InstalledAppFlow is None or build is None or Request is None:
            raise RuntimeError('Gmail dependencies are not installed')
        token_path = self.settings.gmail.gmail_token_dir / f'{user_id}.json'
        creds = None
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if creds is None or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(self.settings.gmail.gmail_credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json(), encoding='utf-8')
            secure_token_file(token_path)
        return build('gmail', 'v1', credentials=creds, cache_discovery=False)

    def _categorize(self, subject: str, snippet: str) -> str:
        combined = f'{subject} {snippet}'.lower()
        if any(keyword in combined for keyword in ('invoice', 'bill', 'payment due')):
            return 'bill'
        if any(keyword in combined for keyword in ('renewal', 'renews', 'subscription')):
            return 'renewal'
        if any(keyword in combined for keyword in ('meeting', 'appointment', 'calendar')):
            return 'appointment'
        if any(keyword in combined for keyword in ('bank', 'statement', 'tax', 'finance')):
            return 'finance'
        if any(keyword in combined for keyword in ('opportunity', 'funding', 'partnership')):
            return 'opportunity'
        if any(keyword in combined for keyword in ('client', 'proposal', 'business')):
            return 'business'
        if any(keyword in combined for keyword in ('family', 'friend', 'party')):
            return 'personal'
        return 'unknown'

    def _format_summary(self, subject: str | None, sender: str | None, snippet: str | None, category: str, translator: Translator) -> str:
        safe_subject = sanitize_plaintext(subject or translator.t('email_no_subject'), max_length=140)
        safe_sender = sanitize_plaintext(sender or translator.t('email_unknown_sender'), max_length=120)
        safe_snippet = sanitize_plaintext(snippet or '', max_length=240)
        return translator.t('email_summary_line', category=category, subject=safe_subject, sender=safe_sender, snippet=safe_snippet)

    def _received_at(self, message: dict) -> datetime:
        internal = message.get('internalDate')
        if internal:
            return datetime.fromtimestamp(int(internal) / 1000, tz=timezone.utc)
        return datetime.now(timezone.utc)
