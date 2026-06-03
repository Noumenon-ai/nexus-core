from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from models import User
from pipeline.types import ServiceResponse
from services.google_auth_service import GoogleAuthError, GoogleAuthService
from utils.i18n import Translator


logger = logging.getLogger(__name__)


_PASTE_ERROR_KEYS = {
    'no_pending_flow': 'google_paste_no_session',
    'flow_expired': 'google_paste_session_expired',
    'state_mismatch': 'google_paste_state_mismatch',
    'no_code_in_paste': 'google_paste_no_code',
    'code_already_redeemed': 'google_paste_already_used',
}


class GoogleIntentHandler:
    """Routes 'connect/disconnect/reconnect google' user intents and accepts
    pasted OAuth redirect URLs (or raw codes) as a code-paste fallback for
    users on devices where the localhost redirect cannot reach the host."""

    def __init__(
        self,
        google_auth_service: GoogleAuthService,
        *,
        notifier: Callable[[str, str], Awaitable[None]] | None = None,
        completion_timeout_seconds: int = 600,
    ) -> None:
        self.google_auth_service = google_auth_service
        self.notifier = notifier
        self.completion_timeout_seconds = completion_timeout_seconds

    def parse_action(self, text: str) -> str | None:
        normalized = text.strip().lower()
        if any(p in normalized for p in ('reconnect google', 'reconnect my google')):
            return 'reconnect'
        if any(p in normalized for p in ('disconnect google', 'disconnect my google', 'unlink google', 'unlink my google')):
            return 'disconnect'
        if any(p in normalized for p in (
            'connect google',
            'connect my google',
            'link google',
            'link my google',
            'set up google',
            'setup google',
        )):
            return 'connect'
        return None

    @staticmethod
    def _looks_like_paste(text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        if 'code=' in normalized.lower():
            return True
        if normalized.startswith('4/') and len(normalized) >= 20 and ' ' not in normalized:
            return True
        return False

    async def handle(self, user: User, text: str, translator: Translator) -> ServiceResponse:
        if self._looks_like_paste(text) and self.google_auth_service.has_pending(user.id):
            return await self._handle_paste(user, text, translator)
        action = self.parse_action(text)
        if action is None:
            if self._looks_like_paste(text):
                return ServiceResponse(text=translator.t('google_paste_no_session'))
            return ServiceResponse(text=translator.t('google_setup_clarify'))
        if action == 'connect':
            return await self._handle_connect_or_reconnect(user, translator, reconnect=False)
        if action == 'reconnect':
            return await self._handle_connect_or_reconnect(user, translator, reconnect=True)
        if action == 'disconnect':
            return await self._handle_disconnect(user, translator)
        return ServiceResponse(text=translator.t('google_setup_clarify'))

    async def _handle_connect_or_reconnect(self, user: User, translator: Translator, *, reconnect: bool) -> ServiceResponse:
        if not self.google_auth_service.settings.google.enabled:
            return ServiceResponse(text=translator.t('google_disabled'))
        try:
            url, future = await self.google_auth_service.begin_connect(user.id, reconnect=reconnect)
        except GoogleAuthError as exc:
            return ServiceResponse(text=translator.t('google_connect_failed', error=str(exc)))
        asyncio.create_task(self._await_completion(user, future, translator))
        return ServiceResponse(
            text=translator.t('google_url_message', url=url),
            voice_appropriate=False,
        )

    async def _handle_disconnect(self, user: User, translator: Translator) -> ServiceResponse:
        if not self.google_auth_service.settings.google.enabled:
            return ServiceResponse(text=translator.t('google_disabled'))
        try:
            await self.google_auth_service.disconnect(user.id)
            return ServiceResponse(text=translator.t('google_disconnect_success'))
        except GoogleAuthError as exc:
            return ServiceResponse(text=translator.t('google_disconnect_failed', error=str(exc)))

    async def _handle_paste(self, user: User, text: str, translator: Translator) -> ServiceResponse:
        try:
            result = await self.google_auth_service.submit_pasted_url(user.id, text)
        except GoogleAuthError as exc:
            error_key = _PASTE_ERROR_KEYS.get(str(exc))
            if error_key is None:
                return ServiceResponse(text=translator.t('google_connect_failed', error=str(exc)))
            return ServiceResponse(text=translator.t(error_key))
        granted = ', '.join(result.scopes_granted) if result.scopes_granted else translator.t('google_no_scopes_granted')
        return ServiceResponse(text=translator.t(
            'google_connect_success',
            email=result.email or translator.t('google_unknown_email'),
            scopes=granted,
        ))

    async def _await_completion(self, user: User, future, translator: Translator) -> None:
        try:
            result = await asyncio.wait_for(future, timeout=self.completion_timeout_seconds)
        except asyncio.TimeoutError:
            self.google_auth_service.cancel_pending(user.id)
            await self._notify(user, translator.t('google_url_timeout'))
            return
        except GoogleAuthError as exc:
            await self._notify(user, translator.t('google_connect_failed', error=str(exc)))
            return
        except Exception as exc:
            logger.exception('google_oauth_unexpected_error')
            await self._notify(user, translator.t('google_connect_failed', error=str(exc)))
            return
        if getattr(result, 'via_paste', False):
            return
        granted = ', '.join(result.scopes_granted) if result.scopes_granted else translator.t('google_no_scopes_granted')
        await self._notify(
            user,
            translator.t(
                'google_connect_success',
                email=result.email or translator.t('google_unknown_email'),
                scopes=granted,
            ),
        )

    async def _notify(self, user: User, text: str) -> None:
        if self.notifier is None:
            return
        try:
            await self.notifier(user.id, text)
        except Exception:
            logger.warning('google_notify_failed', extra={'user_id': user.id})
