from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable
from urllib.parse import parse_qs, urlencode, urlparse
from wsgiref.simple_server import WSGIServer, make_server

import httpx

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
except ImportError:  # pragma: no cover
    Credentials = None  # type: ignore[assignment]
    Flow = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]

from config import Settings
from models import now_utc
from repositories.users_repository import UsersRepository


logger = logging.getLogger(__name__)


GOOGLE_REVOKE_URL = 'https://oauth2.googleapis.com/revoke'
GOOGLE_USERINFO_URL = 'https://openidconnect.googleapis.com/v1/userinfo'

DEFAULT_PENDING_TTL_SECONDS = 600

_REDIRECT_SUCCESS_HTML = (
    b'<html><body style="font-family: sans-serif; padding: 2em">'
    b'<h2>Connected.</h2>'
    b'<p>You can close this tab and return to Nexus on Telegram.</p>'
    b'</body></html>'
)
_REDIRECT_ERROR_HTML = (
    b'<html><body style="font-family: sans-serif; padding: 2em">'
    b'<h2>Connection failed.</h2>'
    b'<p>Return to Nexus on Telegram for next steps.</p>'
    b'</body></html>'
)


async def _run_blocking_without_default_executor(fn, /, *args, **kwargs):
    """Run blocking OAuth calls without touching asyncio's default executor.

    In this environment, `asyncio.to_thread()` can let a default-executor
    worker survive long enough to keep interpreter shutdown hanging after async
    tests already pass. Also, custom `ThreadPoolExecutor` wrappers can stall
    while handing control back to the loop here. Use a one-off thread that
    resolves an asyncio Future directly on the current loop.
    """
    loop = asyncio.get_running_loop()
    result_box: dict[str, object] = {}
    done = threading.Event()

    def worker():
        try:
            result_box['result'] = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - exercised through awaiter
            result_box['exception'] = exc
        else:
            result_box['has_result'] = True
        finally:
            done.set()

    threading.Thread(
        target=worker,
        daemon=True,
        name='google-auth-blocking',
    ).start()
    while not done.is_set():
        await asyncio.sleep(0.01)
    if 'exception' in result_box:
        raise result_box['exception']  # type: ignore[misc]
    return result_box.get('result')


class GoogleAuthError(RuntimeError):
    """Raised when the OAuth flow cannot complete."""


@dataclass(slots=True)
class GoogleConnectionResult:
    user_id: str
    email: str | None
    scopes_granted: tuple[str, ...]
    via_paste: bool = False


@dataclass
class _PendingFlow:
    user_id: str
    flow: Any  # google_auth_oauthlib Flow or test fake
    state: str
    started_at: datetime
    loop: asyncio.AbstractEventLoop | None = None
    future: asyncio.Future | None = None
    server: WSGIServer | None = None
    server_thread: threading.Thread | None = None
    completed: bool = False
    redeeming: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class GoogleAuthService:
    """OAuth flow + per-user token storage for Calendar / Tasks / People.

    Architecture: hybrid localhost + paste-fallback.

    `begin_connect` returns the authorization URL synchronously and starts a
    127.0.0.1 ephemeral local server. Two completion paths converge:

    1. Laptop-seamless: user opens URL on host, browser redirects to localhost,
       the embedded WSGI server captures `code` + `state`, finalizes.
    2. Phone-paste: user opens URL on any device, signs in. The redirect to
       localhost cannot reach the host, but the device address bar shows the
       URL with `?code=...&state=...`. User copies and pastes into Telegram.
       `submit_pasted_url` parses, validates state, finalizes.

    The same `_PendingFlow` (state, flow object, future) is shared by both
    paths under a thread lock so only the first arrival can redeem the code.
    Replay attempts and cross-user state mismatches are rejected.

    Tokens persist at `<token_dir>/<user_id>.json` mode 600. Parent dir 700.
    """

    def __init__(
        self,
        settings: Settings,
        users_repository: UsersRepository,
        *,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        flow_factory: Callable[[list[str]], Any] | None = None,
        local_server_starter: Callable[['_PendingFlow'], tuple[int, WSGIServer | None, threading.Thread | None]] | None = None,
        pending_ttl_seconds: int = DEFAULT_PENDING_TTL_SECONDS,
    ) -> None:
        self.settings = settings
        self.users_repository = users_repository
        self._http_client_factory = http_client_factory or (lambda: httpx.AsyncClient(timeout=15.0))
        self._flow_factory = flow_factory or self._default_flow_factory
        self._local_server_starter = local_server_starter or self._default_local_server_starter
        self._pending_ttl_seconds = pending_ttl_seconds
        self._pending: dict[str, _PendingFlow] = {}
        self._pending_lock = threading.Lock()

    # ------------------------------------------------------------------ public

    async def begin_connect(
        self,
        user_id: str,
        *,
        on_url: Callable[[str], Awaitable[None] | None] | None = None,
        reconnect: bool = False,
    ) -> tuple[str, asyncio.Future]:
        """Start an OAuth flow. Returns (auth_url, future).

        The future resolves to GoogleConnectionResult once either the localhost
        redirect or `submit_pasted_url` completes the flow. Caller is
        responsible for awaiting it (typically in a background task).
        """
        self._require_enabled()
        user = self.users_repository.get_by_id(user_id)
        if user is None:
            raise GoogleAuthError(f'unknown user {user_id}')
        if reconnect:
            await self._best_effort_disconnect(user_id)
        self._cancel_pending(user_id, reason='superseded')

        state = secrets.token_urlsafe(32)
        flow = self._flow_factory(list(self.settings.google.scopes))
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        pending = _PendingFlow(
            user_id=user_id,
            flow=flow,
            state=state,
            started_at=now_utc(),
            loop=loop,
            future=future,
        )

        port, server, server_thread = self._local_server_starter(pending)
        pending.server = server
        pending.server_thread = server_thread
        flow.redirect_uri = f'http://127.0.0.1:{port}/'
        auth_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent',
            state=state,
        )

        with self._pending_lock:
            self._pending[user_id] = pending

        if on_url is not None:
            try:
                emitted = on_url(auth_url)
                if asyncio.iscoroutine(emitted):
                    await emitted
            except Exception:
                logger.warning('google_on_url_callback_failed')
        return auth_url, future

    async def submit_pasted_url(self, user_id: str, text: str) -> GoogleConnectionResult:
        self._require_enabled()
        with self._pending_lock:
            pending = self._pending.get(user_id)
        if pending is None:
            raise GoogleAuthError('no_pending_flow')
        age = (now_utc() - pending.started_at).total_seconds()
        if age > self._pending_ttl_seconds:
            self._cancel_pending(user_id, reason='flow_expired')
            raise GoogleAuthError('flow_expired')

        code, state = self._parse_pasted_code(text)
        if not code:
            raise GoogleAuthError('no_code_in_paste')
        if state is not None and state != pending.state:
            raise GoogleAuthError('state_mismatch')

        return await self._finalize(pending, code, via_paste=True)

    async def connect(
        self,
        user_id: str,
        *,
        on_url: Callable[[str], Awaitable[None] | None] | None = None,
        timeout_seconds: int = DEFAULT_PENDING_TTL_SECONDS,
    ) -> GoogleConnectionResult:
        """Convenience: blocks until either path completes. Used by tests and
        non-Telegram callers; the Telegram pipeline uses begin_connect directly
        so it can return the URL without holding the request open."""
        url, future = await self.begin_connect(user_id, on_url=on_url)
        del url  # caller already saw it via on_url
        try:
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            self._cancel_pending(user_id, reason='timeout')
            raise GoogleAuthError('timeout') from exc

    async def reconnect(
        self,
        user_id: str,
        *,
        on_url: Callable[[str], Awaitable[None] | None] | None = None,
        timeout_seconds: int = DEFAULT_PENDING_TTL_SECONDS,
    ) -> GoogleConnectionResult:
        await self._best_effort_disconnect(user_id)
        return await self.connect(user_id, on_url=on_url, timeout_seconds=timeout_seconds)

    async def disconnect(self, user_id: str) -> None:
        self._require_enabled()
        creds = self._load_token(user_id)
        if creds is not None:
            try:
                await self._revoke(creds)
            except Exception:
                logger.warning('google_revoke_failed', extra={'user_id': user_id})
        self._token_path(user_id).unlink(missing_ok=True)
        self.users_repository.set_google_profile(
            user_id=user_id,
            connected=False,
            email=None,
            scopes_granted=None,
            connected_at=None,
        )

    def cancel_pending(self, user_id: str) -> None:
        self._cancel_pending(user_id, reason='cancelled')

    def has_pending(self, user_id: str) -> bool:
        with self._pending_lock:
            return user_id in self._pending

    # -------------------------------------------------- read-only sibling APIs

    def is_connected(self, user_id: str) -> bool:
        user = self.users_repository.get_by_id(user_id)
        return bool(user and user.google_connected)

    def granted_scopes(self, user_id: str) -> set[str]:
        user = self.users_repository.get_by_id(user_id)
        if not user or not user.google_scopes_granted:
            return set()
        return {scope.strip() for scope in user.google_scopes_granted.split(',') if scope.strip()}

    def has_scope(self, user_id: str, scope: str) -> bool:
        return scope in self.granted_scopes(user_id)

    def get_credentials(self, user_id: str) -> Credentials | None:
        creds = self._load_token(user_id)
        if creds is None:
            return None
        if creds.expired and creds.refresh_token and Request is not None:
            try:
                creds.refresh(Request())
                self._persist_token(user_id, creds)
            except Exception:
                logger.warning('google_token_refresh_failed', extra={'user_id': user_id})
                return None
        return creds

    # ----------------------------------------------------------- internals

    def _require_enabled(self) -> None:
        if not self.settings.google.enabled:
            raise GoogleAuthError('GOOGLE_INTEGRATION_ENABLED is false')
        if Flow is None or Credentials is None:
            raise GoogleAuthError('google-auth-oauthlib is not installed')

    @staticmethod
    def _parse_pasted_code(text: str) -> tuple[str | None, str | None]:
        text = (text or '').strip()
        if not text:
            return None, None
        if '://' in text:
            for token in text.split():
                if '://' in token and 'code=' in token:
                    try:
                        parsed = urlparse(token)
                        query = parsed.query or token.split('?', 1)[1] if '?' in token else ''
                        qs = parse_qs(query)
                        code = (qs.get('code') or [None])[0]
                        state = (qs.get('state') or [None])[0]
                        if code:
                            return code, state
                    except Exception:
                        continue
        if 'code=' in text:
            idx = text.find('?')
            substring = text[idx + 1:] if idx >= 0 else text
            try:
                qs = parse_qs(substring)
                code = (qs.get('code') or [None])[0]
                state = (qs.get('state') or [None])[0]
                if code:
                    return code, state
            except Exception:
                pass
        if text.startswith('4/') and len(text) >= 20 and not any(c.isspace() for c in text):
            return text, None
        return None, None

    async def _finalize(self, pending: _PendingFlow, code: str, *, via_paste: bool = False) -> GoogleConnectionResult:
        with pending.lock:
            if pending.completed or pending.redeeming:
                raise GoogleAuthError('code_already_redeemed')
            pending.redeeming = True
        try:
            await _run_blocking_without_default_executor(pending.flow.fetch_token, code=code)
        except Exception as exc:
            with pending.lock:
                pending.redeeming = False
            raise GoogleAuthError(f'token_exchange_failed: {exc}') from exc
        with pending.lock:
            try:
                pending.completed = True
            finally:
                pending.redeeming = False
        creds = pending.flow.credentials
        granted = tuple(sorted(getattr(creds, 'scopes', None) or ()))
        email = await self._fetch_email(creds)
        self._persist_token(pending.user_id, creds)
        self.users_repository.set_google_profile(
            user_id=pending.user_id,
            connected=True,
            email=email,
            scopes_granted=','.join(granted) if granted else None,
            connected_at=now_utc(),
        )
        result = GoogleConnectionResult(user_id=pending.user_id, email=email, scopes_granted=granted, via_paste=via_paste)
        if pending.future is not None and not pending.future.done():
            try:
                pending.future.set_result(result)
            except Exception:
                logger.warning('google_future_resolve_failed')
        self._teardown_server(pending)
        with self._pending_lock:
            current = self._pending.get(pending.user_id)
            if current is pending:
                self._pending.pop(pending.user_id, None)
        return result

    def _cancel_pending(self, user_id: str, *, reason: str) -> None:
        with self._pending_lock:
            pending = self._pending.pop(user_id, None)
        if pending is None:
            return
        if pending.future is not None and pending.loop is not None and not pending.future.done():
            try:
                pending.loop.call_soon_threadsafe(pending.future.set_exception, GoogleAuthError(reason))
            except Exception:
                pass
        self._teardown_server(pending)

    @staticmethod
    def _teardown_server(pending: _PendingFlow) -> None:
        if pending.server is None:
            return
        try:
            threading.Thread(target=pending.server.shutdown, daemon=True).start()
        except Exception:
            pass

    async def _fetch_email(self, creds) -> str | None:
        token = getattr(creds, 'token', None)
        if not token:
            return None
        try:
            async with self._http_client_factory() as client:
                response = await client.get(
                    GOOGLE_USERINFO_URL,
                    headers={'Authorization': f'Bearer {token}'},
                )
                if response.status_code != 200:
                    return None
                data = response.json()
                return data.get('email')
        except Exception:
            logger.warning('google_userinfo_fetch_failed')
            return None

    async def _revoke(self, creds) -> None:
        token = getattr(creds, 'refresh_token', None) or getattr(creds, 'token', None)
        if not token:
            return
        async with self._http_client_factory() as client:
            response = await client.post(
                GOOGLE_REVOKE_URL,
                content=urlencode({'token': token}),
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
            )
            if response.status_code >= 400:
                logger.warning('google_revoke_status', extra={'status': response.status_code})

    async def _best_effort_disconnect(self, user_id: str) -> None:
        try:
            await self.disconnect(user_id)
        except GoogleAuthError:
            raise
        except Exception:
            logger.warning('google_disconnect_failed', extra={'user_id': user_id})

    def _token_path(self, user_id: str) -> Path:
        return self.settings.google.token_dir / f'{user_id}.json'

    def _persist_token(self, user_id: str, creds) -> None:
        token_dir = self.settings.google.token_dir
        token_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(token_dir, 0o700)
        except OSError:
            pass
        path = self._token_path(user_id)
        path.write_text(_creds_to_json(creds), encoding='utf-8')
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def _load_token(self, user_id: str):
        if Credentials is None:
            return None
        path = self._token_path(user_id)
        if not path.exists():
            return None
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        try:
            return Credentials.from_authorized_user_file(str(path), list(self.settings.google.scopes))
        except Exception:
            logger.warning('google_token_load_failed', extra={'user_id': user_id})
            return None

    # -------------------------------------------------------- defaults

    def _default_flow_factory(self, scopes: list[str]):
        return Flow.from_client_secrets_file(
            str(self.settings.google.credentials_path),
            scopes=scopes,
        )

    def _default_local_server_starter(self, pending: _PendingFlow) -> tuple[int, WSGIServer | None, threading.Thread | None]:
        service = self

        def app(environ, start_response):
            qs = parse_qs(environ.get('QUERY_STRING', ''))
            code = (qs.get('code') or [None])[0]
            state = (qs.get('state') or [None])[0]
            error = (qs.get('error') or [None])[0]
            if error or not code or state != pending.state:
                body = _REDIRECT_ERROR_HTML
                if pending.future is not None and pending.loop is not None and not pending.future.done():
                    pending.loop.call_soon_threadsafe(
                        pending.future.set_exception,
                        GoogleAuthError(error or 'state_mismatch' if state != pending.state else 'no_code'),
                    )
            else:
                body = _REDIRECT_SUCCESS_HTML
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._finalize(pending, code),
                        pending.loop,
                    )
                except Exception:
                    logger.warning('google_local_redirect_finalize_failed')
            start_response('200 OK', [('Content-Type', 'text/html; charset=utf-8'), ('Content-Length', str(len(body)))])
            return [body]

        server = make_server('127.0.0.1', 0, app)
        port = server.server_port

        def serve():
            try:
                server.serve_forever(poll_interval=0.5)
            finally:
                try:
                    server.server_close()
                except Exception:
                    pass

        thread = threading.Thread(target=serve, daemon=True, name=f'google-oauth-{pending.user_id[:8]}')
        thread.start()
        return port, server, thread


def _creds_to_json(creds) -> str:
    if hasattr(creds, 'to_json'):
        return creds.to_json()
    return json.dumps({
        'token': getattr(creds, 'token', None),
        'refresh_token': getattr(creds, 'refresh_token', None),
        'token_uri': getattr(creds, 'token_uri', None),
        'client_id': getattr(creds, 'client_id', None),
        'client_secret': getattr(creds, 'client_secret', None),
        'scopes': list(getattr(creds, 'scopes', ()) or ()),
    })


def scopes_missing(granted: Iterable[str], required: Iterable[str]) -> set[str]:
    return set(required) - set(granted)
