from __future__ import annotations

import json
from typing import Iterable

from services.google_auth_service import GoogleAuthService
from tests.helpers import build_test_container


_DESKTOP_OAUTH_PAYLOAD = {
    'installed': {
        'client_id': 'test-client.apps.googleusercontent.com',
        'project_id': 'nexus-test',
        'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
        'token_uri': 'https://oauth2.googleapis.com/token',
        'client_secret': 'test-secret',
        'redirect_uris': ['http://localhost'],
    }
}


class FakeCredentials:
    def __init__(self, scopes: Iterable[str], *, token: str = 'fake-access', refresh_token: str = 'fake-refresh', expired: bool = False) -> None:
        self.scopes = list(scopes)
        self.token = token
        self.refresh_token = refresh_token
        self.expired = expired
        self.client_id = 'test-client.apps.googleusercontent.com'
        self.client_secret = 'test-secret'
        self.token_uri = 'https://oauth2.googleapis.com/token'

    def to_json(self) -> str:
        return json.dumps({
            'token': self.token,
            'refresh_token': self.refresh_token,
            'token_uri': self.token_uri,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scopes': list(self.scopes),
        })

    def refresh(self, _request) -> None:
        self.expired = False
        self.token = self.token + '-refreshed'


class FakeFlow:
    def __init__(self, granted_scopes: Iterable[str], *, fixed_state: str | None = None) -> None:
        self._granted_scopes = list(granted_scopes)
        self.redirect_uri: str | None = None
        self.fetch_token_calls = 0
        self.last_fetch_code: str | None = None
        self.fixed_state = fixed_state

    def authorization_url(self, **kwargs):
        state = kwargs.get('state', 'unknown')
        url = (
            'https://accounts.google.com/o/oauth2/auth?fake=1'
            f'&state={state}&redirect_uri={self.redirect_uri or ""}'
        )
        return url, state

    def fetch_token(self, *, code: str) -> None:
        self.fetch_token_calls += 1
        self.last_fetch_code = code

    @property
    def credentials(self) -> FakeCredentials:
        return FakeCredentials(self._granted_scopes)


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, *, get_payload: dict | None = None, get_status: int = 200, post_status: int = 200) -> None:
        self._get_payload = get_payload
        self._get_status = get_status
        self._post_status = post_status
        self.get_calls: list[tuple[str, dict | None]] = []
        self.post_calls: list[tuple[str, dict | None]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url, headers=None):
        self.get_calls.append((url, headers))
        return _FakeResponse(self._get_status, self._get_payload)

    async def post(self, url, content=None, headers=None):
        self.post_calls.append((url, headers))
        return _FakeResponse(self._post_status, {})


class _ScopeAwareUserinfoClient:
    """Fakes Google's actual userinfo behavior: 401 unless the bearer token
    was issued with the openid + email scopes."""

    OPENID = 'openid'
    EMAIL = 'https://www.googleapis.com/auth/userinfo.email'

    def __init__(self, granted_scopes: Iterable[str], email: str = 'real@example.com') -> None:
        self._granted = set(granted_scopes)
        self._email = email
        self.get_calls: list[tuple[str, dict | None]] = []
        self.post_calls: list[tuple[str, dict | None]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url, headers=None):
        self.get_calls.append((url, headers))
        if self.OPENID in self._granted and self.EMAIL in self._granted:
            return _FakeResponse(200, {'email': self._email})
        return _FakeResponse(401, {'error': 'unauthorized'})

    async def post(self, url, content=None, headers=None):
        self.post_calls.append((url, headers))
        return _FakeResponse(200, {})


def configure_google_test_env(monkeypatch, tmp_path, *, extra: dict[str, str] | None = None, write_credentials: bool = True):
    creds_path = tmp_path / 'credentials.json'
    if write_credentials:
        creds_path.write_text(json.dumps(_DESKTOP_OAUTH_PAYLOAD), encoding='utf-8')
    google_extra = {
        'GOOGLE_INTEGRATION_ENABLED': 'true',
        'GOOGLE_CREDENTIALS_PATH': str(creds_path),
        'GOOGLE_TOKEN_DIR': str(tmp_path / 'google_tokens'),
    }
    if extra:
        google_extra.update(extra)
    return build_test_container(monkeypatch, tmp_path, extra_env=google_extra, validate_startup=False)


def build_google_auth_service(
    container,
    *,
    granted_scopes: Iterable[str],
    userinfo_email: str = 'test@example.com',
    pending_ttl_seconds: int = 600,
):
    fake_client = _FakeAsyncClient(get_payload={'email': userinfo_email})
    flow_holder: dict[str, FakeFlow] = {}

    def flow_factory(scopes):
        flow = FakeFlow(granted_scopes)
        flow_holder['last'] = flow
        return flow

    def fake_local_server_starter(pending):
        return 0, None, None

    service = GoogleAuthService(
        container.settings,
        container.users_repository,
        http_client_factory=lambda: fake_client,
        flow_factory=flow_factory,
        local_server_starter=fake_local_server_starter,
        pending_ttl_seconds=pending_ttl_seconds,
    )
    return service, fake_client, flow_holder


def pasted_url_for(state: str, code: str = '4/0AeaYS_test_code', port: int = 8080) -> str:
    return f'http://localhost:{port}/?code={code}&scope=foo+bar&state={state}'
