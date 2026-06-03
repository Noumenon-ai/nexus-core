"""H2-043 FIX B — register an additional Gmail account under a chosen label.

Pre-H2-043 each user (Telegram user) had exactly one Gmail token at
``.data/gmail_tokens/<user_id>.json``. This script adds a second (or third,
…) account by running the same OAuth flow against the same OAuth client and
saving the resulting token under a labeled directory:

    .data/gmail_tokens/<label>/<user_id>.json

The default user_id is read from ``NEXUS_MCP_DEFAULT_USER_ID`` (the same env
var the MCP servers use). Pass --user-id to override for a different
Telegram user.

Usage:
    .venv/bin/python scripts/add_gmail_account.py <label>
    .venv/bin/python scripts/add_gmail_account.py <label> --user-id <uuid>

The label is used as a directory name on disk, so it must match
``[a-z][a-z0-9_-]{0,31}``. Examples: ``shop``, ``work_2025``, ``primary``.

After this runs, every nexus-email tool accepts ``account=<label>`` to route
to the new token, and ``list_gmail_accounts`` will surface it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import secure_token_file
from config import get_settings
from services.gmail_accounts import account_token_path, is_valid_label


SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.compose',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.labels',
]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Register an additional Gmail account under a labeled token path.',
    )
    parser.add_argument('label', help='Account label (e.g. "shop", "work_2025"). '
                                      'Must match [a-z][a-z0-9_-]{0,31}.')
    parser.add_argument('--user-id', default=None,
                        help='Override NEXUS_MCP_DEFAULT_USER_ID for this run.')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    label = args.label.strip()
    if not is_valid_label(label):
        print(f'FAIL: label {label!r} must match [a-z][a-z0-9_-]{{0,31}} '
              '(used as a directory name on disk)', file=sys.stderr)
        return 2

    user_id = (args.user_id or os.environ.get('NEXUS_MCP_DEFAULT_USER_ID') or '').strip()
    if not user_id:
        print('FAIL: pass --user-id or set NEXUS_MCP_DEFAULT_USER_ID',
              file=sys.stderr)
        return 2

    try:
        from google.auth.transport.requests import Request  # noqa: F401  (used by flow)
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        print(f'Missing Google libraries: {exc}', file=sys.stderr)
        return 3

    settings = get_settings()
    creds_path = settings.gmail.gmail_credentials_path
    token_path = account_token_path(user_id, label)

    print(f'OAuth client     : {creds_path}')
    print(f'User id          : {user_id}')
    print(f'Account label    : {label}')
    print(f'Token destination: {token_path}')
    print(f'Scopes           : {SCOPES}')
    print()

    if not creds_path.exists():
        print(f'FAIL: OAuth client file does not exist at {creds_path}',
              file=sys.stderr)
        return 4

    token_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(token_path.parent, 0o700)
    except OSError:
        pass

    # If a token already exists for this label, refresh + re-verify instead of
    # forcing the user back through the consent screen.
    existing = None
    if token_path.exists():
        try:
            existing = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception as exc:
            print(f'Existing token at {token_path} did not load; will re-auth: {exc}')
            existing = None
        if existing and existing.expired and existing.refresh_token:
            try:
                existing.refresh(Request())
                print('Existing token refreshed successfully — scope check next.')
            except Exception as exc:
                print(f'Refresh failed ({exc}); running full OAuth flow.')
                existing = None
        if existing and existing.valid:
            granted = set(existing.scopes or [])
            missing = [s for s in SCOPES if s not in granted]
            if not missing:
                print('Existing token already has every requested scope — skipping OAuth flow.')
                token_path.write_text(existing.to_json(), encoding='utf-8')
                secure_token_file(token_path)
            else:
                print(f'Existing token missing scopes: {missing}. Re-running OAuth.')
                existing = None

    if existing is None or not existing.valid:
        print('Starting OAuth flow — a browser tab will open shortly.')
        print('Sign in with the Google account you want to register under this label.')
        print('If no browser opens, copy the URL printed below into a browser yourself.\n')
        flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
        creds = flow.run_local_server(port=0, open_browser=True)
        token_path.write_text(creds.to_json(), encoding='utf-8')
        secure_token_file(token_path)
        existing = creds
        print(f'\nToken saved to {token_path}')

    print('\nProbe: listing profile + 1 message metadata to confirm the token works ...')
    service = build('gmail', 'v1', credentials=existing, cache_discovery=False)
    profile = service.users().getProfile(userId='me').execute()
    print(f'  authenticated as : {profile.get("emailAddress")}')
    print(f'  total messages   : {profile.get("messagesTotal")}')
    msgs = service.users().messages().list(userId='me', maxResults=1).execute().get('messages', [])
    if msgs:
        first = service.users().messages().get(
            userId='me', id=msgs[0]['id'], format='metadata',
            metadataHeaders=['Subject', 'From', 'Date'],
        ).execute()
        headers = {h['name']: h['value'] for h in first.get('payload', {}).get('headers', [])}
        print(f'  most recent      : From: {headers.get("From", "?")} | '
              f'Subject: {headers.get("Subject", "?")[:80]}')
    print(f'\nDone. To route a tool through this account: account="{label}"')
    return 0


if __name__ == '__main__':
    sys.exit(main())
