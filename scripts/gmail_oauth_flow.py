"""Phase 2.5b — one-shot Gmail OAuth flow.

Runs the Google InstalledAppFlow against the OAuth client at
GMAIL_CREDENTIALS_PATH, persists the resulting token to
.data/gmail_tokens/<user_id>.json, then makes a single read-only Gmail API
call to verify the token actually works.

Usage:
    .venv/bin/python scripts/gmail_oauth_flow.py <user_id>

A browser tab will open for Google sign-in. After consent the page redirects
to localhost on a random port; the local server captures the code and exits.

Scopes requested (cover the full Phase 2.5b nexus-email spec):
  - gmail.modify    read, label, archive, trash
  - gmail.compose   draft creation
  - gmail.send      send/forward/reply
  - gmail.labels    list + create labels (subset of modify, kept explicit)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from bootstrap import secure_token_file


SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.labels",
]


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        print("Usage: python scripts/gmail_oauth_flow.py <user_id>", file=sys.stderr)
        return 2
    user_id = sys.argv[1].strip()

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        print(f"Missing Google libraries: {exc}", file=sys.stderr)
        return 3

    settings = get_settings()
    creds_path = settings.gmail.gmail_credentials_path
    token_dir = settings.gmail.gmail_token_dir
    token_path = token_dir / f"{user_id}.json"

    print(f"OAuth client    : {creds_path}")
    print(f"Token destination: {token_path}")
    print(f"Scopes          : {SCOPES}")
    print()

    if not creds_path.exists():
        print(f"FAIL: OAuth client file does not exist at {creds_path}", file=sys.stderr)
        return 4

    token_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(token_dir, 0o700)

    # Skip flow if a valid token with all requested scopes already exists.
    existing = None
    if token_path.exists():
        try:
            existing = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception as exc:
            print(f"Existing token at {token_path} did not load; will re-auth: {exc}")
            existing = None
        if existing and existing.expired and existing.refresh_token:
            try:
                existing.refresh(Request())
                print("Existing token refreshed successfully — scope check next.")
            except Exception as exc:
                print(f"Refresh failed ({exc}); running full OAuth flow.")
                existing = None
        if existing and existing.valid:
            granted = set(existing.scopes or [])
            missing = [s for s in SCOPES if s not in granted]
            if not missing:
                print("Existing token already has every requested scope — skipping OAuth flow.")
                token_path.write_text(existing.to_json(), encoding="utf-8")
                secure_token_file(token_path)
            else:
                print(f"Existing token missing scopes: {missing}. Re-running OAuth.")
                existing = None

    if existing is None or not existing.valid:
        print("Starting OAuth flow — a browser tab will open shortly.")
        print("If no browser opens, copy the URL printed below into a browser yourself.\n")
        flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
        # port=0 → OS picks a free port; matches existing NEXUS pattern.
        creds = flow.run_local_server(port=0, open_browser=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        secure_token_file(token_path)
        existing = creds
        print(f"\nToken saved to {token_path}")

    # ----------------------------------------------------------------- verify
    print("\nProbe: listing 1 message metadata to confirm the token works ...")
    service = build("gmail", "v1", credentials=existing, cache_discovery=False)
    profile = service.users().getProfile(userId="me").execute()
    print(f"  authenticated as: {profile.get('emailAddress')}")
    print(f"  total messages   : {profile.get('messagesTotal')}")
    msgs = service.users().messages().list(userId="me", maxResults=1).execute().get("messages", [])
    if msgs:
        first = service.users().messages().get(
            userId="me", id=msgs[0]["id"], format="metadata",
            metadataHeaders=["Subject", "From", "Date"],
        ).execute()
        headers = {h["name"]: h["value"] for h in first.get("payload", {}).get("headers", [])}
        print(f"  most recent      : From: {headers.get('From', '?')} | Subject: {headers.get('Subject', '?')[:80]}")
    print("\nPhase 2.5b Gmail OAuth flow: COMPLETE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
