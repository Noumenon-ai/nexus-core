"""Phase 2.5b — Google integration OAuth refresh (Calendar + Contacts + Tasks).

The existing token at .data/google_tokens/<user_id>.json has the right scopes
but `invalid_grant` on refresh (Google revoked it). This script re-runs the
InstalledAppFlow with the same scope set so nexus-calendar + nexus-contacts
can authenticate again.

Usage:
    .venv/bin/python scripts/google_oauth_flow.py <user_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from bootstrap import secure_token_file


SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/contacts",
    "https://www.googleapis.com/auth/tasks",
]


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        print("Usage: python scripts/google_oauth_flow.py <user_id>", file=sys.stderr)
        return 2
    user_id = sys.argv[1].strip()

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        print(f"Missing Google libraries: {exc}", file=sys.stderr)
        return 3

    settings = get_settings()
    creds_path = settings.google.credentials_path
    token_dir = settings.google.token_dir
    token_path = token_dir / f"{user_id}.json"

    print(f"OAuth client      : {creds_path}")
    print(f"Token destination : {token_path}")
    print(f"Scopes            : {SCOPES}")
    print()

    if not creds_path.exists():
        print(f"FAIL: OAuth client file does not exist at {creds_path}", file=sys.stderr)
        return 4

    token_dir.mkdir(parents=True, exist_ok=True)

    print("Starting OAuth flow — a browser tab will open shortly.")
    print("If no browser opens, copy the URL printed below into a browser yourself.\n")
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    secure_token_file(token_path)
    print(f"\nToken saved to {token_path}")

    # Probe: list 1 contact group + 1 calendar event
    print("\nProbe (Contacts): list_groups ...")
    people = build("people", "v1", credentials=creds, cache_discovery=False)
    groups = people.contactGroups().list(pageSize=5).execute().get("contactGroups", [])
    print(f"  contact groups visible: {len(groups)}")
    for g in groups[:3]:
        print(f"    - {g.get('formattedName') or g.get('name')}")

    print("\nProbe (Calendar): today's events ...")
    from datetime import datetime, timezone, timedelta
    cal = build("calendar", "v3", credentials=creds, cache_discovery=False)
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=1)
    events = cal.events().list(
        calendarId="primary",
        timeMin=now.isoformat().replace("+00:00", "Z"),
        timeMax=end.isoformat().replace("+00:00", "Z"),
        maxResults=5, singleEvents=True, orderBy="startTime",
    ).execute().get("items", [])
    print(f"  events in next 24h: {len(events)}")
    for e in events[:3]:
        print(f"    - {e.get('summary')}")

    print("\nGoogle OAuth flow: COMPLETE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
