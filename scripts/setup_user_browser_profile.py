"""Phase 5 — one-shot setup for user_2's isolated Chromium profile.

Creates the directory at NEXUS_USER_2_BROWSER_PROFILE (default
.data/browser_profile_user2/) so the first browser_agent invocation doesn't
have to bootstrap from nothing.

Usage:
    .venv/bin/python scripts/setup_user_browser_profile.py [<user_id>]

Defaults to user_2 if no argument is given. Pass user_1's UUID to set up that
profile (creates ~/.config/chromium-nexus/).  Idempotent — leaves an existing
profile alone.

After running, the profile is ready for Playwright Chromium to attach via
launch_persistent_context().  The targeted user must SIGN INTO any services
they want the agent to access (Gmail, WhatsApp Web, etc.) ONCE in this profile;
cookies persist for subsequent agent calls.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.browser_profile_resolver import (
    BrowserProfileError,
    _DEFAULT_USER_2_UUID,
    profile_exists,
    resolve_profile,
)


def main() -> int:
    user_uuid = (sys.argv[1].strip() if len(sys.argv) >= 2
                 else os.environ.get("NEXUS_USER_2_UUID", _DEFAULT_USER_2_UUID).strip())

    try:
        selection = resolve_profile(user_uuid)
    except BrowserProfileError as exc:
        print(f"FAIL: profile resolution: {exc}", file=sys.stderr)
        return 2

    target = selection.profile_path
    print(f"Browser profile target : {target}")
    print(f"User label             : {selection.user_label}")
    print(f"Headless allowed       : {selection.headless_ok}")
    if selection.user_label == "user_2":
        print(f"Root constraint        : {os.environ.get('NEXUS_USER_2_PROFILE_ROOT', '.data/')}")
    print()

    if profile_exists(selection):
        print(f"Profile already exists ({target}). No changes made.")
        return 0

    target.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(target, 0o700)
    except PermissionError as exc:
        print(f"WARN: could not chmod {target}: {exc}")

    # Chromium creates its own profile structure on first launch — we only need
    # the directory to exist.  Marker file lets profile_exists() return True
    # before the first browser run.
    marker = target / "NEXUS_PROFILE.txt"
    marker.write_text(
        f"Created by setup_user_browser_profile.py for {selection.user_label}.\n"
        "Do NOT delete. Browser cookies + state live alongside this file.\n",
        encoding="utf-8",
    )
    print(f"Created profile dir at {target}")
    print()
    print(f"Next step: have {selection.user_label} sign into any sites the agent will need.")
    print("           Run a browser_agent action against this user_id once,")
    print("           Chromium will open, you sign in, then it persists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
