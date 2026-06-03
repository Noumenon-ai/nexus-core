"""Phase 5: per-user Chromium profile resolver.

Maps a user_id (UUID) to the Chromium profile path the browser agent should
launch against.

NOTE on browser choice (2026-05-12): the original spec called for Firefox, but
Ubuntu 24's AppArmor restriction on unprivileged user namespaces blocks
Playwright's bundled Firefox from creating its sandbox process. Chromium uses a
different sandbox model that the kernel permits.  This has the side benefit
that `playwright-stealth` has dramatically better Chromium support.

Constraints baked in:

- USER_1 (the primary human) uses a DEDICATED Chromium profile at
  ~/.config/chromium-nexus/.  This is separate from any existing
  google-chrome / chromium install — we never touch those.  User_1 must
  sign in to Gmail / WhatsApp Web / etc. ONCE in this profile; cookies persist
  for subsequent agent calls.

- USER_2 (secondary human, Sam) uses an ISOLATED profile under .data/.  Path
  must resolve inside the project workspace — refuses any traversal that would
  escape `NEXUS_USER_2_PROFILE_ROOT` (default `.data/`).

Environment overrides:
  NEXUS_USER_1_UUID                  user_1's UUID (DB id)
  NEXUS_USER_2_UUID                  user_2's UUID
  NEXUS_USER_1_BROWSER_PROFILE       absolute path to user_1's Chromium profile
  NEXUS_USER_2_BROWSER_PROFILE       absolute path to user_2's isolated profile
  NEXUS_USER_2_PROFILE_ROOT          directory that user_2 profile MUST live under

Legacy env names (NEXUS_USER_*_FIREFOX_PROFILE) still honored for backward
compatibility from the pre-switch tests; new code should use BROWSER_PROFILE.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Live Nexus user IDs as of 2026-05-24. Env vars still override these when
# an operator needs a different mapping.
_DEFAULT_USER_1_UUID = "00000000-0000-0000-0000-000000000001"
_DEFAULT_USER_2_UUID = "00000000-0000-0000-0000-000000000002"

_DEFAULT_USER_1_PROFILE = "/home/user/.config/chromium-nexus"
_DEFAULT_USER_2_PROFILE = str(PROJECT_ROOT / ".data" / "browser_profile_user2")
_DEFAULT_USER_2_ROOT = str(PROJECT_ROOT / ".data")


def _legacy_or(name_new: str, name_legacy: str, default: str) -> str:
    val = os.environ.get(name_new)
    if val and val.strip():
        return val.strip()
    val = os.environ.get(name_legacy)
    if val and val.strip():
        return val.strip()
    return default


class BrowserProfileError(Exception):
    """Raised when no profile can be safely resolved for a given user_id."""


@dataclass(frozen=True)
class ProfileSelection:
    user_id: str
    user_label: str           # "user_1" / "user_2"
    profile_path: Path
    is_real_profile: bool     # True = the human's real desktop profile
    headless_ok: bool         # True if the agent may run headless for this user


@dataclass(frozen=True)
class _ConfiguredUser:
    user_label: str
    display_name: str
    user_id: str


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val.strip() if val and val.strip() else default


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _configured_users() -> tuple[_ConfiguredUser, _ConfiguredUser]:
    return (
        _ConfiguredUser(
            user_label="user_1",
            display_name="owner",
            user_id=_env("NEXUS_USER_1_UUID", _DEFAULT_USER_1_UUID),
        ),
        _ConfiguredUser(
            user_label="user_2",
            display_name="Sam",
            user_id=_env("NEXUS_USER_2_UUID", _DEFAULT_USER_2_UUID),
        ),
    )


def _configured_users_text(users: tuple[_ConfiguredUser, _ConfiguredUser]) -> str:
    return ", ".join(
        f"{user.user_label}/{user.display_name}={user.user_id}"
        for user in users
    )


def resolve_profile(user_id: str) -> ProfileSelection:
    """Pick the Firefox profile path for this user_id, with safety guards.

    Raises BrowserProfileError on:
      - empty user_id
      - unknown user_id (not user_1 or user_2)
      - user_2 path that escapes NEXUS_USER_2_PROFILE_ROOT
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise BrowserProfileError("user_id is required")
    uid = user_id.strip()

    user_1, user_2 = _configured_users()

    if uid == user_1.user_id:
        raw = _legacy_or("NEXUS_USER_1_BROWSER_PROFILE",
                         "NEXUS_USER_1_FIREFOX_PROFILE", _DEFAULT_USER_1_PROFILE)
        path = Path(raw)
        if not path.is_absolute():
            raise BrowserProfileError(
                f"user_1 profile path must be absolute, got {raw!r}"
            )
        return ProfileSelection(
            user_id=uid,
            user_label="user_1",
            profile_path=path,
            is_real_profile=True,
            headless_ok=False,    # never headless for user_1 — anti-detection
        )

    if uid == user_2.user_id:
        raw = _legacy_or("NEXUS_USER_2_BROWSER_PROFILE",
                         "NEXUS_USER_2_FIREFOX_PROFILE", _DEFAULT_USER_2_PROFILE)
        path = Path(raw)
        if not path.is_absolute():
            raise BrowserProfileError(
                f"user_2 profile path must be absolute, got {raw!r}"
            )
        root_raw = _env("NEXUS_USER_2_PROFILE_ROOT", _DEFAULT_USER_2_ROOT)
        root = Path(root_raw)
        if not root.is_absolute():
            raise BrowserProfileError(
                f"user_2 profile root must be absolute, got {root_raw!r}"
            )
        if not root.exists():
            raise BrowserProfileError(
                f"user_2 profile root does not exist: {root}. Create with scripts/setup_user_browser_profile.py."
            )
        # Path-traversal guard: the resolved path must live under the root.
        if not _is_under(path, root):
            raise BrowserProfileError(
                f"user_2 profile path {path} is outside the allowed root {root}"
            )
        return ProfileSelection(
            user_id=uid,
            user_label="user_2",
            profile_path=path,
            is_real_profile=False,
            headless_ok=True,
        )

    raise BrowserProfileError(
        f"unknown user_id {uid!r}. Configured users: {_configured_users_text((user_1, user_2))}"
    )


def profile_exists(selection: ProfileSelection) -> bool:
    """True if the profile directory exists and is non-empty."""
    p = selection.profile_path
    if not p.exists() or not p.is_dir():
        return False
    return any(p.iterdir())


def profile_is_locked(selection: ProfileSelection) -> bool:
    """True if a browser process is currently holding the profile.

    Chromium leaves `Default/LOCK` behind on every exit (a 0-byte leveldb lock
    file) — its mere presence is NOT proof of an active session.  Use
    SingletonLock (a symlink containing the owning hostname:pid) as the strong
    signal: if its target PID is still alive we treat the profile as locked.
    Firefox-style .parentlock / lock still treated as authoritative for the
    backward-compat path.
    """
    p = selection.profile_path
    if not p.exists():
        return False
    # Firefox legacy markers
    if (p / ".parentlock").exists() or (p / "lock").exists():
        return True
    # Chromium SingletonLock — symlink target is "hostname-<pid>"
    singleton = p / "SingletonLock"
    if singleton.is_symlink():
        try:
            target = os.readlink(singleton)
        except OSError:
            return True   # can't read — assume locked
        # target like "user-12345" — extract the PID
        if "-" in target:
            try:
                pid = int(target.rsplit("-", 1)[-1])
                # PID 0 isn't valid for a real process
                if pid > 0:
                    try:
                        os.kill(pid, 0)   # signal 0 just probes
                        return True
                    except ProcessLookupError:
                        return False
                    except PermissionError:
                        return True
            except ValueError:
                return True
        return True
    return False
