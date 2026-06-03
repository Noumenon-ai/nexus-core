"""Phase 5 — adversarial tests for browser_profile_resolver."""
from __future__ import annotations

from pathlib import Path

import pytest

from services.browser_profile_resolver import (
    _DEFAULT_USER_1_UUID,
    _DEFAULT_USER_2_UUID,
    BrowserProfileError,
    profile_exists,
    profile_is_locked,
    resolve_profile,
)


_DEFAULT_USER_1 = _DEFAULT_USER_1_UUID
_DEFAULT_USER_2 = _DEFAULT_USER_2_UUID


# ---------- input validation ----------

def test_empty_user_id_rejected():
    with pytest.raises(BrowserProfileError, match="user_id is required"):
        resolve_profile("")


def test_whitespace_user_id_rejected():
    with pytest.raises(BrowserProfileError, match="user_id is required"):
        resolve_profile("   ")


def test_unknown_user_id_rejected():
    with pytest.raises(BrowserProfileError, match="unknown user_id"):
        resolve_profile("not-a-valid-uuid")


# ---------- user_1 (real profile) ----------

def test_user_1_default_profile_returned(monkeypatch):
    monkeypatch.delenv("NEXUS_USER_1_BROWSER_PROFILE", raising=False)
    monkeypatch.delenv("NEXUS_USER_1_FIREFOX_PROFILE", raising=False)
    sel = resolve_profile(_DEFAULT_USER_1)
    assert sel.user_label == "user_1"
    assert sel.is_real_profile is True
    assert sel.headless_ok is False
    # Default is the chromium-nexus path (Chromium switch 2026-05-12)
    assert "chromium-nexus" in str(sel.profile_path).lower()


def test_user_1_env_override_browser_profile(monkeypatch):
    monkeypatch.setenv("NEXUS_USER_1_BROWSER_PROFILE", "/tmp/my-profile")
    sel = resolve_profile(_DEFAULT_USER_1)
    assert str(sel.profile_path) == "/tmp/my-profile"


def test_user_1_legacy_firefox_env_still_honored(monkeypatch):
    """Pre-Chromium-switch env name kept working for backward compat."""
    monkeypatch.delenv("NEXUS_USER_1_BROWSER_PROFILE", raising=False)
    monkeypatch.setenv("NEXUS_USER_1_FIREFOX_PROFILE", "/tmp/legacy-profile")
    sel = resolve_profile(_DEFAULT_USER_1)
    assert str(sel.profile_path) == "/tmp/legacy-profile"


def test_user_1_relative_path_rejected(monkeypatch):
    monkeypatch.setenv("NEXUS_USER_1_BROWSER_PROFILE", "relative/path")
    with pytest.raises(BrowserProfileError, match="must be absolute"):
        resolve_profile(_DEFAULT_USER_1)


# ---------- user_2 (isolated profile) ----------

def test_user_2_default_under_data_dir(monkeypatch, tmp_path):
    root = tmp_path / "data_root"
    root.mkdir()
    monkeypatch.setenv("NEXUS_USER_2_PROFILE_ROOT", str(root))
    monkeypatch.setenv("NEXUS_USER_2_BROWSER_PROFILE", str(root / "sam_profile"))
    sel = resolve_profile(_DEFAULT_USER_2)
    assert sel.user_label == "user_2"
    assert sel.is_real_profile is False
    assert sel.headless_ok is True


def test_user_2_path_traversal_blocked(monkeypatch, tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    outside = tmp_path / "escape"
    outside.mkdir()
    monkeypatch.setenv("NEXUS_USER_2_PROFILE_ROOT", str(root))
    monkeypatch.setenv("NEXUS_USER_2_BROWSER_PROFILE", str(outside / "stolen"))
    with pytest.raises(BrowserProfileError, match="outside the allowed root"):
        resolve_profile(_DEFAULT_USER_2)


def test_user_2_missing_root_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("NEXUS_USER_2_PROFILE_ROOT", str(tmp_path / "does_not_exist"))
    monkeypatch.setenv("NEXUS_USER_2_BROWSER_PROFILE", str(tmp_path / "does_not_exist" / "p"))
    with pytest.raises(BrowserProfileError, match="profile root does not exist"):
        resolve_profile(_DEFAULT_USER_2)


def test_user_2_relative_profile_rejected(monkeypatch, tmp_path):
    root = tmp_path / "ok"
    root.mkdir()
    monkeypatch.setenv("NEXUS_USER_2_PROFILE_ROOT", str(root))
    monkeypatch.setenv("NEXUS_USER_2_BROWSER_PROFILE", "relative/path")
    with pytest.raises(BrowserProfileError, match="must be absolute"):
        resolve_profile(_DEFAULT_USER_2)


# ---------- env-mapped UUIDs ----------

def test_uuid_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("NEXUS_USER_1_UUID", "custom-1")
    monkeypatch.setenv("NEXUS_USER_1_FIREFOX_PROFILE", "/tmp/custom1")
    sel = resolve_profile("custom-1")
    assert sel.user_label == "user_1"


# ---------- helper functions ----------

def test_profile_exists_empty_dir_returns_false(tmp_path):
    from services.browser_profile_resolver import ProfileSelection
    empty = tmp_path / "empty"
    empty.mkdir()
    sel = ProfileSelection(user_id="x", user_label="user_2", profile_path=empty,
                           is_real_profile=False, headless_ok=True)
    assert profile_exists(sel) is False


def test_profile_exists_with_content_returns_true(tmp_path):
    from services.browser_profile_resolver import ProfileSelection
    p = tmp_path / "p"
    p.mkdir()
    (p / "prefs.js").write_text("// pretend prefs")
    sel = ProfileSelection(user_id="x", user_label="user_2", profile_path=p,
                           is_real_profile=False, headless_ok=True)
    assert profile_exists(sel) is True


def test_profile_is_locked_detects_chromium_singleton_with_live_pid(tmp_path):
    """SingletonLock pointing at a live PID (our own PID, guaranteed alive) = locked."""
    import os
    from services.browser_profile_resolver import ProfileSelection
    p = tmp_path / "p"
    p.mkdir()
    # SingletonLock is a SYMLINK whose target is "hostname-<pid>"
    target = f"host-{os.getpid()}"
    (p / "SingletonLock").symlink_to(target)
    sel = ProfileSelection(user_id="x", user_label="user_2", profile_path=p,
                           is_real_profile=False, headless_ok=True)
    assert profile_is_locked(sel) is True


def test_profile_is_locked_stale_singleton_dead_pid_returns_false(tmp_path):
    """SingletonLock pointing at a dead PID = stale lock, NOT locked."""
    from services.browser_profile_resolver import ProfileSelection
    p = tmp_path / "p"
    p.mkdir()
    # Pick a PID that cannot exist (max+1)
    (p / "SingletonLock").symlink_to("host-999999999")
    sel = ProfileSelection(user_id="x", user_label="user_2", profile_path=p,
                           is_real_profile=False, headless_ok=True)
    assert profile_is_locked(sel) is False


def test_profile_is_locked_default_lock_alone_returns_false(tmp_path):
    """Default/LOCK is left by every Chromium exit. Its mere presence is NOT a lock."""
    from services.browser_profile_resolver import ProfileSelection
    p = tmp_path / "p"
    (p / "Default").mkdir(parents=True)
    (p / "Default" / "LOCK").touch()
    sel = ProfileSelection(user_id="x", user_label="user_2", profile_path=p,
                           is_real_profile=False, headless_ok=True)
    assert profile_is_locked(sel) is False


def test_profile_is_locked_legacy_firefox_lock_still_detected(tmp_path):
    from services.browser_profile_resolver import ProfileSelection
    p = tmp_path / "p"
    p.mkdir()
    (p / ".parentlock").touch()
    sel = ProfileSelection(user_id="x", user_label="user_2", profile_path=p,
                           is_real_profile=False, headless_ok=True)
    assert profile_is_locked(sel) is True


def test_profile_is_locked_no_lock_returns_false(tmp_path):
    from services.browser_profile_resolver import ProfileSelection
    p = tmp_path / "p"
    p.mkdir()
    sel = ProfileSelection(user_id="x", user_label="user_2", profile_path=p,
                           is_real_profile=False, headless_ok=True)
    assert profile_is_locked(sel) is False
