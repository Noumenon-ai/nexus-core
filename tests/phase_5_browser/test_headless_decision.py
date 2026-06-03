"""Unit tests for _decide_headless — the headed/headless resolution rule.

Real Chromium isn't spawned. We exercise the decision function directly to
lock down the failure-mode behavior: User_1 (real profile) must NOT crash
under systemd with no DISPLAY, and the NEXUS_BROWSER_FORCE_HEADED override
must restore the hard-fail behavior when the operator wants xvfb-run.
"""
from __future__ import annotations

from services.browser_agent import _decide_headless


def _env(**kw) -> dict:
    return kw


# ---------- force_headless explicit override wins ----------

def test_force_headless_true_overrides_everything():
    assert _decide_headless(
        force_headless=True, selection_headless_ok=False,
        is_real_profile=True, env=_env(DISPLAY=":0"),
    ) is True


def test_force_headless_false_overrides_everything():
    assert _decide_headless(
        force_headless=False, selection_headless_ok=True,
        is_real_profile=False, env=_env(),
    ) is False


# ---------- real profile + display present → headed ----------

def test_real_profile_with_display_stays_headed():
    assert _decide_headless(
        force_headless=None, selection_headless_ok=False,
        is_real_profile=True, env=_env(DISPLAY=":0"),
    ) is False


def test_real_profile_with_wayland_display_stays_headed():
    assert _decide_headless(
        force_headless=None, selection_headless_ok=False,
        is_real_profile=True, env=_env(WAYLAND_DISPLAY="wayland-0"),
    ) is False


# ---------- real profile + no display → fall back to headless ----------

def test_real_profile_no_display_falls_back_to_headless():
    """The H2-039 follow-up bug: nexus.service has no DISPLAY env.
    Chromium headed launch would crash. Auto-fallback to headless."""
    assert _decide_headless(
        force_headless=None, selection_headless_ok=False,
        is_real_profile=True, env=_env(),
    ) is True


def test_real_profile_no_display_with_force_headed_stays_headed():
    """Operator override: NEXUS_BROWSER_FORCE_HEADED=1 disables the
    auto-fallback so launches crash loudly, signaling that they need
    to set up xvfb-run / a DISPLAY first."""
    assert _decide_headless(
        force_headless=None, selection_headless_ok=False,
        is_real_profile=True,
        env=_env(NEXUS_BROWSER_FORCE_HEADED="1"),
    ) is False


# ---------- non-real profile uses selection.headless_ok ----------

def test_non_real_profile_uses_selection_default_true():
    assert _decide_headless(
        force_headless=None, selection_headless_ok=True,
        is_real_profile=False, env=_env(),
    ) is True


def test_non_real_profile_uses_selection_default_false():
    assert _decide_headless(
        force_headless=None, selection_headless_ok=False,
        is_real_profile=False, env=_env(),
    ) is False


def test_non_real_profile_ignores_display():
    # Non-real profiles don't get the display-based override.
    assert _decide_headless(
        force_headless=None, selection_headless_ok=True,
        is_real_profile=False, env=_env(DISPLAY=":0"),
    ) is True
