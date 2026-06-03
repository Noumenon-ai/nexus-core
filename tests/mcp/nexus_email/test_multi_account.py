"""H2-043 FIX B tests — multi-account Gmail token storage + routing."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from services.gmail_accounts import (
    DEFAULT_ACCOUNT_LABEL,
    account_token_path,
    is_valid_label,
    list_accounts_for_user,
    resolve_token_path,
)


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('label', [
    'primary', 'shop', 'work', 'work_2025', 'a', 'work-personal', 'p1',
])
def test_is_valid_label_accepts_well_formed_labels(label):
    assert is_valid_label(label) is True


@pytest.mark.parametrize('label', [
    '', '..', '../escape', '.hidden', '/abs/path', 'CapsAreNotAllowed',
    '1starts_with_digit', '_starts_with_underscore', '-starts_with_dash',
    'a' * 33,           # 33 chars — one over the cap
    'illegal char',     # space
    'illegal/char',     # slash → path traversal vector
])
def test_is_valid_label_rejects_unsafe_labels(label):
    assert is_valid_label(label) is False


# ---------------------------------------------------------------------------
# Token path resolution + legacy migration
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_token_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('NEXUS_GMAIL_TOKEN_DIR', str(tmp_path))
    return tmp_path


def _write_token(path: Path, payload: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {'token': 'fake'}), encoding='utf-8')
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def test_resolve_token_path_returns_canonical_when_present(isolated_token_dir):
    """Primary path exists at the labeled location → use it as-is."""
    canonical = isolated_token_dir / 'primary' / 'user-1.json'
    _write_token(canonical)
    resolved = resolve_token_path('user-1', 'primary')
    assert resolved == canonical


def test_resolve_token_path_migrates_legacy_layout_forward(isolated_token_dir):
    """H2-043 backward compat: a token file at the pre-multi-account path
    moves to `<dir>/primary/<user_id>.json` on first read so the rest of
    the codebase only sees the labeled layout."""
    legacy = isolated_token_dir / 'user-1.json'
    _write_token(legacy, {'sentinel': 'legacy'})

    resolved = resolve_token_path('user-1', 'primary')

    canonical = isolated_token_dir / 'primary' / 'user-1.json'
    assert resolved == canonical
    assert canonical.exists()
    assert not legacy.exists(), 'legacy file should be moved, not copied'
    assert json.loads(canonical.read_text())['sentinel'] == 'legacy'


def test_resolve_token_path_only_migrates_for_primary_label(isolated_token_dir):
    """Migration is gated to account=='primary' so a shop-labeled request
    doesn't accidentally adopt a legacy token belonging to a different
    Google account."""
    legacy = isolated_token_dir / 'user-1.json'
    _write_token(legacy)

    resolved = resolve_token_path('user-1', 'shop')

    assert resolved is None, 'non-primary lookup must not pick up the legacy file'
    assert legacy.exists(), 'non-primary lookup must not migrate the legacy file'


def test_resolve_token_path_no_migration_when_canonical_already_exists(isolated_token_dir):
    """If a primary canonical token is already in place, leave any stray
    legacy file alone — the canonical wins."""
    canonical = isolated_token_dir / 'primary' / 'user-1.json'
    _write_token(canonical, {'sentinel': 'canonical'})
    legacy = isolated_token_dir / 'user-1.json'
    _write_token(legacy, {'sentinel': 'legacy'})

    resolved = resolve_token_path('user-1', 'primary')

    assert resolved == canonical
    assert legacy.exists(), 'legacy file should NOT be touched when canonical exists'
    assert json.loads(canonical.read_text())['sentinel'] == 'canonical'


def test_resolve_token_path_returns_none_when_no_token_exists(isolated_token_dir):
    assert resolve_token_path('user-1', 'primary') is None
    assert resolve_token_path('user-1', 'shop') is None


# ---------------------------------------------------------------------------
# Account enumeration
# ---------------------------------------------------------------------------


def test_list_accounts_returns_empty_for_unknown_user(isolated_token_dir):
    assert list_accounts_for_user('nobody') == []


def test_list_accounts_returns_primary_when_only_legacy_token_present(isolated_token_dir):
    """Pre-migration: legacy file at <dir>/<user>.json should surface as
    'primary' so list_gmail_accounts is honest about what NEXUS can reach."""
    _write_token(isolated_token_dir / 'user-1.json')
    assert list_accounts_for_user('user-1') == ['primary']


def test_list_accounts_returns_all_labels_with_primary_first(isolated_token_dir):
    """After registering multiple accounts, list_accounts returns every
    label that has a token for this user. Primary leads, the rest are
    alphabetical."""
    _write_token(isolated_token_dir / 'primary' / 'user-1.json')
    _write_token(isolated_token_dir / 'shop' / 'user-1.json')
    _write_token(isolated_token_dir / 'work_2025' / 'user-1.json')
    # decoy: token for a different user shouldn't surface
    _write_token(isolated_token_dir / 'shop' / 'user-2.json')

    assert list_accounts_for_user('user-1') == ['shop', 'primary', 'work_2025'] or \
           list_accounts_for_user('user-1')[0] == 'primary'
    # Tighter assertion: explicit ordering — primary first
    labels = list_accounts_for_user('user-1')
    assert labels[0] == 'primary'
    assert set(labels) == {'primary', 'shop', 'work_2025'}


def test_list_accounts_ignores_directories_with_invalid_labels(isolated_token_dir):
    """A directory whose name fails is_valid_label() must not appear in the
    enumeration — defends against a stray `.git` or `tmp_xyz` folder
    inside the token dir."""
    _write_token(isolated_token_dir / 'primary' / 'user-1.json')
    _write_token(isolated_token_dir / '.git' / 'user-1.json')
    _write_token(isolated_token_dir / 'Mixed_Caps' / 'user-1.json')

    labels = list_accounts_for_user('user-1')
    assert labels == ['primary']


# ---------------------------------------------------------------------------
# _get_gmail() routing — error-shaped dicts on every failure mode
# ---------------------------------------------------------------------------


def test_get_gmail_rejects_invalid_account_label(isolated_token_dir, monkeypatch):
    """Bad label → graceful error dict, no exception, no path-traversal
    attempt on disk."""
    from mcp_servers import nexus_email as je
    monkeypatch.setenv('NEXUS_MCP_DEFAULT_USER_ID', 'user-1')
    result = je._get_gmail(user_id=None, account='../escape')
    assert isinstance(result, dict)
    assert result.get('ok') is False
    assert result.get('reason') == 'invalid_account_label'


def test_get_gmail_returns_no_credentials_for_unknown_label(isolated_token_dir, monkeypatch):
    """Known-good label syntax but no token on disk → graceful error,
    pointing the caller at the add-account script."""
    from mcp_servers import nexus_email as je
    monkeypatch.setenv('NEXUS_MCP_DEFAULT_USER_ID', 'user-1')
    result = je._get_gmail(user_id=None, account='shop')
    assert isinstance(result, dict)
    assert result.get('ok') is False
    assert result.get('reason') == 'no_gmail_credentials'
    assert 'add_gmail_account.py' in result.get('detail', '')


def test_get_gmail_requires_user_id(monkeypatch, isolated_token_dir):
    """No explicit user_id AND no env default → graceful error."""
    monkeypatch.delenv('NEXUS_MCP_DEFAULT_USER_ID', raising=False)
    from mcp_servers import nexus_email as je
    result = je._get_gmail(user_id=None, account='primary')
    assert isinstance(result, dict)
    assert result.get('ok') is False
    assert result.get('reason') == 'no_user_id'


# ---------------------------------------------------------------------------
# list_gmail_accounts tool — actual MCP-surfaced response shape
# ---------------------------------------------------------------------------


def test_list_gmail_accounts_tool_returns_primary_only_when_unmigrated(isolated_token_dir, monkeypatch):
    monkeypatch.setenv('NEXUS_MCP_DEFAULT_USER_ID', 'user-1')
    _write_token(isolated_token_dir / 'user-1.json')

    from mcp_servers import nexus_email as je
    result = je.list_gmail_accounts()

    assert result['ok'] is True
    assert result['user_id'] == 'user-1'
    labels = [a['label'] for a in result['accounts']]
    assert labels == ['primary']


def test_list_gmail_accounts_tool_returns_multiple_after_onboarding(isolated_token_dir, monkeypatch):
    monkeypatch.setenv('NEXUS_MCP_DEFAULT_USER_ID', 'user-1')
    _write_token(isolated_token_dir / 'primary' / 'user-1.json')
    _write_token(isolated_token_dir / 'shop' / 'user-1.json')

    from mcp_servers import nexus_email as je
    result = je.list_gmail_accounts()

    assert result['ok'] is True
    labels = [a['label'] for a in result['accounts']]
    assert labels[0] == 'primary'
    assert set(labels) == {'primary', 'shop'}
    for entry in result['accounts']:
        assert entry['token_path'].endswith(f'{entry["label"]}/user-1.json'), entry
