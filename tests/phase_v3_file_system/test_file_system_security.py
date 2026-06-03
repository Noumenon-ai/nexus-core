"""Phase 4 file system security tests.

Adversarial input coverage: ensure the security boundary cannot be bypassed.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from services.file_system_tools import (
    FileSystemSecurityError,
    _validate_path,
    list_directory,
    read_file,
    search_files,
)


# ---------- _validate_path: rejection cases ----------

@pytest.mark.parametrize("bad_path", [
    "",
    "   ",
    "relative/path",
    "./file",
    "../etc/passwd",
    "file.txt",
])
def test_relative_or_empty_paths_rejected(bad_path):
    with pytest.raises(FileSystemSecurityError):
        _validate_path(bad_path, must_exist=False)


def test_null_byte_in_path_rejected():
    with pytest.raises(FileSystemSecurityError, match='null byte'):
        _validate_path('/home/user/test\x00.txt', must_exist=False)


def test_other_user_tilde_rejected():
    with pytest.raises(FileSystemSecurityError, match='~user'):
        _validate_path('~root/file', must_exist=False)


@pytest.mark.parametrize("denied_path", [
    '/etc/passwd',
    '/etc/shadow',
    '/var/log/syslog',
    '/sys/class',
    '/proc/1/maps',
    '/root/anything',
    '/home/user/.ssh/id_rsa',
    '/home/user/.ssh/known_hosts',
    '/home/user/.config/anything',
    '/home/user/.aws/credentials',
    '/home/user/.gnupg/pubring.kbx',
])
def test_denied_dir_prefixes_rejected(fs_path, denied_path):
    with pytest.raises(FileSystemSecurityError):
        _validate_path(fs_path(denied_path), must_exist=False)


@pytest.mark.parametrize("denied_substring_path", [
    '/home/user/Documents/my_credentials.txt',
    '/home/user/Documents/api_token_backup.txt',
    '/home/user/Documents/secret_notes.md',
    '/home/user/Documents/private_key_file.txt',
    '/home/user/Documents/id_rsa_backup',
    '/home/user/Documents/gmail_credentials_old.json',
])
def test_denied_substring_paths_rejected(fs_path, denied_substring_path):
    with pytest.raises(FileSystemSecurityError, match='denied substring'):
        _validate_path(fs_path(denied_substring_path), must_exist=False)


@pytest.mark.parametrize("denied_filename_path", [
    '/home/user/Projects/.env',
    '/home/user/Projects/.env.local',
    '/home/user/Projects/.env.production',
    '/home/user/Documents/cert.pem',
    '/home/user/Documents/server.key',
    '/home/user/Documents/google_credentials.json',
    '/home/user/Documents/some_credentials_backup.json',
])
def test_denied_filename_patterns_rejected(fs_path, denied_filename_path):
    with pytest.raises(FileSystemSecurityError):
        _validate_path(fs_path(denied_filename_path), must_exist=False)


@pytest.mark.parametrize("outside_path", [
    '/tmp/anything',
    '/opt/something',
    '/usr/local/bin/anything',
    '/var/anything',
])
def test_paths_outside_allowed_roots_rejected(outside_path):
    with pytest.raises(FileSystemSecurityError, match='outside allowed roots'):
        _validate_path(outside_path, must_exist=False)


# ---------- _validate_path: accept cases ----------

def test_allowed_home_path_accepted(fs_path):
    """A file under /home/user validates OK."""
    test_file = Path(fs_path('/home/user/allowed_test_file.txt'))
    test_file.write_text('hello', encoding='utf-8')
    try:
        resolved = _validate_path(str(test_file), must_exist=True)
        assert resolved == test_file.resolve()
    finally:
        test_file.unlink(missing_ok=True)






def test_denied_substring_still_applies_under_new_root():
    """H2-044: extending ALLOWED_ROOTS to /opt/nexus must NOT loosen the
    DENIED_PATH_SUBSTRINGS filter. Tested both via the /mnt/data symbolic
    path and the /opt/nexus resolved path so the protection is honest
    regardless of which one the LLM tries."""
    for denied in (
        '/mnt/data/Acme Properties/aws_credentials.json',
        '/mnt/data/ExampleBrand/api_secret_notes.txt',
        '/mnt/data/Family/old_private_key.pem',
        '/opt/nexus/secrets/anything.json',
        '/opt/nexus/Acme Properties/aws_credentials.json',
    ):
        with pytest.raises(FileSystemSecurityError):
            _validate_path(denied, must_exist=False)


def test_denied_filename_pattern_still_applies_under_new_root():
    """H2-044 corollary: the .env / id_rsa filename patterns also fire
    inside the newly allowed root. If they didn't, the broader allowlist
    would be a credential bypass."""
    for denied in (
        '/mnt/data/Acme Properties/.env',
        '/mnt/data/Family/id_rsa',
        '/opt/nexus/Acme Properties/.env',
        '/opt/nexus/Family/id_rsa',
    ):
        with pytest.raises(FileSystemSecurityError):
            _validate_path(denied, must_exist=False)


def test_tilde_expansion_works_for_home(fs_path):
    """~/some_file expands to /home/user/some_file."""
    test_file = Path(fs_path('/home/user/tilde_test_file.txt'))
    test_file.write_text('x', encoding='utf-8')
    try:
        resolved = _validate_path('~/tilde_test_file.txt', must_exist=True)
        assert resolved == test_file.resolve()
    finally:
        test_file.unlink(missing_ok=True)


# ---------- symlink attacks ----------

def test_symlink_escaping_allowed_roots_rejected(tmp_path):
    """A symlink in allowed root pointing to /etc must be rejected."""
    symlink = tmp_path / 'home' / 'user' / 'evil_symlink_test'
    target = Path('/etc/hostname')
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    try:
        symlink.symlink_to(target)
        with pytest.raises(FileSystemSecurityError):
            _validate_path(str(symlink), must_exist=True)
    finally:
        if symlink.is_symlink():
            symlink.unlink()


def test_symlink_to_denied_dir_rejected(fs_home_dir):
    """Symlink under allowed root pointing into ~/.ssh must be rejected."""
    symlink = fs_home_dir / 'evil_ssh_link_test'
    ssh_dir = fs_home_dir / '.ssh'
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    try:
        ssh_dir.mkdir(mode=0o700, exist_ok=True)
        symlink.symlink_to(ssh_dir)
        with pytest.raises(FileSystemSecurityError):
            _validate_path(str(symlink), must_exist=True)
    finally:
        if symlink.is_symlink():
            symlink.unlink()


# ---------- read_file behavior ----------

def test_read_file_denied_returns_tool_result_not_exception():
    """Security failures should return ToolResult, not raise."""
    result = read_file(user_id='u1', path='/etc/passwd')
    assert result.data['read'] is False
    assert result.data['reason'] == 'denied'


def test_read_file_text_content(fs_path):
    test_file = Path(fs_path('/home/user/phase4_read_test.txt'))
    test_file.write_text('hello world\nline two', encoding='utf-8')
    try:
        result = read_file(user_id='u1', path=str(test_file))
        assert result.data['read'] is True
        assert result.data['kind'] == 'text'
        assert 'hello world' in result.data['content']
    finally:
        test_file.unlink(missing_ok=True)


def test_read_file_binary_returns_hex_preview(fs_path):
    test_file = Path(fs_path('/home/user/phase4_binary_test.bin'))
    test_file.write_bytes(b'\x00\x01\x02\x03binary\x00data')
    try:
        result = read_file(user_id='u1', path=str(test_file))
        assert result.data['read'] is True
        assert result.data['kind'] == 'binary'
        assert 'preview_hex' in result.data
        assert 'content' not in result.data
    finally:
        test_file.unlink(missing_ok=True)


def test_read_file_too_large_rejected(fs_path):
    test_file = Path(fs_path('/home/user/phase4_large_test.txt'))
    test_file.write_text('a' * (300 * 1024), encoding='utf-8')  # 300KB > 256KB cap
    try:
        result = read_file(user_id='u1', path=str(test_file))
        assert result.data['read'] is False
        assert result.data['reason'] == 'too_large'
    finally:
        test_file.unlink(missing_ok=True)


def test_read_file_nonexistent_returns_denied(fs_path):
    result = read_file(user_id='u1', path=fs_path('/home/user/does_not_exist_phase4.txt'))
    assert result.data['read'] is False
    assert result.data['reason'] == 'denied'


def test_read_file_directory_not_a_file(fs_path):
    result = read_file(user_id='u1', path=fs_path('/home/user'))
    assert result.data['read'] is False
    assert result.data['reason'] == 'not_a_file'


# ---------- list_directory behavior ----------

def test_list_directory_basic(fs_path):
    result = list_directory(user_id='u1', path=fs_path('/home/user'), max_entries=10)
    assert result.data['listed'] is True
    assert isinstance(result.data['entries'], list)


def test_list_directory_hides_denied_entries(fs_home_dir, fs_path):
    """Listing /home/user must NOT include .ssh or .config in entries."""
    (fs_home_dir / '.ssh').mkdir(mode=0o700, exist_ok=True)
    result = list_directory(user_id='u1', path=fs_path('/home/user'), max_entries=1000)
    names = {e['name'] for e in result.data['entries']}
    assert '.ssh' not in names
    assert '.config' not in names
    assert '.aws' not in names


def test_list_directory_denied_path():
    result = list_directory(user_id='u1', path='/etc')
    assert result.data['listed'] is False
    assert result.data['reason'] == 'denied'


def test_list_directory_max_entries_capped(fs_path):
    """Caller asking for 100000 should be capped at hard limit."""
    result = list_directory(user_id='u1', path=fs_path('/home/user'), max_entries=100000)
    assert result.data['listed'] is True
    assert len(result.data['entries']) <= 1000


def test_list_directory_not_a_directory(fs_path):
    test_file = Path(fs_path('/home/user/phase4_not_dir.txt'))
    test_file.write_text('x', encoding='utf-8')
    try:
        result = list_directory(user_id='u1', path=str(test_file))
        assert result.data['listed'] is False
        assert result.data['reason'] == 'not_a_directory'
    finally:
        test_file.unlink(missing_ok=True)


# ---------- search_files behavior ----------

def test_search_files_finds_matching_filename(fs_path):
    test_file = Path(fs_path('/home/user/phase4_unique_searchable_xyzzy.txt'))
    test_file.write_text('x', encoding='utf-8')
    try:
        result = search_files(user_id='u1', root=fs_path('/home/user'), pattern='xyzzy', max_results=10)
        assert result.data['searched'] is True
        assert any('xyzzy' in m for m in result.data['matches'])
    finally:
        test_file.unlink(missing_ok=True)


def test_search_files_excludes_denied_paths(fs_path):
    """Search must not return credential/secret files even if pattern matches."""
    test_file = Path(fs_path('/home/user/phase4_xyzzy_credentials.json'))
    test_file.write_text('{}', encoding='utf-8')
    try:
        result = search_files(user_id='u1', root=fs_path('/home/user'), pattern='xyzzy', max_results=100)
        for m in result.data['matches']:
            assert 'credentials' not in m.lower()
    finally:
        test_file.unlink(missing_ok=True)


def test_search_files_empty_pattern_rejected(fs_path):
    result = search_files(user_id='u1', root=fs_path('/home/user'), pattern='', max_results=10)
    assert result.data['searched'] is False
    assert result.data['reason'] == 'empty_pattern'


def test_search_files_long_pattern_rejected(fs_path):
    result = search_files(user_id='u1', root=fs_path('/home/user'), pattern='x' * 201, max_results=10)
    assert result.data['searched'] is False
    assert result.data['reason'] == 'pattern_too_long'


def test_search_files_denied_root_rejected():
    result = search_files(user_id='u1', root='/etc', pattern='passwd', max_results=10)
    assert result.data['searched'] is False
    assert result.data['reason'] == 'denied'


# ---------- tool registry shape ----------

def test_three_tools_registered():
    from services.file_system_tools import make_file_system_tools
    tools = make_file_system_tools()
    names = {meta['name'] for _, meta in tools}
    assert names == {'read_file', 'list_directory', 'search_files'}


def test_no_tool_requires_approval():
    """Read-only tools must not require approval (would be UX disaster)."""
    from services.file_system_tools import make_file_system_tools
    tools = make_file_system_tools()
    for fn, meta in tools:
        assert meta.get('requires_approval', False) is False, \
            f'{meta["name"]} should not require approval (read-only)'


def test_read_file_description_does_not_trigger_model_refusal():
    """Regression guard for the 2026-05-11 read_file routing bug.

    The pre-fix description listed sensitive file types verbatim
    ('Credential files, .env, .ssh, .pem, .key, and similar are blocked').
    Gemini read 'and similar are blocked' as a catch-all and hallucinated
    'outside allowed roots' refusals without ever invoking the tool —
    even for benign paths inside the allowed roots. The post-fix
    description tells the model to ALWAYS call the tool and let the
    server-side validator return a structured error if a path is
    actually restricted.

    If a future edit reintroduces blocklist language in the description,
    Gemini will revert to pre-emptive refusal in production. This test
    is the canary.
    """
    from services.file_system_tools import make_file_system_tools

    read_meta = next(meta for fn, meta in make_file_system_tools() if meta['name'] == 'read_file')
    desc = read_meta['description']

    forbidden_phrases = (
        'and similar are blocked',
        'are blocked',
        '.env',
        '.ssh',
        '.pem',
        '.key',
        'credential files',
    )
    lower_desc = desc.lower()
    for phrase in forbidden_phrases:
        assert phrase.lower() not in lower_desc, (
            f'read_file description must not contain {phrase!r} — '
            f'this phrasing triggers Gemini to pre-emptively refuse tool calls.'
        )

    assert 'always call this tool' in lower_desc, (
        'read_file description must affirmatively instruct the model to call the tool.'
    )
