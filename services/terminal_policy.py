"""Phase 4 terminal command security policy.

Hard denylist: commands that NEVER reach approval gate — refused at parse time.
Approval-only allowlist of "known-safe" prefixes is intentionally NOT used here;
every non-denied command goes through approval. The denylist is the only refusal.

This module is read-only policy. No imports from services/ to keep it leafy.
"""
from __future__ import annotations

import shlex
import re
from pathlib import Path
from typing import Final


HOME_DIR: Final[Path] = Path.home()


# Argv-level denylist: if argv[0] (or any token) matches, command is refused
DENIED_ARGV_TOKENS: Final[tuple[str, ...]] = (
    'sudo',
    'su',
    'doas',
    'pkexec',
    'shutdown',
    'reboot',
    'halt',
    'poweroff',
    'init',
    'telinit',
    'dd',
    'mkfs',
    'mkfs.ext4',
    'mkfs.btrfs',
    'mkfs.xfs',
    'fdisk',
    'parted',
    'cryptsetup',
    'wipefs',
    'mount',
    'umount',
    'modprobe',
    'insmod',
    'rmmod',
)

# Substring patterns in the full raw command string (catches shell-y forms)
DENIED_COMMAND_SUBSTRINGS: Final[tuple[str, ...]] = (
    '|sh',
    '| sh',
    '|bash',
    '| bash',
    '|zsh',
    '| zsh',
    '> /etc',
    '>/etc',
    '> /usr',
    '>/usr',
    '> /sys',
    '>/sys',
    '> /proc',
    '>/proc',
    '> /dev',
    '>/dev',
    '$(',
    '`',
    'curl ',
    'wget ',
    'nc -',
    'netcat',
    '/etc/passwd',
    '/etc/shadow',
)

# Regex matches for things like `rm -rf /`, `rm -rf ~`, `rm -rf $HOME`
DENIED_REGEX_PATTERNS: Final[tuple[re.Pattern, ...]] = (
    re.compile(r'\brm\s+(-[a-zA-Z]*[rR]|--recursive)'),  # rm with -r/-R/--recursive (recursive delete)
    re.compile(r'\bchmod\s+(-R\s+)?[0-7]?777\b'),
    re.compile(r'\bchown\s+'),
    re.compile(r'\bkill\s+-9\s+1\b'),
    re.compile(r'\bkillall\b'),
    re.compile(r':\(\)\s*\{'),  # fork bomb signature
)


# Working directory allowlist (same as file_system_tools ALLOWED_ROOTS).
# H2-044 added /mnt/data AND /opt/nexus — the user's data partition. /mnt/data
# is a symlink to /opt/nexus on this host, so we list both forms to mirror
# file_system_tools.ALLOWED_ROOTS.
ALLOWED_CWD_ROOTS: Final[tuple[Path, ...]] = (
    HOME_DIR,
    Path('/opt'),
    Path('/mnt/data'),
    Path('/opt/nexus'),
)


DEFAULT_CWD: Final[Path] = HOME_DIR / 'nexus_workspace'


class TerminalSecurityError(Exception):
    """Raised when a command violates terminal security policy."""


def parse_command(raw: str) -> list[str]:
    """Parse a raw command string into argv list (shell=False later)."""
    if not raw or not isinstance(raw, str):
        raise TerminalSecurityError('command must be a non-empty string')
    if '\x00' in raw:
        raise TerminalSecurityError('command contains null byte')
    if len(raw) > 4096:
        raise TerminalSecurityError('command exceeds 4096 character limit')
    try:
        argv = shlex.split(raw)
    except ValueError as exc:
        raise TerminalSecurityError(f'cannot parse command: {exc}')
    if not argv:
        raise TerminalSecurityError('command parses to empty argv')
    return argv


def check_command_allowed(raw: str) -> list[str]:
    """Parse + validate raw command. Returns argv if allowed, else raises."""
    argv = parse_command(raw)

    lowered_full = raw.lower()
    for sub in DENIED_COMMAND_SUBSTRINGS:
        if sub.lower() in lowered_full:
            raise TerminalSecurityError(f'command contains denied pattern: {sub!r}')

    for pat in DENIED_REGEX_PATTERNS:
        if pat.search(raw):
            raise TerminalSecurityError(f'command matches denied pattern: {pat.pattern}')

    for token in argv:
        token_base = Path(token).name.lower() if '/' in token else token.lower()
        if token_base in DENIED_ARGV_TOKENS:
            raise TerminalSecurityError(f'command contains denied argv token: {token_base!r}')

    return argv


def validate_cwd(cwd_str: str | None) -> Path:
    """Validate working directory. Returns DEFAULT_CWD if cwd_str is None."""
    if cwd_str is None or not cwd_str.strip():
        DEFAULT_CWD.mkdir(parents=True, exist_ok=True, mode=0o700)
        return DEFAULT_CWD

    if not cwd_str.startswith('/') and not cwd_str.startswith('~/'):
        raise TerminalSecurityError(f'cwd must be absolute path, got {cwd_str!r}')

    if cwd_str.startswith('~/'):
        candidate = HOME_DIR / cwd_str[2:]
    else:
        candidate = Path(cwd_str)

    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise TerminalSecurityError(f'cwd does not exist: {cwd_str}')
    except (OSError, RuntimeError) as exc:
        raise TerminalSecurityError(f'cannot resolve cwd: {exc}')

    if not resolved.is_dir():
        raise TerminalSecurityError(f'cwd is not a directory: {resolved}')

    in_allowed = False
    for root in ALLOWED_CWD_ROOTS:
        try:
            resolved.relative_to(root)
            in_allowed = True
            break
        except ValueError:
            continue
    if not in_allowed:
        raise TerminalSecurityError(f'cwd outside allowed roots: {resolved}')

    return resolved
