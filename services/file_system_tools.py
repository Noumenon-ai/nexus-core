"""Phase 4 read-only file system tools — security-critical.

Provides read_file, list_directory, search_files. No write/exec.
Write + exec live in destructive_tools.py with approval gates.

Security model:
- ALLOWED_ROOTS: only paths under these prefixes are reachable
- DENIED_DIR_PREFIXES: deny these prefixes even if under an allowed root
- DENIED_PATH_SUBSTRINGS: case-insensitive substring match anywhere in resolved path
- DENIED_FILENAME_PATTERNS: regex match on filename only
- Path normalization: resolve symlinks, reject .. traversal, reject relative paths

Binary handling: read returns metadata + first 1KB hex preview. No binary writes.
"""
from __future__ import annotations

import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Callable

from services.tool_registry import ToolResult


HOME_DIR = Path.home()

ALLOWED_ROOTS: tuple[Path, ...] = (
    HOME_DIR,
    # H2-044: user's data partition. /mnt/data is a symlink to /opt/nexus
    # on this host, so the validator's candidate.resolve() dereferences it
    # before the allowlist check; we list both forms so future readers see
    # the user-facing path (/mnt/data/Acme Properties/...) alongside the real
    # mount point that actually matters for the security check.
    # DENIED_PATH_SUBSTRINGS / DENIED_FILENAME_PATTERNS still apply inside
    # the new root — /opt/nexus/secrets/* is auto-denied via the 'secret'
    # substring filter, same as ~/.aws/credentials would be under HOME_DIR.
    Path('/mnt/data'),
    Path('/opt/nexus'),
)

DENIED_DIR_PREFIXES: tuple[Path, ...] = (
    Path('/etc'),
    Path('/var/log'),
    Path('/sys'),
    Path('/proc'),
    Path('/root'),
    HOME_DIR / '.ssh',
    HOME_DIR / '.config',
    HOME_DIR / '.aws',
    HOME_DIR / '.gnupg',
)

DENIED_PATH_SUBSTRINGS: tuple[str, ...] = (
    'credentials',
    'secret',
    'token',
    'private_key',
    'id_rsa',
    'id_ed25519',
    'gmail_credentials',
)

DENIED_FILENAME_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r'^\.env(\..*)?$'),
    re.compile(r'.*\.pem$'),
    re.compile(r'.*\.key$'),
    re.compile(r'.*credentials.*\.json$', re.IGNORECASE),
)

BINARY_PREVIEW_BYTES = 1024
TEXT_MAX_BYTES = 256 * 1024  # 256KB hard cap on text reads
LIST_MAX_ENTRIES_HARD_CAP = 1000
SEARCH_MAX_RESULTS_HARD_CAP = 500
SEARCH_MAX_TRAVERSAL = 5000  # don't traverse more than 5k entries during search


class FileSystemSecurityError(Exception):
    """Raised when a path or operation violates security policy."""


def _is_under(child: Path, parent: Path) -> bool:
    """True if child is under parent (after both are absolute, resolved)."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _matches_denied_filename(name: str) -> bool:
    return any(pat.match(name) for pat in DENIED_FILENAME_PATTERNS)


def _matches_denied_substring(resolved_str: str) -> bool:
    lowered = resolved_str.lower()
    return any(sub in lowered for sub in DENIED_PATH_SUBSTRINGS)


def _validate_path(raw_path: str, *, must_exist: bool = True) -> Path:
    """The security boundary.

    Returns a resolved Path or raises FileSystemSecurityError.
    """
    if not raw_path or not isinstance(raw_path, str):
        raise FileSystemSecurityError('path must be a non-empty string')
    if '\x00' in raw_path:
        raise FileSystemSecurityError('path contains null byte')

    # Expand ~ but ONLY for the current home dir; reject ~otheruser
    if raw_path.startswith('~/'):
        candidate = HOME_DIR / raw_path[2:]
    elif raw_path == '~':
        candidate = HOME_DIR
    elif raw_path.startswith('~'):
        raise FileSystemSecurityError('~user expansion not allowed; use absolute path or ~/')
    else:
        candidate = Path(raw_path)

    if not candidate.is_absolute():
        raise FileSystemSecurityError(f'path must be absolute, got {raw_path!r}')

    # Resolve symlinks. If must_exist=True, .resolve(strict=True) raises if path missing.
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError:
        raise FileSystemSecurityError(f'path does not exist: {raw_path}')
    except (OSError, RuntimeError) as exc:
        raise FileSystemSecurityError(f'cannot resolve path: {exc}')

    # Reject paths whose resolved form escapes ALLOWED_ROOTS
    if not any(_is_under(resolved, root) for root in ALLOWED_ROOTS):
        raise FileSystemSecurityError(
            f'path is outside allowed roots: {resolved}'
        )

    # Reject denied directory prefixes (symlink-resolved)
    for denied_prefix in DENIED_DIR_PREFIXES:
        try:
            denied_resolved = denied_prefix.resolve()
        except (OSError, RuntimeError):
            denied_resolved = denied_prefix
        if _is_under(resolved, denied_resolved):
            raise FileSystemSecurityError(f'path is in denied directory: {denied_prefix}')

    # Reject denied substrings anywhere in the resolved string
    if _matches_denied_substring(str(resolved)):
        raise FileSystemSecurityError('path matches denied substring (credentials/secret/token/etc)')

    # Reject denied filename patterns (.env, *.pem, *.key, *credentials*.json)
    if _matches_denied_filename(resolved.name):
        raise FileSystemSecurityError(f'filename pattern denied: {resolved.name}')

    return resolved


def _is_likely_binary(content_bytes: bytes) -> bool:
    """Heuristic: contains null byte or fails UTF-8 decode = binary."""
    if b'\x00' in content_bytes[:8192]:
        return True
    try:
        content_bytes[:8192].decode('utf-8')
        return False
    except UnicodeDecodeError:
        return True


def read_file(*, user_id: str, path: str) -> ToolResult:
    try:
        resolved = _validate_path(path, must_exist=True)
    except FileSystemSecurityError as exc:
        return ToolResult.ok(
            data={'read': False, 'reason': 'denied', 'detail': str(exc)},
            announcement=f'Cannot read file: {exc}.',
        )

    if not resolved.is_file():
        return ToolResult.ok(
            data={'read': False, 'reason': 'not_a_file', 'path': str(resolved)},
            announcement=f'Path is not a regular file: {resolved}',
        )

    try:
        stat = resolved.stat()
    except OSError as exc:
        return ToolResult.ok(
            data={'read': False, 'reason': 'stat_failed', 'detail': str(exc)},
            announcement=f'Cannot stat file: {exc}',
        )

    size = stat.st_size
    mime_type, _ = mimetypes.guess_type(str(resolved))

    try:
        with open(resolved, 'rb') as f:
            head = f.read(max(BINARY_PREVIEW_BYTES, TEXT_MAX_BYTES + 1))
    except OSError as exc:
        return ToolResult.ok(
            data={'read': False, 'reason': 'read_failed', 'detail': str(exc)},
            announcement=f'Cannot read file: {exc}',
        )

    if _is_likely_binary(head):
        preview = head[:BINARY_PREVIEW_BYTES].hex()
        return ToolResult.ok(
            data={
                'read': True,
                'kind': 'binary',
                'path': str(resolved),
                'size_bytes': size,
                'mime_type': mime_type,
                'preview_hex': preview,
                'preview_bytes': min(BINARY_PREVIEW_BYTES, len(head)),
            },
            announcement=f'Read binary file {resolved.name} ({size} bytes, {mime_type or "unknown"}).',
        )

    if size > TEXT_MAX_BYTES:
        return ToolResult.ok(
            data={
                'read': False,
                'reason': 'too_large',
                'path': str(resolved),
                'size_bytes': size,
                'max_text_bytes': TEXT_MAX_BYTES,
            },
            announcement=f'File too large to read as text ({size} bytes > {TEXT_MAX_BYTES} byte cap).',
        )

    try:
        text = head[:size].decode('utf-8')
    except UnicodeDecodeError as exc:
        return ToolResult.ok(
            data={'read': False, 'reason': 'decode_failed', 'detail': str(exc)},
            announcement='Cannot decode file as UTF-8 text.',
        )

    return ToolResult.ok(
        data={
            'read': True,
            'kind': 'text',
            'path': str(resolved),
            'size_bytes': size,
            'mime_type': mime_type or 'text/plain',
            'content': text,
        },
        announcement=f'Read text file {resolved.name} ({size} bytes).',
    )


def list_directory(*, user_id: str, path: str, max_entries: int = 200) -> ToolResult:
    max_entries = max(1, min(int(max_entries), LIST_MAX_ENTRIES_HARD_CAP))

    try:
        resolved = _validate_path(path, must_exist=True)
    except FileSystemSecurityError as exc:
        return ToolResult.ok(
            data={'listed': False, 'reason': 'denied', 'detail': str(exc)},
            announcement=f'Cannot list directory: {exc}.',
        )

    if not resolved.is_dir():
        return ToolResult.ok(
            data={'listed': False, 'reason': 'not_a_directory', 'path': str(resolved)},
            announcement=f'Path is not a directory: {resolved}',
        )

    entries = []
    truncated = False
    try:
        for entry in sorted(os.scandir(resolved), key=lambda e: e.name):
            # Hide denied filenames/patterns from the listing entirely
            if _matches_denied_filename(entry.name):
                continue
            full_path = Path(entry.path)
            if _matches_denied_substring(str(full_path)):
                continue
            # Hide entries that are themselves denied-prefix directories
            try:
                entry_resolved = full_path.resolve()
            except (OSError, RuntimeError):
                continue
            entry_denied = False
            for denied_prefix in DENIED_DIR_PREFIXES:
                try:
                    denied_resolved = denied_prefix.resolve()
                except (OSError, RuntimeError):
                    denied_resolved = denied_prefix
                if _is_under(entry_resolved, denied_resolved) or entry_resolved == denied_resolved:
                    entry_denied = True
                    break
            if entry_denied:
                continue
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            entries.append({
                'name': entry.name,
                'kind': 'directory' if is_dir else 'file',
                'size_bytes': stat.st_size if not is_dir else None,
            })
            if len(entries) >= max_entries:
                truncated = True
                break
    except OSError as exc:
        return ToolResult.ok(
            data={'listed': False, 'reason': 'scan_failed', 'detail': str(exc)},
            announcement=f'Cannot list directory: {exc}',
        )

    return ToolResult.ok(
        data={
            'listed': True,
            'path': str(resolved),
            'entry_count': len(entries),
            'truncated': truncated,
            'entries': entries,
        },
        announcement=f'Listed {len(entries)} entries in {resolved.name}.',
    )


def search_files(*, user_id: str, root: str, pattern: str, max_results: int = 100) -> ToolResult:
    if not pattern or not isinstance(pattern, str) or not pattern.strip():
        return ToolResult.ok(
            data={'searched': False, 'reason': 'empty_pattern'},
            announcement='Search pattern is empty.',
        )
    if len(pattern) > 200:
        return ToolResult.ok(
            data={'searched': False, 'reason': 'pattern_too_long'},
            announcement='Search pattern exceeds 200 characters.',
        )

    max_results = max(1, min(int(max_results), SEARCH_MAX_RESULTS_HARD_CAP))

    try:
        resolved_root = _validate_path(root, must_exist=True)
    except FileSystemSecurityError as exc:
        return ToolResult.ok(
            data={'searched': False, 'reason': 'denied', 'detail': str(exc)},
            announcement=f'Cannot search: {exc}.',
        )

    if not resolved_root.is_dir():
        return ToolResult.ok(
            data={'searched': False, 'reason': 'not_a_directory'},
            announcement=f'Search root is not a directory: {resolved_root}',
        )

    pattern_lower = pattern.lower()
    matches = []
    traversed = 0
    truncated = False

    for dirpath, dirnames, filenames in os.walk(resolved_root, followlinks=False):
        # Prune denied dirs from descent
        dirnames[:] = [
            d for d in dirnames
            if not _matches_denied_filename(d)
            and not _matches_denied_substring(str(Path(dirpath) / d))
        ]
        for filename in filenames:
            traversed += 1
            if traversed > SEARCH_MAX_TRAVERSAL:
                truncated = True
                break
            if _matches_denied_filename(filename):
                continue
            full_path = Path(dirpath) / filename
            if _matches_denied_substring(str(full_path)):
                continue
            if pattern_lower in filename.lower():
                matches.append(str(full_path))
                if len(matches) >= max_results:
                    truncated = True
                    break
        if truncated:
            break

    return ToolResult.ok(
        data={
            'searched': True,
            'root': str(resolved_root),
            'pattern': pattern,
            'match_count': len(matches),
            'traversed': traversed,
            'truncated': truncated,
            'matches': matches,
        },
        announcement=f'Found {len(matches)} match{"" if len(matches) == 1 else "es"} for {pattern!r}.',
    )


def make_file_system_tools() -> list[tuple[Callable[..., ToolResult], dict[str, Any]]]:
    return [
        (
            read_file,
            {
                'name': 'read_file',
                'description': (
                    "Read a file from the user's laptop and return its contents. "
                    "Returns text content for text files, or metadata plus a hex preview "
                    "for binary files. Always call this tool when the user asks to read, "
                    "view, or show the contents of a file — do not pre-emptively refuse "
                    "based on how the path looks. The tool itself enforces security policy "
                    "and returns a clear structured error if a path is restricted, so the "
                    "user gets a precise response either way."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'path': {'type': 'string', 'description': 'Absolute path to the file.'},
                    },
                    'required': ['path'],
                },
            },
        ),
        (
            list_directory,
            {
                'name': 'list_directory',
                'description': (
                    "List files and subdirectories in a directory on the user's laptop. "
                    "Same allowed roots and denylist as read_file."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'path': {'type': 'string', 'description': 'Absolute path to the directory.'},
                        'max_entries': {
                            'type': 'integer',
                            'description': 'Max entries to return (default 200, hard cap 1000).',
                            'default': 200,
                        },
                    },
                    'required': ['path'],
                },
            },
        ),
        (
            search_files,
            {
                'name': 'search_files',
                'description': (
                    "Search for files whose name contains a pattern (case-insensitive substring match), "
                    "under the given root directory. Same allowed roots and denylist as read_file."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'root': {'type': 'string', 'description': 'Absolute path to search root.'},
                        'pattern': {'type': 'string', 'description': 'Filename substring to match.'},
                        'max_results': {
                            'type': 'integer',
                            'description': 'Max results (default 100, hard cap 500).',
                            'default': 100,
                        },
                    },
                    'required': ['root', 'pattern'],
                },
            },
        ),
    ]


def register_file_system_tools(registry) -> list:
    specs = []
    for fn, meta in make_file_system_tools():
        specs.append(registry.register(fn, **meta))
    return specs
