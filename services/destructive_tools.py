"""V3.4 destructive (approval-gated) tools — five Nexus-internal write
paths that require user tap-to-approve before execution.

Per V3.4 spec: every tool registered here MUST have requires_approval=True
and a non-empty approval_template. The V3.4 invariant test
(tests/phase_v3_4/test_safety_invariants.py) enforces this across all
registries, so any future tool whose name matches the destructive prefix
set (delete_/forget_/send_/disconnect_) is checked automatically.

Tool fn semantics: in the V3.5 dispatcher, when Gemini calls a tool with
requires_approval=True, the dispatcher creates an Approval row from the
tool's approval_template + args, returns the approval id and preview,
and waits for user tap. ONLY after the tap does the dispatcher (via
approval_service.execute) call the tool fn. So at this layer, the fn is
the post-approval implementation.

Five real tools:
    delete_reminder
    delete_task
    forget_user_memory
    disconnect_google
    append_telos_update

Two stubbed tools (delete_calendar_event, send_telegram_message) live in
services/destructive_tools_stubs.py under the same V3.2.5 phase tag
tracked at HARDENING_PASS_V2.md H2-011.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from googleapiclient.errors import HttpError

from repositories.memories_repository import MemoriesRepository
from repositories.reminders_repository import RemindersRepository
from repositories.tasks_repository import TasksRepository
from services.file_system_tools import (
    DENIED_FILENAME_PATTERNS as _FS_DENIED_FILENAME_PATTERNS,
    FileSystemSecurityError,
    _matches_denied_filename,
    _matches_denied_substring,
    _validate_path as _validate_fs_path,
)
from services.google_auth_service import GoogleAuthError
from services.google_calendar_service import GoogleCalendarService
from services.telos_service import TelosService
from services.terminal_policy import (
    DEFAULT_CWD as _TERMINAL_DEFAULT_CWD,
    TerminalSecurityError,
    check_command_allowed,
    validate_cwd,
)
from services.tool_registry import ToolRegistry, ToolResult, ToolSpec
import asyncio
import re

_BINARY_EXTENSIONS = frozenset({
    '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff',
    '.docx', '.xlsx', '.pptx', '.odt', '.ods', '.odp',
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
    '.mp4', '.mp3', '.ogg', '.wav', '.flac', '.avi', '.mov', '.mkv',
    '.exe', '.dll', '.so', '.dylib', '.bin', '.deb', '.rpm', '.iso',
    '.sqlite', '.db', '.pkl', '.pickle',
})

_WRITE_FILE_MAX_BYTES = 1 * 1024 * 1024  # 1MB text write cap
_TERMINAL_MAX_TIMEOUT_SEC = 300
_TERMINAL_DEFAULT_TIMEOUT_SEC = 60
_TERMINAL_OUTPUT_CAP_BYTES = 64 * 1024  # 64KB stdout+stderr cap returned to model


def make_destructive_tools(
    *,
    reminders_repository: RemindersRepository,
    tasks_repository: TasksRepository,
    memories_repository: MemoriesRepository,
    telos_service: TelosService,
    google_disconnect: Callable[[str], Awaitable[None]],
    google_calendar_service: GoogleCalendarService | None = None,
    scheduler: Any | None = None,
) -> list[tuple[Callable[..., ToolResult], dict[str, Any]]]:
    """Build the five V3.4 destructive tool closures bound to dependencies.

    Each returned function is the post-approval implementation. The V3.5
    dispatcher gates calls on requires_approval=True before invoking.
    """

    def delete_reminder(*, user_id: str, query: str) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            return ToolResult.ok(
                data={'cancelled': False, 'reason': 'empty_query'},
                announcement='Cannot delete reminder: query is empty.',
            )
        reminder = reminders_repository.find_for_cancel(user_id, query)
        if reminder is None:
            return ToolResult.ok(
                data={'cancelled': False, 'reason': 'no_match', 'query': query},
                announcement=f'No matching reminder found for {query!r}.',
            )
        cancelled = reminders_repository.cancel(reminder.id)
        if scheduler is not None:
            scheduler.remove_reminder(reminder.id)
        return ToolResult.ok(
            data={'cancelled': True, 'reminder_id': reminder.id, 'body': cancelled.body if cancelled else reminder.body},
            announcement=f'Reminder cancelled: {reminder.body}',
        )

    def delete_task(*, user_id: str, query: str) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            return ToolResult.ok(
                data={'deleted': False, 'reason': 'empty_query'},
                announcement='Cannot delete task: query is empty.',
            )
        deleted = tasks_repository.delete(user_id, query)
        if deleted is None:
            return ToolResult.ok(
                data={'deleted': False, 'reason': 'no_match', 'query': query},
                announcement=f'No matching pending task found for {query!r}.',
            )
        return ToolResult.ok(
            data={'deleted': True, 'task_id': deleted.id, 'title': deleted.title},
            announcement=f'Task deleted: {deleted.title}',
        )

    def forget_user_memory(*, user_id: str, key: str) -> ToolResult:
        if not isinstance(key, str) or not key.strip():
            return ToolResult.ok(
                data={'forgotten': False, 'reason': 'empty_key'},
                announcement='Cannot forget memory: key is empty.',
            )
        deleted = memories_repository.delete(user_id=user_id, key=key)
        if deleted is None:
            return ToolResult.ok(
                data={'forgotten': False, 'reason': 'no_match', 'key': key},
                announcement=f'No memory found with key {key!r}.',
            )
        return ToolResult.ok(
            data={'forgotten': True, 'key': key},
            announcement=f'Forgot memory: {key}',
        )

    async def disconnect_google(*, user_id: str) -> ToolResult:
        try:
            await google_disconnect(user_id)
        except Exception as exc:
            return ToolResult.ok(
                data={'disconnected': False, 'reason': 'callback_error', 'error_class': type(exc).__name__},
                announcement=f'Google disconnect did not complete cleanly: {type(exc).__name__}.',
            )
        return ToolResult.ok(
            data={'disconnected': True},
            announcement='Google account disconnected for this user.',
        )

    def append_telos_update(*, user_id: str, content: str) -> ToolResult:
        if not isinstance(content, str) or not content.strip():
            return ToolResult.ok(
                data={'appended': False, 'reason': 'empty_content'},
                announcement='Cannot update TELOS: content is empty.',
            )
        try:
            telos_service.append(user_id, content)
        except ValueError as exc:
            return ToolResult.ok(
                data={'appended': False, 'reason': 'invalid_user_id'},
                announcement=f'Cannot update TELOS: {exc}',
            )
        return ToolResult.ok(
            data={'appended': True, 'bytes': len(content.encode('utf-8'))},
            announcement='TELOS updated.',
        )

    async def delete_calendar_event(*, user_id: str, event_id: str) -> ToolResult:
        if google_calendar_service is None:
            return ToolResult.ok(
                data={'deleted': False, 'reason': 'calendar_not_configured'},
                announcement='Cannot delete calendar event: Google Calendar is not connected.',
            )
        if not event_id or not event_id.strip():
            return ToolResult.ok(
                data={'deleted': False, 'reason': 'empty_event_id'},
                announcement='Cannot delete calendar event: event_id is required.',
            )
        try:
            await google_calendar_service.delete_event(user_id, event_id=event_id)
        except GoogleAuthError:
            return ToolResult.ok(
                data={'deleted': False, 'reason': 'auth_error'},
                announcement='Cannot delete calendar event: Google Calendar auth needs re-connection.',
            )
        except HttpError as exc:
            return ToolResult.ok(
                data={'deleted': False, 'reason': 'api_error', 'detail': str(exc)},
                announcement='Cannot delete calendar event: Google API returned an error.',
            )
        return ToolResult.ok(
            data={'deleted': True, 'event_id': event_id},
            announcement=f'Calendar event {event_id!r} deleted.',
        )

    def write_file(*, user_id: str, path: str, content: str) -> ToolResult:
        # path validation: same allowlist/denylist as read
        try:
            # must_exist=False because we may be creating new file
            from pathlib import Path
            if not isinstance(path, str) or not path.strip():
                return ToolResult.ok(
                    data={'written': False, 'reason': 'empty_path'},
                    announcement='Cannot write file: path is empty.',
                )
            if '\x00' in path:
                return ToolResult.ok(
                    data={'written': False, 'reason': 'null_byte'},
                    announcement='Cannot write file: path contains null byte.',
                )
            # If target exists, _validate_fs_path will catch the security check via symlink resolution.
            # If target does NOT exist, we need to validate the parent + the proposed filename separately.
            target = Path(path)
            if path.startswith('~/'):
                target = Path('/home/user') / path[2:]
            elif path.startswith('~'):
                return ToolResult.ok(
                    data={'written': False, 'reason': 'tilde_user_not_allowed'},
                    announcement='Cannot write file: ~user expansion not supported.',
                )
            if not target.is_absolute():
                return ToolResult.ok(
                    data={'written': False, 'reason': 'not_absolute'},
                    announcement='Cannot write file: path must be absolute.',
                )
            # Validate parent directory (must exist + be in allowed root + not denied)
            try:
                parent_resolved = _validate_fs_path(str(target.parent), must_exist=True)
            except FileSystemSecurityError as exc:
                return ToolResult.ok(
                    data={'written': False, 'reason': 'parent_denied', 'detail': str(exc)},
                    announcement=f'Cannot write file: parent directory rejected: {exc}.',
                )
            if not parent_resolved.is_dir():
                return ToolResult.ok(
                    data={'written': False, 'reason': 'parent_not_a_directory'},
                    announcement='Cannot write file: parent path is not a directory.',
                )
            final_path = parent_resolved / target.name
            # Filename + substring checks on the final target
            if _matches_denied_filename(final_path.name):
                return ToolResult.ok(
                    data={'written': False, 'reason': 'filename_denied'},
                    announcement=f'Cannot write file: filename pattern denied ({final_path.name}).',
                )
            if _matches_denied_substring(str(final_path)):
                return ToolResult.ok(
                    data={'written': False, 'reason': 'path_substring_denied'},
                    announcement='Cannot write file: path contains denied substring.',
                )
            # Binary extension denial
            ext = final_path.suffix.lower()
            if ext in _BINARY_EXTENSIONS:
                return ToolResult.ok(
                    data={'written': False, 'reason': 'binary_extension_denied', 'extension': ext},
                    announcement=f'Cannot write file: {ext} is a binary extension (text-only writes).',
                )
        except Exception as exc:
            return ToolResult.fail(f'write_file path validation error: {exc}')

        # Content validation
        if not isinstance(content, str):
            return ToolResult.ok(
                data={'written': False, 'reason': 'content_not_string'},
                announcement='Cannot write file: content must be a string.',
            )
        try:
            encoded = content.encode('utf-8')
        except UnicodeEncodeError as exc:
            return ToolResult.ok(
                data={'written': False, 'reason': 'encode_failed', 'detail': str(exc)},
                announcement='Cannot write file: content cannot be encoded as UTF-8.',
            )
        if len(encoded) > _WRITE_FILE_MAX_BYTES:
            return ToolResult.ok(
                data={
                    'written': False,
                    'reason': 'too_large',
                    'size_bytes': len(encoded),
                    'max_bytes': _WRITE_FILE_MAX_BYTES,
                },
                announcement=f'Cannot write file: content exceeds {_WRITE_FILE_MAX_BYTES} byte cap.',
            )

        # Write atomically via tempfile + rename
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(
                mode='wb', dir=str(final_path.parent), delete=False, prefix='.nexus_write_'
            ) as tmp:
                tmp.write(encoded)
                tmp_path = Path(tmp.name)
            tmp_path.replace(final_path)
        except OSError as exc:
            return ToolResult.fail(f'write_file os error: {exc}')

        return ToolResult.ok(
            data={
                'written': True,
                'path': str(final_path),
                'size_bytes': len(encoded),
            },
            announcement=f'Wrote {len(encoded)} bytes to {final_path}.',
        )

    async def run_terminal_command(
        *,
        user_id: str,
        command: str,
        cwd: str | None = None,
        timeout_sec: int = _TERMINAL_DEFAULT_TIMEOUT_SEC,
    ) -> ToolResult:
        timeout_sec = max(1, min(int(timeout_sec), _TERMINAL_MAX_TIMEOUT_SEC))
        try:
            argv = check_command_allowed(command)
        except TerminalSecurityError as exc:
            return ToolResult.ok(
                data={'executed': False, 'reason': 'denied', 'detail': str(exc)},
                announcement=f'Cannot run command: {exc}.',
            )
        try:
            cwd_resolved = validate_cwd(cwd)
        except TerminalSecurityError as exc:
            return ToolResult.ok(
                data={'executed': False, 'reason': 'cwd_denied', 'detail': str(exc)},
                announcement=f'Cannot run command: {exc}.',
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd_resolved),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult.ok(
                    data={'executed': False, 'reason': 'timeout', 'timeout_sec': timeout_sec},
                    announcement=f'Command timed out after {timeout_sec}s.',
                )
        except FileNotFoundError:
            return ToolResult.ok(
                data={'executed': False, 'reason': 'binary_not_found', 'argv0': argv[0]},
                announcement=f'Command not found: {argv[0]}',
            )
        except OSError as exc:
            return ToolResult.fail(f'run_terminal_command os error: {exc}')

        stdout = stdout_b.decode('utf-8', errors='replace')[:_TERMINAL_OUTPUT_CAP_BYTES]
        stderr = stderr_b.decode('utf-8', errors='replace')[:_TERMINAL_OUTPUT_CAP_BYTES]
        return ToolResult.ok(
            data={
                'executed': True,
                'argv': argv,
                'cwd': str(cwd_resolved),
                'return_code': proc.returncode,
                'stdout': stdout,
                'stderr': stderr,
                'stdout_truncated': len(stdout_b) > _TERMINAL_OUTPUT_CAP_BYTES,
                'stderr_truncated': len(stderr_b) > _TERMINAL_OUTPUT_CAP_BYTES,
            },
            announcement=f'Ran {argv[0]} (exit {proc.returncode}).',
        )

    return [
        (
            delete_reminder,
            {
                'name': 'delete_reminder',
                'description': "Cancel one of the user's active reminders by id or body fragment.",
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string', 'description': 'Reminder id or body fragment.'},
                    },
                    'required': ['query'],
                },
                'requires_approval': True,
                'approval_template': 'Cancel reminder matching {query!r}',
            },
        ),
        (
            delete_task,
            {
                'name': 'delete_task',
                'description': "Delete one of the user's pending tasks by id or title fragment.",
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string', 'description': 'Task id or title fragment.'},
                    },
                    'required': ['query'],
                },
                'requires_approval': True,
                'approval_template': 'Delete task matching {query!r}',
            },
        ),
        (
            forget_user_memory,
            {
                'name': 'forget_user_memory',
                'description': "Delete a memory by key.",
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'key': {'type': 'string', 'description': 'Memory key to delete.'},
                    },
                    'required': ['key'],
                },
                'requires_approval': True,
                'approval_template': 'Forget memory key {key!r}',
            },
        ),
        (
            disconnect_google,
            {
                'name': 'disconnect_google',
                'description': "Revoke and disconnect the user's Google OAuth tokens. The user will need to re-authorize for any Google integration to work again.",
                'parameters': {'type': 'object', 'properties': {}, 'required': []},
                'requires_approval': True,
                'approval_template': 'Disconnect your Google account from this user',
            },
        ),
        (
            append_telos_update,
            {
                'name': 'append_telos_update',
                'description': "Append content to the user's TELOS file (life-OS doc).",
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'content': {'type': 'string', 'description': 'Markdown content to append. Use blank lines to separate sections.'},
                    },
                    'required': ['content'],
                },
                'requires_approval': True,
                'approval_template': 'Append to your TELOS:\n{content}',
            },
        ),
        (
            delete_calendar_event,
            {
                'name': 'delete_calendar_event',
                'description': "Delete an event from the user's primary Google Calendar by event_id.",
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'event_id': {'type': 'string', 'description': 'Google Calendar event id to delete.'},
                    },
                    'required': ['event_id'],
                },
                'requires_approval': True,
                'approval_template': 'Delete Google Calendar event {event_id!r}',
            },
        ),
        (
            write_file,
            {
                'name': 'write_file',
                'description': (
                    "Write text content to a file on the user's laptop. Approval-gated. "
                    "Allowed roots: /home/user and /opt. "
                    "Credential files, .env, binary extensions are blocked. Max 1MB content."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'path': {'type': 'string', 'description': 'Absolute path to write.'},
                        'content': {'type': 'string', 'description': 'UTF-8 text content.'},
                    },
                    'required': ['path', 'content'],
                },
                'requires_approval': True,
                'approval_template': 'Write {len(content)} chars to {path!r}',
            },
        ),
        (
            run_terminal_command,
            {
                'name': 'run_terminal_command',
                'description': (
                    "Run a shell command on the user's laptop. Approval-gated. "
                    "Dangerous commands (sudo, rm -r, dd, etc.) are refused. "
                    "Default cwd is /home/user/nexus_workspace. "
                    "Output capped at 64KB stdout+stderr. Timeout default 60s, max 300s."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'command': {'type': 'string', 'description': 'Command to run.'},
                        'cwd': {'type': 'string', 'description': 'Working directory (optional).'},
                        'timeout_sec': {
                            'type': 'integer',
                            'description': 'Timeout in seconds (default 60, max 300).',
                            'default': 60,
                        },
                    },
                    'required': ['command'],
                },
                'requires_approval': True,
                'approval_template': 'Run command {command!r} in {cwd or "default"}',
            },
        ),
    ]


def register_destructive_tools(registry: ToolRegistry, **deps: Any) -> list[ToolSpec]:
    """Register all five V3.4 destructive tools into the given registry."""
    specs = []
    for fn, meta in make_destructive_tools(**deps):
        specs.append(registry.register(fn, **meta))
    return specs
