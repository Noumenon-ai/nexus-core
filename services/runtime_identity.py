from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import os
import re
import subprocess
import sys
import time


_STARTED_AT = datetime.now(timezone.utc)
_START_MONOTONIC = time.monotonic()
_SERVICE_NAME_RE = re.compile(r'([A-Za-z0-9_.@-]+\.service)\b')


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    git_sha: str
    branch: str
    repo_path: str
    venv_path: str
    venv_label: str
    pid: int
    service_name: str | None
    started_at: str
    uptime_seconds: int


def get_runtime_identity(
    *,
    repo_path: Path | None = None,
    executable_path: str | None = None,
    pid: int | None = None,
) -> RuntimeIdentity:
    resolved_repo = (repo_path or _default_repo_path()).resolve()
    resolved_pid = pid or os.getpid()
    venv_path = _detect_venv_path(executable_path=executable_path)
    return RuntimeIdentity(
        git_sha=_read_git_value(['rev-parse', 'HEAD'], cwd=resolved_repo),
        branch=_read_git_value(['branch', '--show-current'], cwd=resolved_repo),
        repo_path=str(resolved_repo),
        venv_path=str(venv_path),
        venv_label=_render_venv_label(venv_path=venv_path, repo_path=resolved_repo),
        pid=resolved_pid,
        service_name=_detect_service_name(pid=resolved_pid),
        started_at=_STARTED_AT.isoformat(),
        uptime_seconds=max(0, int(time.monotonic() - _START_MONOTONIC)),
    )


def format_runtime_log_line(identity: RuntimeIdentity) -> str:
    return (
        'NEXUS_RUNTIME '
        f'git={identity.git_sha} '
        f'branch={identity.branch} '
        f'repo={identity.repo_path} '
        f'venv={identity.venv_label} '
        f'pid={identity.pid}'
    )


def log_runtime_identity(
    logger: logging.Logger,
    *,
    identity: RuntimeIdentity | None = None,
) -> RuntimeIdentity:
    resolved = identity or get_runtime_identity()
    logger.info(format_runtime_log_line(resolved))
    return resolved


def render_runtime_status_text(identity: RuntimeIdentity) -> str:
    service_name = identity.service_name or 'unknown'
    return (
        'Nexus runtime:\n'
        f'git: {identity.git_sha}\n'
        f'branch: {identity.branch}\n'
        f'repo: {identity.repo_path}\n'
        f'venv: {identity.venv_label}\n'
        f'pid: {identity.pid}\n'
        f'service: {service_name}'
    )


def _default_repo_path() -> Path:
    return Path(__file__).resolve().parents[1]


def _detect_venv_path(*, executable_path: str | None = None) -> Path:
    if executable_path is None:
        return Path(sys.prefix).resolve()
    executable = Path(executable_path).resolve()
    if executable.parent.name == 'bin':
        return executable.parent.parent
    return Path(sys.prefix).resolve()


def _render_venv_label(*, venv_path: Path, repo_path: Path) -> str:
    try:
        relative = venv_path.relative_to(repo_path)
    except ValueError:
        return str(venv_path)
    return str(relative) or '.'


def _read_git_value(args: list[str], *, cwd: Path) -> str:
    try:
        result = subprocess.run(
            ['git', *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return 'unknown'
    except Exception:
        return 'unknown'
    if result.returncode != 0:
        return 'unknown'
    value = (result.stdout or '').strip()
    return value or 'unknown'


def _detect_service_name(*, pid: int) -> str | None:
    text = _read_proc_cgroup(pid)
    if not text:
        return None
    matches = _SERVICE_NAME_RE.findall(text)
    return matches[-1] if matches else None


def _read_proc_cgroup(pid: int) -> str:
    try:
        return Path(f'/proc/{pid}/cgroup').read_text(encoding='utf-8')
    except Exception:
        return ''
