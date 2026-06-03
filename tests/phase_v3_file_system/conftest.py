from __future__ import annotations

from pathlib import Path

import pytest

import services.file_system_tools as file_system_tools
import services.terminal_policy as terminal_policy


_MNT_ALLOWED_ROOTS = (
    Path('/mnt/data'),
    Path('/opt/nexus'),
)


@pytest.fixture(autouse=True)
def fs_home_dir(tmp_path, monkeypatch):
    home_dir = tmp_path / 'home' / 'user'
    workspace_dir = home_dir / 'nexus_workspace'
    for path in (
        home_dir,
        home_dir / 'Documents',
        home_dir / 'Projects',
        workspace_dir,
        home_dir / '.ssh',
        home_dir / '.config',
        home_dir / '.aws',
        home_dir / '.gnupg',
    ):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)

    monkeypatch.setenv('HOME', str(home_dir))

    monkeypatch.setattr(file_system_tools, 'HOME_DIR', home_dir)
    monkeypatch.setattr(file_system_tools, 'ALLOWED_ROOTS', (home_dir, *_MNT_ALLOWED_ROOTS))
    monkeypatch.setattr(
        file_system_tools,
        'DENIED_DIR_PREFIXES',
        (
            Path('/etc'),
            Path('/var/log'),
            Path('/sys'),
            Path('/proc'),
            Path('/root'),
            home_dir / '.ssh',
            home_dir / '.config',
            home_dir / '.aws',
            home_dir / '.gnupg',
        ),
    )

    monkeypatch.setattr(terminal_policy, 'HOME_DIR', home_dir)
    monkeypatch.setattr(terminal_policy, 'ALLOWED_CWD_ROOTS', (home_dir, *_MNT_ALLOWED_ROOTS))
    monkeypatch.setattr(terminal_policy, 'DEFAULT_CWD', workspace_dir)

    return home_dir


@pytest.fixture
def fs_workspace_dir(fs_home_dir):
    return fs_home_dir / 'nexus_workspace'


@pytest.fixture
def fs_path(fs_home_dir):
    def _rewrite(raw_path: str) -> str:
        prefix = '/home/user'
        if raw_path == prefix:
            return str(fs_home_dir)
        if raw_path.startswith(prefix + '/'):
            return str(fs_home_dir / raw_path[len(prefix) + 1:])
        return raw_path

    return _rewrite
