"""V3.4 prerequisite writes:

- TasksRepository.delete(user_id, query) — repo-symmetry hard delete by id
  or title fragment. Counterpart to mark_done; mirrors memories_repository.delete.
- TelosService.append(user_id, content) — file append with mode-600 preserved.
  Counterpart to TelosService.load.

These are the simple write primitives the V3.4 destructive tool wrappers
delegate to. Hardening defenses (validation, secret-scrub) live in the
tool layer, not the repo layer.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

from services.telos_service import TelosService


# -------- TasksRepository.delete --------------------------------------------

def test_tasks_repository_delete_hard_removes_pending_task_by_title_fragment(container):
    user = container.users_repository.get_or_create(111)
    container.tasks_repository.create(user_id=user.id, title='go to gym', due_at=None)
    container.tasks_repository.create(user_id=user.id, title='write report', due_at=None)

    deleted = container.tasks_repository.delete(user.id, 'gym')
    assert deleted is not None
    assert deleted.title == 'go to gym'

    remaining = container.tasks_repository.list_pending(user.id)
    assert [t.title for t in remaining] == ['write report']


def test_tasks_repository_delete_returns_none_when_no_match(container):
    user = container.users_repository.get_or_create(111)
    deleted = container.tasks_repository.delete(user.id, 'nothing here')
    assert deleted is None


def test_tasks_repository_delete_user_isolation(container):
    a = container.users_repository.get_or_create(111)
    b = container.users_repository.get_or_create(222)
    container.tasks_repository.create(user_id=b.id, title='B private task', due_at=None)

    deleted = container.tasks_repository.delete(a.id, 'private')
    assert deleted is None
    assert [t.title for t in container.tasks_repository.list_pending(b.id)] == ['B private task']


def test_tasks_repository_delete_does_not_match_completed_tasks(container):
    user = container.users_repository.get_or_create(111)
    container.tasks_repository.create(user_id=user.id, title='already done', due_at=None)
    container.tasks_repository.mark_done(user.id, 'already done')

    deleted = container.tasks_repository.delete(user.id, 'already done')
    assert deleted is None  # only matches pending; completed tasks are read-only history
    assert [t.title for t in container.tasks_repository.list_completed(user.id)] == ['already done']


# -------- TelosService.append -----------------------------------------------

def test_telos_service_append_creates_file_with_mode_600(tmp_path):
    svc = TelosService(tmp_path / 'telos')
    svc.append('abc-123', '## New section\nGoal: build Nexus.\n')

    p = svc.path_for('abc-123')
    assert p.exists()
    assert p.read_text(encoding='utf-8') == '## New section\nGoal: build Nexus.\n'
    mode = stat.S_IMODE(os.stat(p).st_mode)
    assert mode == 0o600, f'expected mode 600, got {oct(mode)}'


def test_telos_service_append_appends_to_existing_file(tmp_path):
    svc = TelosService(tmp_path / 'telos')
    p = svc.path_for('abc-123')
    p.write_text('# Telos\n', encoding='utf-8')
    os.chmod(p, 0o600)

    svc.append('abc-123', '\n## Update\nFinished V3.4.\n')
    content = p.read_text(encoding='utf-8')
    assert content == '# Telos\n\n## Update\nFinished V3.4.\n'
    mode = stat.S_IMODE(os.stat(p).st_mode)
    assert mode == 0o600


def test_telos_service_append_user_isolation(tmp_path):
    svc = TelosService(tmp_path / 'telos')
    svc.append('user-a', 'A content\n')
    svc.append('user-b', 'B content\n')
    assert svc.load('user-a') == 'A content\n'
    assert svc.load('user-b') == 'B content\n'


def test_telos_service_append_rejects_invalid_user_id(tmp_path):
    import pytest
    svc = TelosService(tmp_path / 'telos')
    with pytest.raises(ValueError):
        svc.append('../escape', 'malicious\n')
