from __future__ import annotations

from datetime import timedelta

from models import now_utc
from utils.i18n import Translator


def test_delete_memory_requests_approval(container):
    user = container.users_repository.get_or_create(111)
    response = container.approval_service.request(user, action_type='delete_memory', preview_text='Delete memory x', payload={'user_id': user.id, 'key': 'x'}, translator=Translator())
    assert 'approval needed' in response.text.lower()
    assert len(response.buttons) == 2


def test_expired_approval_cannot_execute(container):
    user = container.users_repository.get_or_create(111)
    approval = container.approvals_repository.create(user_id=user.id, action_type='delete_memory', preview_text='Delete memory x', payload_json='{}', expires_at=now_utc() - timedelta(minutes=1))
    response = container.approval_service.execute(approval.id, user.id, lambda action_type, payload: None, Translator())
    assert 'expired' in response.text.lower()


def test_approval_sweep_marks_pending_rows_expired(container):
    user = container.users_repository.get_or_create(111)
    approval = container.approvals_repository.create(user_id=user.id, action_type='delete_memory', preview_text='Delete memory x', payload_json='{}', expires_at=now_utc() - timedelta(minutes=1))
    notified: list[tuple[str, str]] = []
    count = container.approval_service.sweep_expired(lambda user_id, text: notified.append((user_id, text)), Translator())
    updated = container.approvals_repository.get(approval.id)
    assert count == 1
    assert updated.status == 'expired'
    assert notified


def test_cross_user_approval_execution_is_rejected(container):
    user_a = container.users_repository.get_or_create(111)
    user_b = container.users_repository.get_or_create(222)
    approval = container.approvals_repository.create(user_id=user_a.id, action_type='delete_memory', preview_text='Delete memory x', payload_json='{}', expires_at=now_utc() + timedelta(minutes=10))
    response = container.approval_service.execute(approval.id, user_b.id, lambda action_type, payload: None, Translator())
    assert 'not allowed' in response.text.lower()


def test_approved_memory_delete_executes(container):
    user = container.users_repository.get_or_create(111)
    container.memories_repository.upsert(user_id=user.id, memory_type='fact', key='secret_plan', value='plan', confidence=1.0, source='explicit')
    approval = container.approvals_repository.create(user_id=user.id, action_type='delete_memory', preview_text='Delete secret_plan', payload_json=f'{{"user_id":"{user.id}","key":"secret_plan"}}', expires_at=now_utc() + timedelta(minutes=10))
    response = container.approval_service.execute(approval.id, user.id, lambda action_type, payload: container.action_service.execute(action_type, payload, Translator()), Translator())
    assert 'removed' in response.text.lower()
    assert container.memories_repository.get(user_id=user.id, memory_type='fact', key='secret_plan') is None
