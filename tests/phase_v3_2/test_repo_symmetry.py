from __future__ import annotations

from datetime import timedelta

from utils.dates import utc_now


def test_list_completed_returns_only_done_tasks_for_user(container):
    user = container.users_repository.get_or_create(111)
    other = container.users_repository.get_or_create(222)

    container.tasks_repository.create(user_id=user.id, title='still open', due_at=None)
    done_task = container.tasks_repository.create(user_id=user.id, title='finished one', due_at=None)
    container.tasks_repository.mark_done(user.id, 'finished')
    container.tasks_repository.create(user_id=other.id, title='other user done', due_at=None)
    container.tasks_repository.mark_done(other.id, 'other user done')

    completed = container.tasks_repository.list_completed(user.id)
    assert [t.title for t in completed] == ['finished one']
    assert all(t.status == 'done' for t in completed)
    assert all(t.user_id == user.id for t in completed)


def test_list_completed_orders_most_recent_first(container):
    user = container.users_repository.get_or_create(111)
    container.tasks_repository.create(user_id=user.id, title='first', due_at=None)
    container.tasks_repository.mark_done(user.id, 'first')
    container.tasks_repository.create(user_id=user.id, title='second', due_at=None)
    container.tasks_repository.mark_done(user.id, 'second')
    container.tasks_repository.create(user_id=user.id, title='third', due_at=None)
    container.tasks_repository.mark_done(user.id, 'third')

    completed = container.tasks_repository.list_completed(user.id)
    assert [t.title for t in completed] == ['third', 'second', 'first']


def test_list_completed_returns_empty_when_no_done_tasks(container):
    user = container.users_repository.get_or_create(111)
    container.tasks_repository.create(user_id=user.id, title='open', due_at=None)
    assert container.tasks_repository.list_completed(user.id) == []


def test_list_active_pending_for_user_returns_only_unexpired_pending(container):
    user = container.users_repository.get_or_create(111)
    other = container.users_repository.get_or_create(222)
    now = utc_now()

    active_for_user = container.approvals_repository.create(
        user_id=user.id, action_type='delete_reminder', preview_text='delete X',
        payload_json='{}', expires_at=now + timedelta(minutes=5),
    )
    expired_for_user = container.approvals_repository.create(
        user_id=user.id, action_type='delete_reminder', preview_text='old',
        payload_json='{}', expires_at=now - timedelta(minutes=1),
    )
    other_user_active = container.approvals_repository.create(
        user_id=other.id, action_type='send_email', preview_text='send Y',
        payload_json='{}', expires_at=now + timedelta(minutes=5),
    )

    listed = container.approvals_repository.list_active_pending_for_user(user.id)
    listed_ids = {a.id for a in listed}
    assert active_for_user.id in listed_ids
    assert expired_for_user.id not in listed_ids
    assert other_user_active.id not in listed_ids
    assert all(a.status == 'pending' for a in listed)
    assert all(a.user_id == user.id for a in listed)


def test_list_active_pending_for_user_returns_empty_when_none(container):
    user = container.users_repository.get_or_create(111)
    assert container.approvals_repository.list_active_pending_for_user(user.id) == []
