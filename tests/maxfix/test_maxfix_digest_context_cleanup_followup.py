from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

import pipeline.tool_dispatcher as dispatcher_module
import services.task_service as task_service_module
from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.dispatcher_registry import build_dispatcher_registry
from services.telos_service import TelosService
from utils.i18n import Translator


_TIMEOUT_TEXT = (
    "Approved workflow could not continue because the provider/router timed out."
)
_PROMPT = "I already did it, remove it"


class HangingLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_with_tools(
        self,
        *,
        user_id: str,
        system_prompt: str,
        contents: list[dict[str, Any]],
        tool_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.calls += 1
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class StubMem0:
    def search(self, *args, **kwargs):
        return []

    def add(self, *args, **kwargs):
        return None


async def _async_noop_disconnect(_user_id: str) -> None:
    return None


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / "telos")


@pytest.fixture
def registry(container, telos_service):
    return build_dispatcher_registry(
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        emails_repository=container.emails_repository,
        approvals_repository=container.approvals_repository,
        telos_service=telos_service,
        scheduler=container.scheduler,
        habit_service=container.habit_service,
        google_disconnect=_async_noop_disconnect,
        app_timezone="America/New_York",
        onboarding_repository=container.onboarding_repository,
        users_repository=container.users_repository,
    )


def _approve_button_id(buttons) -> str:
    return next(
        button.callback_data.split(":", 2)[2]
        for button in buttons
        if "approve" in button.callback_data
    )


def _make_dispatcher(
    container,
    registry,
    telos_service,
) -> tuple[ToolDispatcher, HangingLLM]:
    llm = HangingLLM()
    dispatcher = ToolDispatcher(
        llm=llm,
        registry=registry,
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        approvals_repository=container.approvals_repository,
        proactive_notifications_repository=container.proactive_repository,
        max_iterations=5,
        app_timezone="America/New_York",
    )
    return dispatcher, llm


def _force_timeout(monkeypatch, *, fixed_now: datetime) -> None:
    monkeypatch.setattr(dispatcher_module, "utc_now", lambda: fixed_now)
    monkeypatch.setattr(task_service_module, "utc_now", lambda: fixed_now)
    monkeypatch.setattr(
        dispatcher_module,
        "_POST_APPROVAL_CONTINUATION_TIMEOUT_SEC",
        0.01,
        raising=False,
    )


async def _record_morning_digest(container, user) -> str:
    response = await container.proactive_service.morning_briefing(
        user, Translator(), explicit=False
    )
    assert response.metadata.get("should_send") is True
    return response.text


@pytest.mark.asyncio
async def test_digest_context_followup_marks_single_digest_task_done(
    container,
    registry,
    telos_service,
    monkeypatch,
):
    fixed_now = datetime(2026, 5, 24, 14, 0, tzinfo=timezone.utc)
    _force_timeout(monkeypatch, fixed_now=fixed_now)

    user = container.users_repository.get_or_create(111)
    task = container.tasks_repository.create(
        user_id=user.id,
        title="Complete NEXUS audit",
        due_at=fixed_now - timedelta(days=1),
        priority=2,
        source="user",
    )
    digest_text = await _record_morning_digest(container, user)
    assert "Top tasks:" in digest_text
    assert "Overdue: Complete NEXUS audit" in digest_text

    unrelated = container.tasks_repository.create(
        user_id=user.id,
        title="Pay HOA invoice",
        due_at=fixed_now + timedelta(days=2),
        priority=1,
        source="user",
    )
    dispatcher, llm = _make_dispatcher(container, registry, telos_service)

    gate_out = await dispatcher.handle(DispatcherInput(user=user, text=_PROMPT))
    approval_id = _approve_button_id(gate_out.buttons)
    post_out = await dispatcher.handle(
        DispatcherInput(user=user, text=f"approval:approve:{approval_id}")
    )

    assert llm.calls == 1
    assert post_out.metadata.get("post_approval_timeout") is True
    assert post_out.text != _TIMEOUT_TEXT
    assert "complete nexus audit" in post_out.text.lower()
    assert task.id == container.tasks_repository.list_completed(user.id)[0].id
    assert {item.id for item in container.tasks_repository.list_pending(user.id)} == {
        unrelated.id
    }


@pytest.mark.asyncio
async def test_digest_context_followup_multiple_digest_items_asks_for_clarification(
    container,
    registry,
    telos_service,
    monkeypatch,
):
    fixed_now = datetime(2026, 5, 24, 14, 0, tzinfo=timezone.utc)
    _force_timeout(monkeypatch, fixed_now=fixed_now)

    user = container.users_repository.get_or_create(111)
    task = container.tasks_repository.create(
        user_id=user.id,
        title="Complete NEXUS audit",
        due_at=fixed_now - timedelta(days=1),
        priority=2,
        source="user",
    )
    reminder = container.reminders_repository.create(
        user_id=user.id,
        body="Call bank",
        next_fire_at=fixed_now + timedelta(minutes=30),
        recurrence=None,
    )
    digest_text = await _record_morning_digest(container, user)
    assert "Top tasks:" in digest_text
    assert "Overdue: Complete NEXUS audit" in digest_text
    assert "Call bank" in digest_text

    dispatcher, llm = _make_dispatcher(container, registry, telos_service)

    gate_out = await dispatcher.handle(DispatcherInput(user=user, text=_PROMPT))
    approval_id = _approve_button_id(gate_out.buttons)
    post_out = await dispatcher.handle(
        DispatcherInput(user=user, text=f"approval:approve:{approval_id}")
    )

    assert llm.calls == 1
    assert post_out.metadata.get("post_approval_timeout") is True
    assert post_out.text != _TIMEOUT_TEXT
    assert "what \"it\" means" in post_out.text.lower() or "which one" in post_out.text.lower()
    assert {item.id for item in container.tasks_repository.list_pending(user.id)} == {task.id}
    assert {item.id for item in container.reminders_repository.list_active(user.id)} == {
        reminder.id
    }
    assert container.tasks_repository.list_completed(user.id) == []


@pytest.mark.asyncio
async def test_digest_context_followup_without_recent_digest_asks_for_target(
    container,
    registry,
    telos_service,
    monkeypatch,
):
    fixed_now = datetime(2026, 5, 24, 14, 0, tzinfo=timezone.utc)
    _force_timeout(monkeypatch, fixed_now=fixed_now)

    user = container.users_repository.get_or_create(111)
    task = container.tasks_repository.create(
        user_id=user.id,
        title="Complete NEXUS audit",
        due_at=fixed_now - timedelta(days=1),
        priority=2,
        source="user",
    )
    unrelated = container.tasks_repository.create(
        user_id=user.id,
        title="Pay HOA invoice",
        due_at=fixed_now + timedelta(days=2),
        priority=1,
        source="user",
    )
    dispatcher, llm = _make_dispatcher(container, registry, telos_service)

    gate_out = await dispatcher.handle(DispatcherInput(user=user, text=_PROMPT))
    approval_id = _approve_button_id(gate_out.buttons)
    post_out = await dispatcher.handle(
        DispatcherInput(user=user, text=f"approval:approve:{approval_id}")
    )

    assert llm.calls == 1
    assert post_out.metadata.get("post_approval_timeout") is True
    assert post_out.text != _TIMEOUT_TEXT
    assert "what \"it\" means" in post_out.text.lower()
    assert {item.id for item in container.tasks_repository.list_pending(user.id)} == {
        task.id,
        unrelated.id,
    }
    assert container.tasks_repository.list_completed(user.id) == []
