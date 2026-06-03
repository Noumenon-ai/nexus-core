from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

import pipeline.tool_dispatcher as dispatcher_module
import services.auto_write_tools as auto_write_tools_module
from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.auto_write_tools import register_auto_write_tools
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry


VOICE_PROMPT = (
    "Tell Sarah I will send someone tomorrow. "
    "Wait no. First check if Mike replied. "
    "Remind me to follow up Friday morning. "
    "No, not Sarah, unit 200 and 4. "
    "Actually this is urgent because of water damage."
)


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


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / "telos")


@pytest.fixture
def write_registry(container):
    registry = ToolRegistry()
    register_auto_write_tools(
        registry,
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        scheduler=container.scheduler,
        habit_service=container.habit_service,
        app_timezone="UTC",
    )
    return registry


@pytest.mark.asyncio
async def test_post_approval_timeout_creates_local_followup_reminder(
    container,
    write_registry,
    telos_service,
    monkeypatch,
):
    fixed_now = datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(dispatcher_module, "utc_now", lambda: fixed_now)
    monkeypatch.setattr(auto_write_tools_module, "utc_now", lambda: fixed_now)
    monkeypatch.setattr(
        dispatcher_module,
        "_POST_APPROVAL_CONTINUATION_TIMEOUT_SEC",
        0.01,
        raising=False,
    )

    user = container.users_repository.get_or_create(111)
    llm = HangingLLM()
    dispatcher = ToolDispatcher(
        llm=llm,
        registry=write_registry,
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        approvals_repository=container.approvals_repository,
        max_iterations=5,
    )

    gate_out = await dispatcher.handle(
        DispatcherInput(user=user, text=VOICE_PROMPT)
    )

    assert gate_out.metadata.get("destructive_gate") is True
    approval_id = next(
        button.callback_data.split(":", 2)[2]
        for button in gate_out.buttons
        if "approve" in button.callback_data
    )

    post_out = await dispatcher.handle(
        DispatcherInput(user=user, text=f"approval:approve:{approval_id}")
    )

    reminders = container.reminders_repository.list_active(user.id)

    assert llm.calls == 1
    assert post_out.text != (
        "Approved workflow could not continue because the provider/router timed out."
    )
    assert "couldn't finish the send path" in post_out.text.lower()
    assert "created the safe reminder part" in post_out.text.lower()
    assert "friday morning follow-up for mike" in post_out.text.lower()
    assert "water damage in unit 204" in post_out.text.lower()
    assert "no message was sent" in post_out.text.lower()
    assert len(reminders) == 1
    assert len(container.scheduler.scheduled) == 1

    reminder = reminders[0]
    assert "mike" in reminder.body.lower()
    assert "water damage" in reminder.body.lower()
    assert "unit 204" in reminder.body.lower()
    assert reminder.next_fire_at.isoformat() == "2026-05-22T09:00:00+00:00"
    assert post_out.metadata.get("post_approval_local_fallback") is True
    assert post_out.metadata.get("outbound_send_blocked") is True
