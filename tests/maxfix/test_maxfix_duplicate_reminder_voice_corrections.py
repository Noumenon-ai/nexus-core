from __future__ import annotations

from datetime import timedelta
from typing import Any
import pytest

from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.auto_write_tools import register_auto_write_tools
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry
from utils.dates import utc_now


VOICE_PROMPT = (
    "Tell Sarah I'll send someone tomorrow. Actually no, ask if Thursday works. "
    "Wait no, first check if Mike replied. Remind me to follow up Friday morning. "
    "No not Sarah, unit 204. Actually this is urgent because of water damage."
)


class ScriptedLLM:
    def __init__(self, script: list[dict[str, Any]]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def generate_with_tools(
        self,
        *,
        user_id: str,
        system_prompt: str,
        contents: list[dict[str, Any]],
        tool_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append({
            "user_id": user_id,
            "system_prompt": system_prompt,
            "contents": contents,
            "tool_catalog": tool_catalog,
        })
        if not self.script:
            raise AssertionError("ScriptedLLM exhausted")
        return self.script.pop(0)


class StubMem0:
    def search(self, query: str, *, user_id, limit: int = 5):
        return []

    def add(self, messages, *, user_id):
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


def _user(container, telegram_id: int):
    return container.users_repository.get_or_create(telegram_id)


@pytest.mark.asyncio
async def test_multi_correction_voice_flow_creates_one_followup_reminder(
    container,
    write_registry,
    telos_service,
    monkeypatch,
):
    user = _user(container, 111)
    when = (utc_now() + timedelta(days=2)).replace(second=0, microsecond=0).isoformat()
    latest_body = "Follow up with Mike about urgent water damage in units 200 and 4."
    llm = ScriptedLLM([
        {
            "tool_calls": [
                {
                    "name": "create_reminder",
                    "arguments": {
                        "body": "Follow up with Mike about urgent water damage for units 200 and 4.",
                        "next_fire_at": when,
                    },
                },
                {
                    "name": "create_reminder",
                    "arguments": {
                        "body": "Follow up with Mike about the urgent water damage in units 200 and 4.",
                        "next_fire_at": when,
                    },
                },
                {
                    "name": "create_reminder",
                    "arguments": {
                        "body": latest_body,
                        "next_fire_at": when,
                    },
                },
            ]
        },
        {
            "tool_calls": [
                {
                    "name": "create_reminder",
                    "arguments": {
                        "body": latest_body,
                        "next_fire_at": when,
                    },
                }
            ]
        },
        {"text": "I created the Friday follow-up reminder."},
    ])
    dispatcher = ToolDispatcher(
        llm=llm,
        registry=write_registry,
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        max_iterations=10,
    )

    async def _archive_stub(*args, **kwargs):
        return None

    monkeypatch.setattr(ToolDispatcher, "_archive_memory_async", _archive_stub)

    out = await dispatcher.handle(DispatcherInput(user=user, text=VOICE_PROMPT))
    await dispatcher.wait_for_archival_idle()

    reminders = container.reminders_repository.list_active(user.id)
    assert out.text == "I created the Friday follow-up reminder."
    assert len(reminders) == 1
    assert reminders[0].body == latest_body
    assert reminders[0].next_fire_at.isoformat() == when
    assert len(container.scheduler.scheduled) == 1

    serialized_second_iteration = repr(llm.calls[2]["contents"])
    assert "deduplicated" in serialized_second_iteration.lower()
