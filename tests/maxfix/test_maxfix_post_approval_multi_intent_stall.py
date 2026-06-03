from __future__ import annotations

import asyncio
from typing import Any

import pytest

import pipeline.tool_dispatcher as dispatcher_module
from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry


VOICE_STYLE_PROMPT = """Tell Sarah I'll send someone tomorrow—
Actually no, ask if Thursday works.
Wait no, first check if Mike replied.
Remind me to follow up Friday morning.
No not Sarah, unit 204.
Actually this is urgent because of water damage.
"""

def _last_user_text(contents: list[dict[str, Any]]) -> str:
    for item in reversed(contents):
        if item.get("role") != "user":
            continue
        parts = item.get("parts") or []
        for part in reversed(parts):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return ""


class HangingLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.last_user_text = ""

    async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
        self.last_user_text = _last_user_text(contents)
        self.calls.append(
            {
                "user_id": user_id,
                "system_prompt": system_prompt,
                "contents": contents,
                "tool_catalog": tool_catalog,
            }
        )
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class ClarifyingLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.last_user_text = ""

    async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
        self.last_user_text = _last_user_text(contents)
        self.calls.append(
            {
                "user_id": user_id,
                "system_prompt": system_prompt,
                "contents": contents,
                "tool_catalog": tool_catalog,
            }
        )
        return {
            "text": (
                "I still need one clarification:\n"
                "Should I draft the message for Unit 204, or only create the "
                "Friday morning follow-up and urgent water-damage task?"
            )
        }


class StubMem0:
    def search(self, *args, **kwargs):
        return []

    def add(self, *args, **kwargs):
        return None


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / "telos")


def _build_dispatcher(container, telos_service, llm):
    return ToolDispatcher(
        llm=llm,
        registry=ToolRegistry(),
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        approvals_repository=container.approvals_repository,
        max_iterations=5,
    )


@pytest.mark.asyncio
async def test_post_approval_multi_intent_hang_returns_controlled_timeout(
    container, telos_service, monkeypatch,
):
    user = container.users_repository.get_or_create(111)
    monkeypatch.setattr(dispatcher_module, "build_vague_clarification", lambda _text: None)
    monkeypatch.setattr(
        dispatcher_module,
        "_POST_APPROVAL_CONTINUATION_TIMEOUT_SEC",
        0.01,
        raising=False,
    )
    llm = HangingLLM()
    dispatcher = _build_dispatcher(container, telos_service, llm)

    gate_out = await dispatcher.handle(
        DispatcherInput(user=user, text=VOICE_STYLE_PROMPT)
    )

    assert gate_out.metadata.get("destructive_gate") is True
    approve_button = next(
        button for button in gate_out.buttons
        if "approve" in button.callback_data
    )
    approval_id = approve_button.callback_data.split(":", 2)[2]

    post_out = await asyncio.wait_for(
        dispatcher.handle(
            DispatcherInput(user=user, text=f"approval:approve:{approval_id}")
        ),
        timeout=0.2,
    )

    assert "send path" in post_out.text.lower()
    assert "did not send anything" in post_out.text.lower()
    assert post_out.metadata.get("post_approval_timeout") is True
    assert (post_out.metadata.get("structured_failure") or {}).get("route") == "contact_send"
    assert len(llm.calls) == 1
    assert "no not sarah, unit 204." in llm.last_user_text.lower()
    assert "water damage" in llm.last_user_text.lower()
    assert "friday morning" in llm.last_user_text.lower()
    assert "check if mike replied" in llm.last_user_text.lower()


@pytest.mark.asyncio
async def test_post_approval_multi_intent_resume_preserves_latest_state(
    container, telos_service, monkeypatch,
):
    user = container.users_repository.get_or_create(111)
    monkeypatch.setattr(dispatcher_module, "build_vague_clarification", lambda _text: None)
    llm = ClarifyingLLM()
    dispatcher = _build_dispatcher(container, telos_service, llm)

    gate_out = await dispatcher.handle(
        DispatcherInput(user=user, text=VOICE_STYLE_PROMPT)
    )
    approval_id = next(
        button.callback_data.split(":", 2)[2]
        for button in gate_out.buttons
        if "approve" in button.callback_data
    )

    post_out = await dispatcher.handle(
        DispatcherInput(user=user, text=f"approval:approve:{approval_id}")
    )

    assert "unit 204" in post_out.text.lower()
    assert "friday morning" in post_out.text.lower()
    assert "water-damage" in post_out.text.lower()
    assert "no not sarah, unit 204." in llm.last_user_text.lower()
    assert "water damage" in llm.last_user_text.lower()
    assert "friday morning" in llm.last_user_text.lower()
