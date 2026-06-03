from __future__ import annotations

import re

import pytest

import pipeline.tool_dispatcher as dispatcher_module
from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry, ToolResult


GENERIC_PROVIDER_FAILURE = (
    "Having trouble connecting right now. Please try again in a moment."
)

VOICE_STYLE_PROMPT = """Tell Sarah I'll send someone tomorrow—
Actually no, ask if Thursday works.
Wait no, first check if Mike replied.
Remind me to follow up Friday morning.
No not Sarah, unit 204.
Actually this is urgent because of water damage.
"""


class ProviderFailureLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_with_tools(self, *, user_id, system_prompt, contents, tool_catalog):
        self.calls += 1
        return {"text": GENERIC_PROVIDER_FAILURE}


class StubMem0:
    def search(self, *args, **kwargs):
        return []

    def add(self, *args, **kwargs):
        return None


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / "telos")


def _read_registry(container, telos_service) -> ToolRegistry:
    del container, telos_service
    registry = ToolRegistry()

    def get_current_time() -> ToolResult:
        return ToolResult.ok(data={"iso": "2026-05-20T12:34:56+00:00", "timezone": "UTC"})

    registry.register(
        get_current_time,
        name="get_current_time",
        description="Return the current time as ISO 8601 in the user app timezone.",
    )
    return registry


def _dispatcher(container, telos_service, *, registry: ToolRegistry | None = None) -> ToolDispatcher:
    return ToolDispatcher(
        llm=ProviderFailureLLM(),
        registry=registry or ToolRegistry(),
        telos_service=telos_service,
        mem0=StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        approvals_repository=container.approvals_repository,
        max_iterations=5,
    )


@pytest.mark.asyncio
async def test_provider_error_routes_self_audit_to_local_audit_guidance(
    container, telos_service,
):
    user = container.users_repository.get_or_create(111)
    dispatcher = _dispatcher(container, telos_service)

    out = await dispatcher.handle(
        DispatcherInput(user=user, text="Run a self-audit on Nexus.")
    )

    assert out.text != GENERIC_PROVIDER_FAILURE
    assert "audit" in out.text.lower()
    assert "chaos_audit_runner --quick" in out.text
    assert "utility_audit_runner --quick" in out.text


@pytest.mark.asyncio
async def test_provider_error_on_approved_workflow_becomes_controlled_failure(
    container, telos_service,
):
    user = container.users_repository.get_or_create(111)
    dispatcher = _dispatcher(container, telos_service)

    gate_out = await dispatcher.handle(
        DispatcherInput(user=user, text="Send the lease renewal to unit 204 now.")
    )
    assert gate_out.metadata.get("destructive_gate") is True
    approval_id = next(
        button.callback_data.split(":", 2)[2]
        for button in gate_out.buttons
        if "approve" in button.callback_data
    )

    out = await dispatcher.handle(
        DispatcherInput(user=user, text=f"approval:approve:{approval_id}")
    )

    assert out.text != GENERIC_PROVIDER_FAILURE
    assert "send path" in out.text.lower()
    assert "provider" in out.text.lower()
    assert "did not send anything" in out.text.lower()
    assert (out.metadata.get("structured_failure") or {}).get("root_reason") == "provider_unavailable"


@pytest.mark.asyncio
async def test_provider_error_on_messy_multi_intent_becomes_clarification(
    container, telos_service, monkeypatch,
):
    user = container.users_repository.get_or_create(111)
    dispatcher = _dispatcher(container, telos_service)
    monkeypatch.setattr(dispatcher_module, "build_vague_clarification", lambda _text: None)

    out = await dispatcher.handle(
        DispatcherInput(
            user=user,
            text=VOICE_STYLE_PROMPT,
            bypass_destructive_approval=True,
        )
    )

    assert out.text != GENERIC_PROVIDER_FAILURE
    assert "which tenant or unit" in out.text.lower()
    assert "what item or case" in out.text.lower()


@pytest.mark.asyncio
async def test_provider_error_on_local_time_command_routes_locally(
    container, telos_service,
):
    user = container.users_repository.get_or_create(111)
    dispatcher = _dispatcher(
        container,
        telos_service,
        registry=_read_registry(container, telos_service),
    )

    out = await dispatcher.handle(
        DispatcherInput(user=user, text="what time is it")
    )

    assert out.text != GENERIC_PROVIDER_FAILURE
    assert out.text.lower().startswith("current time:")
    assert re.search(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}", out.text.lower())
