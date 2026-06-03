from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.tool_dispatcher import DispatcherInput, ToolDispatcher
from services.capability_registry import CapabilityRegistry, CapabilityStatus
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry


class _FailIfCalledLLM:
    async def generate_with_tools(self, **_kwargs):
        raise AssertionError('LLM should not be called for capability-awareness prelude tests')


class _StubMem0:
    def search(self, *args, **kwargs):
        return []

    def add(self, *args, **kwargs):
        return {'results': []}


class _RecordingCapabilityRegistry:
    def __init__(self, statuses: dict[str, CapabilityStatus], status_text: str | None = None) -> None:
        self.statuses = statuses
        self.status_text = status_text or 'Capabilities:\n- rentals read: not_wired'
        self.calls: list[str] = []

    def get_capability(self, name: str, *, user=None, registry=None) -> CapabilityStatus:
        del user, registry
        self.calls.append(name)
        return self.statuses[name]

    def render_status_text(self, *, user=None, registry=None) -> str:
        del user, registry
        return self.status_text


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


def _build_dispatcher(
    container,
    telos_service,
    *,
    registry: ToolRegistry | None = None,
    capability_registry=None,
) -> ToolDispatcher:
    return ToolDispatcher(
        llm=_FailIfCalledLLM(),
        registry=registry or ToolRegistry(),
        telos_service=telos_service,
        mem0=_StubMem0(),
        approval_service=container.approval_service,
        conversation_turns_repository=container.conversation_turns_repository,
        conversation_service=container.conversation_service,
        approvals_repository=container.approvals_repository,
        capability_registry=capability_registry or CapabilityRegistry(),
        max_iterations=5,
    )


def _status(
    name: str,
    state: str,
    *,
    reason: str = '',
    service: str = 'test',
    tool_name: str = 'tool',
    safe_to_attempt: bool = False,
    manual_fix: str = '',
    details: dict | None = None,
) -> CapabilityStatus:
    return CapabilityStatus(
        name=name,
        state=state,
        reason=reason,
        service=service,
        tool_name=tool_name,
        safe_to_attempt=safe_to_attempt,
        manual_fix=manual_fix,
        details=details or {},
    )


@pytest.mark.asyncio
async def test_rentals_status_request_checks_rentals_read_first(container, telos_service):
    user = container.users_repository.get_or_create(111)
    capabilities = _RecordingCapabilityRegistry(
        {
            'rentals_read': _status(
                'rentals_read',
                'not_wired',
                reason='dashboard exists but Telegram has no rentals-read tool',
                service='nexus-dashboard.service',
                tool_name='',
                safe_to_attempt=False,
                manual_fix='Add a rentals-read tool.',
            ),
        }
    )
    dispatcher = _build_dispatcher(
        container,
        telos_service,
        capability_registry=capabilities,
    )

    out = await dispatcher.handle(
        DispatcherInput(user=user, text='Check whether my 3 rentals were updated.')
    )

    assert capabilities.calls == ['rentals_read']
    assert out.metadata.get('capability_checked') == 'rentals_read'
    assert out.metadata.get('capability_state') == 'not_wired'
    assert "isn't wired to read it yet" in out.text
    assert container.approvals_repository.list_active_pending_for_user(user.id) == []


@pytest.mark.asyncio
async def test_rentals_read_available_routes_to_local_checker_stub(container, telos_service, tmp_path):
    user = container.users_repository.get_or_create(111)
    monkey_registry = CapabilityRegistry(
        rentals_reader=lambda: [],
    )
    dispatcher = _build_dispatcher(
        container,
        telos_service,
        capability_registry=monkey_registry,
    )

    out = await dispatcher.handle(
        DispatcherInput(user=user, text='did u update my 3 rentals m')
    )

    assert out.metadata.get('capability_checked') == 'rentals_read'
    assert out.metadata.get('capability_state') == 'available'
    assert 'I can access the rentals store, but I found no rental records yet.' in out.text
    assert 'not wired' not in out.text.lower()


@pytest.mark.asyncio
async def test_explicit_whatsapp_send_checks_bridge_health_before_approval(container, telos_service):
    user = container.users_repository.get_or_create(111)
    capabilities = _RecordingCapabilityRegistry(
        {
            'whatsapp_send': _status(
                'whatsapp_send',
                'auth_required',
                reason='bridge waiting for QR',
                service='whatsapp-bridge.service',
                tool_name='services.contact_reminder_dispatcher.dispatch',
                safe_to_attempt=False,
                manual_fix='Scan the WhatsApp QR.',
            ),
        }
    )
    dispatcher = _build_dispatcher(
        container,
        telos_service,
        capability_registry=capabilities,
    )

    out = await dispatcher.handle(
        DispatcherInput(user=user, text='Send a WhatsApp message to Mike saying hello.')
    )

    assert capabilities.calls == ['whatsapp_send']
    assert out.metadata.get('capability_checked') == 'whatsapp_send'
    assert out.metadata.get('destructive_gate') is not True
    assert out.buttons == []
    assert 'bridge still needs pairing' in out.text


@pytest.mark.asyncio
async def test_calendar_request_with_missing_token_returns_auth_required(container, telos_service):
    user = container.users_repository.get_or_create(111)
    registry = ToolRegistry()
    registry.register(
        lambda *, user_id, time_min, time_max, max_results=10: {'ok': True},
        name='list_calendar_events',
        description='List calendar events.',
        parameters={'type': 'object', 'properties': {}, 'required': []},
    )
    capabilities = _RecordingCapabilityRegistry(
        {
            'calendar_read': _status(
                'calendar_read',
                'auth_required',
                reason='missing Google token',
                service='Google Calendar',
                tool_name='list_calendar_events',
                safe_to_attempt=False,
                manual_fix='Reconnect Google.',
            ),
        }
    )
    dispatcher = _build_dispatcher(
        container,
        telos_service,
        registry=registry,
        capability_registry=capabilities,
    )

    out = await dispatcher.handle(
        DispatcherInput(user=user, text="What's on my calendar tomorrow?")
    )

    assert capabilities.calls == ['calendar_read']
    assert out.metadata.get('capability_checked') == 'calendar_read'
    assert out.metadata.get('capability_state') == 'auth_required'
    assert 'Calendar access needs Google auth' in out.text


@pytest.mark.asyncio
async def test_status_capabilities_lists_states(container, telos_service):
    user = container.users_repository.get_or_create(111)
    capabilities = _RecordingCapabilityRegistry(
        statuses={},
        status_text=(
            'Capabilities:\n'
            '- reminders read: available\n'
            '- whatsapp send: auth_required\n'
            '- rentals read: not_wired'
        ),
    )
    dispatcher = _build_dispatcher(
        container,
        telos_service,
        capability_registry=capabilities,
    )

    out = await dispatcher.handle(DispatcherInput(user=user, text='/status capabilities'))

    assert out.metadata.get('capability_status') is True
    assert out.text.startswith('Capabilities:\n')
    assert '- reminders read: available' in out.text
    assert '- whatsapp send: auth_required' in out.text


@pytest.mark.asyncio
async def test_capability_response_does_not_claim_success_without_backing_tool(container, telos_service):
    user = container.users_repository.get_or_create(111)
    capabilities = _RecordingCapabilityRegistry(
        {
            'rentals_read': _status(
                'rentals_read',
                'not_wired',
                reason='dashboard exists but Telegram has no rentals-read tool',
                service='nexus-dashboard.service',
                tool_name='',
                safe_to_attempt=False,
                manual_fix='Add a rentals-read tool.',
            ),
        }
    )
    dispatcher = _build_dispatcher(
        container,
        telos_service,
        capability_registry=capabilities,
    )

    out = await dispatcher.handle(
        DispatcherInput(user=user, text='did u update my 3 rentals m')
    )

    assert "I'll check what I can see." not in out.text
    assert "isn't wired to read it yet" in out.text


