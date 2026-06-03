"""Tests for the deferred destructive tool stub (V3.4).

After V3.2.5.3 promoted delete_calendar_event to a real implementation,
only send_telegram_message remains as a stub here. Outbound messaging
deferred to a separate phase with explicit external-comm approval design.
"""
from __future__ import annotations

import pytest

from services.destructive_tools_stubs import (
    _PHASE_TAG,
    register_google_destructive_stubs,
    send_telegram_message,
)
from services.tool_registry import ToolRegistry, ToolResult


_DEFERRAL_PHRASE = 'not yet shipped'


def test_send_telegram_message_returns_deferral_envelope_with_announcement():
    result = send_telegram_message(user_id='u1', target='u2', text='hello')
    assert isinstance(result, ToolResult)
    assert result.success is True
    payload = result.data
    assert payload.get('deferred') is True
    assert _DEFERRAL_PHRASE in payload['message'].lower()
    assert isinstance(result.announcement, str) and result.announcement.strip()


@pytest.mark.parametrize(
    'kwargs',
    [
        {'user_id': 'u1', 'target': 't', 'text': 'x'},
        {'user_id': 'u-2', 'target': '@alice', 'text': 'long message'},
    ],
)
def test_send_telegram_uses_success_not_fail(kwargs):
    result = send_telegram_message(**kwargs)
    assert result.success is True
    assert result.error is None


def test_send_telegram_does_not_invent_destructive_outcome():
    """Counter-test: stub must NOT pretend the message was sent."""
    result = send_telegram_message(user_id='u1', target='t', text='x')
    payload = result.data
    forbidden = {'sent', 'delivered_at', 'message_id', 'sent_at'}
    assert not (forbidden & set(payload.keys())), f'stub claims outbound: {payload.keys()}'
    assert set(payload.keys()) <= {'deferred', 'message', 'phase'}


def test_register_google_destructive_stubs_only_send_telegram():
    registry = ToolRegistry()
    specs = register_google_destructive_stubs(registry)
    names = {s.name for s in specs}
    assert names == {'send_telegram_message'}
    for spec in specs:
        assert spec.requires_approval is True, f'{spec.name} must be approval-gated'
        assert spec.approval_template, f'{spec.name} missing approval_template'


def test_destructive_stub_phase_tag_matches_other_stubs():
    """All remaining stubs across V3.2 + V3.3 + V3.4 must share V3.2.5 tag."""
    from services.read_tools_stubs import _PHASE_TAG as READ_TAG
    from services.auto_write_tools_stubs import _PHASE_TAG as WRITE_TAG
    assert READ_TAG == WRITE_TAG == _PHASE_TAG == 'V3.2.5'
