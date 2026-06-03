"""Deferred Telegram destructive-tool stub — outbound message send.

Calendar event deletion was promoted to a real implementation in V3.2.5.3
and now lives in services/destructive_tools.py. The remaining
send_telegram_message stub is tracked under H2-011 (separate Sprint
target — outbound messaging needs explicit external-comm approval design).
"""
from __future__ import annotations

from typing import Any

from services.tool_registry import ToolRegistry, ToolResult, ToolSpec


_PHASE_TAG = 'V3.2.5'

DEFERRAL_TELEGRAM_SEND = (
    'Outbound Telegram message sending from a tool call is not yet shipped. '
    'The inbound bot exists; the dispatcher-driven outbound DM path is '
    f'queued for a future phase ({_PHASE_TAG}). Tell the user this honestly '
    'if they ask to send a Telegram message on their behalf.'
)

_ANNOUNCE_TELEGRAM_SEND = 'Outbound Telegram send not yet shipped — outbound DM path queued for a future phase.'


def _deferred(message: str, announcement: str) -> ToolResult:
    return ToolResult.ok(
        data={'deferred': True, 'message': message, 'phase': _PHASE_TAG},
        announcement=announcement,
    )


def send_telegram_message(*, user_id: str, target: str, text: str) -> ToolResult:
    """Send a Telegram message to a target chat or contact on behalf of the user."""
    return _deferred(DEFERRAL_TELEGRAM_SEND, _ANNOUNCE_TELEGRAM_SEND)


_PARAMS_TELEGRAM_SEND: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'target': {'type': 'string', 'description': 'Telegram chat id or contact handle.'},
        'text': {'type': 'string', 'description': 'Message text to send.'},
    },
    'required': ['target', 'text'],
}


def register_google_destructive_stubs(registry: ToolRegistry) -> list[ToolSpec]:
    """Register the deferred destructive tool stub (send_telegram_message)."""
    return [
        registry.register(
            send_telegram_message,
            name='send_telegram_message',
            description='Send a Telegram message on behalf of the user. Currently deferred (returns deferral notice).',
            parameters=_PARAMS_TELEGRAM_SEND,
            requires_approval=True,
            approval_template='Send Telegram message to {target}: {text}',
        ),
    ]
