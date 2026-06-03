"""V3.9 unified-pipeline integration: dispatcher-only routing.

V3.5 originally shipped this file to verify the feature-flagged
dispatcher entry. V3.9 deleted intent_classifier + assistant_router
and the NEXUS_DISPATCHER_ENABLED flag became a no-op tombstone, so
the original four tests no longer make sense in their original form:
- `test_flag_off_uses_existing_intent_classifier_path` was deleted
  outright (no classic path to fall back to).
- `test_flag_on_routes_to_dispatcher_and_skips_classifier` rewritten
  as `test_dispatcher_handles_authorized_message` (no flag, only path).
- `test_flag_on_blocks_unauthorized_telegram_id` rewritten as
  `test_dispatcher_path_blocks_unauthorized_telegram_id` (still the
  same auth-before-dispatcher property; flag references removed).
- `test_flag_on_dispatcher_none_falls_back_to_classic` rewritten as
  `test_dispatcher_none_returns_clarify_generic_sentinel` (no classic
  fallback path; instead returns a benign clarify message).

Together with `tests/phase_v3_9/test_classic_path_deleted.py`, these
3 tests pin the V3.9 invariant: there is only one message-handling
path, and it goes through ToolDispatcher.
"""
from __future__ import annotations

from typing import Any

import pytest

from pipeline.tool_dispatcher import DispatcherOutput
from pipeline.types import PipelineInput
from pipeline.unified import UnifiedPipeline


class RecordingDispatcher:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def handle(self, input_data):
        self.calls.append({'user_id': input_data.user.id, 'text': input_data.text})
        return DispatcherOutput(text='dispatcher reply', iterations=1)


def _make_pipeline(container, *, dispatcher=None) -> UnifiedPipeline:
    return UnifiedPipeline(
        users_repository=container.users_repository,
        conversation_service=container.conversation_service,
        voice_input_service=None,
        voice_output_service=None,
        allowed_telegram_ids=container.settings.core.allowed_telegram_ids,
        tool_dispatcher=dispatcher,
    )


@pytest.mark.asyncio
async def test_dispatcher_handles_authorized_message(container):
    """The dispatcher is invoked unconditionally for every authorized
    message. No flag check, no classic-path branch."""
    dispatcher = RecordingDispatcher()
    pipeline = _make_pipeline(container, dispatcher=dispatcher)

    inp = PipelineInput(kind='text', text='whats my reminders', telegram_id=111, username=None, full_name=None)
    out = await pipeline.handle(inp)

    assert len(dispatcher.calls) == 1
    assert dispatcher.calls[0]['text'] == 'whats my reminders'
    assert out.text == 'dispatcher reply'


@pytest.mark.asyncio
async def test_dispatcher_path_blocks_unauthorized_telegram_id(container):
    """Authorization gate runs BEFORE the dispatcher: a non-allowed
    telegram_id never reaches it. This property is unchanged from V3.5
    — the unauth check fires before any routing."""
    dispatcher = RecordingDispatcher()
    pipeline = _make_pipeline(container, dispatcher=dispatcher)

    inp = PipelineInput(kind='text', text='hi', telegram_id=999, username=None, full_name=None)
    out = await pipeline.handle(inp)

    assert dispatcher.calls == []
    assert 'not allowed' in out.text.lower() or 'unauthorized' in out.text.lower()


@pytest.mark.asyncio
async def test_dispatcher_none_returns_clarify_generic_sentinel(container):
    """V3.9 deletion removed the classic fallback. If the dispatcher
    fails to construct at boot (LLM creds missing, mem0 unavailable,
    Google not connected), `tool_dispatcher` is None. Authorized
    messages must NOT crash; they get a benign clarify-generic reply
    so the user sees text rather than silence.
    """
    pipeline = _make_pipeline(container, dispatcher=None)
    inp = PipelineInput(kind='text', text='hi', telegram_id=111, username=None, full_name=None)
    out = await pipeline.handle(inp)

    assert isinstance(out.text, str)
    assert out.text  # non-empty
    # Sentinel should NOT include the unauthorized phrase (this user
    # IS authorized; the issue is dispatcher boot failure, not auth).
    assert 'not allowed' not in out.text.lower()
    assert 'unauthorized' not in out.text.lower()
