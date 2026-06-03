"""Phase-1 pipeline integration tests.

V3.9 deleted 3 tests from this file (test_how_are_you_returns_general_
assistant_reply, test_good_morning_routes_to_general_briefing,
test_voice_input_uses_same_pipeline_and_echoes_transcript) — all
exercised classic-path routing through `intent_classifier` +
`assistant_router`, which V3.9 deleted. Authorization-gate tests
remain because that property survives the V3 dispatcher rewrite
(unified.py still rejects non-allowed telegram_ids before any
routing). See HARDENING_PASS_V2.md H2-022 for the categorization
of the 6 deleted classic-path integration tests.
"""
from __future__ import annotations

import pytest

from pipeline.types import PipelineInput


@pytest.mark.asyncio
async def test_unauthorized_user_is_rejected(container):
    output = await container.pipeline.handle(PipelineInput(kind='text', telegram_id=999, text='hello'))
    assert 'not allowed' in output.text.lower()


@pytest.mark.asyncio
async def test_unauthorized_user_is_not_persisted(container):
    await container.pipeline.handle(PipelineInput(kind='text', telegram_id=999, text='hello'))
    assert container.users_repository.list_all() == []
