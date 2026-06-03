from __future__ import annotations

import pipeline.tool_dispatcher as dispatcher_module
from services.fallback_manager import (
    PROVIDER_UNAVAILABLE_USER_TEXT,
    FallbackContext,
    FallbackManager,
)


EXPECTED_PROVIDER_UNAVAILABLE_TEXT = (
    "Having trouble connecting right now. Please try again in a moment."
)


def test_provider_unavailable_copy_is_exact_and_legacy_copy_is_absent():
    assert PROVIDER_UNAVAILABLE_USER_TEXT == EXPECTED_PROVIDER_UNAVAILABLE_TEXT
    assert dispatcher_module._GENERIC_PROVIDER_FAILURE_TEXT == EXPECTED_PROVIDER_UNAVAILABLE_TEXT
    assert dispatcher_module._GENERAL_PROVIDER_UNAVAILABLE == EXPECTED_PROVIDER_UNAVAILABLE_TEXT

    combined = " ".join(
        (
            PROVIDER_UNAVAILABLE_USER_TEXT,
            dispatcher_module._GENERIC_PROVIDER_FAILURE_TEXT,
            dispatcher_module._GENERAL_PROVIDER_UNAVAILABLE,
        )
    ).lower()
    assert "smaller step" not in combined
    assert "breaking it into smaller steps" not in combined
    assert "trouble reaching one of my providers" not in combined


def test_fallback_manager_returns_exact_provider_unavailable_copy():
    manager = FallbackManager()
    text, mode, structured_failure = manager.normalize_provider_failure(
        context=FallbackContext(
            route='tool_loop',
            stage='tool_loop',
            provider='brain_router',
            root_reason='provider_unavailable',
            raw_text='hello',
            recovered_text='hello',
        ),
        provider_text=EXPECTED_PROVIDER_UNAVAILABLE_TEXT,
        is_provider_failure_text=True,
        local_time_text=None,
        audit_guidance=None,
        vague_clarification=None,
        post_approval_resume=False,
    )

    assert text == EXPECTED_PROVIDER_UNAVAILABLE_TEXT
    assert mode == 'retry_safe'
    assert structured_failure is not None
