"""V3.7 source-inspection invariant: TOOL_STAGE_MESSAGES coverage.

Spec halt condition: every tool registered by
`services.dispatcher_registry.build_dispatcher_registry` MUST have an
entry in `pipeline.tool_dispatcher.TOOL_STAGE_MESSAGES`. Empty
allow-list — every tool deserves a human-readable stage rather than
falling through to the generic 'Working on it...' default.

A failure here is the test working as designed: a tool was added to
the registry without a corresponding stage string. Fix is one line.
"""
from __future__ import annotations

import pytest

from pipeline.tool_dispatcher import TOOL_STAGE_MESSAGES, _GENERIC_TOOL_STAGE
from services.dispatcher_registry import build_dispatcher_registry
from services.telos_service import TelosService


async def _async_noop_disconnect(uid):
    return None


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


@pytest.fixture
def registry(container, telos_service):
    return build_dispatcher_registry(
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        emails_repository=container.emails_repository,
        approvals_repository=container.approvals_repository,
        telos_service=telos_service,
        scheduler=container.scheduler,
        habit_service=container.habit_service,
        google_disconnect=_async_noop_disconnect,
        app_timezone='UTC',
        onboarding_repository=container.onboarding_repository,
        users_repository=container.users_repository,
    )


def test_tool_stage_messages_covers_every_registered_tool(registry):
    """Source-inspection invariant: TOOL_STAGE_MESSAGES has an entry
    for every tool name in the dispatcher registry. Drift here means
    a new tool was added without a stage string — fix by adding the
    one-liner to TOOL_STAGE_MESSAGES.
    """
    registered_names = set(registry.names())
    mapped_names = set(TOOL_STAGE_MESSAGES.keys())

    missing = registered_names - mapped_names
    assert missing == set(), (
        f'TOOL_STAGE_MESSAGES is missing entries for: {sorted(missing)}. '
        f'Add a human-readable stage for each tool in '
        f'pipeline/tool_dispatcher.py. Empty allow-list — every '
        f'registered tool deserves a non-generic stage.'
    )


def test_tool_stage_messages_has_no_orphans(registry):
    """Reverse direction: TOOL_STAGE_MESSAGES should not list tools
    that are no longer in the registry. Orphans suggest a tool was
    deleted from registration but its stage entry was forgotten —
    cosmetic, but worth catching."""
    registered_names = set(registry.names())
    mapped_names = set(TOOL_STAGE_MESSAGES.keys())

    orphans = mapped_names - registered_names
    assert orphans == set(), (
        f'TOOL_STAGE_MESSAGES has stale entries for tools no longer '
        f'in the registry: {sorted(orphans)}. Remove them.'
    )


def test_tool_stage_messages_all_values_are_non_empty_strings():
    """Sanity: no entry should be an empty string or None — that
    would silently degrade UX to a blank message."""
    for name, stage in TOOL_STAGE_MESSAGES.items():
        assert isinstance(stage, str), f'{name}: stage must be str, got {type(stage)}'
        assert stage.strip(), f'{name}: stage must be non-empty/non-whitespace'


def test_generic_fallback_stage_is_distinguishable():
    """The generic fallback must be different from any tool-specific
    stage string. Otherwise the unknown-tool fallback path is silently
    indistinguishable from the registered-tool path in user-visible
    output — a bug-hiding hazard."""
    assert _GENERIC_TOOL_STAGE not in TOOL_STAGE_MESSAGES.values(), (
        f'Generic fallback {_GENERIC_TOOL_STAGE!r} collides with a '
        f'registered tool stage. Pick a different generic string.'
    )
