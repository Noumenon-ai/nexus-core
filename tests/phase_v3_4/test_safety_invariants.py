"""V3.4 safety-invariant structural tests.

CRITICAL invariant: every tool name matching delete_/forget_/send_/disconnect_
MUST have requires_approval=True. This is the safety boundary that
prevents future @tool functions from accidentally bypassing approval.

SECOND structural test: every approval-gated tool must have either an
explicit approval_template OR a sensible default preview generated from
tool name + args. Complements the V3.3 announcement contract.

Both tests scan the union of all four V3 tool registries (reads, writes,
destructive, plus all stubs) so any new tool added in V3.5+ is checked
without further test-file changes.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from services.auto_write_tools import register_auto_write_tools
from services.auto_write_tools_stubs import register_google_write_stubs
from services.destructive_tools import register_destructive_tools
from services.destructive_tools_stubs import register_google_destructive_stubs
from services.read_tools import register_read_tools
from services.read_tools_stubs import register_google_stubs
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry, render_approval_preview
from utils.dates import utc_now


_DESTRUCTIVE_PREFIXES = ('delete_', 'forget_', 'send_', 'disconnect_')


async def _async_noop_disconnect(uid):
    """V3.5: disconnect_google is async at source. Tests inject async."""
    return None


@pytest.fixture
def telos_service(tmp_path):
    return TelosService(tmp_path / 'telos')


@pytest.fixture
def all_tools_registry(container, telos_service):
    """One shared registry holding every V3.2/V3.3/V3.4 tool — real and
    stubbed. Future V3 tools registered alongside these will be scanned by
    the same invariants without test-file edits."""
    registry = ToolRegistry()
    register_read_tools(
        registry,
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        emails_repository=container.emails_repository,
        approvals_repository=container.approvals_repository,
        telos_service=telos_service,
        app_timezone='UTC',
    )
    register_google_stubs(registry)
    register_auto_write_tools(
        registry,
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        scheduler=container.scheduler,
        habit_service=container.habit_service,
        app_timezone='UTC',
    )
    register_google_write_stubs(registry)
    register_destructive_tools(
        registry,
        reminders_repository=container.reminders_repository,
        tasks_repository=container.tasks_repository,
        memories_repository=container.memories_repository,
        telos_service=telos_service,
        google_disconnect=_async_noop_disconnect,
        scheduler=container.scheduler,
    )
    register_google_destructive_stubs(registry)
    return registry


# CRITICAL invariant — never relax without explicit user sign-off.
def test_every_destructive_named_tool_has_requires_approval(all_tools_registry):
    """SAFETY BOUNDARY: any tool whose name matches the destructive prefix
    set (delete_/forget_/send_/disconnect_) MUST have requires_approval=True.
    A future tool that accidentally registers without approval gating fails
    this test immediately — defense against silent regression."""
    violations = []
    for spec in all_tools_registry.all():
        if any(spec.name.startswith(p) for p in _DESTRUCTIVE_PREFIXES):
            if not spec.requires_approval:
                violations.append(spec.name)
    assert violations == [], (
        f'V3.4 INVARIANT VIOLATION: destructive-named tools missing '
        f'requires_approval=True: {violations}'
    )


# Approval-template contract.
def test_every_approval_gated_tool_has_approval_template_or_default(all_tools_registry):
    """For every spec with requires_approval=True, render_approval_preview()
    must produce a non-empty string for representative happy-path args.
    Source-inspection style: introspects the registry, doesn't enumerate
    tools. New approval-gated tools are auto-checked."""
    sample_args_by_name = {
        'delete_reminder': {'user_id': 'u', 'query': 'meds'},
        'delete_task': {'user_id': 'u', 'query': 'gym'},
        'forget_user_memory': {'user_id': 'u', 'key': 'reminder_time_preference'},
        'disconnect_google': {'user_id': 'u'},
        'append_telos_update': {'user_id': 'u', 'content': '## new\n'},
        'delete_calendar_event': {'user_id': 'u', 'event_id': 'evt-42'},
        'send_telegram_message': {'user_id': 'u', 'target': 'someone', 'text': 'hi'},
    }
    missing = []
    for spec in all_tools_registry.all():
        if not spec.requires_approval:
            continue
        sample = sample_args_by_name.get(spec.name, {'user_id': 'u'})
        preview = render_approval_preview(spec, sample)
        if not (isinstance(preview, str) and preview.strip()):
            missing.append((spec.name, preview))
    assert missing == [], f'Approval-gated tools producing empty preview: {missing}'


def test_every_approval_gated_tool_template_references_at_least_one_arg(all_tools_registry):
    """Soft invariant: explicit approval_templates should reference at
    least one runtime arg so the user sees what is being approved. A
    template with no placeholder is allowed but flagged here so it is
    intentional. disconnect_google has no args other than user_id and
    is exempt."""
    EXEMPT = {'disconnect_google'}
    weak = []
    for spec in all_tools_registry.all():
        if not spec.requires_approval or spec.name in EXEMPT:
            continue
        if spec.approval_template and '{' not in spec.approval_template:
            weak.append(spec.name)
    assert weak == [], (
        f'These approval-gated tools have approval_template without any '
        f'placeholder, leaving the user with a generic preview: {weak}. '
        f'Either add a placeholder or add to EXEMPT.'
    )


# Counter-test: full registry sanity — every spec has fn callable.
def test_every_registered_tool_function_is_callable(all_tools_registry):
    for spec in all_tools_registry.all():
        assert callable(spec.fn), f'{spec.name}: fn is not callable'
