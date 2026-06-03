"""H2-048 — destructive_intent_classifier must surface the specific
"Update rental unit?" template for update_unit prompts.

Pre-H2-048 the H2-047 update_unit MCP tool shipped with the right
@mcp.tool(meta={"approval_template": …}) decorator, but the operational
gate (DESTRUCTIVE_TOOL_REGISTRY) had no entry — so "update unit 1
lease_end to 2027-03-09" fell through to the generic-verb path with
the meaningless "This may modify or send data" template.
"""
from __future__ import annotations

import pytest

from services.destructive_intent_classifier import (
    DESTRUCTIVE_TOOL_REGISTRY,
    classify,
)


def test_update_unit_is_registered_in_destructive_registry():
    """Symbolic check — the registry must carry an update_unit entry
    or every "update unit X" prompt falls through to generic."""
    assert 'update_unit' in DESTRUCTIVE_TOOL_REGISTRY
    actions, targets, template = DESTRUCTIVE_TOOL_REGISTRY['update_unit']
    assert 'update' in actions
    assert 'unit' in targets
    assert 'Update' in template  # "Update rental unit?"


@pytest.mark.parametrize('prompt', [
    'update unit 1 lease_end to 2027-03-09',
    'change unit 2 rent to 2200',
    'modify unit 3 tenant_name',
    'edit the lease on unit 1',
])
def test_update_unit_prompts_route_to_specific_template(prompt):
    intent = classify(prompt)
    assert intent.is_destructive is True
    assert 'update_unit' in intent.matched_tools, (
        f'prompt={prompt!r} did not match update_unit. '
        f'matched_tools={intent.matched_tools}, '
        f'template={intent.suggested_approval_template!r}'
    )
    assert intent.suggested_approval_template == 'Update rental unit?', (
        f'wrong template surfaced: {intent.suggested_approval_template!r}'
    )


def test_update_unit_does_not_collide_with_update_reminder():
    """The registry has multiple update_* tools — make sure the resolver
    picks update_unit when unit is the target, not update_reminder."""
    intent = classify('update unit 5 monthly_rent to 2500')
    assert 'update_unit' in intent.matched_tools
    assert 'update_reminder' not in intent.matched_tools


def test_update_reminder_is_ungated():
    """Owner directive 2026-06-02: reminders are ungated. A reminder-targeted
    update must NOT match a destructive tool and must NOT gate — even though
    'update unit X' (a different target) still does."""
    intent = classify('update reminder 3 body to call mom')
    assert intent.is_destructive is False
    assert 'update_reminder' not in intent.matched_tools
