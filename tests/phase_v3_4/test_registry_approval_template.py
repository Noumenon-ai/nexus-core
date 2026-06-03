"""V3.4 ToolRegistry extension: approval_template + render_approval_preview.

Approval-gated tools need user-facing preview text describing what will
happen if the tap-to-approve confirmation goes through. The template is
either explicit (string with {arg} placeholders) or sensibly defaulted
from tool name + args.
"""
from __future__ import annotations

import pytest

from services.tool_registry import (
    ToolRegistry,
    ToolResult,
    ToolSpec,
    render_approval_preview,
    tool,
)


@pytest.fixture
def registry():
    return ToolRegistry()


def test_register_with_approval_template(registry):
    def delete_thing():
        return None

    spec = registry.register(
        delete_thing,
        requires_approval=True,
        approval_template='Delete thing {thing_id}',
    )
    assert isinstance(spec, ToolSpec)
    assert spec.requires_approval is True
    assert spec.approval_template == 'Delete thing {thing_id}'


def test_register_default_no_approval_template(registry):
    def harmless():
        return None

    spec = registry.register(harmless)
    assert spec.approval_template is None


def test_decorator_carries_approval_template():
    reg = ToolRegistry()

    @tool(name='delete_x', requires_approval=True, approval_template='Delete x={x}', registry=reg)
    def _impl(*, x):
        return ToolResult.ok()

    spec = reg.get('delete_x')
    assert spec is not None
    assert spec.approval_template == 'Delete x={x}'
    assert spec.requires_approval is True


def test_render_preview_uses_explicit_template(registry):
    def delete_thing(*, thing_id):
        return None

    spec = registry.register(
        delete_thing,
        requires_approval=True,
        approval_template='Delete thing {thing_id}',
    )
    preview = render_approval_preview(spec, {'thing_id': 'abc-42'})
    assert preview == 'Delete thing abc-42'


def test_render_preview_falls_back_to_default_when_template_absent(registry):
    def delete_undocumented(*, x):
        return None

    spec = registry.register(delete_undocumented, requires_approval=True)
    preview = render_approval_preview(spec, {'x': 'value'})
    assert isinstance(preview, str) and preview.strip()
    assert 'delete_undocumented' in preview


def test_render_preview_handles_template_missing_placeholder_arg(registry):
    """If the template references {missing} but args don't have it, fall
    back to the raw template string (still a non-empty preview)."""
    def delete_thing():
        return None

    spec = registry.register(
        delete_thing,
        requires_approval=True,
        approval_template='Delete thing {thing_id}',
    )
    preview = render_approval_preview(spec, {})
    assert isinstance(preview, str) and preview.strip()


def test_render_preview_for_non_approval_tool_returns_default(registry):
    """Reading or non-destructive tools don't normally need a preview, but
    callers may still ask for one (e.g., dry-run UX). Default suffices."""
    def list_things():
        return None

    spec = registry.register(list_things)
    preview = render_approval_preview(spec, {})
    assert isinstance(preview, str) and preview.strip()
