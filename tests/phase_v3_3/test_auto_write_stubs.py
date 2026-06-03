"""V3.3 auto-write-tool stubs file is empty after V3.2.5 phases promoted
all Google auto-write tools to real implementations. This test pins
that contract."""
from __future__ import annotations

from services.auto_write_tools_stubs import _PHASE_TAG, register_google_write_stubs
from services.tool_registry import ToolRegistry


def test_register_google_write_stubs_is_empty_after_v3_2_5():
    registry = ToolRegistry()
    specs = register_google_write_stubs(registry)
    assert specs == []


def test_phase_tag_aligned():
    assert _PHASE_TAG == 'V3.2.5'
