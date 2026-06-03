"""V3.2 read-tool stubs file is empty after V3.2.5 phases promoted all
read tools to real implementations. This test pins that contract."""
from __future__ import annotations

from services.read_tools_stubs import _PHASE_TAG, register_google_stubs
from services.tool_registry import ToolRegistry


def test_register_google_stubs_is_empty_after_v3_2_5():
    registry = ToolRegistry()
    specs = register_google_stubs(registry)
    assert specs == []


def test_phase_tag_aligned():
    assert _PHASE_TAG == 'V3.2.5'
