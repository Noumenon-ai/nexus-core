"""All Google auto-write-tool stubs were promoted to real implementations.

V3.2.5.3: create_calendar_event, update_calendar_event → services/auto_write_tools.py
V3.2.5.5: create_contact                               → services/auto_write_tools.py

This file is kept as an empty registration stub so other modules can
still import register_google_write_stubs without breaking.
"""
from __future__ import annotations

from services.tool_registry import ToolRegistry, ToolSpec


_PHASE_TAG = 'V3.2.5'


def register_google_write_stubs(registry: ToolRegistry) -> list[ToolSpec]:
    """No-op: all V3.3 auto-write stubs have been promoted to real implementations."""
    return []
