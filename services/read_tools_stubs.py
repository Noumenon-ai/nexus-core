"""All Google read-tool stubs were promoted to real implementations.

V3.2.5.1: list_calendar_events  → services/google_calendar_service.py
V3.2.5.2: check_freebusy        → services/google_calendar_service.py
V3.2.5.4: list_google_tasks     → services/google_tasks_service.py
V3.2.5.5: lookup_contact        → services/google_people_service.py

This file is kept as an empty registration stub so other modules can
still import register_google_stubs without breaking. Future stubs (if
added) would land here.
"""
from __future__ import annotations

from services.tool_registry import ToolRegistry, ToolSpec


_PHASE_TAG = 'V3.2.5'


def register_google_stubs(registry: ToolRegistry) -> list[ToolSpec]:
    """No-op: all V3.2 read stubs have been promoted to real implementations."""
    return []
