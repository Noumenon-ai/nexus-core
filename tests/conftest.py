from __future__ import annotations

import pytest

from tests.helpers import build_test_container, configure_test_env


@pytest.fixture(autouse=True)
def _disable_mcp_routing_by_default(monkeypatch):
    """Phase 2.5a: brain_router.generate_with_tools now routes TOOL_LIKELY turns
    to Claude CLI subprocess with MCP servers loaded. Unit tests should NOT
    spawn that subprocess — disable MCP path by default. Individual tests that
    exercise the MCP path can monkeypatch NEXUS_MCP_CONFIG back to a valid file.
    """
    monkeypatch.setenv("NEXUS_MCP_CONFIG", "disabled")


@pytest.fixture
def settings(monkeypatch, tmp_path):
    return configure_test_env(monkeypatch, tmp_path)


@pytest.fixture
def container(monkeypatch, tmp_path):
    return build_test_container(monkeypatch, tmp_path)
