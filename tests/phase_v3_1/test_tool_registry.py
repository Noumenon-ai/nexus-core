from __future__ import annotations

import pytest

from services.tool_registry import (
    ToolRegistry,
    ToolResult,
    ToolSpec,
    get_default_registry,
    tool,
)


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture(autouse=True)
def _isolate_default_registry():
    reg = get_default_registry()
    reg.clear()
    yield
    reg.clear()


def test_tool_result_ok_no_data():
    r = ToolResult.ok()
    assert r.success is True
    assert r.data is None
    assert r.announcement is None
    assert r.error is None


def test_tool_result_ok_with_data_and_announcement():
    r = ToolResult.ok(data={"id": 1}, announcement="Created reminder for 3 PM")
    assert r.success is True
    assert r.data == {"id": 1}
    assert r.announcement == "Created reminder for 3 PM"
    assert r.error is None


def test_tool_result_fail_carries_error():
    r = ToolResult.fail("boom")
    assert r.success is False
    assert r.error == "boom"
    assert r.data is None
    assert r.announcement is None


def test_tool_result_is_frozen():
    r = ToolResult.ok()
    with pytest.raises(Exception):
        r.success = False  # type: ignore[misc]


def test_register_function_directly(registry):
    def get_time():
        """Returns current time."""
        return "now"

    spec = registry.register(get_time)
    assert isinstance(spec, ToolSpec)
    assert spec.name == "get_time"
    assert spec.description == "Returns current time."
    assert spec.requires_approval is False
    assert spec.fn is get_time
    assert registry.get("get_time") is spec
    assert registry.names() == ["get_time"]


def test_register_with_explicit_metadata(registry):
    def fn():
        return None

    spec = registry.register(
        fn,
        name="custom",
        description="overridden",
        requires_approval=True,
        parameters={"type": "object", "properties": {}},
    )
    assert spec.name == "custom"
    assert spec.description == "overridden"
    assert spec.requires_approval is True
    assert spec.parameters == {"type": "object", "properties": {}}


def test_duplicate_registration_raises(registry):
    def fn():
        return None

    registry.register(fn, name="dup")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(fn, name="dup")


def test_clear_resets_registry(registry):
    def fn():
        return None

    registry.register(fn, name="x")
    assert registry.names() == ["x"]
    registry.clear()
    assert registry.names() == []


def test_decorator_registers_into_default_registry():
    @tool()
    def sample():
        """Sample read tool."""
        return ToolResult.ok("hi")

    reg = get_default_registry()
    spec = reg.get("sample")
    assert spec is not None
    assert spec.name == "sample"
    assert spec.description == "Sample read tool."
    assert spec.requires_approval is False
    assert sample() == ToolResult.ok("hi")


def test_decorator_with_kwargs():
    @tool(name="delete_thing", description="Destructive", requires_approval=True)
    def _impl(thing_id: str):
        return ToolResult.ok(thing_id, announcement=f"Deleted {thing_id}")

    spec = get_default_registry().get("delete_thing")
    assert spec is not None
    assert spec.requires_approval is True
    assert spec.description == "Destructive"


def test_decorator_attaches_spec_to_function():
    @tool()
    def has_meta():
        """meta"""
        return None

    assert hasattr(has_meta, "__tool_spec__")
    assert has_meta.__tool_spec__.name == "has_meta"


def test_decorator_uses_explicit_registry(registry):
    @tool(registry=registry, name="iso")
    def isolated():
        return None

    assert registry.get("iso") is not None
    assert get_default_registry().get("iso") is None


def test_all_returns_every_registered_spec(registry):
    def a():
        return None

    def b():
        return None

    registry.register(a)
    registry.register(b)
    specs = registry.all()
    assert {s.name for s in specs} == {"a", "b"}


def test_source_inspection_can_filter_by_approval(registry):
    """Foundation for V3.4 mandatory source-inspection test:
    delete_*/send_*/disconnect_* MUST have requires_approval=True."""

    def list_things():
        return None

    def delete_thing():
        return None

    registry.register(list_things)
    registry.register(delete_thing, requires_approval=True)
    approval_gated = [s.name for s in registry.all() if s.requires_approval]
    assert approval_gated == ["delete_thing"]
