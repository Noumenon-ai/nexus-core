from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True, slots=True)
class ToolResult:
    success: bool
    data: Any = None
    announcement: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, data: Any = None, announcement: Optional[str] = None) -> "ToolResult":
        return cls(success=True, data=data, announcement=announcement)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        return cls(success=False, error=error)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    requires_approval: bool = False
    parameters: dict = field(default_factory=dict)
    approval_template: Optional[str] = None


def render_approval_preview(spec: "ToolSpec", args: dict) -> str:
    """Render the user-facing preview text shown before tap-to-approve.

    If the tool was registered with an explicit ``approval_template``
    (e.g. ``"Delete reminder: {body}"``), format it with ``args``. If the
    template references a placeholder not present in args, fall back to
    the raw template string. If no template was registered, fall back to
    a generic ``"Approve <name> with: <args>"`` default so every tool —
    approval-gated or not — has a non-empty preview available.
    """
    if spec.approval_template:
        try:
            return spec.approval_template.format(**args)
        except (KeyError, IndexError):
            return spec.approval_template
    return f"Approve {spec.name} with: {args}"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        fn: Callable[..., Any],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        requires_approval: bool = False,
        parameters: Optional[dict] = None,
        approval_template: Optional[str] = None,
    ) -> ToolSpec:
        tool_name = name or fn.__name__
        if tool_name in self._tools:
            raise ValueError(f"Tool already registered: {tool_name}")
        spec = ToolSpec(
            name=tool_name,
            description=(description if description is not None else (fn.__doc__ or "").strip()),
            fn=fn,
            requires_approval=requires_approval,
            parameters=parameters if parameters is not None else {},
            approval_template=approval_template,
        )
        self._tools[tool_name] = spec
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def clear(self) -> None:
        self._tools.clear()


_default_registry = ToolRegistry()


def get_default_registry() -> ToolRegistry:
    return _default_registry


def tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    requires_approval: bool = False,
    parameters: Optional[dict] = None,
    approval_template: Optional[str] = None,
    registry: Optional[ToolRegistry] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        target = registry if registry is not None else _default_registry
        spec = target.register(
            fn,
            name=name,
            description=description,
            requires_approval=requires_approval,
            parameters=parameters,
            approval_template=approval_template,
        )
        try:
            fn.__tool_spec__ = spec  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass
        return fn

    return deco
