"""App-local remote method entry registry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Literal, cast

from zcu_tools.gui.remote.method_spec import (
    BoundMethod,
    Handler,
    MethodSpec,
    build_method_registry,
)
from zcu_tools.gui.remote.observation import ResourceObservationPolicy
from zcu_tools.gui.remote.param_spec import build_input_schema

AgentExposure = Literal["rpc", "tool", "internal"]


@dataclass(frozen=True, slots=True)
class AgentMethodPolicy(ResourceObservationPolicy):
    """Measure exposure metadata plus the shared resource observation contract.

    exposure is rpc, tool or internal. tool_names names fixed tools only for tool
    exposure; operation_key is an agent-tracking resource key template, such as
    tab:{tab_id}, or None when the method exposes no operation. Inherited fields declare GUI-owned observations and are
    never exported to MCP. Invalid exposure/tool declarations raise ValueError.
    """

    exposure: AgentExposure = "rpc"
    tool_names: tuple[str, ...] = ()
    operation_key: str | None = None

    def __post_init__(self) -> None:
        ResourceObservationPolicy.__post_init__(self)
        if self.exposure not in ("rpc", "tool", "internal"):
            raise ValueError(f"unknown agent exposure {self.exposure!r}")
        if (self.exposure == "tool") != bool(self.tool_names):
            raise ValueError("tool exposure requires tool_names only")
        if len(set(self.tool_names)) != len(self.tool_names):
            raise ValueError("duplicate tool names")


@dataclass(frozen=True, slots=True)
class RemoteMethodEntry:
    """Single registration record for one measure-gui wire method."""

    method: str
    handler_ref: str
    spec: MethodSpec
    agent: AgentMethodPolicy = field(default_factory=AgentMethodPolicy)


def method_entry(
    method: str,
    handler_ref: str,
    spec: MethodSpec,
    *,
    agent: AgentMethodPolicy | None = None,
) -> RemoteMethodEntry:
    return RemoteMethodEntry(method, handler_ref, spec, agent or AgentMethodPolicy())


def build_agent_catalog(
    entries: tuple[RemoteMethodEntry, ...],
) -> list[dict[str, object]]:
    """Project the live GUI's agent-facing method contract, never its handlers."""
    build_method_specs(entries)  # Reject duplicates at the same boundary as dispatch.
    return [
        {
            "method": entry.method,
            "description": entry.spec.description,
            "params": build_input_schema(entry.spec.params),
            "timeout_seconds": entry.spec.timeout_seconds,
            "exposure": entry.agent.exposure,
            "tool_names": list(entry.agent.tool_names),
            "operation_key": entry.agent.operation_key,
        }
        for entry in entries
        if entry.agent.exposure != "internal"
    ]


def build_method_specs(
    entries: tuple[RemoteMethodEntry, ...],
) -> dict[str, MethodSpec]:
    specs: dict[str, MethodSpec] = {}
    duplicates: set[str] = set()
    for entry in entries:
        if entry.method in specs:
            duplicates.add(entry.method)
        specs[entry.method] = entry.spec
    if duplicates:
        methods = ", ".join(sorted(duplicates))
        raise RuntimeError(f"duplicate remote method entries: {methods}")
    return specs


def build_dispatch_registry(
    entries: tuple[RemoteMethodEntry, ...],
) -> dict[str, BoundMethod]:
    specs = build_method_specs(entries)
    handlers: dict[str, Handler] = {}
    for entry in entries:
        entry.agent.validate_owner_thread(off_main_thread=entry.spec.off_main_thread)
        handlers[entry.method] = _resolve_handler_ref(entry.handler_ref)
    return build_method_registry(handlers, specs)


def _resolve_handler_ref(handler_ref: str) -> Handler:
    module_name, function_name = _parse_handler_ref(handler_ref)
    package = __package__
    if package is None:
        raise RuntimeError("remote method entry package is unavailable")
    remote_package = package.rsplit(".", 1)[0]
    import_name = f"{remote_package}.handlers.{module_name}"
    try:
        module = import_module(import_name)
    except ImportError as exc:
        raise RuntimeError(
            f"cannot import remote handler module {module_name!r} "
            f"for handler ref {handler_ref!r}"
        ) from exc
    try:
        handler = getattr(module, function_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"remote handler ref {handler_ref!r} does not name an attribute"
        ) from exc
    if not callable(handler):
        raise RuntimeError(f"remote handler ref {handler_ref!r} is not callable")
    return cast(Handler, handler)


def _parse_handler_ref(handler_ref: str) -> tuple[str, str]:
    if handler_ref.count(":") != 1:
        raise RuntimeError(
            "remote handler refs must use '<handler_module>:<function_name>'"
        )
    module_name, function_name = handler_ref.split(":", 1)
    if not module_name or not function_name:
        raise RuntimeError(
            "remote handler refs must use '<handler_module>:<function_name>'"
        )
    return module_name, function_name
