"""App-local remote method entry registry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Literal, cast

from zcu_tools.gui.remote.method_spec import (
    BoundMethod,
    Handler,
    MethodSpec,
    build_method_registry,
)
from zcu_tools.gui.remote.param_spec import build_input_schema

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.services.remote.service import RemoteControlAdapter

AgentExposure = Literal["rpc", "tool", "internal"]


@dataclass(frozen=True, slots=True)
class AgentMethodPolicy:
    """Measure-only exposure and concurrency contract for one wire method."""

    exposure: AgentExposure = "rpc"
    tool_names: tuple[str, ...] = ()
    guard_deps: tuple[str, ...] = ()
    reveals: tuple[str, ...] = ()
    # A partial query cannot reveal the entire named resource.
    reveals_without: tuple[str, ...] = ()
    # A successful write reports the versions it changed on the owner thread.
    refresh_after_write: bool = False
    operation_key: str | None = None

    def __post_init__(self) -> None:
        if self.exposure not in ("rpc", "tool", "internal"):
            raise ValueError(f"unknown agent exposure {self.exposure!r}")
        if (self.exposure == "tool") != bool(self.tool_names):
            raise ValueError("tool exposure requires tool_names only")
        if len(set(self.tool_names)) != len(self.tool_names):
            raise ValueError("duplicate tool names")
        if self.reveals_without and not self.reveals:
            raise ValueError("reveals_without requires revealed resources")


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
            "guard_deps": list(entry.agent.guard_deps),
            "reveals": list(entry.agent.reveals),
            "reveals_without": list(entry.agent.reveals_without),
            "refresh_after_write": entry.agent.refresh_after_write,
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
        handler = _resolve_handler_ref(entry.handler_ref)
        if entry.agent.refresh_after_write:
            if entry.spec.off_main_thread:
                raise ValueError("write-version receipts require the owner thread")
            handler = _with_write_versions(handler)
        handlers[entry.method] = handler
    return build_method_registry(handlers, specs)


def _with_write_versions(handler: Handler) -> Handler:
    def wrapped(
        adapter: RemoteControlAdapter, params: dict[str, object]
    ) -> dict[str, object]:
        # The handler and both samples share one owner-thread dispatch. Versions
        # obtained by a separate RPC after the reply may include unseen GUI edits.
        before = adapter.ctrl.resources_versions()
        result = handler(adapter, params)
        after = adapter.ctrl.resources_versions()
        return {
            **result,
            "__agent_write_versions": {
                key: [before.get(key, 0), version]
                for key, version in after.items()
                if version != before.get(key, 0)
            },
        }

    return wrapped


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
