"""App-local remote method entry registry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import cast

from zcu_tools.gui.remote.method_spec import (
    BoundMethod,
    Handler,
    MethodSpec,
    build_method_registry,
)


@dataclass(frozen=True, slots=True)
class RemoteMethodEntry:
    """Single registration record for one measure-gui wire method."""

    method: str
    handler_ref: str
    spec: MethodSpec


def method_entry(method: str, handler_ref: str, spec: MethodSpec) -> RemoteMethodEntry:
    return RemoteMethodEntry(method=method, handler_ref=handler_ref, spec=spec)


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
    handlers = {
        entry.method: _resolve_handler_ref(entry.handler_ref) for entry in entries
    }
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
