"""Shared wire-method registry mechanics — MethodSpec, BoundMethod, registry.

App-agnostic: every GUI app (main / fluxdep / dispersive) declares its own
``METHOD_SPECS`` (the Qt-free contract table) and ``_HANDLERS`` (the domain
handler map), then builds its runtime registry with ``build_method_registry``.
This module owns only the *mechanism* — the spec dataclass, the spec↔handler
binding, and the fail-fast drift check — and knows nothing of any concrete
method, handler signature, or Controller.

A handler's first argument is the per-app RemoteControlAdapter that hosts it;
this layer does not name that type (it is app-specific). ``Handler`` therefore
leaves the call signature unconstrained — each app keeps a precise
``Callable[["RemoteControlAdapter", Mapping], Mapping]`` alias locally for its
own handlers, which is assignable to this one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .param_spec import ParamSpec

# The adapter that hosts a handler is app-specific, so this shared alias does
# not name it; the return is always a wire dict.
Handler = Callable[..., Mapping[str, object]]


@dataclass(frozen=True)
class MethodSpec:
    """Contract for one wire method, independent of its handler.

    ``timeout_seconds`` is the main-thread handler budget. ``params`` is the
    parameter contract used both for runtime validation (dispatch/service) and
    MCP ``inputSchema`` generation (mcp_server). ``tool_name`` overrides the
    derived ``<prefix>_<method>`` MCP tool name when non-empty.

    ``off_main_thread`` marks a blocking handler that must NOT be marshalled
    onto the Qt main thread — it runs on the IO worker thread instead. Required
    for handlers that block waiting on a worker-thread completion (e.g.
    ``operation.await``): marshalling them onto the main thread would deadlock
    (the handler occupies the event loop that must dispatch the very signal it
    awaits). Off-main handlers must only do thread-safe waiting and must not
    touch main-thread-owned state, the stale guard, or the origin scope. A
    read-only app keeps the flag for mechanism parity (no method sets it True).
    """

    timeout_seconds: float
    description: str
    params: tuple[ParamSpec, ...] = ()
    tool_name: str = ""
    off_main_thread: bool = False


@dataclass(frozen=True)
class BoundMethod:
    """Runtime registry entry — binds a synchronous handler to a MethodSpec."""

    handler: Handler
    spec: MethodSpec

    @property
    def timeout_seconds(self) -> float:
        return self.spec.timeout_seconds

    @property
    def params(self) -> tuple[ParamSpec, ...]:
        return self.spec.params

    @property
    def off_main_thread(self) -> bool:
        return self.spec.off_main_thread


def build_method_registry(
    handlers: Mapping[str, Handler],
    specs: Mapping[str, MethodSpec],
) -> dict[str, BoundMethod]:
    """Bind each spec to its handler, failing fast on drift.

    Every spec must have a handler and vice versa — a new handler without a
    spec (or a spec without a handler) is a wiring bug, raised at import time
    rather than discovered at dispatch. The ``auth`` sentinel is handled by the
    service before the registry, so it is intentionally absent from both maps.
    """
    if set(handlers) != set(specs):
        missing_spec = sorted(set(handlers) - set(specs))
        missing_handler = sorted(set(specs) - set(handlers))
        raise RuntimeError(
            "dispatch/method_specs drift — "
            f"handlers without spec: {missing_spec}; specs without handler: {missing_handler}"
        )
    return {
        method: BoundMethod(handler=handlers[method], spec=specs[method])
        for method in specs
    }
