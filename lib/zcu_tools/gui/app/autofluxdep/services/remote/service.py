"""RemoteControlAdapter — autofluxdep-gui's second View (driving adapter).

The RPC face onto the autofluxdep ``Controller``, peer to the Qt ``MainWindow``
(ADR-0013). autofluxdep is read-only over the wire: it adds no dispatch policy
beyond the shared scaffolding — no version guard, no editor sessions, no off-main
handlers, a bare main-thread marshal. So this module is just the thin binding of
:class:`RemoteControlServiceBase` to autofluxdep's domain (its method registry,
event serializers, versions, server name); all the mechanism lives in the base.

Reading the run-lived ``run_results`` on the main thread is safe even while a run
worker is filling rows — the worker only writes into pre-allocated numpy arrays in
place (not a State semantic write), so the marshalled handler sees a consistent
snapshot. No off-main handler is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.remote.control_service import (
    ControlOptions,
    RemoteControlServiceBase,
)

if TYPE_CHECKING:
    # Type-only: the string annotation keeps the import graph lean and lets
    # pyright check handler/ctrl method names without a runtime import.
    from zcu_tools.gui.app.autofluxdep.controller import Controller

from .dispatch import METHOD_REGISTRY
from .events import EVENT_SERIALIZERS, wire_event_name
from .wire_version import GUI_VERSION, WIRE_VERSION


class RemoteControlAdapter(RemoteControlServiceBase):
    """Driving adapter: an NDJSON RPC face onto the autofluxdep ``Controller``.

    Dispatch handlers receive *this adapter*, so they reach commands through
    ``adapter.ctrl.<m>``. Construct after the Controller exists; inert until
    ``start()``. autofluxdep's EventBus is reached via the base default
    (``ctrl.bus`` — a ``BaseEventBus``) and its serializers are keyed by payload
    ``type``.
    """

    ctrl: Controller

    def __init__(self, controller: Controller, opts: ControlOptions) -> None:
        super().__init__(
            controller,
            opts,
            wire_version=WIRE_VERSION,
            gui_version=GUI_VERSION,
            server_name="AutoFluxDepRemoteServer",
            method_registry=METHOD_REGISTRY,
            event_serializers=EVENT_SERIALIZERS,
            wire_event_name=wire_event_name,
        )


__all__ = ["ControlOptions", "RemoteControlAdapter"]
