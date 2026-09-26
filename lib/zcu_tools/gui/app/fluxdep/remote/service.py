"""RemoteControlAdapter — fluxdep-gui's second View (driving adapter).

The RPC face onto the fluxdep ``Controller``, peer to the Qt ``MainWindow``
(ADR-0013). fluxdep is read-only over the wire: it adds no dispatch policy beyond
the shared scaffolding — no version guard, no editor sessions, no off-main
handlers, a bare main-thread marshal. So this module is just the thin binding of
:class:`RemoteControlServiceBase` to fluxdep's domain (its method registry, event
serializers, versions, server name); all the mechanism lives in the base.
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
    from zcu_tools.gui.app.fluxdep.controller import Controller
    from zcu_tools.gui.session.ports import OwnerScheduler

from .dispatch import METHOD_REGISTRY
from .events import EVENT_SERIALIZERS, wire_event_name
from .wire_version import GUI_VERSION, WIRE_VERSION


class RemoteControlAdapter(RemoteControlServiceBase):
    """Driving adapter: an NDJSON RPC face onto the fluxdep ``Controller``.

    Dispatch handlers receive *this adapter*, so they reach commands through
    ``adapter.ctrl.<m>``. Construct after the Controller exists; inert until
    ``start()``. fluxdep's EventBus is reached via the base default
    (``ctrl.bus``) and its serializers are keyed by payload ``type``.
    """

    ctrl: Controller

    def __init__(
        self,
        controller: Controller,
        opts: ControlOptions,
        *,
        owner_scheduler: OwnerScheduler,
    ) -> None:
        super().__init__(
            controller,
            opts,
            owner_scheduler=owner_scheduler,
            wire_version=WIRE_VERSION,
            gui_version=GUI_VERSION,
            server_name="FluxDepRemoteServer",
            method_registry=METHOD_REGISTRY,
            event_serializers=EVENT_SERIALIZERS,
            wire_event_name=wire_event_name,
        )


__all__ = ["ControlOptions", "RemoteControlAdapter"]
