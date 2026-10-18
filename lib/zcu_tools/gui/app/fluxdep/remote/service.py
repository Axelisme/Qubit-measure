"""RemoteControlAdapter — fluxdep-gui's second View (driving adapter).

The RPC face onto the fluxdep ``Controller``, peer to the Qt ``MainWindow``
(ADR-0068). This module binds the app method registry, event serializers and
control identity owner. Shared dispatch checks control authority; handlers invoke
the existing Controller commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QApplication

from zcu_tools.gui.remote.control_service import (
    ControlOptions,
    RemoteAppSpec,
    RemoteControlServiceBase,
)
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.wire import Request

if TYPE_CHECKING:
    # Type-only: the string annotation keeps the import graph lean and lets
    # pyright check handler/ctrl method names without a runtime import.
    from zcu_tools.gui.app.fluxdep.controller import Controller
    from zcu_tools.gui.remote.rpc_endpoint import ClientLink
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

    def _route_extra(self, link: ClientLink, req: Request) -> bool:
        del link
        # ParamSpec normalizes absent and null optional names alike. Reject the
        # forbidden key before that information is lost, without accessing State.
        if (
            req.method in ("interactive.read", "interactive.render")
            and req.params.get("target") == "selection"
            and "name" in req.params
        ):
            raise RemoteError(ErrorCode.INVALID_PARAMS, "selection target forbids name")
        return False

    def _local_modal_open(self) -> bool:
        return QApplication.activeModalWidget() is not None

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
            app=RemoteAppSpec(
                wire_version=WIRE_VERSION,
                gui_version=GUI_VERSION,
                server_name="FluxDepRemoteServer",
                method_registry=METHOD_REGISTRY,
                event_serializers=EVENT_SERIALIZERS,
                wire_event_name=wire_event_name,
            ),
            owner_scheduler=owner_scheduler,
            control_identity=controller.control_identity,
        )


__all__ = ["ControlOptions", "RemoteControlAdapter"]
