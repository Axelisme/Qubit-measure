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

from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.remote.control_service import (
    ControlOptions,
    RemoteControlServiceBase,
)

if TYPE_CHECKING:
    # Type-only: the string annotation keeps the import graph lean and lets
    # pyright check handler/ctrl method names without a runtime import.
    from zcu_tools.gui.app.autofluxdep.controller import Controller
    from zcu_tools.gui.session.ports import OwnerScheduler

from .dispatch import METHOD_REGISTRY
from .events import EVENT_SERIALIZERS, wire_event_name
from .read_model import ControllerRemoteReadModel, RemoteReadModel
from .wire_version import GUI_VERSION, WIRE_VERSION


class ScreenshotView(Protocol):
    """Narrow read-only view surface exposed to remote handlers."""

    def take_window_screenshot(self) -> bytes: ...


class RemoteControlAdapter(RemoteControlServiceBase):
    """Driving adapter: an NDJSON RPC face onto the autofluxdep ``Controller``.

    Dispatch handlers receive *this adapter*, so they reach read-only queries through
    ``adapter.read_model.<m>``. Construct after the Controller exists; inert until
    ``start()``. autofluxdep's EventBus is reached via the base default
    (``ctrl.bus`` — a ``BaseEventBus``) and its serializers are keyed by payload
    ``type``.
    """

    ctrl: Controller
    read_model: RemoteReadModel

    def __init__(
        self,
        controller: Controller,
        opts: ControlOptions,
        *,
        owner_scheduler: OwnerScheduler,
        view: ScreenshotView | None = None,
    ) -> None:
        super().__init__(
            controller,
            opts,
            owner_scheduler=owner_scheduler,
            wire_version=WIRE_VERSION,
            gui_version=GUI_VERSION,
            server_name="AutoFluxDepRemoteServer",
            method_registry=METHOD_REGISTRY,
            event_serializers=EVENT_SERIALIZERS,
            wire_event_name=wire_event_name,
        )
        self.read_model = ControllerRemoteReadModel(controller)
        self._view = view

    def take_screenshot(self, target: str) -> bytes:
        if target != "window":
            raise ValueError(f"unsupported screenshot target {target!r}")
        if self._view is None:
            raise RuntimeError("remote screenshot requires a mounted MainWindow view")
        return self._view.take_window_screenshot()


__all__ = ["ControlOptions", "RemoteControlAdapter", "ScreenshotView"]
