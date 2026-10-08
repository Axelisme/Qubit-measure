"""RemoteControlAdapter — fluxdep-gui's second View (driving adapter).

The RPC face onto the fluxdep ``Controller``, peer to the Qt ``MainWindow``
(ADR-0068). This module binds the app method registry, observation policies,
event serializers and current resource versions. Shared dispatch owns all seen
maps and guard mechanics; handlers invoke the existing Controller commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.app.fluxdep.state import (
    FIT_VERSION_KEY,
    PROJECT_VERSION_KEY,
    SELECTION_VERSION_KEY,
    SPECTRUM_SET_VERSION_KEY,
)
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
from .method_specs import OBSERVATION_POLICIES
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
            resource_versions=self._resource_snapshot,
            observation_policies=OBSERVATION_POLICIES,
        )

    def _resource_snapshot(self) -> dict[str, int]:
        """Return all app guard keys, including fixed resources at version zero."""
        state = self.ctrl.state
        state.assert_owner_thread()
        versions: dict[str, int] = dict.fromkeys(
            (
                PROJECT_VERSION_KEY,
                FIT_VERSION_KEY,
                SELECTION_VERSION_KEY,
                SPECTRUM_SET_VERSION_KEY,
            ),
            0,
        )
        versions.update(state.version.snapshot())
        return versions


__all__ = ["ControlOptions", "RemoteControlAdapter"]
