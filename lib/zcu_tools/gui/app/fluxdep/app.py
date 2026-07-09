"""Composition root for the fluxdep-gui.

Wires State + Controller + MainWindow behind the shared GUI process runtime.
Unlike measure-gui there is no registry/role-catalog (no experiment adapters)
and no session-restore caretaker (v1 does not persist a session — the analysis
tool starts fresh by loading an hdf5; exported spectrums.hdf5 / params.json are
the only persisted artifacts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.runtime import (
    GuiAssembly,
    GuiRuntimeBehavior,
    GuiRuntimeSpec,
    PlotPolicy,
)

if TYPE_CHECKING:
    from zcu_tools.gui.project import ProjectInfo
    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions


class FluxDepGuiBehavior(GuiRuntimeBehavior):
    spec = GuiRuntimeSpec(
        app_name="fluxdep",
        app_slug="fluxdep",
        plot_policy=PlotPolicy.EMBEDDED_BACKEND,
        default_control_port=8766,
    )

    def __init__(
        self,
        project: ProjectInfo | None = None,
        project_root: str | None = None,
    ) -> None:
        self._project = project
        self._project_root = project_root

    def assemble(self, control: ControlOptions | None) -> GuiAssembly:
        from zcu_tools.gui.app.fluxdep.controller import Controller
        from zcu_tools.gui.app.fluxdep.event_bus import EventBus
        from zcu_tools.gui.app.fluxdep.services.remote.service import (
            RemoteControlAdapter,
        )
        from zcu_tools.gui.app.fluxdep.state import FluxDepState
        from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow

        ctrl = Controller(
            FluxDepState(self._project),
            EventBus(),
            project_root=self._project_root,
        )
        window = MainWindow(ctrl)
        adapter = RemoteControlAdapter(ctrl, control) if control is not None else None
        return GuiAssembly(controller=ctrl, window=window, control_adapter=adapter)


__all__ = ["FluxDepGuiBehavior"]
