"""Composition root — assemble and launch the dispersive-fit-gui.

Wires State + Controller + MainWindow behind the shared GUI process runtime. Like
fluxdep-gui there is no registry/role-catalog and no session-restore caretaker:
the tool starts fresh (load params.json + a one-tone hdf5); the written
``dispersive`` section of params.json is the only persisted artifact.

When ``control`` is given, a read-only ``RemoteControlAdapter`` is started so an
automation agent (or the MCP server) can *observe* the GUI over TCP; it is stopped
on the Qt main thread when the app quits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.runtime import (
    GuiAssembly,
    GuiRuntimeBehavior,
    GuiRuntimeSpec,
    PlotPolicy,
    run_gui_runtime,
)

if TYPE_CHECKING:
    from zcu_tools.gui.project import ProjectInfo
    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions


class DispersiveGuiBehavior(GuiRuntimeBehavior):
    spec = GuiRuntimeSpec(
        app_name="dispersive",
        app_slug="dispersive",
        plot_policy=PlotPolicy.EMBEDDED_BACKEND,
        default_control_port=8767,
    )

    def __init__(
        self,
        project: ProjectInfo | None = None,
        project_root: str | None = None,
    ) -> None:
        self._project = project
        self._project_root = project_root

    def assemble(self, control: ControlOptions | None) -> GuiAssembly:
        from zcu_tools.gui.app.dispersive.controller import Controller
        from zcu_tools.gui.app.dispersive.event_bus import EventBus
        from zcu_tools.gui.app.dispersive.state import DispersiveState
        from zcu_tools.gui.app.dispersive.ui.main_window import MainWindow

        ctrl = Controller(
            DispersiveState(self._project),
            EventBus(),
            project_root=self._project_root,
        )
        window = MainWindow(ctrl)
        adapter = None
        if control is not None:
            from zcu_tools.gui.app.dispersive.services.remote.service import (
                RemoteControlAdapter,
            )

            adapter = RemoteControlAdapter(ctrl, control)
        return GuiAssembly(controller=ctrl, window=window, control_adapter=adapter)


def run_app(
    project: ProjectInfo | None = None,
    control: ControlOptions | None = None,
    project_root: str | None = None,
) -> int:
    """Build and launch the dispersive-fit-gui. Blocks until the window is closed.

    ``project_root`` is the base dir the default result/database paths anchor
    under; the entry script passes the repo root so a .bat launcher that cd's
    into script/ does not scope defaults under script/. None falls back to cwd.
    """
    return run_gui_runtime(
        DispersiveGuiBehavior(project=project, project_root=project_root),
        control,
    )


__all__ = ["DispersiveGuiBehavior", "run_app"]
