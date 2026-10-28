"""Composition root for the dispersive-fit-gui.

Wires State + Controller + MainWindow behind the shared GUI process runtime. Like
fluxdep-gui there is no registry/role-catalog and no session-restore caretaker:
the tool starts fresh with an entry target, predictor and one-tone hdf5.
Export writes the entry dispersive artifact and kind-declared setup fields.

When ``control`` is given, a read-only ``RemoteControlAdapter`` is started so an
automation agent (or the MCP server) can *observe* the GUI over TCP; it is stopped
on the Qt main thread when the app quits.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget
from zcu_tools.gui.app.analysis_project.selection import AnalysisSelection
from zcu_tools.gui.app.dispersive.lab import DispersiveGuiLab
from zcu_tools.gui.runtime import (
    GuiAssembly,
    GuiRuntimeBehavior,
    GuiRuntimeSpec,
)
from zcu_tools.resources.entry.registry import component_registry

if TYPE_CHECKING:
    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions


class DispersiveGuiBehavior(GuiRuntimeBehavior):
    spec = GuiRuntimeSpec(
        app_name="dispersive",
        app_slug="dispersive",
        default_control_port=8767,
    )

    def __init__(
        self,
        lab: DispersiveGuiLab,
        selection: AnalysisSelection,
    ) -> None:
        """Register lab components, then open the requested entry.

        selection supplies the project root, entry and optional component. Empty
        entry leaves the GUI unopened. AnalysisSelection construction rejects a
        component without an entry before this behavior is constructed.
        The project root is resolved at construction so later cwd changes do
        not move dialog defaults. Registration and target errors propagate
        before window assembly.
        Dispersive reads predictor frames, not component flux policy.
        """
        lab.register_components(component_registry)
        self._project = AnalysisTarget.open(selection) if selection.entry else None
        self._project_root = str(Path(selection.project_root).resolve())

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
            from zcu_tools.gui.app.dispersive.remote.service import (
                RemoteControlAdapter,
            )
            from zcu_tools.gui.session.adapters.qt_owner_scheduler import (
                QtOwnerScheduler,
            )

            adapter = RemoteControlAdapter(
                ctrl,
                control,
                owner_scheduler=QtOwnerScheduler(),
            )
        return GuiAssembly(controller=ctrl, window=window, control_adapter=adapter)


__all__ = ["DispersiveGuiBehavior"]
