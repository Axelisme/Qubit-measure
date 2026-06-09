"""Composition root — assemble and launch the dispersive-fit-gui.

Wires State + Controller + MainWindow and runs the Qt event loop. Like fluxdep-gui
there is no registry/role-catalog and no session-restore caretaker: the tool starts
fresh (load params.json + a one-tone hdf5); the written ``dispersive`` section of
params.json is the only persisted artifact.

When ``control`` is given, a read-only ``RemoteControlAdapter`` is started so an
automation agent (or the MCP server) can *observe* the GUI over TCP; it is stopped
on the Qt main thread when the app quits. (The remote layer is wired in Phase 5;
passing ``control`` before then raises a clear ImportError.)
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Optional

from zcu_tools.gui.app.dispersive.controller import Controller
from zcu_tools.gui.app.dispersive.event_bus import EventBus
from zcu_tools.gui.app.dispersive.state import DispersiveState
from zcu_tools.gui.project import ProjectInfo
from zcu_tools.gui.run_app import run_qt_app

if TYPE_CHECKING:
    from zcu_tools.gui.app.dispersive.services.remote.service import (
        ControlOptions,
        RemoteControlAdapter,
    )


def run_app(
    project: Optional[ProjectInfo] = None,
    control: Optional["ControlOptions"] = None,
    project_root: Optional[str] = None,
) -> None:
    """Build and launch the dispersive-fit-gui. Blocks until the window is closed.

    ``project_root`` is the base dir the default result/database paths anchor
    under; the entry script passes the repo root so a .bat launcher that cd's
    into script/ does not scope defaults under script/. None falls back to cwd.
    """
    from zcu_tools.gui.app.dispersive.ui.main_window import MainWindow

    def controller_factory() -> Controller:
        return Controller(
            DispersiveState(project), EventBus(), project_root=project_root
        )

    def adapter_factory(
        ctrl: Controller, opts: "ControlOptions"
    ) -> "RemoteControlAdapter":
        # Lazily import the remote layer so it is pulled in only when a control
        # socket is requested (the read-only bridge / MCP path), not on a plain
        # GUI launch.
        from zcu_tools.gui.app.dispersive.services.remote.service import (
            RemoteControlAdapter,
        )

        return RemoteControlAdapter(ctrl, opts)

    sys.exit(
        run_qt_app(
            controller_factory=controller_factory,
            window_factory=MainWindow,
            control=control,
            adapter_factory=adapter_factory,
        )
    )
