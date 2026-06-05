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
from zcu_tools.gui.app.dispersive.state import DispersiveState, ProjectInfo

if TYPE_CHECKING:
    from zcu_tools.gui.app.dispersive.services.remote.service import ControlOptions


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
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    from zcu_tools.gui.app.dispersive.ui.main_window import MainWindow
    from zcu_tools.gui.plotting import ensure_host, set_shutting_down

    app = QApplication.instance() or QApplication(sys.argv)

    # Create the plot host on the Qt main thread before any worker plots. (The
    # tuning / result figures are drawn on local canvases, but a notebook helper
    # could still touch pyplot, so the shared embedded backend must be ready.)
    ensure_host()
    app.aboutToQuit.connect(lambda: set_shutting_down(True))

    state = DispersiveState(project)
    ctrl = Controller(state, EventBus(), project_root=project_root)
    window = MainWindow(ctrl)
    window.show()

    if control is not None:
        from zcu_tools.gui.app.dispersive.services.remote.service import (
            RemoteControlAdapter,
        )

        adapter = RemoteControlAdapter(ctrl, control)
        adapter.start()
        app.aboutToQuit.connect(adapter.stop)  # type: ignore[attr-defined]

    sys.exit(app.exec())
