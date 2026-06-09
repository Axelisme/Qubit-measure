"""Composition root — assemble and launch the fluxdep-gui.

Wires State + Controller + MainWindow and runs the Qt event loop. Unlike
measure-gui there is no registry/role-catalog (no experiment adapters) and no
session-restore caretaker (v1 does not persist a session — the analysis tool
starts fresh by loading an hdf5; exported spectrums.hdf5 / params.json are the
only persisted artifacts).
"""

from __future__ import annotations

import sys
from typing import Optional

from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import EventBus
from zcu_tools.gui.app.fluxdep.services.remote.service import (
    ControlOptions,
    RemoteControlAdapter,
)
from zcu_tools.gui.app.fluxdep.state import FluxDepState
from zcu_tools.gui.project import ProjectInfo


def run_app(
    project: Optional[ProjectInfo] = None,
    control: Optional[ControlOptions] = None,
    project_root: Optional[str] = None,
) -> None:
    """Build and launch the fluxdep-gui. Blocks until the window is closed.

    When ``control`` is given, a ``RemoteControlAdapter`` is started so an
    automation agent (or the MCP server) can drive the GUI over TCP; it is
    stopped on the Qt main thread when the app quits.

    ``project_root`` is the base dir the default result/database paths anchor
    under; the entry script passes the repo root so a .bat launcher that cd's
    into script/ does not scope defaults under script/. None falls back to cwd.
    """
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow
    from zcu_tools.gui.plotting import ensure_host, set_shutting_down

    app = QApplication.instance() or QApplication(sys.argv)

    # Create the plot host now, on the Qt main thread: the shared embedded
    # matplotlib backend marshals worker-thread figure work through it, and it
    # must be built on the GUI thread before any worker plots (e.g. the search).
    ensure_host()
    # On teardown, mark the plot host down BEFORE Qt deletes its widgets, so the
    # matplotlib atexit hook (Gcf.destroy_all → backend destroy → remove_canvas)
    # becomes a no-op instead of touching a deleted container.
    app.aboutToQuit.connect(lambda: set_shutting_down(True))

    state = FluxDepState(project)
    ctrl = Controller(state, EventBus(), project_root=project_root)
    window = MainWindow(ctrl)
    window.show()

    if control is not None:
        adapter = RemoteControlAdapter(ctrl, control)
        adapter.start()
        # stop() unsubscribes EventBus synchronously, so it must run on the Qt
        # main thread; aboutToQuit fires there.
        app.aboutToQuit.connect(adapter.stop)  # type: ignore[attr-defined]

    sys.exit(app.exec())
