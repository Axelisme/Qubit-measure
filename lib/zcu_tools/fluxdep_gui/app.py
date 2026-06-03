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

from zcu_tools.fluxdep_gui.controller import Controller
from zcu_tools.fluxdep_gui.event_bus import EventBus
from zcu_tools.fluxdep_gui.services.remote.service import (
    ControlOptions,
    RemoteControlAdapter,
)
from zcu_tools.fluxdep_gui.state import FluxDepState, ProjectInfo


def run_app(
    project: Optional[ProjectInfo] = None,
    control: Optional[ControlOptions] = None,
) -> None:
    """Build and launch the fluxdep-gui. Blocks until the window is closed.

    When ``control`` is given, a ``RemoteControlAdapter`` is started so an
    automation agent (or the MCP server) can drive the GUI over TCP; it is
    stopped on the Qt main thread when the app quits.
    """
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    from zcu_tools.fluxdep_gui.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)

    state = FluxDepState(project)
    ctrl = Controller(state, EventBus())
    window = MainWindow(ctrl)
    window.show()

    if control is not None:
        adapter = RemoteControlAdapter(ctrl, control)
        adapter.start()
        # stop() unsubscribes EventBus synchronously, so it must run on the Qt
        # main thread; aboutToQuit fires there.
        app.aboutToQuit.connect(adapter.stop)  # type: ignore[attr-defined]

    sys.exit(app.exec())
