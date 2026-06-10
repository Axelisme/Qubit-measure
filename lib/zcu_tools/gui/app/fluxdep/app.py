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
from zcu_tools.gui.run_app import run_qt_app


def run_app(
    project: ProjectInfo | None = None,
    control: ControlOptions | None = None,
    project_root: str | None = None,
) -> None:
    """Build and launch the fluxdep-gui. Blocks until the window is closed.

    When ``control`` is given, a ``RemoteControlAdapter`` is started so an
    automation agent (or the MCP server) can drive the GUI over TCP; it is
    stopped on the Qt main thread when the app quits.

    ``project_root`` is the base dir the default result/database paths anchor
    under; the entry script passes the repo root so a .bat launcher that cd's
    into script/ does not scope defaults under script/. None falls back to cwd.
    """
    from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow

    def controller_factory() -> Controller:
        return Controller(FluxDepState(project), EventBus(), project_root=project_root)

    sys.exit(
        run_qt_app(
            controller_factory=controller_factory,
            window_factory=MainWindow,
            control=control,
            adapter_factory=RemoteControlAdapter,
        )
    )
