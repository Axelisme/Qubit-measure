"""Composition root for autofluxdep-gui.

``build_core`` assembles State + EventBus + Controller (the testable domain
core). ``run_app`` adds the Qt MainWindow and runs the event loop.

The app composes the shared session services (connection / context / device /
startup) and uses the shared setup / device / predictor dialogs; the run drives
the orchestrator over the node graph, each node building its run cfg from the
active context and simulating the acquire (no hardware in this phase).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.state import AutoFluxDepState, ProjectInfo
from zcu_tools.gui.event_bus import BaseEventBus as EventBus

if TYPE_CHECKING:
    from zcu_tools.gui.app.autofluxdep.services.remote.service import ControlOptions
    from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow


def _make_empty_ctx():
    """Minimal startup context: real empty MetaDict/ModuleLibrary, no file sync."""
    from zcu_tools.gui.session.types import ExpContext
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

    return ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)


def build_core(
    project: ProjectInfo | None = None,
    project_root: str | None = None,
) -> Controller:
    """Wire State + EventBus + Controller — the testable domain core.

    ``project_root`` is the base directory default result/database paths anchor
    under (the setup dialog derives defaults against it). None falls back to cwd —
    fine for tests / a ``python -m`` run from the repo root; ``run_app`` injects
    the repo root so the defaults are correct regardless of the working directory.
    """
    state = AutoFluxDepState(_make_empty_ctx(), project=project)
    return Controller(state, EventBus(), project_root=project_root)


def run_app(
    project: ProjectInfo | None = None,
    control: ControlOptions | None = None,
) -> None:
    """Build and launch the autofluxdep-gui. Blocks until the window closes.

    ``control`` (a ``ControlOptions`` with a TCP port/token) opts the run into the
    read-only remote-control bridge — the RPC face an agent/MCP observes the
    workflow through. None (the default) leaves the GUI un-instrumented.
    """
    from pathlib import Path

    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow

    # Anchor default paths under the repo root (this file is at
    # <repo>/lib/zcu_tools/gui/app/autofluxdep/app.py), not the launch cwd.
    repo_root = str(Path(__file__).resolve().parents[5])

    app = QApplication.instance() or QApplication(sys.argv)
    ctrl = build_core(project, project_root=repo_root)
    window = MainWindow(ctrl)
    window.show()

    # Start the read-only remote-control adapter in-place (decision 6: app-local
    # start/stop, mirroring the shared run_qt_app's three lines, rather than
    # routing through run_qt_app — autofluxdep owns its own startup-dialog loop and
    # uses Agg, so it does not run through that shared helper). Inert until the
    # GUI is up; stop() unsubscribes the EventBus synchronously, so it must run on
    # the Qt main thread — aboutToQuit fires there.
    if control is not None:
        from zcu_tools.gui.app.autofluxdep.services.remote.service import (
            RemoteControlAdapter,
        )

        adapter = RemoteControlAdapter(ctrl, control)
        adapter.start()
        app.aboutToQuit.connect(adapter.stop)

    # Mirror measure-gui: open the setup dialog non-modally on startup so the
    # user can configure the project (chip/qub names, connection) immediately.
    # Non-modal keeps the Qt event loop pumping (required for the run worker and
    # any background operations) while the dialog is visible (ADR mirrors main/app.py).
    _show_startup_dialog(ctrl, parent=window)

    sys.exit(app.exec())


def _show_startup_dialog(
    ctrl: Controller,
    parent: MainWindow,
) -> None:
    """Open the setup dialog non-modally on first launch.

    Mirrors ``zcu_tools.gui.app.main.app._show_startup_dialog``.  Non-modal is
    required so the Qt event loop keeps pumping while the dialog is visible —
    this lets background session operations (mock-soc connect, device setup)
    complete without deadlocking.
    """
    from qtpy.QtCore import Qt  # type: ignore[attr-defined]

    from zcu_tools.gui.session.ui.setup_dialog import SetupDialog

    dlg = SetupDialog(ctrl, parent=parent, startup_mode=True)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dlg.open()
