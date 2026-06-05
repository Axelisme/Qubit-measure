"""Composition root for autofluxdep-gui.

``build_core`` assembles State + EventBus + Controller (the testable domain
core). ``run_app`` adds the Qt MainWindow and runs the event loop.

Prototype (Phase C): Setup builds fake resources and Run drives the orchestrator
on fake data — no hardware. Unlike fluxdep/dispersive this app is *control type*:
Phase B wires the real soc / device connection + acquire (the heavy half aligned
with measure-gui).
"""

from __future__ import annotations

import sys

from typing_extensions import Optional

from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.event_bus import EventBus
from zcu_tools.gui.app.autofluxdep.state import AutoFluxDepState, ProjectInfo


def build_core(project: Optional[ProjectInfo] = None) -> Controller:
    """Wire State + EventBus + Controller — the testable domain core."""
    state = AutoFluxDepState(project=project)
    return Controller(state, EventBus())


def run_app(project: Optional[ProjectInfo] = None) -> None:
    """Build and launch the autofluxdep-gui. Blocks until the window closes."""
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    ctrl = build_core(project)
    window = MainWindow(ctrl)
    window.show()
    sys.exit(app.exec())
