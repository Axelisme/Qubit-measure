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

from typing_extensions import Optional

from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.state import AutoFluxDepState, ProjectInfo
from zcu_tools.gui.event_bus import BaseEventBus as EventBus


def _make_empty_ctx():
    """Minimal startup context: real empty MetaDict/ModuleLibrary, no file sync."""
    from zcu_tools.gui.session.types import ExpContext
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

    return ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)


def build_core(
    project: Optional[ProjectInfo] = None,
    project_root: Optional[str] = None,
) -> Controller:
    """Wire State + EventBus + Controller — the testable domain core.

    ``project_root`` is the base directory default result/database paths anchor
    under (the setup dialog derives defaults against it). None falls back to cwd —
    fine for tests / a ``python -m`` run from the repo root; ``run_app`` injects
    the repo root so the defaults are correct regardless of the working directory.
    """
    state = AutoFluxDepState(_make_empty_ctx(), project=project)
    return Controller(state, EventBus(), project_root=project_root)


def run_app(project: Optional[ProjectInfo] = None) -> None:
    """Build and launch the autofluxdep-gui. Blocks until the window closes."""
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
    sys.exit(app.exec())
