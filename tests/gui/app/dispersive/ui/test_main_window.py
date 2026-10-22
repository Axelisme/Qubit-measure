"""Tests for the dispersive MainWindow shell."""

from __future__ import annotations

from zcu_tools.gui.app.dispersive.controller import Controller
from zcu_tools.gui.app.dispersive.state import DispersiveState


def test_main_window_builds_with_pipeline_panel(qapp):
    from zcu_tools.gui.app.dispersive.ui.main_window import MainWindow
    from zcu_tools.gui.app.dispersive.ui.pipeline_panel import PipelinePanelWidget

    state = DispersiveState()
    window = MainWindow(Controller(state))

    assert isinstance(window.centralWidget(), PipelinePanelWidget)
