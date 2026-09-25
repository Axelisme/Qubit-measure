"""Tests for the dispersive MainWindow shell."""

from __future__ import annotations

from zcu_tools.gui.app.dispersive.controller import Controller
from zcu_tools.gui.app.dispersive.state import DispersiveState
from zcu_tools.gui.project import ProjectInfo


def test_main_window_builds_with_pipeline_panel(qapp):
    from zcu_tools.gui.app.dispersive.ui.main_window import MainWindow
    from zcu_tools.gui.app.dispersive.ui.pipeline_panel import PipelinePanelWidget

    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    window = MainWindow(Controller(state))

    assert window.windowTitle() == "dispersive-fit-gui"
    assert isinstance(window.centralWidget(), PipelinePanelWidget)
