"""MainWindow — the dispersive-fit-gui shell.

A thin QMainWindow holding the single ``PipelinePanelWidget`` as its central
widget (no list / stacked-editor — dispersive is a single linear flow, not a
collection). The panel owns the pipeline; the window just frames it.
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import QMainWindow, QWidget  # type: ignore[attr-defined]

from zcu_tools.gui.app.dispersive.controller import Controller

from .pipeline_panel import PipelinePanelWidget


class MainWindow(QMainWindow):
    """The dispersive-fit-gui main window: one single-flow pipeline panel."""

    def __init__(self, ctrl: Controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self.setWindowTitle("dispersive-fit-gui")
        self.resize(1100, 900)
        self._panel = PipelinePanelWidget(ctrl)
        self.setCentralWidget(self._panel)
