"""MainWindow — the dispersive-fit-gui shell.

A thin QMainWindow holding the single ``PipelinePanelWidget`` as its central
widget (no list / stacked-editor — dispersive is a single linear flow, not a
collection). The panel owns the pipeline; the window just frames it.
"""

from __future__ import annotations

from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QMainWindow, QWidget

from zcu_tools.gui.app.dispersive.controller import Controller

from .pipeline_panel import PipelinePanelWidget


class MainWindow(QMainWindow):
    """The dispersive-fit-gui main window: one single-flow pipeline panel."""

    def __init__(self, ctrl: Controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self.setWindowTitle("dispersive-fit-gui")
        self.resize(1100, 900)
        self._panel = PipelinePanelWidget(ctrl)
        self.setCentralWidget(self._panel)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Attempt the panel's default background drain before continuing Qt close."""
        # Preserve the existing close policy: a drain timeout does not veto close.
        self._panel.quiesce_background_work()
        super().closeEvent(a0)
