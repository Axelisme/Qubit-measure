"""Independent results window and its read-only browser component."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QMainWindow

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.controller import Controller


class ResultsBrowserWindow(QMainWindow):
    """Retain one read-only results view; the measure window owns its lifetime."""

    def __init__(self, controller: Controller, parent: QMainWindow | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setObjectName("results.browser")
        self.setWindowTitle("Results browser")
        self.resize(1280, 800)
        index = controller.query_results()
        label = QLabel(f"All results · {len(index.runs)} runs", self)
        label.setObjectName("results.focus")
        self.setCentralWidget(label)
