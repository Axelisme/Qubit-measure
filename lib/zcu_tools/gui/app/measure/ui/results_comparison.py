"""Pinned result selection shared by the browser and comparison presentation."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from qtpy.QtWidgets import QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget

from zcu_tools.gui.app.measure.services.results_index import ResultRun, ResultsIndex
from zcu_tools.gui.widgets.selectable_text import make_label_selectable


class ResultsComparison(QWidget):
    """Own ordered unique run pins for one browser index, on the Qt owner thread.

    pins_changed emits after adding/removing a pin or changing its figure. Re-pinning
    an existing run preserves order and updates its selected figure only when supplied.
    """

    pins_changed = pyqtSignal()

    def __init__(self, index: ResultsIndex, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self._pins: dict[str, str | None] = {}
        layout = QVBoxLayout(self)
        self._count = QLabel("Pinned runs: 0")
        self._count.setObjectName("results.pin_count")
        make_label_selectable(self._count)
        layout.addWidget(self._count)
        self._list = QListWidget()
        self._list.setObjectName("results.pins")
        layout.addWidget(self._list)
        remove = QPushButton("Unpin selected")
        remove.clicked.connect(self._remove_selected)
        layout.addWidget(remove)

    def pin_run(self, run: ResultRun, figure_selector: str | None = None) -> None:
        """Pin an indexed run; unknown run/figure raises KeyError without mutation."""
        run = self._index.get_run(run.key)
        if figure_selector is not None and figure_selector not in {
            path.name for path in run.figure_paths
        }:
            raise KeyError(figure_selector)
        if run.key not in self._pins or figure_selector is not None:
            self._pins[run.key] = figure_selector
            self._render()
            self.pins_changed.emit()

    def pinned_runs(self) -> tuple[ResultRun, ...]:
        """Return pinned runs in insertion order, without exposing mutable selection."""
        return tuple(self._index.get_run(key) for key in self._pins)

    def selected_figure(self, run_key: str) -> str | None:
        """Return the selected figure filename, or None; unknown pin raises KeyError."""
        return self._pins[run_key]

    def _render(self) -> None:
        self._count.setText(f"Pinned runs: {len(self._pins)}")
        self._list.clear()
        for run in self.pinned_runs():
            self._list.addItem(f"{run.entry} / {run.run_path.name}")

    def _remove_selected(self) -> None:
        row = self._list.currentRow()
        if row >= 0:
            del self._pins[tuple(self._pins)[row]]
            self._render()
            self.pins_changed.emit()
