"""Pinned result tables and four-run saved-image presentation."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.measure.services.results_comparison import build_comparison
from zcu_tools.gui.app.measure.services.results_index import ResultRun, ResultsIndex
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

from .results_gallery import ResultImage


class ResultsComparison(QWidget):
    """Own ordered unique run pins for one browser index, on the Qt owner thread.

    pins_changed emits after adding/removing a pin or changing its figure. Re-pinning
    an existing run preserves order and updates its selected figure only when supplied.
    Table/figure mode and the at-most-four checked runs never change pin membership.
    """

    pins_changed = pyqtSignal()

    def __init__(self, index: ResultsIndex, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self._pins: dict[str, str | None] = {}
        self._shown: set[str] = set()
        self._selected_key: str | None = None
        layout = QVBoxLayout(self)
        self._count = QLabel("Pinned runs: 0")
        self._count.setObjectName("results.pin_count")
        make_label_selectable(self._count)
        layout.addWidget(self._count)
        self._list = QListWidget()
        self._list.setObjectName("results.pins")
        self._list.setMaximumHeight(100)
        self._list.setToolTip(
            "Check at most 4 runs for the figure grid. All pins stay in the table."
        )
        self._list.itemChanged.connect(self._toggle_image)
        self._list.currentRowChanged.connect(self._select_list_row)
        layout.addWidget(self._list)
        layout.addLayout(self._build_actions())
        self._feedback = QLabel("Check at most 4 runs for figures")
        self._feedback.setObjectName("results.compare.feedback")
        self._feedback.setWordWrap(True)
        layout.addWidget(self._feedback)
        self._views = QStackedWidget()
        self._table = QTableWidget()
        self._table.setObjectName("results.comparison_table")
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.currentCellChanged.connect(self._select_table_row)
        self._views.addWidget(self._table)
        self._images = QWidget()
        self._grid = QGridLayout(self._images)
        self._views.addWidget(self._images)
        layout.addWidget(self._views, 1)
        self._render()

    def _build_actions(self) -> QHBoxLayout:
        actions = QHBoxLayout()
        for caption, callback in (
            ("Comparison table", lambda: self._set_mode(0)),
            ("2 × 2 figures", lambda: self._set_mode(1)),
            ("Unpin selected", self._remove_selected),
            ("Clear comparison", self._clear),
        ):
            button = QPushButton(caption)
            button.clicked.connect(callback)
            actions.addWidget(button)
        return actions

    def pin_run(self, run: ResultRun, figure_selector: str | None = None) -> None:
        """Pin an indexed run; unknown run/figure raises KeyError without mutation.

        A new pin defaults to its first indexed PNG, or None without PNGs.
        Re-pinning without a selector preserves an existing exact selection.
        """
        run = self._index.get_run(run.key)
        if figure_selector is not None and figure_selector not in {
            path.name for path in run.figure_paths
        }:
            raise KeyError(figure_selector)
        if run.key in self._pins and figure_selector is None:
            return
        selected = figure_selector
        if selected is None and run.figure_paths:
            selected = run.figure_paths[0].name
        if run.key in self._pins and self._pins[run.key] == selected:
            return
        if run.key not in self._pins and len(self._shown) < 4:
            self._shown.add(run.key)
        self._pins[run.key] = selected
        self._render()
        self.pins_changed.emit()

    def pinned_runs(self) -> tuple[ResultRun, ...]:
        """Return pinned runs in insertion order, without exposing mutable selection."""
        return tuple(self._index.get_run(key) for key in self._pins)

    def selected_figure(self, run_key: str) -> str | None:
        """Return the exact selected PNG filename, or None; unknown pin raises KeyError."""
        return self._pins[run_key]

    def _render(self) -> None:
        self._count.setText(f"Pinned runs: {len(self._pins)}")
        self._list.blockSignals(True)
        self._list.clear()
        for run in self.pinned_runs():
            item = QListWidgetItem(f"{run.entry} / {run.run_path.name}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if run.key in self._shown
                else Qt.CheckState.Unchecked
            )
            item.setToolTip(self._pins[run.key] or "No saved PNG for this run")
            self._list.addItem(item)
        self._list.blockSignals(False)
        table = build_comparison(self._index, tuple(self._pins))
        self._table.blockSignals(True)
        self._table.setRowCount(len(table.rows))
        self._table.setColumnCount(len(table.columns))
        self._table.setHorizontalHeaderLabels(table.columns)
        for row, values in enumerate(table.rows):
            for column, value in enumerate(values.cells):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self._table.setItem(row, column, item)
        self._table.resizeRowsToContents()
        self._table.blockSignals(False)
        if self._views.currentIndex() == 1:
            self._render_images()

    def _select_list_row(self, row: int) -> None:
        if row >= 0:
            self._selected_key = tuple(self._pins)[row]

    def _select_table_row(
        self, row: int, _column: int, _old_row: int, _old_column: int
    ) -> None:
        if row >= 0:
            self._selected_key = tuple(self._pins)[row]

    def _toggle_image(self, item: QListWidgetItem) -> None:
        key = tuple(self._pins)[self._list.row(item)]
        if item.checkState() == Qt.CheckState.Checked:
            if len(self._shown) == 4 and key not in self._shown:
                self._list.blockSignals(True)
                item.setCheckState(Qt.CheckState.Unchecked)
                self._list.blockSignals(False)
                self._feedback.setText(
                    "Select at most 4 runs. Uncheck one before adding another."
                )
                return
            self._shown.add(key)
        else:
            self._shown.discard(key)
        self._feedback.setText(f"Figures selected: {len(self._shown)} / 4")
        if self._views.currentIndex() == 1:
            self._render_images()

    def _set_mode(self, mode: int) -> None:
        self._views.setCurrentIndex(mode)
        if mode == 1:
            self._render_images()

    def _render_images(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item is not None and (widget := item.widget()) is not None:
                widget.setParent(None)
                widget.deleteLater()
        for number, key in enumerate(key for key in self._pins if key in self._shown):
            run = self._index.get_run(key)
            selector = self._pins[key]
            card = QWidget()
            layout = QVBoxLayout(card)
            reason = "No saved PNG for this run"
            if selector is not None:
                try:
                    image = ResultImage(self._index.read_figure(key, selector))
                except (OSError, ValueError, KeyError) as error:
                    reason = f"{selector}: {error}"
                else:
                    image.setObjectName(f"results.compare.image.{number}")
                    image.setMinimumSize(60, 60)
                    image.setSizePolicy(
                        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
                    )
                    image.setCursor(Qt.CursorShape.ArrowCursor)
                    layout.addWidget(image, 1)
                    reason = selector
            caption = QLabel(
                f"{run.entry} / {run.point or '—'}\n{run.experiment or '—'}\n{reason}"
            )
            caption.setObjectName(f"results.compare.caption.{number}")
            caption.setWordWrap(True)
            make_label_selectable(caption)
            layout.addWidget(caption)
            self._grid.addWidget(card, number // 2, number % 2)
        if not self._shown:
            self._grid.addWidget(
                QLabel("Check up to four pinned runs to compare figures"), 0, 0
            )
        for position in range(2):
            self._grid.setColumnStretch(position, 1)
            self._grid.setRowStretch(position, 1)

    def _remove_selected(self) -> None:
        if self._selected_key in self._pins:
            key = self._selected_key
            del self._pins[key]
            self._shown.discard(key)
            self._selected_key = None
            self._render()
            self.pins_changed.emit()

    def _clear(self) -> None:
        if self._pins:
            self._pins.clear()
            self._shown.clear()
            self._selected_key = None
            self._render()
            self.pins_changed.emit()
