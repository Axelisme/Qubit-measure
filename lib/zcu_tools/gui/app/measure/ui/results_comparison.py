"""Pinned summary-JSON tables and filename-aligned saved PNG comparisons."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.measure.services.results_comparison import build_comparison
from zcu_tools.gui.app.measure.services.results_index import ResultRun, ResultsIndex
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

from .results_gallery import ResultImage


class _RunHeader(QLabel):
    activated = pyqtSignal()

    def mouseDoubleClickEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is not None and a0.button() == Qt.MouseButton.LeftButton:
            self.activated.emit()
        super().mouseDoubleClickEvent(a0)


class ResultsComparison(QWidget):
    """Own ordered unique run pins, with a table and all saved PNGs together.

    pins_changed emits after membership changes. run_activated emits the opaque
    run key when a table row or image-column header is double-clicked. All
    callbacks run synchronously on the Qt owner thread. Pins belong only to this
    instance; no selected-image or image-count limit is applied. image_activated
    emits the clicked cell's opaque run key, exact PNG basename and native raster
    on left click. Missing or unreadable cells do not emit it.
    """

    pins_changed = pyqtSignal()
    run_activated = pyqtSignal(str)
    image_activated = pyqtSignal(str, str, QPixmap)

    def __init__(self, index: ResultsIndex, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self._pins: dict[str, ResultRun] = {}
        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        self._count = QLabel("Pinned runs: 0")
        self._count.setObjectName("results.pin_count")
        make_label_selectable(self._count)
        actions.addWidget(self._count, 1)
        clear = QPushButton("Clear comparison")
        clear.clicked.connect(self._clear)
        actions.addWidget(clear)
        layout.addLayout(actions)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setObjectName("results.compare.splitter")
        self._table = QTableWidget()
        self._table.setObjectName("results.comparison_table")
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.cellDoubleClicked.connect(self._activate_row)
        splitter.addWidget(self._table)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("results.compare.images")
        self._scroll.setWidgetResizable(True)
        splitter.addWidget(self._scroll)
        splitter.setSizes([220, 500])
        layout.addWidget(splitter, 1)
        self._rows: dict[str, QLabel] = {}
        self._render()

    def pin_run(self, run: ResultRun) -> None:
        """Pin an indexed run once; unknown keys raise KeyError without mutation."""
        run = self._index.get_run(run.key)
        if run.key not in self._pins:
            self._pins[run.key] = run
            self._render()
            self.pins_changed.emit()

    def pinned_runs(self) -> tuple[ResultRun, ...]:
        """Return pinned runs in insertion order without exposing mutable state."""
        return tuple(self._pins.values())

    def reveal_figure(self, selector: str) -> None:
        """Scroll to the exact PNG filename row; an absent filename raises KeyError.

        Call after this widget is shown so the scroll viewport has its final size.
        This changes only the scroll position, never pin membership or selection.
        """
        row = self._rows[selector]
        self._scroll.ensureWidgetVisible(row, 0, 0)

    def _render(self) -> None:
        self._count.setText(f"Pinned runs: {len(self._pins)}")
        table = build_comparison(self._index, tuple(self._pins))
        self._table.setRowCount(len(table.rows))
        self._table.setColumnCount(len(table.columns))
        self._table.setHorizontalHeaderLabels(table.columns)
        for row, values in enumerate(table.rows):
            for column, value in enumerate(values.cells):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self._table.setItem(row, column, item)
        self._table.resizeRowsToContents()
        self._render_images()

    def _render_images(self) -> None:
        content = QWidget()
        grid = QGridLayout(content)
        self._rows.clear()
        filenames = dict.fromkeys(
            path.name for run in self._pins.values() for path in run.figure_paths
        )
        for row, filename in enumerate(filenames, 1):
            caption = QLabel(filename)
            caption.setObjectName(f"results.compare.row.{row - 1}")
            caption.setWordWrap(True)
            make_label_selectable(caption)
            grid.addWidget(caption, row, 0)
            self._rows[filename] = caption
        for column, run in enumerate(self._pins.values(), 1):
            header = QWidget()
            line = QHBoxLayout(header)
            label = _RunHeader(
                f"{run.run_path.name[-6:]} · {run.at or 'Unknown time'}\n"
                f"{run.point or 'Global / unknown'} / {run.experiment or 'Unknown experiment'}"
            )
            label.setObjectName(f"results.compare.header.{column - 1}")
            label.setWordWrap(True)
            label.setToolTip(
                f"{run.entry} / {run.run_path.name}\nDouble-click to open run"
            )
            label.activated.connect(lambda key=run.key: self.run_activated.emit(key))
            line.addWidget(label, 1)
            remove = QPushButton("✕")
            remove.setObjectName(f"results.compare.remove.{column - 1}")
            remove.setToolTip(f"Unpin {run.run_path.name}")
            remove.clicked.connect(
                lambda _checked=False, key=run.key: self._remove(key)
            )
            line.addWidget(remove)
            grid.addWidget(header, 0, column)
            grid.setColumnMinimumWidth(column, 200)
            grid.setColumnStretch(column, 1)
            paths = {path.name for path in run.figure_paths}
            for row, filename in enumerate(filenames, 1):
                cell: QWidget
                if filename in paths:
                    try:
                        image = ResultImage(self._index.read_figure(run.key, filename))
                    except (OSError, ValueError, KeyError) as error:
                        cell = QLabel(f"{filename}: {error}")
                    else:
                        image.setMinimumWidth(60)
                        image.setObjectName(
                            f"results.compare.image.{row - 1}.{column - 1}"
                        )
                        image.setToolTip(f"{run.run_path.name}\n{filename}")
                        image.activated.connect(
                            lambda key=run.key, name=filename, source=image: (
                                self.image_activated.emit(
                                    key, name, source.original_pixmap()
                                )
                            )
                        )
                        cell = image
                else:
                    cell = QLabel("—")
                    cell.setObjectName(
                        f"results.compare.missing.{row - 1}.{column - 1}"
                    )
                grid.addWidget(cell, row, column)
        if not self._pins:
            grid.addWidget(QLabel("Add runs to compare saved results"), 0, 0)
        elif not filenames:
            grid.addWidget(
                QLabel("No saved PNGs for these runs"), 1, 0, 1, len(self._pins) + 1
            )
        content.setMinimumWidth(120 + 200 * len(self._pins))
        grid.setRowStretch(len(filenames) + 1, 1)
        self._scroll.setWidget(content)

    def _activate_row(self, row: int, _column: int) -> None:
        self.run_activated.emit(tuple(self._pins)[row])

    def _remove(self, key: str) -> None:
        del self._pins[key]
        self._render()
        self.pins_changed.emit()

    def _clear(self) -> None:
        if self._pins:
            self._pins.clear()
            self._render()
            self.pins_changed.emit()
