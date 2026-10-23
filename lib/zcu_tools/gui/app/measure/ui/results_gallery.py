"""Scrollable saved PNG gallery, independent of the measure figure pager."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import pyqtSignal
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QMouseEvent, QPixmap, QResizeEvent
from qtpy.QtWidgets import (
    QGridLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.measure.services.results_index import ResultRun, ResultsIndex
from zcu_tools.gui.widgets.selectable_text import make_label_selectable


class ResultImage(QLabel):
    """Clickable raster preview; left click selects, right click opens member actions."""

    activated = pyqtSignal()
    action_requested = pyqtSignal(str)

    def __init__(self, png: bytes, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = QPixmap()
        if not self._image.loadFromData(png):
            raise ValueError("Cannot decode saved PNG")
        self.setMinimumSize(180, 160)
        self.setMaximumHeight(230)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def mousePressEvent(self, ev: QMouseEvent | None) -> None:
        if ev is not None and ev.button() == Qt.MouseButton.LeftButton:
            self.activated.emit()
        super().mousePressEvent(ev)

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self.setPixmap(
            self._image.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _context_menu(self, point: QPoint) -> None:
        menu = QMenu(self)
        for caption in ("Copy path", "Add to comparison", "Copy image"):
            menu.addAction(caption)
        action = menu.exec(self.mapToGlobal(point))
        if action is not None:
            self.action_requested.emit(action.text())
        menu.deleteLater()


class ResultsGallery(QScrollArea):
    """Tile saved previews in two columns; the browser owns focus and commands."""

    def __init__(self, index: ResultsIndex, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self.setObjectName("results.gallery")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def present(
        self,
        runs: tuple[ResultRun, ...],
        activate: Callable[[ResultRun, str], None],
        action: Callable[[ResultRun, str, str], None],
        *,
        all_figures: bool = False,
    ) -> None:
        """Replace cards with each run's first PNG, or all PNGs when all_figures.

        Runs without PNGs have no card. Unreadable members retain their filename
        and error. Left clicks call activate; context-menu actions call action.
        """
        old = self.takeWidget()
        if old is not None:
            old.setParent(None)
            old.deleteLater()
        content = QWidget()
        grid = QGridLayout(content)
        number = 0
        for run in runs:
            paths = run.figure_paths if all_figures else run.figure_paths[:1]
            for path in paths:
                card = QWidget()
                layout = QVBoxLayout(card)
                caption = QLabel(
                    f"{run.entry} / {run.point or 'Global / unknown'}\n{path.name}"
                )
                caption.setWordWrap(True)
                make_label_selectable(caption)
                try:
                    image = ResultImage(self._index.read_figure(run.key, path.name))
                except (OSError, ValueError, KeyError) as error:
                    layout.addWidget(QLabel(f"{path.name}: {error}"))
                else:
                    image.setObjectName(f"results.figure.{number}")
                    image.setToolTip(
                        f"{run.entry} / {run.point or 'Global / unknown'}\n"
                        f"{run.experiment or 'Unknown experiment'}\n{run.run_path.name}\n"
                        f"{run.at or 'Unknown date'}\n{path}"
                    )
                    image.activated.connect(lambda r=run, f=path.name: activate(r, f))
                    image.action_requested.connect(
                        lambda a, r=run, f=path.name: action(r, f, a)
                    )
                    layout.addWidget(image)
                layout.addWidget(caption)
                grid.addWidget(card, number // 2, number % 2)
                number += 1
        if number == 0:
            grid.addWidget(QLabel("No figures in this selection"), 0, 0)
        grid.setRowStretch(grid.rowCount(), 1)
        self.setWidget(content)
