"""Scrollable saved PNG gallery, independent of the measure figure pager."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from PyQt6.QtCore import pyqtSignal
from qtpy.QtCore import QPoint, QSize, Qt
from qtpy.QtGui import QMouseEvent, QPixmap, QResizeEvent
from qtpy.QtWidgets import (
    QFrame,
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


class ResultImageAction(Enum):
    """Saved-image commands, independent of menu captions.

    COPY_PATH copies the exact member path; ADD_TO_COMPARISON pins its run and
    selected member; COPY_IMAGE copies decoded PNG pixels to the clipboard.
    Values are stable command identifiers, never presentation text.
    """

    COPY_PATH = "copy_path"
    ADD_TO_COMPARISON = "add_to_comparison"
    COPY_IMAGE = "copy_image"


class ResultImage(QLabel):
    """Clickable PNG raster; invalid PNG bytes raise ValueError.

    activated emits on left click. action_requested emits one ResultImageAction
    on context-menu selection; dismissing the menu emits nothing. The receiver
    owns command execution; this widget only presents and requests actions.
    """

    activated = pyqtSignal()
    action_requested = pyqtSignal(ResultImageAction)

    def __init__(self, png: bytes, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = QPixmap()
        if not self._image.loadFromData(png):
            raise ValueError("Cannot decode saved PNG")
        self.setMinimumWidth(180)
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def mousePressEvent(self, ev: QMouseEvent | None) -> None:
        if ev is not None and ev.button() == Qt.MouseButton.LeftButton:
            self.activated.emit()
        super().mousePressEvent(ev)

    def heightForWidth(self, a0: int) -> int:
        """Return the saved raster's aspect-preserving height at width pixels."""
        return round(a0 * self._image.height() / self._image.width())

    def sizeHint(self) -> QSize:
        """Prefer a small initial preview; layouts grow it with the available width."""
        return QSize(180, self.heightForWidth(180))

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
        actions = [
            (menu.addAction(caption), command)
            for caption, command in (
                ("Copy path", ResultImageAction.COPY_PATH),
                ("Add to comparison", ResultImageAction.ADD_TO_COMPARISON),
                ("Copy image", ResultImageAction.COPY_IMAGE),
            )
        ]
        selected = menu.exec(self.mapToGlobal(point))
        for action, command in actions:
            if selected is action:
                self.action_requested.emit(command)
                break
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
        action: Callable[[ResultRun, str, ResultImageAction], None],
        *,
        all_figures: bool = False,
    ) -> None:
        """Replace cards with each run's first PNG, or all PNGs when all_figures.

        runs contains indexed runs from this gallery's ResultsIndex. By default
        show each run's first indexed PNG; all_figures=True shows every member.
        Runs without PNGs have no card. Unreadable members retain their filename
        and error instead of an interactive image.

        Left clicks call activate(run, filename). Context-menu selections call
        action(run, filename, command), where command is COPY_PATH,
        ADD_TO_COMPARISON or COPY_IMAGE from ResultImageAction. filename is the
        exact indexed PNG basename, suitable for ResultsIndex.read_figure.
        Callbacks run synchronously on the Qt owner thread and may change focus,
        replace this presentation, pin the run/member or copy its path/pixels.
        This gallery does not execute commands or write saved results. Callback
        exceptions are not caught; dismissing the menu invokes no callback.
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
                card = QFrame()
                card.setFrameShape(QFrame.Shape.StyledPanel)
                layout = QVBoxLayout(card)
                caption = QLabel(
                    path.name
                    if all_figures
                    else f"{run.experiment or 'Unknown experiment'} · {run.run_path.name[-6:]}\n"
                    f"{run.entry} / {run.point or 'Global / unknown'} · {run.at or 'Unknown time'}"
                )
                caption.setWordWrap(True)
                make_label_selectable(caption)
                caption.setStyleSheet("font-weight: 600; padding-bottom: 3px;")
                layout.addWidget(caption)
                card.setObjectName(f"results.card.{number}")
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
                grid.addWidget(card, number // 2, number % 2)
                number += 1
        if number == 0:
            grid.addWidget(QLabel("No figures in this selection"), 0, 0)
        grid.setRowStretch(grid.rowCount(), 1)
        self.setWidget(content)
