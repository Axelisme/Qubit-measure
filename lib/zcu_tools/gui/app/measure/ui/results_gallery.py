"""Scrollable saved PNG gallery, independent of the measure figure pager."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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
    reveals that PNG row; COPY_IMAGE copies decoded PNG pixels to the clipboard.
    Values are stable command identifiers, never presentation text.
    """

    COPY_PATH = "copy_path"
    ADD_TO_COMPARISON = "add_to_comparison"
    COPY_IMAGE = "copy_image"


class ResultImage(QLabel):
    """Clickable saved raster, from PNG bytes or a decoded QPixmap.

    Invalid PNG bytes or an empty pixmap raise ValueError.

    activated emits on left click. action_requested emits one ResultImageAction
    on context-menu selection; dismissing the menu emits nothing. The receiver
    owns command execution; this widget only presents and requests actions.
    """

    activated = pyqtSignal()
    action_requested = pyqtSignal(ResultImageAction)

    def __init__(self, png: bytes | QPixmap, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = QPixmap(png) if isinstance(png, QPixmap) else QPixmap()
        if isinstance(png, bytes) and not self._image.loadFromData(png):
            raise ValueError("Cannot decode saved PNG")
        if self._image.isNull():
            raise ValueError("Cannot display an empty raster")
        self.setMinimumWidth(180)
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def original_pixmap(self) -> QPixmap:
        """Return an implicitly shared copy of the unscaled decoded raster."""
        return QPixmap(self._image)

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


@dataclass
class _SavedCard:
    signature: tuple[int, int, int, int, int]
    frame: QFrame
    caption: QLabel
    image: ResultImage


class ResultsGallery(QScrollArea):
    """Tile saved previews in two columns; the browser owns focus and commands."""

    def __init__(self, index: ResultsIndex, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        # At most one card per indexed PNG, owned by this gallery's lifetime.
        self._cards: dict[tuple[str, str], _SavedCard] = {}
        self._retained = QWidget(self)
        self._retained.hide()
        self._presentation: tuple[
            tuple[str, str, tuple[int, int, int, int, int] | None], ...
        ] = ()
        self._all_figures = False
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
        """Show each run's first PNG, or all PNGs when all_figures.

        Reuse cards and decoded rasters for unchanged indexed members within this
        gallery. File identity, size and modification/change times invalidate a
        retained card; failed reads are retried on the next presentation.

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
        self._activate = activate
        self._action = action
        members = [
            (run, path, self._signature(path))
            for run in runs
            for path in (run.figure_paths if all_figures else run.figure_paths[:1])
        ]
        presentation = tuple(
            (run.key, path.name, signature) for run, path, signature in members
        )
        if (
            self.widget() is not None
            and presentation == self._presentation
            and all_figures == self._all_figures
            and all(
                signature is not None and (key, name) in self._cards
                for key, name, signature in presentation
            )
        ):
            return
        for saved in self._cards.values():
            saved.frame.setParent(self._retained)
            saved.frame.hide()
            saved.frame.setObjectName("")
            saved.image.setObjectName("")
        old = self.takeWidget()
        if old is not None:
            old.setParent(None)
            old.deleteLater()
        content = QWidget()
        grid = QGridLayout(content)
        grid.setColumnStretch(0, 1)
        # Stretch a second column only when it has cards; a lone card uses all width.
        grid.setColumnStretch(1, int(len(members) > 1))
        number = 0
        for run, path, signature in members:
            try:
                saved = self._saved_card(run, path, signature)
            except (OSError, ValueError, KeyError) as error:
                grid.addWidget(QLabel(f"{path.name}: {error}"), number // 2, number % 2)
                number += 1
                continue
            saved.caption.setText(
                path.name
                if all_figures
                else f"{run.experiment or 'Unknown experiment'} · {run.run_path.name[-6:]}\n"
                f"{run.entry} / {run.point or 'Global / unknown'} · {run.at or 'Unknown time'}"
            )
            saved.frame.setObjectName(f"results.card.{number}")
            saved.image.setObjectName(f"results.figure.{number}")
            grid.addWidget(saved.frame, number // 2, number % 2)
            saved.frame.show()
            number += 1
        if number == 0:
            grid.addWidget(QLabel("No figures in this selection"), 0, 0)
        grid.setRowStretch(grid.rowCount(), 1)
        self.setWidget(content)
        self._presentation = presentation
        self._all_figures = all_figures

    def original_pixmap(self, run: ResultRun, selector: str) -> QPixmap:
        """Return the indexed PNG's native raster, reusing an unchanged card.

        Unknown run/member raises KeyError; read/decode errors propagate.
        The caller owns presentation; this gallery never opens another window.
        """
        run = self._index.get_run(run.key)
        path = next((path for path in run.figure_paths if path.name == selector), None)
        if path is None:
            raise KeyError(selector)
        return self._saved_card(
            run, path, self._signature(path)
        ).image.original_pixmap()

    @staticmethod
    def _signature(path: Path) -> tuple[int, int, int, int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def _saved_card(
        self,
        run: ResultRun,
        path: Path,
        signature: tuple[int, int, int, int, int] | None,
    ) -> _SavedCard:
        key = (run.key, path.name)
        saved = self._cards.get(key)
        if saved is not None and saved.signature != signature:
            saved.frame.deleteLater()
            del self._cards[key]
            saved = None
        if saved is None:
            saved = self._read_card(run, path)
            saved.frame.setParent(self._retained)
            self._cards[key] = saved
        return saved

    def _read_card(self, run: ResultRun, path: Path) -> _SavedCard:
        image = ResultImage(self._index.read_figure(run.key, path.name))
        signature = self._signature(path)
        if signature is None:
            image.deleteLater()
            raise OSError(f"Saved PNG disappeared: {path}")
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(card)
        caption = QLabel()
        caption.setWordWrap(True)
        make_label_selectable(caption)
        caption.setStyleSheet("font-weight: 600; padding-bottom: 3px;")
        layout.addWidget(caption)
        image.setToolTip(
            f"{run.entry} / {run.point or 'Global / unknown'}\n"
            f"{run.experiment or 'Unknown experiment'}\n{run.run_path.name}\n"
            f"{run.at or 'Unknown date'}\n{path}"
        )
        image.activated.connect(lambda: self._activate(run, path.name))
        image.action_requested.connect(
            lambda command: self._action(run, path.name, command)
        )
        layout.addWidget(image)
        return _SavedCard(signature, card, caption, image)
