"""Workflow figure area: the focused figure large, every figure as a thumbnail."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from matplotlib.figure import Figure
from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.plotting.container import FigureContainer
from zcu_tools.gui.plotting.explicit import QtPlotHost
from zcu_tools.gui.session.ports import OwnerScheduler
from zcu_tools.plotting.plots import Plots

from ._palette import PRIMARY_COLOR

# Thumbnails redraw from the canvases at most this often; the focused figure
# keeps the shared 50 ms presentation coalescing.
_THUMBNAIL_INTERVAL_MS = 250
_THUMBNAIL_SIZE = QSize(132, 112)
_PICTURE_SIZE = QSize(120, 78)
_STRIP_MARGIN = 4
# The focused thumbnail carries the blue frame of the primary action colour.
_THUMBNAIL_STYLE = (
    "QToolButton { border: 1px solid #c8ced6; border-radius: 4px;"
    " background: white; color: #4a5260; padding: 2px; }"
    "QToolButton:hover { border-color: #8fb3e6; }"
    f"QToolButton:checked {{ border: 2px solid {PRIMARY_COLOR}; background: #eaf2fd;"
    " color: #111827; font-weight: 600; }"
)


@dataclass
class _Thumbnail:
    """One presented figure: its native name, creation index and strip button."""

    figure: Figure
    name: str
    order: int
    button: QToolButton


class WorkflowFigureArea(QWidget):
    """Show one run's figures: the focused one large above a thumbnail strip.

    This widget is the PlotHost of every run's Plots. Each presented figure gets
    a thumbnail captioned with its env.plots name, ordered by creation in that
    Plots; the focused thumbnail is checked (blue frame) and the title reads
    "name   (k / n)". Follow latest starts checked: the focus then moves to
    each figure when it is first presented and whenever its live content
    refreshes. The final flush when a run finishes is not an update. Clicking
    a thumbnail focuses it and unchecks Follow latest; checking it again
    focuses the most recently updated figure. The strip scrolls horizontally
    and keeps the focused thumbnail in view. Thumbnails redraw from their
    canvases every 250 ms at most. Figures stay until their Plots release them.

    Object names: "workflow.figures" is the stack showing the focused canvas,
    "workflow.figures.title", "workflow.figures.follow",
    "workflow.figures.thumbnails" (the strip) and "workflow.figures.thumbnail"
    (each button). Construct, call and use on the Qt owner of owner.
    """

    def __init__(self, owner: OwnerScheduler, parent: QWidget | None = None) -> None:
        """Build the empty area; owner runs live plotting callbacks.

        Requires a QApplication on its GUI thread. No figure is shown until
        begin_run names a run and its Plots presents a figure.
        """
        super().__init__(parent)
        self.setObjectName("workflow.figure_area")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        self._title = QLabel(self)
        self._title.setObjectName("workflow.figures.title")
        self._follow = QCheckBox("Follow latest", self)
        self._follow.setObjectName("workflow.figures.follow")
        self._follow.setChecked(True)
        header.addWidget(self._title, 1)
        header.addWidget(self._follow)
        layout.addLayout(header)
        stack = QStackedWidget(self)
        stack.setObjectName("workflow.figures")
        placeholder = QLabel("Workflow plots", stack)
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stack.addWidget(placeholder)
        layout.addWidget(stack, 1)
        self._strip, self._content, self._row = self._thumbnail_strip()
        layout.addWidget(self._strip)
        self._buttons = QButtonGroup(self)
        self._buttons.setExclusive(True)
        self._buttons.buttonClicked.connect(self._thumbnail_clicked)
        self._follow.toggled.connect(self._follow_toggled)
        self._container = FigureContainer(stack, placeholder)
        self._qt = QtPlotHost(self._container, owner)
        self._plots: Plots | None = None
        self._thumbnails: list[_Thumbnail] = []
        self._focus: Figure | None = None
        self._latest: Figure | None = None
        self._stale: set[Figure] = set()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_THUMBNAIL_INTERVAL_MS)
        self._timer.timeout.connect(self._redraw_thumbnails)

    def _thumbnail_strip(self) -> tuple[QScrollArea, QWidget, QHBoxLayout]:
        strip = QScrollArea(self)
        strip.setObjectName("workflow.figures.thumbnails")
        # The row is sized to its buttons, so scrolling to a new thumbnail
        # works synchronously instead of after a deferred layout pass.
        strip.setWidgetResizable(False)
        strip.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget(strip)
        content.setStyleSheet(_THUMBNAIL_STYLE)
        row = QHBoxLayout(content)
        margin = _STRIP_MARGIN
        row.setContentsMargins(margin, margin, margin, margin)
        row.setSpacing(6)
        content.adjustSize()
        strip.setWidget(content)
        bar = strip.horizontalScrollBar()
        bar_height = 0 if bar is None else bar.sizeHint().height()
        strip.setFixedHeight(
            _THUMBNAIL_SIZE.height()
            + 2 * _STRIP_MARGIN
            + bar_height
            + 2 * strip.frameWidth()
        )
        return strip, content, row

    def begin_run(self, plots: Plots) -> None:
        """Take the next run's figure names from plots and check Follow latest.

        Call after the previous run's Plots released every figure and before
        plots presents one; remaining thumbnails raise RuntimeError.
        """
        if self._thumbnails:
            raise RuntimeError("Release the previous run's figures first")
        self._plots = plots
        self._latest = None
        self._follow.setChecked(True)

    def call[T](self, callback: Callable[[], T]) -> T:
        """Run callback synchronously on the owner and return its result."""
        return self._qt.call(callback)

    def present(self, figure: Figure) -> None:
        """Show figure and add its thumbnail in creation order.

        Call on the owner for a figure of the begun run's Plots; another
        container's figure raises ValueError, a figure without a name in that
        Plots raises ValueError and no begun run raises RuntimeError. The
        figure takes focus when Follow latest is checked or nothing has focus.
        """
        if self._record(figure) is None:
            # Reject an unnamed figure before its canvas is attached.
            name, order = self._name_of(figure)
            self._qt.present(figure)
            self._add_thumbnail(figure, name, order)
        else:
            self._qt.present(figure)
        self._latest = figure
        if self._focus is None or self._follow.isChecked():
            self._focus_on(figure)
        else:
            # Presenting selected the new canvas; keep showing the pinned one.
            self._container.set_current_canvas(self._canvas(self._focus))
            self._update_title()
        self._mark_stale(figure)

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        """Coalesce a redraw of a presented figure and schedule its thumbnail.

        Call on the owner (else RuntimeError) for a figure this area presents
        (else ValueError). A live refresh (final=False) marks figure the latest
        update and focuses it when Follow latest is checked. final draws
        immediately and never moves focus.
        """
        self._qt.refresh(figure, final=final)
        self._mark_stale(figure)
        if final:
            return
        self._latest = figure
        if figure is not self._focus and self._follow.isChecked():
            self._focus_on(figure)

    def release(self, figure: Figure) -> None:
        """Remove figure's canvas and thumbnail, keeping its artists for saving.

        Call on the owner (else RuntimeError) after the producer stops; a
        figure presented by another container raises ValueError. An
        unpresented or already released figure only clears pending work. If
        figure had focus, the last remaining thumbnail takes it.
        """
        self._qt.release(figure)
        self._stale.discard(figure)
        if self._latest is figure:
            self._latest = None
        record = self._record(figure)
        if record is None:
            return
        self._thumbnails.remove(record)
        self._buttons.removeButton(record.button)
        self._row.removeWidget(record.button)
        # Leave the strip now; deleteLater alone waits for the event loop.
        record.button.setParent(None)
        record.button.deleteLater()
        self._content.adjustSize()
        if self._focus is figure:
            self._focus = None
            if self._thumbnails:
                self._focus_on(self._thumbnails[-1].figure)
        self._update_title()

    def _name_of(self, figure: Figure) -> tuple[str, int]:
        """Return figure's env.plots name and its creation index in the run."""
        if self._plots is None:
            raise RuntimeError("No run's figures have begun")
        for order, name in enumerate(self._plots):
            if self._plots[name] is figure:
                return name, order
        raise ValueError("Figure is not one of this run's named figures")

    def _add_thumbnail(self, figure: Figure, name: str, order: int) -> None:
        button = QToolButton(self._content)
        button.setObjectName("workflow.figures.thumbnail")
        button.setCheckable(True)
        button.setFixedSize(_THUMBNAIL_SIZE)
        button.setIconSize(_PICTURE_SIZE)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        caption_width = _THUMBNAIL_SIZE.width() - 8
        button.setText(
            button.fontMetrics().elidedText(
                name, Qt.TextElideMode.ElideMiddle, caption_width
            )
        )
        button.setToolTip(name)
        place = sum(1 for record in self._thumbnails if record.order < order)
        self._thumbnails.insert(place, _Thumbnail(figure, name, order, button))
        self._buttons.addButton(button)
        self._row.insertWidget(place, button)
        # A child added to a shown parent appears only after a queued event;
        # show it now so the row's size, and scrolling, account for it.
        button.show()
        self._content.adjustSize()

    def _focus_on(self, figure: Figure) -> None:
        """Show figure large, frame its thumbnail and scroll the strip to it."""
        record = self._record(figure)
        if record is None:
            raise RuntimeError("Only a presented figure can take focus")
        self._focus = figure
        self._container.set_current_canvas(self._canvas(figure))
        record.button.setChecked(True)
        self._strip.ensureWidgetVisible(record.button, _STRIP_MARGIN, 0)
        self._update_title()

    def _thumbnail_clicked(self, button: QAbstractButton) -> None:
        for record in self._thumbnails:
            if record.button is button:
                self._follow.setChecked(False)
                self._focus_on(record.figure)
                return

    def _follow_toggled(self) -> None:
        latest = self._latest
        if (
            self._follow.isChecked()
            and latest is not None
            and latest is not self._focus
        ):
            self._focus_on(latest)

    def _update_title(self) -> None:
        for place, record in enumerate(self._thumbnails, start=1):
            if record.figure is self._focus:
                count = len(self._thumbnails)
                self._title.setText(f"{record.name}   ({place} / {count})")
                return
        self._title.setText("")

    def _mark_stale(self, figure: Figure) -> None:
        self._stale.add(figure)
        if not self._timer.isActive():
            self._timer.start()

    def _redraw_thumbnails(self) -> None:
        stale, self._stale = self._stale, set()
        for record in self._thumbnails:
            if record.figure not in stale:
                continue
            canvas = record.figure.canvas
            # A canvas the workflow replaced has no picture to grab; the
            # thumbnail keeps its last one rather than stopping the GUI.
            if isinstance(canvas, QWidget):
                picture = canvas.grab().scaled(
                    _PICTURE_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                record.button.setIcon(QIcon(picture))

    def _record(self, figure: Figure) -> _Thumbnail | None:
        for record in self._thumbnails:
            if record.figure is figure:
                return record
        return None

    @staticmethod
    def _canvas(figure: Figure) -> QWidget:
        canvas = figure.canvas
        if not isinstance(canvas, QWidget):
            raise RuntimeError("Presented Figure has no Qt canvas")
        return canvas
