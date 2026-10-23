"""Tile small figure sets without changing FigureContainer stack membership."""

from PyQt6.QtCore import pyqtSignal
from qtpy.QtCore import QEvent, QObject, QRect
from qtpy.QtGui import QResizeEvent, QShowEvent
from qtpy.QtWidgets import QStackedWidget, QWidget

MAX_TILED_FIGURES = 4


class FigurePaneLayout(QStackedWidget):
    """Tile named canvases in pages of four, or show one interactive widget.

    Page zero is the placeholder. Canvases remain direct stack members so the
    existing FigureContainer retains attachment, cleanup and current-page rules.
    Call set_tiled after changing membership on the Qt owner thread. Interactive
    presentations use ordinary single-page mode. No Figures are owned here.
    presentationChanged fires after selection, membership removal or mode changes
    so the shared pager can refresh its count and button availability.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tiled = False
        self.currentChanged.connect(self._presentation_changed)
        self.widgetRemoved.connect(self._presentation_changed)

    presentationChanged = pyqtSignal()

    def set_tiled(self, *, tiled: bool) -> None:
        """Enable paged tiling, or disable it for an interactive presentation."""
        self._tiled = tiled
        for index in range(1, self.count()):
            widget = self.widget(index)
            if widget is not None:
                widget.installEventFilter(self)
        self._presentation_changed()

    def page_count(self) -> int:
        """Return the number of tiled pages, or zero in single-widget mode."""
        return (
            (self.count() - 1 + MAX_TILED_FIGURES - 1) // MAX_TILED_FIGURES
            if self._tiled
            else 0
        )

    def current_page(self) -> int:
        """Return the zero-based page of the current canvas, or zero when empty."""
        return max(0, self.currentIndex() - 1) // MAX_TILED_FIGURES

    def set_page(self, index: int) -> None:
        """Select a zero-based page and its first canvas; reject invalid indices."""
        if not 0 <= index < self.page_count():
            raise IndexError(f"Figure page {index} out of range")
        self.setCurrentIndex(1 + index * MAX_TILED_FIGURES)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if (
            self._tiled
            and a1 is not None
            and a1.type() == QEvent.Type.MouseButtonPress
            and isinstance(a0, QWidget)
            and self.indexOf(a0) > 0
        ):
            self.setCurrentWidget(a0)
        return super().eventFilter(a0, a1)

    def _presentation_changed(self) -> None:
        self._arrange()
        self.presentationChanged.emit()

    def setCurrentWidget(self, w: QWidget | None) -> None:
        super().setCurrentWidget(w)
        self._arrange()

    def _arrange(self) -> None:
        count = self.count() - 1
        tiled = self._tiled and count > 0 and self.currentIndex() > 0
        start = 1 + self.current_page() * MAX_TILED_FIGURES
        page_size = min(MAX_TILED_FIGURES, count - start + 1)
        columns = 2 if page_size > 1 else 1
        rows = (page_size + columns - 1) // columns
        for index in range(self.count()):
            widget = self.widget(index)
            if widget is None:
                continue
            visible = (
                start <= index < start + page_size
                if tiled
                else index == self.currentIndex()
            )
            widget.setVisible(visible)
            if tiled and visible:
                row, column = divmod(index - start, columns)
                x0 = column * self.width() // columns
                x1 = (column + 1) * self.width() // columns
                y0 = row * self.height() // rows
                y1 = (row + 1) * self.height() // rows
                widget.setGeometry(QRect(x0, y0, x1 - x0, y1 - y0))
                # Stack selection raises one child; restore visual reading order
                # for Qt accessibility traversal when all pages are peers.
                widget.raise_()
            elif visible:
                widget.setGeometry(self.rect())

    def event(self, e: QEvent | None) -> bool:
        handled = super().event(e)
        if e is not None and e.type() == QEvent.Type.LayoutRequest:
            # Raster resize changes size hints; QStackedLayout then expands its
            # current child again. Reapply the peer layout after that delivery.
            self._arrange()
        return handled

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self._arrange()

    def showEvent(self, a0: QShowEvent | None) -> None:
        super().showEvent(a0)
        self._arrange()
