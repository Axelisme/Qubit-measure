"""Tile small figure sets without changing FigureContainer stack membership."""

from qtpy.QtCore import QEvent, QRect
from qtpy.QtGui import QResizeEvent, QShowEvent
from qtpy.QtWidgets import QStackedWidget, QWidget


class FigurePaneLayout(QStackedWidget):
    """Present up to four stack pages together, otherwise only the current page.

    Page zero is the placeholder. Canvases remain direct stack members so the
    existing FigureContainer retains attachment, cleanup and current-page rules.
    Call set_tiled after changing membership on the Qt owner thread. Interactive
    presentations use ordinary single-page mode. No Figures are owned here.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tiled = False
        self.currentChanged.connect(self._arrange)

    def set_tiled(self, *, tiled: bool) -> None:
        self._tiled = tiled
        self._arrange()

    def setCurrentWidget(self, w: QWidget | None) -> None:
        super().setCurrentWidget(w)
        self._arrange()

    def _arrange(self) -> None:
        count = self.count() - 1
        tiled = self._tiled and 0 < count <= 4 and self.currentIndex() > 0
        columns = 2 if count > 1 else 1
        rows = (count + columns - 1) // columns
        for index in range(self.count()):
            widget = self.widget(index)
            if widget is None:
                continue
            visible = index > 0 if tiled else index == self.currentIndex()
            widget.setVisible(visible)
            if tiled and index > 0:
                row, column = divmod(index - 1, columns)
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
