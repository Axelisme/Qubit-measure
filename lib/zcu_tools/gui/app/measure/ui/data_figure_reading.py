"""One scrollable reading surface for the Measure Data gallery."""

import logging
from collections.abc import Callable

from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence, QPixmap, QShortcut
from qtpy.QtWidgets import QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from zcu_tools.gui.lifetime import LifetimeScope

logger = logging.getLogger(__name__)


class DataFigureReading(QWidget):
    """Display one owner-rendered raster at native pixel size, or a local error.

    This pane owns no Figure or canvas. The gallery hides the pane and restores
    focus when `on_dismiss` is called by the Back button or Escape.
    """

    def __init__(
        self, owner_scope: LifetimeScope, on_dismiss: Callable[[], None]
    ) -> None:
        """Mount controls whose callbacks are retired with the Data gallery."""
        super().__init__()
        self.setObjectName("dataFigureReading")
        layout = QVBoxLayout(self)
        self._back = QPushButton("Back to previews")
        layout.addWidget(self._back, alignment=Qt.AlignmentFlag.AlignLeft)
        owner_scope.connect(self._back.clicked, on_dismiss)
        escape = QShortcut(QKeySequence("Esc"), self)
        escape.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        owner_scope.connect(escape.activated, on_dismiss)

        self._caption = QLabel()
        self._caption.setTextFormat(Qt.TextFormat.PlainText)
        self._caption.setWordWrap(True)
        layout.addWidget(self._caption)
        self._error = QLabel("Unable to render this figure.")
        self._error.setWordWrap(True)
        layout.addWidget(self._error)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("dataFigureReadingScroll")
        self._scroll.setAccessibleName("Figure image; scroll to see the full figure")
        viewport = self._scroll.viewport()
        if viewport is not None:
            viewport.setObjectName("dataFigureReadingViewport")
        self._image = QLabel()
        self._image.setObjectName("dataFigureReadingImage")
        self._scroll.setWidget(self._image)
        layout.addWidget(self._scroll, stretch=1)

    def show_figure(self, caption: str, render: Callable[[], bytes]) -> None:
        """Show `caption` and render once on the calling Qt thread, without I/O.

        `render` returns PNG bytes. Draw or decode errors display an inline
        notice without raising to the gallery or blocking the rest of Data.
        """
        self.clear()
        self._caption.setText(caption)
        try:
            pixmap = QPixmap()
            if not pixmap.loadFromData(render(), "PNG"):
                raise ValueError("Figure reading renderer returned invalid PNG")
        except Exception:
            logger.exception("Could not render Data figure %s", caption)
            self._error.show()
            return
        self._image.setPixmap(pixmap)
        self._image.resize(pixmap.size())
        self._scroll.show()

    def focus_back(self) -> None:
        """Start keyboard navigation at the return button after showing the pane."""
        self._back.setFocus()

    def clear(self) -> None:
        """Drop the displayed raster and error and reset scroll for the next figure."""
        self._caption.clear()
        self._image.clear()
        self._scroll.hide()
        self._error.hide()
        horizontal = self._scroll.horizontalScrollBar()
        vertical = self._scroll.verticalScrollBar()
        if horizontal is not None:
            horizontal.setValue(0)
        if vertical is not None:
            vertical.setValue(0)
