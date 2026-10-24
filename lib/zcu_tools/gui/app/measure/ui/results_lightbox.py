"""Browser-owned, aspect-fit saved-image overlay shared by both result views."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import (
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPixmap,
    QResizeEvent,
)
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from .results_gallery import ResultImage


class ResultLightbox(QWidget):
    """Cover the parent browser with one saved raster until dismissed.

    The browser must keep this child's geometry equal to its content rectangle.
    present accepts an unscaled raster and one-line identity caption. While
    visible it ignores further presentations. Clicks outside the fitted image,
    including the caption, or Escape hide it; image clicks keep it open.
    All operations run on the Qt owner thread. No independent window is created.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("results.lightbox")
        self.setStyleSheet("background: transparent; color: white;")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 12)
        self._caption = QLabel()
        self._caption.setObjectName("results.lightbox.caption")
        self._caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption.setStyleSheet(
            "background: transparent; color: white; font-size: 11px;"
        )
        layout.addWidget(self._caption)
        self._area = QWidget()
        self._area.setObjectName("results.lightbox.area")
        layout.addWidget(self._area, 1)
        self._image: ResultImage | None = None
        self.hide()

    def present(self, raster: QPixmap, caption: str) -> None:
        """Show a valid native raster with its short run id / exact PNG filename.

        An empty raster raises ValueError. Ignore requests while already visible,
        so an overlay cannot silently swap its image. Reuse no top-level window.
        """
        if self.isVisible():
            return
        if self._image is not None:
            self._image.hide()
            self._image.setObjectName("")
            self._image.deleteLater()
        self._image = ResultImage(raster, self._area)
        self._image.setMinimumWidth(0)
        self._image.setObjectName("results.lightbox.image")
        self._image.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self._image.setCursor(Qt.CursorShape.ArrowCursor)
        self._caption.setText(caption)
        self.show()
        self.raise_()
        self._fit_image()
        self._image.show()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self._fit_image()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        super().paintEvent(a0)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 185))
        painter.end()

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is not None:
            image = self._image
            point = self.mapToGlobal(a0.pos())
            if image is None or not image.rect().contains(image.mapFromGlobal(point)):
                self.hide()
            a0.accept()

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        if a0 is not None and a0.key() == Qt.Key.Key_Escape:
            self.hide()
            a0.accept()
        else:
            super().keyPressEvent(a0)

    def _fit_image(self) -> None:
        if self._image is None:
            return
        size = (
            self._image.original_pixmap()
            .size()
            .scaled(self._area.size(), Qt.AspectRatioMode.KeepAspectRatio)
        )
        self._image.setGeometry(
            (self._area.width() - size.width()) // 2,
            (self._area.height() - size.height()) // 2,
            size.width(),
            size.height(),
        )
