"""Canonical Measure figure rendering and viewport fitting (ADR-0069)."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QRectF, QSizeF, Qt
from qtpy.QtGui import QImage, QPainter, QPaintEvent, QPixmap, QResizeEvent, QShowEvent
from qtpy.QtWidgets import QLabel, QSizePolicy, QWidget

from ..figure_export import SAVE_DPI, SAVE_FIGSIZE


def _fit_figure(viewport: QRectF, source_size: QSizeF) -> QRectF:
    """Center a positive-size figure in the viewport, preserving its aspect ratio."""
    size = source_size.scaled(viewport.size(), Qt.AspectRatioMode.KeepAspectRatio)
    fitted = QRectF(viewport.topLeft(), size)
    fitted.moveCenter(viewport.center())
    return fitted


class FigureTileRaster(QLabel):
    """Fit a fixed-geometry image into a resizable Data preview or reader."""

    def __init__(self) -> None:
        super().__init__()
        self._original: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    @property
    def original_pixmap(self) -> QPixmap | None:
        """Return the unscaled source pixmap, or None if empty."""
        return self._original

    def present(self, pixmap: QPixmap | None) -> None:
        """Show the supplied raster; None clears it. Call on the Qt owner thread."""
        self._original = pixmap
        self._fit_image()

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self._fit_image()

    def _fit_image(self) -> None:
        if self._original is None or self._original.isNull():
            self.clear()
            return
        fitted = _fit_figure(QRectF(self.contentsRect()), QSizeF(self._original.size()))
        self.setPixmap(
            self._original.scaled(
                fitted.size().toSize(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class FigureTileCanvas(FigureCanvasQTAgg):
    """Render a native Figure at save geometry and only scale its on-screen image.

    Construct on the Qt owner thread. The Figure is installed on this canvas and
    set to the canonical inches and dpi. Normal Matplotlib draws and callbacks
    still operate on the original Figure, with Qt input mapped back to its pixels.
    """

    def __init__(self, figure: Figure) -> None:
        self._image = QImage()
        self._scaled = QImage()
        figure.set_size_inches(*SAVE_FIGSIZE, forward=False)
        super().__init__(figure)
        figure.set_dpi(SAVE_DPI)

    def _figure_rect(self) -> QRectF:
        return _fit_figure(QRectF(self.rect()), QSizeF(*SAVE_FIGSIZE))

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        # Do not let the Qt backend resize the native Figure or queue a render.
        QWidget.resizeEvent(self, event)
        self.update()

    def showEvent(self, event: QShowEvent | None) -> None:
        # The Qt backend tracks monitor DPI by changing the renderer size.
        # Here Qt scales a fixed-resolution image, even on a high-DPI display.
        QWidget.showEvent(self, event)
        if self._image.isNull():
            self.draw_idle()

    def draw(self) -> None:
        """Render current artists and retain a frame independent of savefig draws."""
        if self.is_saving():
            super().draw()
            return
        # Live artists can request their own size; presentation still uses the
        # same geometry as a save, regardless of the widget or authored size.
        if tuple(self.figure.get_size_inches()) != SAVE_FIGSIZE:
            self.figure.set_size_inches(*SAVE_FIGSIZE, forward=False)
        if self.figure.dpi != SAVE_DPI:
            self.figure.set_dpi(SAVE_DPI)
        super().draw()
        self._copy_frame()

    def draw_idle(self) -> None:
        """Schedule a draw, including if QtAgg already marked it pending on save."""
        super().draw_idle()
        self.update()

    def _copy_frame(self) -> None:
        width, height = self.get_width_height(physical=True)
        rgba = self.buffer_rgba()
        self._image = QImage(
            rgba.tobytes(), width, height, QImage.Format.Format_RGBA8888
        ).copy()
        self._scaled = QImage()
        self.update()

    def blit(self, bbox=None) -> None:
        """Refresh the full scaled image, even when only bbox changed."""
        del bbox
        self._copy_frame()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        # QtAgg marks its renderer dirty on print_figure. Flush before painting
        # so visible artists and the pointer transform agree after an export.
        if not self.is_saving():
            self._draw_idle()
        painter = QPainter(self)
        try:
            painter.eraseRect(self.rect() if event is None else event.rect())
            if not self._image.isNull():
                fitted = self._figure_rect()
                painter.drawImage(fitted, self._scaled_frame(fitted.size()))
                # The inherited toolbar draws its zoom rectangle in canvas pixels.
                scale = fitted.width() / self._image.width()
                painter.translate(fitted.topLeft())
                painter.scale(scale, scale)
                self._draw_rect_callback(painter)
        finally:
            painter.end()

    def _scaled_frame(self, logical_size: QSizeF) -> QImage:
        # Pre-filter downsampling. Direct painter scaling can drop fine lines
        # and tick glyphs when a tile is much smaller than the source image.
        device_size = (logical_size * self.devicePixelRatioF()).toSize()
        if self._scaled.size() != device_size:
            self._scaled = self._image.scaled(
                device_size,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        return self._scaled

    def mouseEventCoords(self, pos=None) -> tuple[float, float]:
        """Map Qt events/positions to Matplotlib's bottom-left renderer pixels."""
        x, y = super().mouseEventCoords(pos)
        fitted = self._figure_rect()
        if fitted.isEmpty():
            return (-1, -1)
        height = float(self.figure.bbox.height)
        scale = fitted.width() / self.figure.bbox.width
        return (x - fitted.left()) / scale, height + (y - height + fitted.top()) / scale
