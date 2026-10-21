"""Renderers and geometry assertions shared by gallery and host contracts."""

from __future__ import annotations

import base64

from qtpy.QtCore import QBuffer, QByteArray, QIODevice, QRect, QSize
from qtpy.QtGui import QColor, QPixmap
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.measure.ui.data_figure_preview_gallery import (
    DataFigurePreviewGallery,
)

# Minimal 1x1 PNG for deterministic fake renderer (opaque, valid for QPixmap)
FAKE_PNG_BYTES = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def fake_renderer(_fig: object) -> bytes:
    """Ignore the figure and return a deterministic valid 1x1 PNG."""
    return bytes(FAKE_PNG_BYTES)


def make_png_bytes(width: int, height: int, color: str = "#286ac7") -> bytes:
    """Deterministic PNG bytes of exact size (for aspect-fit regression)."""
    pix = QPixmap(width, height)
    pix.fill(QColor(color))
    ba = QByteArray()
    buf = QBuffer(ba)
    # tolerated: QtPy promotes the binding OpenModeFlag members onto QIODevice at runtime.
    buf.open(QIODevice.WriteOnly)  # pyright: ignore[reportAttributeAccessIssue]
    pix.save(buf, "PNG")
    buf.close()
    # tolerated: SIP supports QByteArray buffer conversion to bytes, which the binding stub omits.
    return bytes(ba)  # pyright: ignore[reportArgumentType]


def renderer_640x480(_fig: object) -> bytes:
    """Ignore the figure and return a fresh 640x480 PNG; requires a Qt application."""
    # Each call returns fresh 640x480 PNG (production screenshot size).
    return make_png_bytes(640, 480, "#286ac7")


def _aspect_ratio(size: QSize) -> float:
    if size.height() == 0:
        return 0.0
    return float(size.width()) / float(size.height())


def assert_aspect_fit(
    viewport: QSize, displayed: QSize, original: QSize, *, tol: float = 0.02
) -> None:
    """Assert nonzero maximal viewport fit and original aspect ratio within tol."""
    # Displayed must fit within viewport (allow 1px rounding)
    assert displayed.width() <= viewport.width() + 1, (
        f"width {displayed.width()} > viewport {viewport.width()}"
    )
    assert displayed.height() <= viewport.height() + 1, (
        f"height {displayed.height()} > viewport {viewport.height()}"
    )
    # Must not be cropped to zero
    assert displayed.width() > 0 and displayed.height() > 0
    # Aspect ratio preserved
    orig_ar = _aspect_ratio(original)
    disp_ar = _aspect_ratio(displayed)
    assert abs(orig_ar - disp_ar) < tol, (
        f"aspect {disp_ar:.3f} vs original {orig_ar:.3f}"
    )
    # Displayed should be maximal within viewport: at least one dimension tight (within 2px)
    # For 640x480 -> 396x120 viewport, height should be ~120, width ~160.
    assert (
        abs(displayed.width() - viewport.width()) <= 2
        or abs(displayed.height() - viewport.height()) <= 2
    ), (
        f"not tight {displayed.width()}x{displayed.height()} in {viewport.width()}x{viewport.height()}"
    )


def gallery_geometry_cards(g: DataFigurePreviewGallery) -> dict[str, QRect]:
    """Deliver Qt events and return existing card rectangles in gallery coordinates."""
    # Force layout
    QApplication.processEvents()
    QApplication.processEvents()
    geoms: dict[str, QRect] = {}
    for key in ["run", "analysis", "post_analysis"]:
        c = g.find_card(key)
        if c is not None:
            geoms[key] = c.geometry()
    return geoms
