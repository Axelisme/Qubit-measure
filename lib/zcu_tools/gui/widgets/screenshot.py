"""Shared QWidget screenshot encoding mechanics."""

from __future__ import annotations

from qtpy.QtCore import QBuffer, QIODevice  # type: ignore[attr-defined]
from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]


def widget_to_png_bytes(widget: QWidget, *, subject: str = "widget") -> bytes:
    """Capture ``widget`` and encode it as PNG bytes."""
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not widget.grab().save(buffer, "PNG"):
        raise RuntimeError(f"Failed to encode {subject} screenshot as PNG")
    return bytes(buffer.data().data())  # type: ignore[arg-type]
