from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QLabel  # type: ignore[attr-defined]
from zcu_tools.gui.widgets import widget_to_png_bytes


def test_widget_to_png_bytes_returns_png(qapp) -> None:  # noqa: ARG001
    assert widget_to_png_bytes(QLabel("capture")).startswith(b"\x89PNG\r\n\x1a\n")


def test_widget_to_png_bytes_fails_when_encoder_rejects() -> None:
    widget = MagicMock()
    widget.grab.return_value.save.return_value = False

    with pytest.raises(RuntimeError, match="test subject"):
        widget_to_png_bytes(widget, subject="test subject")
