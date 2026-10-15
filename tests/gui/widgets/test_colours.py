"""Theme-aware error foreground shared by GUI widgets."""

import pytest
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication, QWidget


@pytest.mark.parametrize(
    "window, expected", [("#ffffff", "#b3261e"), ("#353535", "#ff8a80")]
)
def test_error_colour_uses_widget_window_lightness(
    qapp: QApplication, window: str, expected: str
) -> None:
    from zcu_tools.gui.widgets import error_colour

    widget = QWidget()
    try:
        palette = widget.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(window))
        widget.setPalette(palette)
        assert error_colour(widget).name() == expected
    finally:
        widget.deleteLater()
        qapp.processEvents()
