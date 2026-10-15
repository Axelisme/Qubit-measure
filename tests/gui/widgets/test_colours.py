"""Theme-aware error foreground shared by GUI widgets."""

from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication, QWidget


def test_error_colour_uses_widget_window_lightness(qapp: QApplication) -> None:
    from zcu_tools.gui.widgets import error_colour

    widget = QWidget()
    try:
        palette = widget.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#ffffff"))
        widget.setPalette(palette)
        light_theme_red = error_colour(widget)
        palette.setColor(QPalette.ColorRole.Window, QColor("#353535"))
        widget.setPalette(palette)
        dark_theme_red = error_colour(widget)
        assert dark_theme_red != light_theme_red
        assert dark_theme_red.lightness() > light_theme_red.lightness()
    finally:
        widget.deleteLater()
        qapp.processEvents()
