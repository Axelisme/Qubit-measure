"""Theme-aware foreground colours shared by Qt presentation widgets."""

from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QWidget

# Forms also mix this fixed red into Base for invalid direct-input backgrounds.
ERROR_RED_LIGHT = QColor("#b3261e")
_ERROR_RED_DARK = QColor("#ff8a80")


def error_colour(widget: QWidget) -> QColor:
    """Return an error foreground readable against widget's current Window colour.

    A Window lightness below 128 selects the dark-theme red; all other
    palettes use the light-theme red. Call again after a palette change.
    """
    window = widget.palette().color(QPalette.ColorRole.Window)
    return QColor(_ERROR_RED_DARK if window.lightness() < 128 else ERROR_RED_LIGHT)
