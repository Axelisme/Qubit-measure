"""Theme-aware foreground colours shared by Qt presentation widgets."""

from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QWidget

# Light-theme error foreground and fixed red mixed into error backgrounds.
ERROR_RED_LIGHT = QColor("#b3261e")
_ERROR_RED_DARK = QColor("#ff8a80")


def mixed_colour(base: QColor, accent: QColor, share: float) -> QColor:
    """Mix RGB channels toward accent by share, from 0 (base) to 1 (accent).

    Channels are rounded to integers; the returned colour is opaque.
    """
    return QColor(
        round(base.red() + (accent.red() - base.red()) * share),
        round(base.green() + (accent.green() - base.green()) * share),
        round(base.blue() + (accent.blue() - base.blue()) * share),
    )


def error_colour(widget: QWidget) -> QColor:
    """Return an error foreground readable against widget's current Window colour.

    A Window lightness below 128 selects the dark-theme red; all other
    palettes use the light-theme red. Call again after a palette change.
    """
    window = widget.palette().color(QPalette.ColorRole.Window)
    return QColor(_ERROR_RED_DARK if window.lightness() < 128 else ERROR_RED_LIGHT)
