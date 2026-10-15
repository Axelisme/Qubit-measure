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


def success_colour(widget: QWidget) -> QColor:
    """Return a success foreground for the current Window palette."""
    dark = widget.palette().color(QPalette.ColorRole.Window).lightness() < 128
    return QColor("#81c995" if dark else "#1e7d32")


def warning_colour(widget: QWidget) -> QColor:
    """Return an attention foreground for the current Window palette."""
    dark = widget.palette().color(QPalette.ColorRole.Window).lightness() < 128
    return QColor("#f6c26b" if dark else "#8a5a00")


def accent_colour(widget: QWidget) -> QColor:
    """Return an emphasized foreground for the current Window palette."""
    dark = widget.palette().color(QPalette.ColorRole.Window).lightness() < 128
    return QColor("#8ab4f8" if dark else "#1f5fae")


def secondary_colour(widget: QWidget) -> QColor:
    """Mute Text toward Window, retaining the palette's own text hue."""
    palette = widget.palette()
    return mixed_colour(
        palette.color(QPalette.ColorRole.Text),
        palette.color(QPalette.ColorRole.Window),
        0.42,
    )


def set_foreground(widget: QWidget, colour: QColor) -> None:
    """Set WindowText, Text and the widget's foregroundRole, keeping other roles.

    Supports labels, buttons and text edits. Item views' individual items use
    setForeground instead. Callers keep colour declarations out of stylesheets
    and calculate the colour from the owner of the visible Window background.
    """
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.WindowText, colour)
    palette.setColor(QPalette.ColorRole.Text, colour)
    palette.setColor(widget.foregroundRole(), colour)
    widget.setPalette(palette)
