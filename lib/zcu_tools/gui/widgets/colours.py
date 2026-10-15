"""Theme-aware foreground colours shared by Qt presentation widgets."""

from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QWidget

# Light-theme error foreground and fixed red mixed into error backgrounds.
ERROR_RED_LIGHT = QColor("#b3261e")
# Error foreground on dark Window backgrounds.
_ERROR_RED_DARK = QColor("#ff8a80")
# Success foreground on light Window backgrounds.
_SUCCESS_GREEN_LIGHT = QColor("#1e7d32")
# Success foreground on dark Window backgrounds.
_SUCCESS_GREEN_DARK = QColor("#81c995")
# Attention foreground on light Window backgrounds.
_WARNING_AMBER_LIGHT = QColor("#8a5a00")
# Attention foreground on dark Window backgrounds.
_WARNING_AMBER_DARK = QColor("#f6c26b")
# Emphasized foreground on light Window backgrounds.
_ACCENT_BLUE_LIGHT = QColor("#1f5fae")
# Emphasized foreground on dark Window backgrounds.
_ACCENT_BLUE_DARK = QColor("#8ab4f8")
# Window share mixed into Text for muted descriptions and paths.
_SECONDARY_WINDOW_SHARE = 0.42


def _is_dark(widget: QWidget) -> bool:
    return widget.palette().color(QPalette.ColorRole.Window).lightness() < 128


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
    return QColor(_ERROR_RED_DARK if _is_dark(widget) else ERROR_RED_LIGHT)


def success_colour(widget: QWidget) -> QColor:
    """Return a success foreground for the current Window palette."""
    return QColor(_SUCCESS_GREEN_DARK if _is_dark(widget) else _SUCCESS_GREEN_LIGHT)


def warning_colour(widget: QWidget) -> QColor:
    """Return an attention foreground for the current Window palette."""
    return QColor(_WARNING_AMBER_DARK if _is_dark(widget) else _WARNING_AMBER_LIGHT)


def accent_colour(widget: QWidget) -> QColor:
    """Return an emphasized foreground for the current Window palette."""
    return QColor(_ACCENT_BLUE_DARK if _is_dark(widget) else _ACCENT_BLUE_LIGHT)


def secondary_colour(widget: QWidget) -> QColor:
    """Mute Text toward Window, retaining the palette's own text hue."""
    palette = widget.palette()
    return mixed_colour(
        palette.color(QPalette.ColorRole.Text),
        palette.color(QPalette.ColorRole.Window),
        _SECONDARY_WINDOW_SHARE,
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


def reset_foreground(widget: QWidget) -> None:
    """Restore inherited WindowText, Text and foregroundRole colours on widget.

    Clears the explicit roles set by set_foreground, so later parent or
    application palette changes reach the widget. Other explicitly set palette
    brushes are preserved; no stylesheet declarations are changed.
    """
    current = widget.palette()
    inherited_roles = {
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        widget.foregroundRole(),
    }
    # QPalette has no per-role unset; rebuild from the other explicit brushes.
    palette = QPalette()
    for group in (
        QPalette.ColorGroup.Active,
        QPalette.ColorGroup.Inactive,
        QPalette.ColorGroup.Disabled,
    ):
        for role in QPalette.ColorRole:
            if (
                role not in inherited_roles
                and role
                not in (QPalette.ColorRole.NoRole, QPalette.ColorRole.NColorRoles)
                and current.isBrushSet(group, role)
            ):
                palette.setBrush(group, role, current.brush(group, role))
    widget.setPalette(palette)
