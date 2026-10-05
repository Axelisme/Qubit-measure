"""Project run activity into main-window tab marker presentation."""

from __future__ import annotations

from qtpy.QtGui import QColor

_ACTIVITY_MARKER_PREFIX = "● "
_ACTIVITY_MARKER_COLOR = "#286ac7"
_ACTIVITY_MARKER_TOOLTIP = "Run in progress"


def activity_marker_presentation(
    label: str, *, active: bool
) -> tuple[str, QColor, str]:
    """Return the tab label, text color, and tooltip for run activity.

    The tooltip always carries the full label because the tab bar elides text.
    """
    if label.startswith(_ACTIVITY_MARKER_PREFIX):
        label = label[len(_ACTIVITY_MARKER_PREFIX) :]
    return (
        f"{_ACTIVITY_MARKER_PREFIX}{label}" if active else label,
        QColor(_ACTIVITY_MARKER_COLOR) if active else QColor(),
        f"{label} — {_ACTIVITY_MARKER_TOOLTIP}" if active else label,
    )
