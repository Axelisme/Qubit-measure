"""Theme-aware control holder foregrounds at the widget interface."""

from collections.abc import Callable

import pytest
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QWidget


@pytest.mark.parametrize(
    "holder, locked, colour_name",
    [
        ("gui", False, "success_colour"),
        ("gui", True, "warning_colour"),
        ("rpc:agent", False, "accent_colour"),
    ],
)
def test_indicator_uses_theme_status_foreground(
    qapp, window_palette, holder, locked, colour_name
):
    from zcu_tools.gui.control_identity import ControlIdentity, ControlStatus
    from zcu_tools.gui.widgets import accent_colour, success_colour, warning_colour
    from zcu_tools.gui.widgets.control_indicator import ControlIndicator

    colours: dict[str, Callable[[QWidget], QColor]] = {
        "success_colour": success_colour,
        "warning_colour": warning_colour,
        "accent_colour": accent_colour,
    }
    colour = colours[colour_name]
    indicator = ControlIndicator(ControlIdentity())
    try:
        indicator.setPalette(window_palette)
        indicator.set_status(ControlStatus(holder, locked, True))
        indicator.ensurePolished()
        assert indicator.palette().color(indicator.foregroundRole()) == colour(
            indicator
        )
    finally:
        indicator.deleteLater()
        qapp.processEvents()
