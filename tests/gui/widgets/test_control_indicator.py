"""Theme-aware control holder foregrounds at the widget interface."""

import pytest


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
    from zcu_tools.gui import widgets
    from zcu_tools.gui.control_identity import ControlIdentity, ControlStatus
    from zcu_tools.gui.widgets.control_indicator import ControlIndicator

    colour = getattr(widgets, colour_name, None)
    assert callable(colour), f"{colour_name} must be public"
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
