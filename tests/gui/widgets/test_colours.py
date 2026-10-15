"""Theme-aware error foreground shared by GUI widgets."""

from collections.abc import Callable

import pytest
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


@pytest.mark.parametrize(
    "name", ["success_colour", "warning_colour", "accent_colour", "secondary_colour"]
)
def test_semantic_foregrounds_follow_window_palette(qapp, name):
    from zcu_tools.gui.widgets import (
        accent_colour,
        secondary_colour,
        success_colour,
        warning_colour,
    )

    colours: dict[str, Callable[[QWidget], QColor]] = {
        "success_colour": success_colour,
        "warning_colour": warning_colour,
        "accent_colour": accent_colour,
        "secondary_colour": secondary_colour,
    }
    colour = colours[name]
    widget = QWidget()
    try:
        palette = widget.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("white"))
        palette.setColor(QPalette.ColorRole.Text, QColor("black"))
        widget.setPalette(palette)
        light = colour(widget)
        palette.setColor(QPalette.ColorRole.Window, QColor("#353535"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#eeeeee"))
        widget.setPalette(palette)
        dark = colour(widget)
        assert dark != light
        assert dark.lightness() > light.lightness()
        if name == "secondary_colour":
            for channel in ("red", "green", "blue"):
                assert (
                    getattr(palette.color(QPalette.ColorRole.Window), channel)()
                    < getattr(dark, channel)()
                    < getattr(palette.color(QPalette.ColorRole.Text), channel)()
                )
    finally:
        widget.deleteLater()
        qapp.processEvents()


def test_set_foreground_updates_label_and_edit_text_without_changing_window(qapp):
    from zcu_tools.gui.widgets import set_foreground

    widget = QWidget()
    try:
        original_window = widget.palette().color(QPalette.ColorRole.Window)
        colour = QColor("purple")
        set_foreground(widget, colour)
        assert widget.palette().color(QPalette.ColorRole.WindowText) == colour
        assert widget.palette().color(QPalette.ColorRole.Text) == colour
        assert widget.palette().color(QPalette.ColorRole.Window) == original_window
    finally:
        widget.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(
    "role", [QPalette.ColorRole.WindowText, QPalette.ColorRole.Text]
)
def test_reset_foreground_restores_live_inheritance(qapp, role):
    from qtpy.QtWidgets import QLabel
    from zcu_tools.gui.widgets import reset_foreground, set_foreground

    owner = QWidget()
    label = QLabel("Normal", owner)
    try:
        palette = owner.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor("black"))
        palette.setColor(QPalette.ColorRole.Text, QColor("black"))
        owner.setPalette(palette)
        set_foreground(label, QColor("purple"))
        assert label.palette().color(role) == QColor("purple")
        reset_foreground(label)
        palette.setColor(QPalette.ColorRole.WindowText, QColor("white"))
        palette.setColor(QPalette.ColorRole.Text, QColor("white"))
        palette.setColor(QPalette.ColorRole.Window, QColor("#353535"))
        owner.setPalette(palette)
        qapp.processEvents()
        assert label.palette().color(role) == owner.palette().color(role)
    finally:
        owner.deleteLater()
        qapp.processEvents()
