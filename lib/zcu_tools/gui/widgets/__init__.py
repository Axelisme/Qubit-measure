"""Shared GUI widgets reused across the analysis apps.

App-agnostic Qt widgets that the sibling GUIs (fluxdep-gui, dispersive-gui)
would otherwise copy verbatim. Each widget is parameterised by the small bits an
app needs to vary (e.g. a label string), not forked. Unlike ``zcu_tools.gui``
itself this package DOES pull in Qt — import a widget only after the matplotlib
backend is selected (i.e. when building the UI).
"""

from zcu_tools.gui.widgets.dialog_lifecycle import DialogRefStore
from zcu_tools.gui.widgets.dialog_presenter import DialogPresenter, QtDialogPresenter
from zcu_tools.gui.widgets.screenshot import widget_to_png_bytes

__all__ = [
    "DialogPresenter",
    "DialogRefStore",
    "QtDialogPresenter",
    "widget_to_png_bytes",
]
