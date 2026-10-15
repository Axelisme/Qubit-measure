"""Palette-aware foreground declarations shared by the ledger and its editors."""

from collections.abc import Callable
from weakref import ref

from PyQt6 import sip
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QWidget

from zcu_tools.gui.widgets import set_foreground


class ThemedForegrounds:
    """Keep foreground colour functions for a writeback view's live widgets.

    owner supplies the background palette passed to every colour function.
    Use on the Qt GUI thread; the caller invokes refresh after palette changes.
    refresh drops destroyed Qt objects. Keep widgets registered until then so
    pending-delete children also follow palette changes.
    """

    def __init__(self, owner: QWidget) -> None:
        self._owner = ref(owner)
        self._colours: dict[QWidget, Callable[[QWidget], QColor]] = {}

    def set(self, widget: QWidget, colour: Callable[[QWidget], QColor]) -> None:
        """Record or replace widget's colour function and apply it using owner.

        colour accepts the owner widget and returns the desired foreground.
        Other palette roles remain unchanged. Both widgets must be live.
        """
        self._colours[widget] = colour
        owner = self._owner()
        if owner is not None:
            set_foreground(widget, colour(owner))

    def refresh(self) -> None:
        """Reapply foregrounds using owner's palette, removing destroyed widgets."""
        owner = self._owner()
        if owner is None:
            return
        # Avoid destroyed slots: Qt may destroy children during Python cyclic GC,
        # when a Python callback's closure has already been cleared.
        for widget, colour in list(self._colours.items()):
            if sip.isdeleted(widget):
                del self._colours[widget]
            else:
                set_foreground(widget, colour(owner))
