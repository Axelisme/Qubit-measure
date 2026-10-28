"""FigureContainer — a plot stack widget the host attaches canvases into.

A thin, **passive** wrapper around a ``QStackedWidget`` with a fixed placeholder.
It holds no signals and does no thread marshalling — the host (``host.py``) owns
the single main-thread bridge QObject and calls these synchronous methods on the
main thread. Worker code uses an explicit plot host, which schedules artist
updates and presentation on the owner thread.

Constructing a Container ensures the host bridge is initialised on the main
thread first (the canvas thread-affinity invariant — see ``host.ensure_host``).
"""

from __future__ import annotations

from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QStackedWidget, QWidget

from . import host as _host


class FigureContainer:
    """Thin host wrapper around a plot stack with a fixed placeholder widget."""

    def __init__(self, stack: QStackedWidget, placeholder: QWidget) -> None:
        self._stack = stack
        self._placeholder = placeholder
        # Initialise the host bridge on the main thread now (this Container is
        # built on the GUI thread), so a later worker-thread emit marshals
        # correctly instead of taking the worker's affinity.
        _host.ensure_host()

    @property
    def object_name(self) -> str:
        """Return the live stack's Qt objectName on the GUI thread.

        An unnamed stack returns an empty string. A deleted stack raises
        RuntimeError; query membership before requesting its diagnostic name.
        """
        return self._stack.objectName()

    def contains_figure(self, fig: Figure) -> bool:
        """Inspect membership of ``fig`` by Figure identity on the GUI thread.

        This read-only query does not change selection, registry or ownership.
        Inspect canvases in stack order and return whether the first identity
        match is a live Qt widget; do not skip a deleted first match. Return
        False when there is no match or the stack or matching canvas is deleted.
        Only RuntimeError from accessing a deleted stack is isolated; other
        failures propagate. The caller must run synchronously on the GUI thread.
        """
        from qtpy import sip

        try:
            count = self._stack.count()
        except RuntimeError:
            return False
        for index in range(count):
            widget = self._stack.widget(index)
            figure = getattr(widget, "figure", None)
            if figure is fig:
                return widget is not None and not sip.isdeleted(widget)
        return False

    def contains_canvas(self, canvas: QWidget) -> bool:
        """Inspect whether ``canvas`` belongs to this stack on the GUI thread.

        Accept a QWidget canvas and use Qt indexOf membership, independently of
        current selection. This read-only query changes neither selection nor
        ownership. Return False for a nonmember or when a deleted stack/canvas
        wrapper makes indexOf raise RuntimeError. Other failures propagate.
        The caller must run synchronously on the GUI thread.
        """
        try:
            return self._stack.indexOf(canvas) >= 0
        except RuntimeError:
            return False

    def attach_canvas(self, canvas: QWidget) -> None:
        if self._stack.indexOf(canvas) < 0:
            self._stack.addWidget(canvas)
        self._stack.setCurrentWidget(canvas)

    def detach_canvas(self, canvas: QWidget) -> None:
        was_current = self._stack.currentWidget() is canvas
        if self._stack.indexOf(canvas) >= 0:
            self._stack.removeWidget(canvas)
        if was_current:
            self._stack.setCurrentWidget(self._placeholder)

    def set_current_canvas(self, canvas: QWidget) -> None:
        if self._stack.indexOf(canvas) < 0:
            raise RuntimeError("Canvas is not attached to this FigureContainer")
        self._stack.setCurrentWidget(canvas)

    def clear_dynamic_canvases(self) -> None:
        """Remove presentation while keeping retained Figures saveable."""
        while self._stack.count() > 1:
            widget = self._stack.widget(self._stack.count() - 1)
            if widget is None:
                break
            self._stack.removeWidget(widget)
            figure = getattr(widget, "figure", None)
            if isinstance(figure, Figure):
                _host.drop_from_registry(figure)
                if isinstance(widget, FigureCanvasBase) and figure.canvas is widget:
                    FigureCanvasAgg(figure)
            widget.deleteLater()
        self._stack.setCurrentWidget(self._placeholder)
