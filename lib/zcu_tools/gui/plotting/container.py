"""FigureContainer — a plot stack widget the host attaches canvases into.

A thin, **passive** wrapper around a ``QStackedWidget`` with a fixed placeholder.
It holds no signals and does no thread marshalling — the host (``host.py``) owns
the single main-thread bridge QObject and calls these synchronous methods on the
main thread. Worker code never touches a Container directly; it only *selects*
one via routing (``routing.routing_scope``) and the host marshals the attach/
activate/refresh onto the main thread.

Constructing a Container ensures the host bridge is initialised on the main
thread first (the canvas thread-affinity invariant — see ``host.ensure_host``).
"""

from __future__ import annotations

from matplotlib.figure import Figure
from qtpy.QtWidgets import QStackedWidget, QWidget  # type: ignore[attr-defined]

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
        while self._stack.count() > 1:
            widget = self._stack.widget(self._stack.count() - 1)
            self._stack.removeWidget(widget)
            if widget is not None:
                figure = getattr(widget, "figure", None)
                if isinstance(figure, Figure):
                    _host.drop_from_registry(figure)
                widget.deleteLater()
        self._stack.setCurrentWidget(self._placeholder)
