"""plot_host — host side of fluxdep's embedded-matplotlib substrate.

A slim, fluxdep-local version of measure-gui's plotting substrate, kept the SAME
shape so a shared layer can be lifted later: the host lives on the main (Qt)
thread and owns a ``FigureContainer`` (a QStackedWidget that shows one figure
canvas); the *client* is the custom matplotlib backend (``mpl_backend``) that
experiment code reaches via plain ``plt.figure()`` / ``plt.show()``. The client
may run on a worker thread, so every host operation it triggers is marshalled to
the main thread through a Qt bridge (a QObject + queued signals) and awaited with
a ``threading.Event``.

What fluxdep drops vs measure-gui: there is no task-local routing of multiple
containers (fluxdep has a single diagnostic-figure slot), so "the current
container" is one module-level value set for the duration of a ``with
use_container(...)`` block, rather than a per-task stack.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Generator, Optional

from matplotlib.figure import Figure
from qtpy.QtCore import (  # type: ignore[attr-defined]
    QCoreApplication,
    QObject,
    QThread,
    Signal,  # type: ignore[attr-defined]
)
from qtpy.QtWidgets import QStackedWidget, QWidget  # type: ignore[attr-defined]

# figure id -> the container it is attached to (host-thread bookkeeping).
_fig_container_registry: dict[int, "FigureContainer"] = {}

# The container new pyplot figures attach to (set by use_container()).
_current_container: Optional["FigureContainer"] = None

_bridge: Optional[QObject] = None
_shutting_down = False


class FigureContainer:
    """A QStackedWidget host that shows one dynamic figure canvas at a time.

    Index 0 is a fixed placeholder; attaching a canvas adds it and makes it
    current. Mirrors measure-gui's ``FigureContainer`` (minus multi-canvas
    history — fluxdep shows one diagnostic figure).
    """

    def __init__(self, stack: QStackedWidget, placeholder: QWidget) -> None:
        self._stack = stack
        self._placeholder = placeholder
        ensure_bridge()

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

    def clear(self) -> None:
        """Drop every dynamic canvas, back to the placeholder."""
        while self._stack.count() > 1:
            widget = self._stack.widget(self._stack.count() - 1)
            self._stack.removeWidget(widget)
            if widget is not None:
                figure = getattr(widget, "figure", None)
                if isinstance(figure, Figure):
                    _fig_container_registry.pop(id(figure), None)
                widget.deleteLater()
        self._stack.setCurrentWidget(self._placeholder)


def set_current_container(container: Optional[FigureContainer]) -> None:
    """Set (or clear, with None) the container new pyplot figures attach to.

    For the async flow where plotting happens on a worker thread: the main thread
    sets the container before starting the worker and clears it after the worker
    finishes. (Module-level, so the worker thread reads the same value.) Prefer
    ``use_container`` for synchronous/test code.
    """
    global _current_container
    _current_container = container


@contextmanager
def use_container(container: FigureContainer) -> Generator[None, None, None]:
    """Route pyplot figures created in this block to ``container`` (sync use).

    The backend's ``plt.figure()`` interception attaches the new figure to the
    container current here. Restores the previous current container on exit
    (supports the rare nested case without losing the outer one).
    """
    global _current_container
    previous = _current_container
    _current_container = container
    try:
        yield
    finally:
        _current_container = previous


def require_current_container() -> FigureContainer:
    if _current_container is None:
        raise RuntimeError(
            "no current FigureContainer — wrap plotting in use_container(...)"
        )
    return _current_container


def is_main_thread() -> bool:
    """True on the GUI (main) thread — the authority for 'may touch Qt directly'."""
    app = QCoreApplication.instance()
    if app is None:
        return False
    return QThread.currentThread() is app.thread()


def get_figure_container(fig: Figure) -> Optional[FigureContainer]:
    return _fig_container_registry.get(id(fig))


def attach_figure_to_current_container(fig: Figure, canvas_class: Any = None) -> Any:
    """Attach ``fig`` to the current container on the main thread; return its canvas.

    Called by the backend when ``plt.figure()`` runs (possibly on a worker). The
    actual widget work is marshalled to the main thread via the bridge and awaited.
    """
    container = require_current_container()
    done = threading.Event()
    result: list[Any] = []
    errors: list[BaseException] = []
    _get_bridge().attach_requested.emit(  # type: ignore[attr-defined]
        {
            "container": container,
            "fig": fig,
            "canvas_class": canvas_class,
            "result": result,
            "errors": errors,
            "done": done,
        }
    )
    done.wait(timeout=5.0)
    if errors:
        raise RuntimeError("failed to attach figure to container") from errors[0]
    if not result:
        raise RuntimeError("timed out attaching figure to container")
    return result[0]


def activate_figure(fig: Figure) -> None:
    """Make ``fig``'s canvas the current one in its container (plt.show())."""
    if _shutting_down:
        return
    done = threading.Event()
    _get_bridge().activate_requested.emit({"fig": fig, "done": done})  # type: ignore[attr-defined]
    done.wait(timeout=5.0)


def remove_canvas(canvas: QWidget) -> None:
    if _shutting_down:
        return
    done = threading.Event()
    _get_bridge().remove_canvas_requested.emit(  # type: ignore[attr-defined]
        {"canvas": canvas, "done": done}
    )
    done.wait(timeout=5.0)


def refresh_figure_in_main_thread(fig: Figure) -> None:
    """Fire-and-forget draw_idle() on the main thread (worker→main marshalling)."""
    if _shutting_down:
        return
    _get_bridge().refresh_requested.emit(fig)  # type: ignore[attr-defined]


def _attach_figure_canvas(
    container: FigureContainer, fig: Figure, canvas_class: Any = None
) -> Any:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    canvas = fig.canvas
    expected = canvas_class if canvas_class is not None else FigureCanvasQTAgg
    if not isinstance(canvas, expected):
        canvas = expected(fig)

    previous = _fig_container_registry.get(id(fig))
    if previous is not None and previous is not container:
        previous.detach_canvas(canvas)

    container.attach_canvas(canvas)
    _fig_container_registry[id(fig)] = container
    return canvas


def _remove_canvas_impl(canvas: QWidget) -> None:
    figure = getattr(canvas, "figure", None)
    if not isinstance(figure, Figure):
        raise RuntimeError(f"cannot remove canvas {canvas!r}: not a matplotlib canvas")
    container = _fig_container_registry.pop(id(figure), None)
    if container is not None:
        container.detach_canvas(canvas)


def _get_bridge() -> QObject:
    global _bridge
    if _bridge is None:
        app = QCoreApplication.instance()
        if app is None:
            raise RuntimeError(
                "QCoreApplication must exist before the plot-host bridge"
            )
        if QThread.currentThread() is not app.thread():
            raise RuntimeError(
                "the plot-host bridge must be created on the GUI thread "
                "(before any worker plotting)"
            )

        class _Bridge(QObject):
            attach_requested = Signal(object)
            remove_canvas_requested = Signal(object)
            activate_requested = Signal(object)
            refresh_requested = Signal(object)

            def __init__(self) -> None:
                super().__init__()
                self.attach_requested.connect(self._on_attach)
                self.remove_canvas_requested.connect(self._on_remove_canvas)
                self.activate_requested.connect(self._on_activate)
                self.refresh_requested.connect(self._on_refresh)

            def _on_attach(self, payload: Any) -> None:
                try:
                    canvas = _attach_figure_canvas(
                        payload["container"],
                        payload["fig"],
                        payload.get("canvas_class"),
                    )
                    payload["result"].append(canvas)
                except BaseException as exc:  # noqa: BLE001 — relay to the waiter
                    payload["errors"].append(exc)
                finally:
                    payload["done"].set()

            def _on_remove_canvas(self, payload: Any) -> None:
                try:
                    canvas = payload["canvas"]
                    _remove_canvas_impl(canvas)
                    canvas.deleteLater()
                finally:
                    payload["done"].set()

            def _on_activate(self, payload: Any) -> None:
                try:
                    fig = payload["fig"]
                    container = _fig_container_registry.get(id(fig))
                    if container is not None and isinstance(fig.canvas, QWidget):
                        container.attach_canvas(fig.canvas)
                finally:
                    payload["done"].set()

            def _on_refresh(self, fig: object) -> None:
                if isinstance(fig, Figure):
                    fig.canvas.draw_idle()

        _bridge = _Bridge()
    return _bridge


def ensure_bridge() -> None:
    """Create the main-thread bridge now (call once on the GUI thread at startup)."""
    _get_bridge()


def shutdown() -> None:
    """Mark the plot host as shutting down and drop the bridge.

    Call on app teardown (aboutToQuit). After this, ``remove_canvas`` /
    ``activate_figure`` / ``refresh_figure_in_main_thread`` are no-ops, so the
    matplotlib atexit hook (``Gcf.destroy_all`` → backend ``destroy`` →
    ``remove_canvas``) does not touch the already-deleted Qt ``_Bridge`` and
    raise "wrapped C/C++ object ... has been deleted".
    """
    global _shutting_down, _bridge
    _shutting_down = True
    _bridge = None


__all__ = [
    "FigureContainer",
    "use_container",
    "set_current_container",
    "require_current_container",
    "is_main_thread",
    "get_figure_container",
    "attach_figure_to_current_container",
    "activate_figure",
    "remove_canvas",
    "refresh_figure_in_main_thread",
    "ensure_bridge",
    "shutdown",
]
