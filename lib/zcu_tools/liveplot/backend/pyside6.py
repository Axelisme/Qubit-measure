from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable, TypeVar, cast

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

T = TypeVar("T")


class _GuiThreadBridge(QObject):
    execute = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
        # AutoConnection is queued when crossing threads.
        self.execute.connect(self._execute_in_gui)

    def _execute_in_gui(self, fn: Callable[[], Any], state: dict[str, Any]) -> None:
        try:
            state["result"] = fn()
        except Exception as exc:
            state["error"] = exc
        finally:
            state["event"].set()


_BRIDGE: Any = None
_BRIDGE_LOCK = threading.Lock()
_HOST_LOCK = threading.Lock()
_FIGURE_CANVAS_MAP: dict[int, FigureCanvasQTAgg] = {}
_PLOT_HOST: dict[str, Any] = {"container": None, "clear_container": True}


def _ensure_bridge() -> Any:
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("QApplication is not initialized")

    global _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is None:
            bridge = _GuiThreadBridge()
            bridge.moveToThread(app.thread())
            _BRIDGE = bridge
    return _BRIDGE


def _run_in_gui_thread(fn: Callable[[], T]) -> T:
    bridge = _ensure_bridge()

    app = QApplication.instance()
    if app is None:
        raise RuntimeError("QApplication is not initialized")

    if QThread.currentThread() == app.thread():
        return fn()

    state: dict[str, Any] = {"event": threading.Event(), "result": None, "error": None}
    cast(Any, bridge).execute.emit(fn, state)
    if not state["event"].wait(timeout=60):
        raise TimeoutError("Timed out while waiting for GUI thread to run plot action.")
    if state["error"] is not None:
        raise state["error"]
    return state["result"]


def set_plot_host(container: QWidget, *, clear_container: bool = True) -> None:
    with _HOST_LOCK:
        _PLOT_HOST["container"] = container
        _PLOT_HOST["clear_container"] = clear_container


def clear_plot_host() -> None:
    with _HOST_LOCK:
        _PLOT_HOST["container"] = None
        _PLOT_HOST["clear_container"] = True


@contextmanager
def plot_host_scope(container: QWidget, *, clear_container: bool = True):
    set_plot_host(container, clear_container=clear_container)
    try:
        yield
    finally:
        clear_plot_host()


def _host_snapshot() -> tuple[Any, bool]:
    with _HOST_LOCK:
        return _PLOT_HOST["container"], bool(_PLOT_HOST["clear_container"])


def _ensure_host_layout(container: QWidget) -> QVBoxLayout:
    layout = container.layout()
    if layout is None:
        created = QVBoxLayout(container)
        created.setContentsMargins(0, 0, 0, 0)
        created.setSpacing(0)
        return created
    if not isinstance(layout, QVBoxLayout):
        raise TypeError("Plot host layout must be QVBoxLayout")
    return layout


def _clear_host_widgets(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()


def _attach_figure_to_host(fig: Figure) -> bool:
    container, clear_container = _host_snapshot()
    if not isinstance(container, QWidget):
        return False
    try:
        layout = _ensure_host_layout(container)
    except Exception:
        return False

    if clear_container:
        _clear_host_widgets(layout)

    canvas = FigureCanvasQTAgg(fig)
    layout.addWidget(canvas)
    canvas.draw_idle()
    _FIGURE_CANVAS_MAP[id(fig)] = canvas
    return True


def make_plot_frame(
    n_row: int, n_col: int, plot_instant: bool = False, **kwargs
) -> tuple[Figure, list[list[Axes]]]:
    def _create() -> tuple[Figure, list[list[Axes]]]:
        kwargs.setdefault("squeeze", False)
        kwargs.setdefault("figsize", (6 * n_col, 4 * n_row))
        fig, axs = plt.subplots(n_row, n_col, **kwargs)

        if plot_instant and not _attach_figure_to_host(fig):
            fig.show(warn=False)

        return fig, axs

    return _run_in_gui_thread(_create)


def refresh_figure(fig: Figure) -> None:
    def _refresh() -> None:
        # Keep refresh non-blocking in Qt contexts.
        fig.canvas.draw_idle()

    _run_in_gui_thread(_refresh)


def close_figure(fig: Figure) -> None:
    def _close() -> None:
        canvas = _FIGURE_CANVAS_MAP.pop(id(fig), None)
        if canvas is not None:
            canvas.setParent(None)
            canvas.deleteLater()
        plt.close(fig)

    _run_in_gui_thread(_close)
