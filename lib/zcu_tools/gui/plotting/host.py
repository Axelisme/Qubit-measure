"""Plot host — the single main-thread bridge + figure registry.

The host owns:
- a **single module-level QObject** (``_host``) with the forwarding signals
  (attach/activate/refresh/remove/close). The client (``backend.py``) forwards a
  plotting operation by calling a host entry function; the host runs the work on
  the main thread (Qt AutoConnection: a same-thread emit runs the slot inline; a
  worker emit is queued onto the host's thread). Communication is one-way
  worker→host plus a ``threading.Event`` round-trip that hands the attached
  canvas back to the worker. Containers (``container.py``) are passive widgets
  the host operates — they hold no signals.
- the **figure registry** (``id(figure) → FigureContainer``). ``show`` / draw /
  ``close`` start from a bare ``Figure`` and resolve its container here; routing
  (``routing.py``) is only consulted at *attach* (which container new figures go
  to). Resolving refresh/activate/close from the registry — never from the
  routing ContextVar — is what keeps concurrent workers from cross-routing.

Invariants:
- **The host QObject MUST be created first on the GUI (main) thread.** A worker
  triggering the first init would give every later canvas the wrong thread
  affinity and crash repaints. ``FigureContainer.__init__`` calls ``ensure_host``
  on the main thread to guarantee this.
- ``plt.show()`` only activates an already-created canvas; it never takes over
  the QApplication event loop.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)

from matplotlib.figure import Figure
from qtpy.QtCore import (  # type: ignore[attr-defined]
    QCoreApplication,
    QObject,
    QThread,
    Signal,  # type: ignore[reportPrivateImportUsage]
)
from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]

from .routing import require_current_container

if TYPE_CHECKING:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    from .container import FigureContainer

_fig_container_registry: dict[int, FigureContainer] = {}
_host: Any = None
_shutting_down: bool = False


def set_shutting_down(value: bool) -> None:
    global _shutting_down
    _shutting_down = value


def drop_from_registry(fig: Figure) -> None:
    """Forget a figure→container mapping (called by Container.clear)."""
    _fig_container_registry.pop(id(fig), None)


@dataclass(frozen=True)
class PlotStateSnapshot:
    active_figure_count: int
    attached_figure_ids: tuple[int, ...]


def attach_existing_figure_to_container(
    fig: Figure, container: FigureContainer
) -> QWidget:
    done = threading.Event()
    result: list[Any] = []
    errors: list[BaseException] = []
    _get_host().attach_requested.emit(
        {
            "container": container,
            "fig": fig,
            "result": result,
            "errors": errors,
            "done": done,
        }
    )
    done.wait(timeout=5.0)
    if errors:
        raise RuntimeError("Failed to attach figure to FigureContainer") from errors[0]
    if not result:
        raise RuntimeError("Timed out attaching figure to FigureContainer")
    return result[0]


def attach_figure_to_current_container(
    fig: Figure,
    canvas_class: type[FigureCanvasQTAgg] | None = None,
) -> FigureCanvasQTAgg:
    container = require_current_container()
    done = threading.Event()
    result: list[Any] = []
    errors: list[BaseException] = []
    _get_host().attach_requested.emit(
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
        raise RuntimeError(
            "Failed to attach figure to active FigureContainer"
        ) from errors[0]
    if not result:
        raise RuntimeError("Timed out attaching figure to active FigureContainer")
    return result[0]


def close_figure(fig: Figure) -> None:
    if _shutting_down:
        return
    done = threading.Event()
    errors: list[BaseException] = []
    _get_host().close_requested.emit({"fig": fig, "errors": errors, "done": done})
    if not done.wait(timeout=5.0):
        raise RuntimeError("Timed out closing figure")
    if errors:
        raise RuntimeError("Failed to close figure") from errors[0]


def remove_canvas(canvas: QWidget) -> None:
    if _shutting_down:
        return
    done = threading.Event()
    errors: list[BaseException] = []
    _get_host().remove_canvas_requested.emit(
        {"canvas": canvas, "errors": errors, "done": done}
    )
    if not done.wait(timeout=5.0):
        raise RuntimeError("Timed out removing canvas")
    if errors:
        raise RuntimeError("Failed to remove canvas") from errors[0]


def activate_figure(fig: Figure) -> None:
    if _shutting_down:
        return
    done = threading.Event()
    errors: list[BaseException] = []
    _get_host().activate_requested.emit({"fig": fig, "errors": errors, "done": done})
    if not done.wait(timeout=5.0):
        raise RuntimeError("Timed out activating figure")
    if errors:
        raise RuntimeError("Failed to activate figure") from errors[0]


def is_main_thread() -> bool:
    """True if called on the GUI (main) thread.

    The authority for "am I on the thread that may touch Qt widgets directly".
    ``GuiFigureCanvas`` uses it to decide whether a draw can run inline or must
    be marshalled to the main thread.
    """
    app = QCoreApplication.instance()
    if app is None:
        return False
    return QThread.currentThread() is app.thread()


def refresh_figure_in_main_thread(fig: Figure) -> None:
    """Dispatch draw_idle() to the main thread via the host (fire-and-forget)."""
    if _shutting_down:
        return
    _get_host().refresh_requested.emit(fig)


def get_figure_container(fig: Figure) -> FigureContainer | None:
    return _fig_container_registry.get(id(fig))


def _purge_stale_registry_entries() -> None:
    stale_ids: list[int] = []
    for fig_id, container in _fig_container_registry.items():
        try:
            container._stack.count()
        except RuntimeError:
            stale_ids.append(fig_id)
    for fig_id in stale_ids:
        _fig_container_registry.pop(fig_id, None)


def dump_plot_state() -> PlotStateSnapshot:
    _purge_stale_registry_entries()
    attached_figure_ids = tuple(sorted(_fig_container_registry))
    return PlotStateSnapshot(
        active_figure_count=len(attached_figure_ids),
        attached_figure_ids=attached_figure_ids,
    )


def assert_plot_invariants() -> None:
    _purge_stale_registry_entries()
    for fig_id, container in _fig_container_registry.items():
        stack = container._stack
        found_canvas = False
        for index in range(stack.count()):
            widget = stack.widget(index)
            figure = getattr(widget, "figure", None)
            if isinstance(figure, Figure) and id(figure) == fig_id:
                found_canvas = True
                break
        if not found_canvas:
            raise RuntimeError(
                f"Figure registry invariant broken for figure id {fig_id}"
            )


def _attach_figure_canvas(
    container: FigureContainer,
    fig: Figure,
    canvas_class: type[FigureCanvasQTAgg] | None = None,
) -> FigureCanvasQTAgg:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    canvas = fig.canvas
    expected_canvas_class = (
        canvas_class if canvas_class is not None else FigureCanvasQTAgg
    )
    if not isinstance(canvas, expected_canvas_class):
        canvas = expected_canvas_class(fig)

    previous_container = _fig_container_registry.get(id(fig))
    if previous_container is not None and previous_container is not container:
        previous_container.detach_canvas(canvas)

    container.attach_canvas(canvas)
    _fig_container_registry[id(fig)] = container
    return canvas


def _remove_canvas_impl(canvas: QWidget) -> None:
    figure = getattr(canvas, "figure", None)
    if not isinstance(figure, Figure):
        raise RuntimeError(f"Cannot remove canvas {canvas!r}: not a matplotlib canvas")
    container = _fig_container_registry.pop(id(figure), None)
    if container is None:
        raise RuntimeError(
            f"Cannot remove canvas {canvas!r}: figure not tracked in registry"
        )
    container.detach_canvas(canvas)


def _get_host() -> Any:
    global _host
    if _host is None:
        app = QCoreApplication.instance()
        if app is None:
            raise RuntimeError(
                "QCoreApplication must exist before initializing the plot host"
            )
        if QThread.currentThread() is not app.thread():
            raise RuntimeError(
                "Plot host must be initialized from the GUI thread before worker plotting"
            )

        class _PlotHost(QObject):
            attach_requested = Signal(object)
            close_requested = Signal(object)
            remove_canvas_requested = Signal(object)
            activate_requested = Signal(object)
            refresh_requested = Signal(object)

            def __init__(self) -> None:
                super().__init__()
                self.attach_requested.connect(self._on_attach)
                self.close_requested.connect(self._on_close)
                self.remove_canvas_requested.connect(self._on_remove_canvas)
                self.activate_requested.connect(self._on_activate)
                self.refresh_requested.connect(self._on_refresh)

            def _on_attach(self, payload: Any) -> None:
                try:
                    container = payload["container"]
                    fig = payload["fig"]
                    canvas_class = payload.get("canvas_class")
                    result = payload["result"]

                    result.append(_attach_figure_canvas(container, fig, canvas_class))
                except BaseException as exc:
                    payload["errors"].append(exc)
                finally:
                    payload["done"].set()

            def _on_close(self, payload: Any) -> None:
                import matplotlib.pyplot as _plt

                try:
                    fig = payload["fig"]
                    canvas = fig.canvas
                    container = _fig_container_registry.pop(id(fig), None)
                    if container is not None and isinstance(canvas, QWidget):
                        container.detach_canvas(canvas)
                        canvas.deleteLater()
                    _plt.close(fig)
                except BaseException as exc:
                    payload["errors"].append(exc)
                finally:
                    payload["done"].set()

            def _on_remove_canvas(self, payload: Any) -> None:
                try:
                    canvas = payload["canvas"]
                    _remove_canvas_impl(canvas)
                    canvas.deleteLater()
                except BaseException as exc:
                    payload["errors"].append(exc)
                finally:
                    payload["done"].set()

            def _on_activate(self, payload: Any) -> None:
                try:
                    fig = payload["fig"]
                    container = _fig_container_registry.get(id(fig))
                    if container is None:
                        raise RuntimeError(
                            "Figure is not attached to any FigureContainer"
                        )
                    canvas = fig.canvas
                    if not isinstance(canvas, QWidget):
                        raise RuntimeError("Figure canvas is not a QWidget")
                    container.set_current_canvas(canvas)
                except BaseException as exc:
                    payload["errors"].append(exc)
                finally:
                    payload["done"].set()

            def _on_refresh(self, fig: object) -> None:
                if isinstance(fig, Figure):
                    fig.canvas.draw_idle()

        _host = _PlotHost()
    return _host


def ensure_host() -> None:
    _get_host()
