"""Main-thread bridge for explicit figure attachment and canvas removal.

The host forwards worker requests to one GUI-thread QObject and tracks the
Figure-to-container relation with weak keys. Containers remain passive widgets.
Create the host on the GUI thread before workers use it; runtime initialization
and FigureContainer construction both ensure this prerequisite.
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from matplotlib.figure import Figure
from qtpy.QtCore import (
    QCoreApplication,
    QObject,
    QThread,
    Signal,  # type: ignore[reportPrivateImportUsage]
)
from qtpy.QtWidgets import QWidget

if TYPE_CHECKING:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    from .container import FigureContainer

# Keyed by the Figure itself (identity hash + weakref). A WeakKeyDictionary is
# load-bearing, not a convenience: an ``id(fig)``-keyed dict never purges entries
# on the normal render path, so after a figure is GC'd CPython can reuse its id
# and a NEW figure aliases a stale entry pointing at a DIFFERENT container —
# _attach_figure_canvas would then detach the wrong container's canvas and flip
# it to its placeholder (the intermittent "analyze figure not displayed" bug).
# Weak keys make GC'd figures vanish automatically, so id reuse cannot alias.
_fig_container_registry: weakref.WeakKeyDictionary[Figure, FigureContainer] = (
    weakref.WeakKeyDictionary()
)
_host: Any = None
_shutting_down: bool = False


def set_shutting_down(value: bool) -> None:
    global _shutting_down
    _shutting_down = value


def drop_from_registry(fig: Figure) -> None:
    """Forget a figure→container mapping (called by Container.clear)."""
    _fig_container_registry.pop(fig, None)


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


def get_figure_container(fig: Figure) -> FigureContainer | None:
    return _fig_container_registry.get(fig)


def _purge_stale_registry_entries() -> None:
    from qtpy import sip

    # GC-driven weakref eviction can mutate the registry mid-iteration, so snapshot
    # to a plain list first (a live WeakKeyDictionary iterator would raise
    # RuntimeError if a key is collected during the loop).
    stale_figs: list[Figure] = []
    for fig, container in list(_fig_container_registry.items()):
        try:
            container._stack.count()
        except RuntimeError:
            # The container's QStackedWidget itself was deleted.
            stale_figs.append(fig)
            continue
        # Also evict entries whose canvas wrapper is dead: the registry maps a
        # figure to its container, but if the canvas was deleted the mapping is
        # stale and would resurrect a dead wrapper on the next attachment.
        fig_canvas: QWidget | None = None
        for index in range(container._stack.count()):
            widget = container._stack.widget(index)
            figure = getattr(widget, "figure", None)
            if figure is fig:
                fig_canvas = widget
                break
        if fig_canvas is None or sip.isdeleted(fig_canvas):
            stale_figs.append(fig)
    for fig in stale_figs:
        _fig_container_registry.pop(fig, None)


def dump_plot_state() -> PlotStateSnapshot:
    _purge_stale_registry_entries()
    # Snapshot keys to a list before taking ids: WeakKeyDictionary iteration can
    # raise if a figure is GC'd mid-loop.
    attached_figure_ids = tuple(
        sorted(id(fig) for fig in list(_fig_container_registry))
    )
    return PlotStateSnapshot(
        active_figure_count=len(attached_figure_ids),
        attached_figure_ids=attached_figure_ids,
    )


def assert_plot_invariants() -> None:
    _purge_stale_registry_entries()
    # Snapshot before iterating: a key GC'd mid-loop would raise on a live
    # WeakKeyDictionary iterator.
    for fig, container in list(_fig_container_registry.items()):
        stack = container._stack
        found_canvas = False
        for index in range(stack.count()):
            widget = stack.widget(index)
            figure = getattr(widget, "figure", None)
            if figure is fig:
                found_canvas = True
                break
        if not found_canvas:
            raise RuntimeError(
                f"Figure registry invariant broken for figure id {id(fig)}"
            )


def _attach_figure_canvas(
    container: FigureContainer,
    fig: Figure,
) -> FigureCanvasQTAgg:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from qtpy import sip

    canvas = fig.canvas
    # ``fig.canvas`` can be a dead Qt wrapper if a previous render path deleted
    # the canvas widget while matplotlib still holds the Python reference. Reusing
    # it would crash at the first C++ call (e.g. stack.indexOf). Treat a deleted
    # wrapper as "no usable canvas" and rebuild a fresh one. This keeps the system
    # self-healing even if some path deletes a canvas out from under its figure.
    if (
        not isinstance(canvas, FigureCanvasQTAgg) or sip.isdeleted(canvas)  # type: ignore[attr-defined]
    ):
        canvas = FigureCanvasQTAgg(fig)

    previous_container = _fig_container_registry.get(fig)
    if previous_container is not None and previous_container is not container:
        # Defense in depth: only detach from the previous container if that entry
        # is genuinely live — the container's stack must still exist (not a
        # deleted Qt object) and must actually host this canvas. A stale/aliased
        # entry (e.g. the canvas already moved, or the container was torn down)
        # must NOT trigger detach_canvas, which would flip an unrelated
        # container back to its placeholder. If stale, just drop the entry.
        if _container_hosts_canvas(previous_container, canvas):
            previous_container.detach_canvas(canvas)
        else:
            _fig_container_registry.pop(fig, None)

    container.attach_canvas(canvas)
    _fig_container_registry[fig] = container
    return canvas


def _container_hosts_canvas(container: FigureContainer, canvas: QWidget) -> bool:
    """True if ``container``'s stack is live and currently holds ``canvas``.

    Guards the detach path against stale/aliased registry entries: a dead
    QStackedWidget raises on ``indexOf`` (treated as not-hosting), and a live
    stack that does not contain the canvas means the mapping is stale.
    """
    try:
        return container._stack.indexOf(canvas) >= 0
    except RuntimeError:
        # The container's QStackedWidget itself was deleted.
        return False


def _remove_canvas_impl(canvas: QWidget) -> None:
    figure = getattr(canvas, "figure", None)
    if not isinstance(figure, Figure):
        raise RuntimeError(f"Cannot remove canvas {canvas!r}: not a matplotlib canvas")
    container = _fig_container_registry.pop(figure, None)
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
            remove_canvas_requested = Signal(object)

            def __init__(self) -> None:
                super().__init__()
                self.attach_requested.connect(self._on_attach)
                self.remove_canvas_requested.connect(self._on_remove_canvas)

            def _on_attach(self, payload: Any) -> None:
                try:
                    container = payload["container"]
                    fig = payload["fig"]
                    result = payload["result"]

                    result.append(_attach_figure_canvas(container, fig))
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

        _host = _PlotHost()
    return _host


def ensure_host() -> None:
    _get_host()
