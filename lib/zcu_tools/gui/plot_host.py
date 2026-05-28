from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qtpy.QtCore import (  # type: ignore[attr-defined]
    QCoreApplication,
    QObject,
    QThread,
    Signal,  # type: ignore[reportPrivateImportUsage]
)
from qtpy.QtWidgets import QStackedWidget, QWidget  # type: ignore[attr-defined]

from .plot_routing import require_current_container

if TYPE_CHECKING:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

_fig_container_registry: dict[int, "FigureContainer"] = {}
_bridge: Any = None
_shutting_down: bool = False


def set_shutting_down(value: bool) -> None:
    global _shutting_down
    _shutting_down = value


class FigureContainer:
    """Thin host wrapper around a plot stack with a fixed placeholder widget."""

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
                    _fig_container_registry.pop(id(figure), None)
                widget.deleteLater()
        self._stack.setCurrentWidget(self._placeholder)


@dataclass(frozen=True)
class PlotStateSnapshot:
    active_figure_count: int
    attached_figure_ids: tuple[int, ...]


def create_figure_in_current_container(
    n_row: int, n_col: int, **kwargs: Any
) -> tuple[Figure, list[list[Axes]]]:
    container = require_current_container()
    kwargs.setdefault("squeeze", False)
    kwargs.setdefault("figsize", (6 * n_col, 4 * n_row))
    done = threading.Event()
    result: list[Any] = []
    errors: list[BaseException] = []
    _get_bridge().create_requested.emit(
        {
            "container": container,
            "n_row": n_row,
            "n_col": n_col,
            "kwargs": kwargs,
            "result": result,
            "errors": errors,
            "done": done,
        }
    )
    done.wait(timeout=5.0)
    if errors:
        raise RuntimeError("Failed to create figure in FigureContainer") from errors[0]
    if not result:
        raise RuntimeError("Timed out creating figure in FigureContainer")
    fig, axs = result[0]
    return fig, axs


def attach_existing_figure_to_container(
    fig: Figure, container: FigureContainer
) -> QWidget:
    done = threading.Event()
    result: list[Any] = []
    errors: list[BaseException] = []
    _get_bridge().attach_requested.emit(
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
    canvas_class: Optional[type[FigureCanvasQTAgg]] = None,
) -> FigureCanvasQTAgg:
    container = require_current_container()
    done = threading.Event()
    result: list[Any] = []
    errors: list[BaseException] = []
    _get_bridge().attach_requested.emit(
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
    _get_bridge().close_requested.emit({"fig": fig, "done": done})
    done.wait(timeout=5.0)


def remove_canvas(canvas: QWidget) -> None:
    if _shutting_down:
        return
    done = threading.Event()
    _get_bridge().remove_canvas_requested.emit({"canvas": canvas, "done": done})
    done.wait(timeout=5.0)


def activate_figure(fig: Figure) -> None:
    if _shutting_down:
        return
    done = threading.Event()
    _get_bridge().activate_requested.emit({"fig": fig, "done": done})
    done.wait(timeout=5.0)


def refresh_figure_in_main_thread(fig: Figure) -> None:
    """Dispatch draw_idle() to the main thread via the Qt bridge (fire-and-forget)."""
    if _shutting_down:
        return
    _get_bridge().refresh_requested.emit(fig)


def get_figure_container(fig: Figure) -> Optional[FigureContainer]:
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
    canvas_class: Optional[type[FigureCanvasQTAgg]] = None,
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


def _get_bridge() -> Any:
    global _bridge
    if _bridge is None:
        app = QCoreApplication.instance()
        if app is None:
            raise RuntimeError(
                "QCoreApplication must exist before initializing plot host bridge"
            )
        if QThread.currentThread() is not app.thread():
            raise RuntimeError(
                "Plot host bridge must be initialized from the GUI thread before worker plotting"
            )

        class _Bridge(QObject):
            create_requested = Signal(object)
            attach_requested = Signal(object)
            close_requested = Signal(object)
            remove_canvas_requested = Signal(object)
            activate_requested = Signal(object)
            refresh_requested = Signal(object)

            def __init__(self) -> None:
                super().__init__()
                self.create_requested.connect(self._on_create)
                self.attach_requested.connect(self._on_attach)
                self.close_requested.connect(self._on_close)
                self.remove_canvas_requested.connect(self._on_remove_canvas)
                self.activate_requested.connect(self._on_activate)
                self.refresh_requested.connect(self._on_refresh)

            def _on_create(self, payload: Any) -> None:
                try:
                    kwargs = dict(payload["kwargs"])
                    subplot_kwargs = {
                        "sharex": kwargs.pop("sharex", False),
                        "sharey": kwargs.pop("sharey", False),
                        "squeeze": kwargs.pop("squeeze", False),
                        "subplot_kw": kwargs.pop("subplot_kw", None),
                        "gridspec_kw": kwargs.pop("gridspec_kw", None),
                    }
                    if kwargs:
                        fig = Figure(**kwargs)
                    else:
                        fig = Figure()

                    container = payload["container"]
                    n_row = payload["n_row"]
                    n_col = payload["n_col"]
                    result = payload["result"]

                    axs = fig.subplots(n_row, n_col, **subplot_kwargs)
                    _attach_figure_canvas(container, fig)
                    result.append((fig, axs))
                except BaseException as exc:
                    payload["errors"].append(exc)
                finally:
                    payload["done"].set()

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

                fig = payload["fig"]
                done = payload["done"]

                canvas = fig.canvas
                container = _fig_container_registry.pop(id(fig), None)
                if container is not None and isinstance(canvas, QWidget):
                    container.detach_canvas(canvas)
                    canvas.deleteLater()
                _plt.close(fig)
                done.set()

            def _on_remove_canvas(self, payload: Any) -> None:
                canvas = payload["canvas"]
                done = payload["done"]
                _remove_canvas_impl(canvas)
                canvas.deleteLater()
                done.set()

            def _on_activate(self, payload: Any) -> None:
                fig = payload["fig"]
                done = payload["done"]
                container = _fig_container_registry.get(id(fig))
                if container is None:
                    raise RuntimeError("Figure is not attached to any FigureContainer")
                canvas = fig.canvas
                if not isinstance(canvas, QWidget):
                    raise RuntimeError("Figure canvas is not a QWidget")
                container.set_current_canvas(canvas)
                done.set()

            def _on_refresh(self, fig: object) -> None:
                if isinstance(fig, Figure):
                    fig.canvas.draw_idle()

        _bridge = _Bridge()
    return _bridge


def ensure_bridge() -> None:
    _get_bridge()


__all__ = [
    "FigureContainer",
    "activate_figure",
    "assert_plot_invariants",
    "attach_existing_figure_to_container",
    "attach_figure_to_current_container",
    "close_figure",
    "create_figure_in_current_container",
    "dump_plot_state",
    "ensure_bridge",
    "get_figure_container",
    "PlotStateSnapshot",
    "refresh_figure_in_main_thread",
    "remove_canvas",
]
