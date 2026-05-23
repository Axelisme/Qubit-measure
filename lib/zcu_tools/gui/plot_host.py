from __future__ import annotations

import threading
from typing import Any, Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import QStackedWidget, QWidget  # type: ignore[attr-defined]

_container_stack: list["FigureContainer"] = []
_fig_container_registry: dict[int, "FigureContainer"] = {}
_bridge: Any = None


class FigureContainer:
    """Thin host wrapper around a plot stack with a fixed placeholder widget."""

    def __init__(self, stack: QStackedWidget, placeholder: QWidget) -> None:
        self._stack = stack
        self._placeholder = placeholder

    def attach_canvas(self, canvas: QWidget) -> None:
        if self._stack.indexOf(canvas) < 0:
            self._stack.addWidget(canvas)
        self._stack.setCurrentWidget(canvas)

    def detach_canvas(self, canvas: QWidget) -> None:
        if self._stack.indexOf(canvas) >= 0:
            self._stack.removeWidget(canvas)
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


def push_container(container: FigureContainer) -> None:
    _container_stack.append(container)
    _get_bridge()


def pop_container(container: Optional[FigureContainer] = None) -> FigureContainer:
    if not _container_stack:
        raise RuntimeError("No active FigureContainer to pop")
    top = _container_stack[-1]
    if container is not None and top is not container:
        raise RuntimeError("FigureContainer stack corruption: non-top pop attempted")
    return _container_stack.pop()


def peek_container() -> Optional[FigureContainer]:
    if not _container_stack:
        return None
    return _container_stack[-1]


def has_container() -> bool:
    return bool(_container_stack)


def create_figure_in_active_container(
    n_row: int, n_col: int, **kwargs: Any
) -> tuple[Figure, list[list[Axes]]]:
    container = peek_container()
    if container is None:
        raise RuntimeError("No active FigureContainer")

    kwargs.setdefault("squeeze", False)
    kwargs.setdefault("figsize", (6 * n_col, 4 * n_row))
    done = threading.Event()
    result: list[Any] = []
    _get_bridge().create_requested.emit(
        {
            "container": container,
            "n_row": n_row,
            "n_col": n_col,
            "kwargs": kwargs,
            "result": result,
            "done": done,
        }
    )
    done.wait(timeout=5.0)
    if not result:
        raise RuntimeError("Timed out creating figure in FigureContainer")
    fig, axs = result[0]
    return fig, axs


def attach_existing_figure_to_container(
    fig: Figure, container: FigureContainer
) -> QWidget:
    done = threading.Event()
    result: list[Any] = []
    _get_bridge().attach_requested.emit(
        {
            "container": container,
            "fig": fig,
            "result": result,
            "done": done,
        }
    )
    done.wait(timeout=5.0)
    if not result:
        raise RuntimeError("Timed out attaching figure to FigureContainer")
    return result[0]


def close_figure(fig: Figure) -> None:
    done = threading.Event()
    _get_bridge().close_requested.emit({"fig": fig, "done": done})
    done.wait(timeout=5.0)


def remove_canvas(canvas: QWidget) -> None:
    done = threading.Event()
    _get_bridge().remove_canvas_requested.emit({"canvas": canvas, "done": done})
    done.wait(timeout=5.0)


def _attach_figure_canvas(container: FigureContainer, fig: Figure) -> QWidget:
    from matplotlib.backends.backend_qtagg import (  # type: ignore[import-untyped]
        FigureCanvasQTAgg,
    )

    canvas = fig.canvas
    if not isinstance(canvas, FigureCanvasQTAgg):
        canvas = FigureCanvasQTAgg(fig)

    previous_container = _fig_container_registry.get(id(fig))
    if previous_container is not None and previous_container is not container:
        previous_container.detach_canvas(canvas)

    container.attach_canvas(canvas)
    _fig_container_registry[id(fig)] = container
    return canvas


def _remove_canvas_impl(canvas: QWidget) -> None:
    figure = getattr(canvas, "figure", None)
    if isinstance(figure, Figure):
        container = _fig_container_registry.pop(id(figure), None)
        if container is not None:
            container.detach_canvas(canvas)
            return
    for mapped_container in _fig_container_registry.values():
        if mapped_container._stack.indexOf(canvas) >= 0:
            mapped_container.detach_canvas(canvas)
            return


def _get_bridge() -> Any:
    global _bridge
    if _bridge is None:

        class _Bridge(QObject):
            create_requested = Signal(object)
            attach_requested = Signal(object)
            close_requested = Signal(object)
            remove_canvas_requested = Signal(object)

            def __init__(self) -> None:
                super().__init__()
                self.create_requested.connect(self._on_create)
                self.attach_requested.connect(self._on_attach)
                self.close_requested.connect(self._on_close)
                self.remove_canvas_requested.connect(self._on_remove_canvas)

            def _on_create(self, payload: Any) -> None:
                import matplotlib.pyplot as _plt

                container = payload["container"]
                n_row = payload["n_row"]
                n_col = payload["n_col"]
                kwargs = payload["kwargs"]
                result = payload["result"]
                done = payload["done"]

                fig, axs = _plt.subplots(n_row, n_col, **kwargs)
                _attach_figure_canvas(container, fig)
                result.append((fig, axs))
                done.set()

            def _on_attach(self, payload: Any) -> None:
                container = payload["container"]
                fig = payload["fig"]
                result = payload["result"]
                done = payload["done"]

                result.append(_attach_figure_canvas(container, fig))
                done.set()

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

        _bridge = _Bridge()
    return _bridge


__all__ = [
    "FigureContainer",
    "attach_existing_figure_to_container",
    "close_figure",
    "create_figure_in_active_container",
    "has_container",
    "peek_container",
    "pop_container",
    "push_container",
    "remove_canvas",
]
