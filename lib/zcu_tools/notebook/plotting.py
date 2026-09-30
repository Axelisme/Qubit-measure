"""Explicit Notebook presentation without pyplot figure registration."""

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, cast

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ipympl.backend_nbagg import Canvas, FigureManager

_T = TypeVar("_T")


class NotebookPlotHost:
    """Initialize and synchronously render widgets until explicitly released.

    Calls run synchronously on the caller's thread. Callers serialize operations;
    this host does not schedule work or detect frontend availability. Figures
    with an existing manager must be released by that owner before presentation.
    """

    def __init__(self) -> None:
        self._managers: dict[Figure, FigureManager] = {}

    def call(self, callback: Callable[[], _T]) -> _T:
        return callback()

    def present(self, figure: Figure) -> None:
        if figure in self._managers:
            return
        if figure.canvas.manager is not None:
            raise ValueError("Figure already has a presentation manager")

        from ipympl.backend_nbagg import Canvas, FigureManager
        from IPython.display import display

        # Construct directly: pyplot/Gcf registration would redisplay at cell end.
        canvas = Canvas(figure)
        try:
            manager = FigureManager(canvas, num=0)
            _initialize_canvas(canvas)
        except Exception:
            # Release acquired widgets without downgrading the original failure.
            _release_canvas(canvas)
            raise
        self._managers[figure] = manager
        # Retain the manager even if publishing fails, so release can clean up.
        display(canvas)

    def refresh(
        self,
        figure: Figure,
        *,
        final: bool = False,  # noqa: ARG002 - Both PlotHost modes render synchronously.
    ) -> None:
        manager = self._managers.get(figure)
        if manager is None:
            raise ValueError("Figure is not presented by this host")
        # Idle draws require frontend requests that can wait until the cell ends.
        # Both ordinary updates and final draws use the synchronous path.
        manager.canvas.draw()

    def release(self, figure: Figure) -> None:
        manager = self._managers.get(figure)
        if manager is None:
            return
        # Matplotlib types only the base of this concrete ipympl canvas.
        try:
            _release_canvas(cast("Canvas", manager.canvas))
        finally:
            del self._managers[figure]


def _initialize_canvas(canvas: "Canvas") -> None:
    """Finish the ipympl handshake before a busy cell can defer its requests."""
    width, height = canvas.figure.get_size_inches()
    canvas.toolbar_visible = False
    canvas.header_visible = False
    canvas.footer_visible = False
    canvas.layout.width = f"{int(width * canvas.figure.dpi)}px"
    canvas.layout.height = f"{int(height * canvas.figure.dpi)}px"
    # This third-party protocol is the verified legacy Notebook initialization.
    # Keep it at the widget adapter, without importing the retiring routing backend.
    for message_type in ("refresh", "draw", "send_image_mode", "initialized"):
        canvas._handle_message(canvas, {"type": message_type}, [])  # pyright: ignore[reportPrivateUsage]


def _release_canvas(canvas: "Canvas") -> None:
    from ipympl.backend_nbagg import Toolbar
    from ipywidgets import Widget

    errors: list[Exception] = []

    def close(widget: Widget) -> None:
        try:
            widget.close()
        except Exception as exc:  # noqa: BLE001 - finish cleanup and report failures
            errors.append(exc)
            # Close the comm even if a widget's specialized close failed first.
            try:
                Widget.close(widget)
            except Exception as cleanup_exc:  # noqa: BLE001 - retain both errors
                errors.append(cleanup_exc)

    # A failed manager constructor may not have installed a widget toolbar yet.
    toolbar = canvas.toolbar
    if isinstance(toolbar, Toolbar):
        close(toolbar.layout)
        close(toolbar)
    close(canvas.layout)
    close(canvas)
    FigureCanvasAgg(canvas.figure)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup("Notebook widget release failed", errors)
