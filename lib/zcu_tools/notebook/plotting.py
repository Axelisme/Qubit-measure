"""Explicit Notebook presentation without pyplot figure registration."""

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, cast

from matplotlib import style
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from zcu_tools.plotting import DEFAULT_STYLE
from zcu_tools.plotting.presentation import FigureShowManager

if TYPE_CHECKING:
    from ipympl.backend_nbagg import Canvas, FigureManager

    from zcu_tools.plotting.plots import Plots

_T = TypeVar("_T")


class NotebookPlotHost:
    """Initialize and synchronously render widgets until explicitly released.

    Construction applies the shared Matplotlib style to process-wide rcParams
    for subsequent figures. Apply native style overrides after construction.
    Calls run synchronously on the caller's thread. Callers serialize operations;
    this host does not schedule work or detect frontend availability. Figures
    with another presentation manager must be released by that owner first.
    Framework FigureShowManager is accepted and restored on the released Agg
    canvas. The ipympl manager delegates show to that original operation.
    """

    def __init__(self) -> None:
        style.use(DEFAULT_STYLE)
        self._managers: dict[Figure, FigureManager] = {}
        self._show_managers: dict[Figure, FigureShowManager] = {}

    def call(self, callback: Callable[[], _T]) -> _T:
        return callback()

    def present(self, figure: Figure) -> None:
        if figure in self._managers:
            return
        previous = figure.canvas.manager
        if previous is not None and not isinstance(previous, FigureShowManager):
            raise ValueError("Figure already has a presentation manager")

        from ipympl.backend_nbagg import Canvas, FigureManager
        from IPython.display import display

        # Construct directly: pyplot/Gcf registration would redisplay at cell end.
        canvas = Canvas(figure)
        try:
            if isinstance(previous, FigureShowManager):

                class _NotebookManager(FigureManager):
                    """Keep native show routed through the operation manager."""

                    def show(self) -> None:
                        """Delegate show, including the finished-operation error."""
                        previous.show()

                manager = _NotebookManager(canvas, num=0)
            else:
                manager = FigureManager(canvas, num=0)
            _initialize_canvas(canvas)
        except Exception:
            # Release acquired widgets without downgrading the original failure.
            try:
                _release_canvas(canvas)
            finally:
                if isinstance(previous, FigureShowManager):
                    previous.attach(figure.canvas)
            raise
        if isinstance(previous, FigureShowManager):
            self._show_managers[figure] = previous
        self._managers[figure] = manager
        # Retain the manager even if publishing fails, so release can clean up.
        display(canvas)

    def refresh(
        self,
        figure: Figure,
        *,
        # tolerated: PlotHost requires this keyword; both modes draw synchronously.
        final: bool = False,  # noqa: ARG002
    ) -> None:
        from ipympl.backend_nbagg import Canvas

        manager = self._managers.get(figure)
        if manager is None:
            raise ValueError("Figure is not presented by this host")
        canvas = manager.canvas
        if not isinstance(canvas, Canvas):
            raise TypeError("Presented figure lost its ipympl canvas")
        # Idle draws require frontend requests that can wait until the cell ends.
        # Both ordinary updates and final draws use the synchronous path.
        # The frontend's base-image request waits for the busy cell too, so a diff
        # frame would land on a blank canvas; send a complete frame every time.
        # tolerated: ipympl offers full-frame mode only through this trait.
        canvas._force_full = True  # pyright: ignore[reportPrivateUsage]
        canvas.draw()

    def release(self, figure: Figure) -> None:
        manager = self._managers.get(figure)
        if manager is None:
            return
        try:
            # tolerated(cast): FigureManager.canvas is typed as FigureCanvasBase, but
            # present() installs ipympl Canvas; release needs its widget layout/toolbar.
            _release_canvas(cast("Canvas", manager.canvas))
        finally:
            del self._managers[figure]
            previous = self._show_managers.pop(figure, None)
            if previous is not None:
                previous.attach(figure.canvas)


def finish_failed_plots(
    plots: "Plots", error: BaseException, *, operation: str
) -> None:
    """Release this failed operation's plots without losing its original error."""
    try:
        try:
            plots.finish(present=False)
        finally:
            plots.release()
    # tolerated: Retain producer and cleanup failures, including interrupts.
    except BaseException as cleanup_error:  # noqa: BLE001
        raise BaseExceptionGroup(
            f"{operation} and plot cleanup failed", [error, cleanup_error]
        ) from None


def _initialize_canvas(canvas: "Canvas") -> None:
    """Finish the ipympl handshake before a busy cell can defer its requests."""
    width, height = canvas.figure.get_size_inches()
    canvas.toolbar_visible = False
    canvas.header_visible = False
    canvas.footer_visible = False
    canvas.layout.width = f"{int(width * canvas.figure.dpi)}px"
    canvas.layout.height = f"{int(height * canvas.figure.dpi)}px"
    for message_type in ("refresh", "draw", "send_image_mode", "initialized"):
        # tolerated: ipympl exposes its pre-publication handshake only via this hook.
        canvas._handle_message(canvas, {"type": message_type}, [])  # pyright: ignore[reportPrivateUsage]


def _release_canvas(canvas: "Canvas") -> None:
    from ipympl.backend_nbagg import Toolbar
    from ipywidgets import Widget

    errors: list[Exception] = []

    def close(widget: Widget) -> None:
        try:
            widget.close()
        # tolerated: Finish releasing all widgets and report each close failure.
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            # Close the comm even if a widget's specialized close failed first.
            try:
                Widget.close(widget)
            # tolerated: Retain fallback comm-close failure alongside widget-close error.
            except Exception as cleanup_exc:  # noqa: BLE001
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
