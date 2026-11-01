"""Named figure management with explicit owner execution and presentation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import copy, deepcopy
from functools import wraps
from pathlib import Path
from typing import Protocol, TypeVar

from matplotlib import rc_context
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .figures import FigureCollection, NamedFigures
from .presentation import FigureShowManager

_T = TypeVar("_T")


class PlotHost(Protocol):
    """Execute plotting and presentation on one explicit owner.

    call(callback) synchronously returns the callback's result or propagates its
    exception, preserving each producer's call order. The implementation owns
    thread dispatch. present(figure) attaches its view; refresh(figure) requests
    rendering, with final=True flushing the latest frame before return.
    release(figure) detaches presentation without destroying the native figure.
    Hosts preserve FigureShowManager across canvas swaps and restore it on the
    released Agg canvas; replacement backend managers delegate show to it.
    Host errors propagate to the producer. No method infers a current figure.
    """

    def call(self, callback: Callable[[], _T]) -> _T: ...

    def present(self, figure: Figure) -> None: ...

    def refresh(self, figure: Figure, *, final: bool = False) -> None: ...

    def release(self, figure: Figure) -> None: ...


class NonPresentingHost:
    """Keep native artists and savefig usable without rendering or opening a view."""

    def call(self, callback: Callable[[], _T]) -> _T:
        return callback()

    def present(self, figure: Figure) -> None:
        pass

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        pass

    def release(self, figure: Figure) -> None:
        pass


class Plots(FigureCollection):
    """One producer's plotting lifetime, shared by frontend adapters.

    Stop the producer before finish. Host calls are synchronous and preserve
    that producer's order. subplots accepts Matplotlib dimensions and optional
    unique num names, without presenting. Figure.show presents once through the
    host, even after a canvas swap. It raises RuntimeError after finish, including
    after release. Release drops the manager's reference to this operation.
    Finishing fixes membership and stops owner updates; it does not declare
    analysis success. Release presentation separately
    after finishing. Neither finish nor release destroys retained Figure objects.
    """

    def __init__(self, host: PlotHost) -> None:
        super().__init__()
        self._host = host
        self._live: list[Figure] = []
        self._finished = False
        self._finished_figures: NamedFigures | None = None
        self._released: set[Figure] = set()

    def _create_canvas(self, figure: Figure) -> None:
        FigureShowManager(FigureCanvasAgg(figure), lambda: self._show(figure))

    def _show(self, figure: Figure) -> None:
        self.ensure_active()

        def present() -> None:
            self.ensure_active()
            self._present_live(figure)

        self._host.call(present)

    def fork(self) -> Plots:
        """Return an empty independent producer using this same presentation host.

        Membership, Figure ownership, updates, finish and release are not
        shared. The caller owns the new producer's lifecycle. Always return a
        plain Plots, even when this producer has a frontend-specific subtype.
        """
        return Plots(self._host)

    def ensure_active(self) -> None:
        """Raise RuntimeError after finish, before creating or updating artists."""
        if self._finished:
            raise RuntimeError("Plot operation has finished")

    def _present_live(self, figure: Figure) -> None:
        if figure not in self._live:
            self._host.present(figure)
            self._live.append(figure)

    def on_plot_thread(
        self, callback: Callable[[_T], Iterable[Artist] | None]
    ) -> Callable[[_T], None]:
        """Decorate a one-payload matplotlib update for synchronous owner execution.

        Use ``@plots.on_plot_thread`` without arguments. The resulting function
        accepts one deepcopy-compatible payload and returns None. It checks the
        operation lifetime before copying and again on the host owner, then
        calls the body with the snapshot. Copy, body and host errors propagate.
        Finish/release rejects calls with RuntimeError, before copying data.

        The body must return an iterable of changed Artists (Figure included),
        and should mutate only figures belonging to those returned artists.
        Refresh each distinct root figure already presented by this Plots once.
        Before show, only retain mutations: do not present, refresh or load a
        backend. Later show or finish(present=True) displays the latest content.
        Mixed results refresh only presented figures. An empty iterable does
        not refresh. None raises TypeError; artists outside this Plots raise
        ValueError. Validate all results before any
        refresh. Errors do not roll back mutations. Binding claims no axes.
        Use the same decorator for repeated updates and one-off mutations.
        """

        @wraps(callback)
        def push(data: _T) -> None:
            self.ensure_active()
            snapshot = deepcopy(data)

            def apply() -> None:
                self.ensure_active()
                changed = callback(snapshot)
                if changed is None:
                    raise TypeError("Plot callback must return an iterable of artists")
                figures: dict[Figure, None] = {}
                for artist in changed:
                    figure = artist.get_figure(root=True)
                    if not isinstance(figure, Figure) or figure not in self.values():
                        raise ValueError("Changed artist must belong to this Plots")
                    figures[figure] = None
                for figure in figures:
                    if figure in self._live:
                        self._host.refresh(figure)

            self._host.call(apply)

        return push

    def save(
        self,
        name: str,
        path: str | Path,
        *,
        figsize: tuple[float, float] = (6.4, 4.8),
        dpi: float = 100,
    ) -> None:
        """Save an existing named figure on the host owner without refreshing.

        Execution is synchronous on the host owner thread and blocks the calling
        worker thread until saving completes. With a GUI host it occupies the
        GUI thread; expensive rendering can freeze the interface. This method
        invokes no callback and does not refresh. Saving does not restore
        artists removed by an earlier owner callback.
        figsize gives positive finite output dimensions in inches; dpi is the
        positive finite output resolution. Defaults produce a 640 x 480 PNG,
        independently of display size and savefig.bbox settings.
        Preserve an existing layout engine, or use tight layout when absent.
        Restore display size, layout and axes positions even if saving raises.
        Infer format from path. Saving is allowed before or after finish.
        Parent directories are not created. Existing files follow native savefig
        overwrite behavior. Unknown names
        raise KeyError; host, rendering and I/O errors propagate.
        """
        figure = self[name]

        def save_figure() -> None:
            original_width, original_height = figure.get_size_inches()
            original_layout = figure.get_layout_engine()
            original_subplots = copy(figure.subplotpars)
            positions = [
                (
                    axis,
                    axis.get_position(original=True).frozen(),
                    axis.get_position().frozen(),
                    axis.get_in_layout(),
                )
                for axis in figure.axes
            ]
            with rc_context(
                {
                    "savefig.bbox": None,
                    "figure.autolayout": False,
                    "figure.constrained_layout.use": False,
                }
            ):
                try:
                    figure.set_size_inches(*figsize, forward=False)
                    if original_layout is None:
                        figure.set_layout_engine("tight")
                    figure.savefig(path, dpi=dpi)
                finally:
                    figure.set_size_inches(
                        float(original_width), float(original_height), forward=False
                    )
                    figure.set_layout_engine(original_layout)
                    figure.subplotpars.update(
                        left=original_subplots.left,
                        right=original_subplots.right,
                        bottom=original_subplots.bottom,
                        top=original_subplots.top,
                        wspace=original_subplots.wspace,
                        hspace=original_subplots.hspace,
                    )
                    for axis, original, active, in_layout in positions:
                        axis.set_position(original, which="original")
                        axis.set_position(active, which="active")
                        axis.set_in_layout(in_layout)

        self._host.call(save_figure)

    def finish(self, *, present: bool = True) -> NamedFigures:
        if self._finished_figures is not None:
            return self._finished_figures
        self._finished = True
        figures = NamedFigures(self)
        self._finished_figures = figures

        def complete() -> None:
            for figure in self._live:
                self._host.refresh(figure, final=True)
            if present:
                for figure in self.values():
                    if figure not in self._live:
                        self._host.present(figure)

        self._host.call(complete)
        return figures

    def release(self) -> None:
        """Release every presentation, reporting failures after trying all figures."""
        if not self._finished:
            raise RuntimeError(
                "Finish the plot operation before releasing presentation"
            )

        def cleanup() -> None:
            errors: list[Exception] = []
            for figure in self.values():
                if figure in self._released:
                    continue
                try:
                    self._host.release(figure)
                # tolerated: Try every host release, aggregate failures and leave failed figures retryable.
                except Exception as error:  # noqa: BLE001
                    errors.append(error)
                else:
                    manager = figure.canvas.manager
                    if isinstance(manager, FigureShowManager):
                        manager.retire()
                    self._released.add(figure)
            if errors:
                raise ExceptionGroup("Failed to release plot presentation", errors)

        self._host.call(cleanup)
