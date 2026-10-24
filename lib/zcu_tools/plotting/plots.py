"""Named figure management with explicit owner execution and presentation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .figures import FigureCollection, NamedFigures

_T = TypeVar("_T")


class PlotHost(Protocol):
    """Execute plotting and presentation on one explicit owner.

    call(callback) synchronously returns the callback's result or propagates its
    exception, preserving each producer's call order. The implementation owns
    thread dispatch. present(figure) attaches its view; refresh(figure) requests
    rendering, with final=True flushing the latest frame before return.
    release(figure) detaches presentation without destroying the native figure.
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
    that producer's order. General figures stay unpresented until finish unless
    drawn. Live factories and the first successful draw present immediately.
    Finishing fixes membership and stops typed updates; it does not declare
    analysis success. Release presentation separately
    after finishing. Neither finish nor release destroys retained Figure objects.
    """

    def __init__(self, host: PlotHost) -> None:
        super().__init__()
        self._host = host
        self._live: list[Figure] = []
        self._live_axes: set[Axes] = set()
        self._finished = False
        self._finished_figures: NamedFigures | None = None
        self._released: set[Figure] = set()

    def fork(self) -> Plots:
        """Return an empty independent producer using this same presentation host.

        Membership, Figure ownership, typed handles, finish and release are not
        shared. The caller owns the new producer's lifecycle. Always return a
        plain Plots, even when this producer has a frontend-specific subtype.
        """
        return Plots(self._host)

    def ensure_active(self) -> None:
        """Raise RuntimeError after finish, before creating or updating artists."""
        if self._finished:
            raise RuntimeError("Plot operation has finished")

    def _resolve_axes(
        self, name: str, axes: tuple[Axes, ...] | None, count: int
    ) -> tuple[Figure, tuple[Axes, ...]]:
        if axes is None:
            figure, _ = self.subplots(name, ncols=count, squeeze=False)
            resolved = tuple(figure.axes)
        else:
            figure = self[name]
            resolved = axes
        if len(resolved) != count or len(set(resolved)) != count:
            raise ValueError("Live plot requires distinct axes of the expected count")
        if any(ax.figure is not figure or ax not in figure.axes for ax in resolved):
            raise ValueError("Live axes must belong to the named figure")
        if any(ax in self._live_axes for ax in resolved):
            raise ValueError("Axes already belong to a live plot")
        self._live_axes.intersection_update(
            ax for current in self.values() for ax in current.axes
        )
        self._live_axes.update(resolved)
        return figure, resolved

    def _present_live(self, figure: Figure) -> None:
        if figure not in self._live:
            self._host.present(figure)
            self._live.append(figure)

    def create_live(
        self,
        name: str,
        initialize: Callable[
            [PlotHost, Figure, tuple[Axes, ...], Callable[[], None]], _T
        ],
        *,
        axes: tuple[Axes, ...] | None = None,
        count: int = 1,
    ) -> _T:
        """Initialize a live figure on its owner and return the callback's handle.

        Without axes, create a unique named figure with count columns. Otherwise
        claim count distinct, unused axes of self[name]. Conflicts raise
        ValueError; an unknown existing name raises KeyError. A finished
        collection raises RuntimeError. The callback receives the host, figure,
        claimed axes and an active-lifetime checker for its update handle.
        It may return a handle that retains artists for owner-managed updates,
        but workers must not mutate those artists directly.
        Present only after initialization succeeds, once per figure. Callback
        and host errors propagate without presentation or rollback.
        """
        self.ensure_active()

        def create() -> _T:
            self.ensure_active()
            figure, resolved = self._resolve_axes(name, axes, count)
            handle = initialize(self._host, figure, resolved, self.ensure_active)
            self._present_live(figure)
            return handle

        return self._host.call(create)

    def draw(self, name: str, callback: Callable[[Figure], None]) -> None:
        """Draw on an existing named figure, then refresh on the host owner.

        Execution is synchronous on the host owner thread and blocks the calling
        worker thread until callback, presentation and refresh return.
        The first successful draw presents the figure and makes it live; later
        draws refresh the same presentation. Untouched general figures still
        wait until finish to present. With a GUI host it
        occupies the GUI thread; heavy drawing can freeze the interface.
        The callback may freely modify axes shared with caller-defined live
        tools, but must not retain artists for later worker-side mutation.
        Calling cla() removes their artists; those tools decide how subsequent
        updates handle removed artists. Callback exceptions propagate without
        rollback, presentation or refresh. Unknown names raise KeyError;
        finished plots raise RuntimeError. Host errors also propagate.
        """
        self.ensure_active()
        figure = self[name]

        def draw_frame() -> None:
            self.ensure_active()
            callback(figure)
            self._present_live(figure)
            self._host.refresh(figure)

        self._host.call(draw_frame)

    def save(self, name: str, path: str | Path) -> None:
        """Save an existing named figure on the host owner without refreshing.

        Execution is synchronous on the host owner thread and blocks the calling
        worker thread until saving completes. With a GUI host it occupies the
        GUI thread; expensive rendering can freeze the interface. This method
        invokes no callback and does not refresh. Saving does not restore
        artists removed by an earlier draw callback.
        Use native Figure.savefig defaults and infer format from path. Saving is
        allowed before or after finish. Parent directories are not created and
        existing files follow native savefig overwrite behavior. Unknown names
        raise KeyError; host, rendering and I/O errors propagate.
        """
        figure = self[name]
        self._host.call(lambda: figure.savefig(path))

    def refresh(self, name: str) -> None:
        """Refresh a complete live frame after a batch of typed updates."""
        self.ensure_active()
        figure = self[name]
        if figure not in self._live:
            raise ValueError("Figure is not live")

        def refresh_frame() -> None:
            self.ensure_active()
            self._host.refresh(figure)

        self._host.call(refresh_frame)

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
                    self._released.add(figure)
            if errors:
                raise ExceptionGroup("Failed to release plot presentation", errors)

        self._host.call(cleanup)
