"""Workflow environments and yield-from helpers."""

from __future__ import annotations

from collections.abc import Callable, Generator
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...progress_bar import BaseProgressBar
from .display import Live1D, Live2D, Live2DRow
from .effects import DeviceEffect, RunEffect, WaitEffect
from .models import Completed, Effect, Failed, MissingCapability, ProgressSnapshot
from .ports import PlotPort, ProgressFactory
from .run import Run


class InitEnv[C]:
    """Init-only data: started_at is aware UTC, context is detached read-only data.

    The host owns context's deep immutability. context=None means the capability
    is absent and access raises MissingCapability. Naive or non-UTC started_at
    raises ValueError. No files, hardware, or plots are exposed to init.
    """

    def __init__(self, started_at: datetime, context: C | None) -> None:
        if started_at.utcoffset() != timedelta(0):
            raise ValueError("Workflow start must be timezone-aware UTC")
        self._started_at = started_at
        self._context = context

    @property
    def started_at(self) -> datetime:
        """Return the original aware UTC start, unchanged by Resume."""
        return self._started_at

    @property
    def context(self) -> C:
        """Return host-owned deeply read-only data, or raise MissingCapability."""
        if self._context is None:
            raise MissingCapability("context")
        return self._context


class Displays:
    """Engine-owned named axes and bars shared across step environments.

    plots captures figures on the execution thread. progress is the concrete
    runtime bar factory, not make_pbar. This owner retains bars during Pause and
    closes them at terminal; environments do not own their lifecycle.
    """

    def __init__(self, plots: PlotPort, progress: ProgressFactory) -> None:
        self._plots = plots
        self._progress = progress
        self._axes: dict[str, Axes] = {}
        self._bars: dict[str, BaseProgressBar] = {}
        self._lock = RLock()

    def axes(self, name: str) -> Axes:
        """Return stable named axes; reject empty names with ValueError."""
        if not name.strip():
            raise ValueError("Axes name must not be empty")
        if name not in self._axes:
            self._axes[name] = self._plots.axes(name)
        return self._axes[name]

    def pbar(self, desc: str, total: int) -> BaseProgressBar:
        """Return a retained named bar, requiring a consistent nonnegative total."""
        if not desc.strip() or type(total) is not int or total < 0:
            raise ValueError(
                "Progress requires a nonempty name and nonnegative integer total"
            )
        with self._lock:
            if desc not in self._bars:
                self._bars[desc] = self._progress(total=total, desc=desc)
            bar = self._bars[desc]
            if bar.total != total:
                raise ValueError(f"Progress total changed for {desc!r}")
            return bar

    def snapshot(self) -> tuple[ProgressSnapshot, ...]:
        """Return detached display values; they are not commit evidence.

        Bar creation and snapshot iteration are serialized. A backend may
        update numeric values concurrently; this is a display-only observation.
        """
        with self._lock:
            return tuple(
                ProgressSnapshot(name, bar.total, bar.n)
                for name, bar in self._bars.items()
            )

    def refresh(self) -> None:
        """Snapshot each used figure once; propagate plotting failures."""
        figures = tuple(
            dict.fromkeys(axis.get_figure(root=True) for axis in self._axes.values())
        )
        for figure in figures:
            if not isinstance(figure, Figure):
                raise ValueError("Workflow axes must belong to a root Figure")
            self._plots.refresh(figure)

    def close(self) -> None:
        """Close all retained workflow bars after producers stop."""
        for bar in self._bars.values():
            bar.close()


class WorkflowEnv[C](InitEnv[C]):
    """One step's detached context, files directory, and effect capabilities.

    started_at is the run's aware UTC start, unchanged by Resume. context is
    host-owned deeply read-only data. iter_dir is this invocation's files/
    directory, not the run root. displays is the engine's retained display owner.
    Use yield from for every effect; the engine discards the generator on cancel.
    """

    def __init__(
        self,
        started_at: datetime,
        context: C | None,
        iter_dir: Path,
        displays: Displays,
    ) -> None:
        super().__init__(started_at, context)
        self.iter_dir = iter_dir
        self._displays = displays

    def run[Cfg, Result](
        self,
        experiment: Callable[[Run[Cfg]], Result],
        cfg: Cfg,
        *,
        save: Callable[[Completed[Cfg, Result], Path], None],
        live: Live1D | Live2D | Live2DRow | None = None,
        setup_devices: bool = True,
    ) -> Generator[Effect, None, Completed[Cfg, Result] | Failed]:
        """Execute and save an experiment; return its typed Completed or Failed.

        experiment must have an ASCII letter/digit/underscore function name;
        anonymous or unnamed callables raise ValueError at effect entry.
        cfg must be a deepcopy-able dataclass. save must write and close the
        exact temporary path; the engine publishes it before returning Completed.
        live optionally projects a unique buffer. setup_devices=False skips
        automatic cfg.dev setup. Outer, live, saver, and deepcopy errors
        propagate; Schedule failures return Failed without data or a run file.
        Cancellation discards the step and never returns a cancellation value.
        """
        request = RunEffect(experiment, cfg, save, live, setup_devices)
        yield request
        if request.outcome is None:
            raise RuntimeError("Engine did not complete the run effect")
        return request.outcome

    def set_device(self, name: str, value: float) -> Generator[Effect, None, None]:
        """Set a finite absolute native-device value, or propagate adapter failure.

        Empty names and nonfinite values raise ValueError. The adapter enforces
        connectivity and limits. Cancellation discards the entire step.
        """
        request = DeviceEffect(name, value)
        yield request
        if not request.done:
            raise RuntimeError("Engine did not complete the device effect")

    def wait_until(self, target: datetime) -> Generator[Effect, None, None]:
        """Wait for an aware deadline, returning immediately for past targets.

        Naive datetimes raise ValueError. Cancellation discards the entire step,
        leaving its state-derived deadline unchanged for a later Resume.
        """
        request = WaitEffect(target)
        yield request
        if not request.done:
            raise RuntimeError("Engine did not complete the wait effect")

    def axes(self, name: str) -> Axes:
        """Return the engine's stable axes for a nonempty name."""
        return self._displays.axes(name)

    def pbar(self, desc: str, total: int) -> BaseProgressBar:
        """Return a retained named bar with a consistent nonnegative integer total."""
        return self._displays.pbar(desc, total)
