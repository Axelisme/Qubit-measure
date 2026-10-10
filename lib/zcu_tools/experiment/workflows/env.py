"""Workflow environments and yield-from helpers."""

from __future__ import annotations

from collections.abc import Callable, Generator
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Literal

from ...plotting.plots import Plots
from ...progress_bar import BaseProgressBar
from .effects import DeviceEffect, RunEffect, WaitEffect
from .models import Completed, Effect, Failed, MissingCapability, ProgressSnapshot
from .ports import DevicePort, EnginePorts, ProgressFactory
from .run import LiveUpdate, Run


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
    """Engine-owned named bars shared across step environments.

    progress is the concrete runtime bar factory, not make_pbar. This owner
    retains bars during Pause and closes them at terminal; environments do not
    own their lifecycle. Plotting belongs to the injected Plots instead.
    """

    def __init__(self, progress: ProgressFactory) -> None:
        self._progress = progress
        self._bars: dict[str, BaseProgressBar] = {}
        self._lock = RLock()

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

    def close(self) -> None:
        """Close all retained workflow bars after producers stop."""
        for bar in self._bars.values():
            bar.close()


class WorkflowEnv[C](InitEnv[C]):
    """One step's detached context, files directory, and effect capabilities.

    started_at is the run's aware UTC start, unchanged by Resume. context is
    host-owned deeply read-only data. iter_dir is this invocation's files/
    directory, not the run root. displays is the engine's retained display owner.
    soc/devices are read-only accessors for author-owned availability checks.
    Hardware operations should still use effects to retain journal and cancellation
    behavior; the framework trusts authors and does not enforce that restriction.
    Use yield from for every effect; the engine discards the generator on cancel.
    """

    def __init__(
        self,
        started_at: datetime,
        ports: EnginePorts[C],
        iter_dir: Path,
        displays: Displays,
    ) -> None:
        """Bind host ports and retained displays to one step's files directory.

        ports supplies detached context and native handles without copying or
        connecting them. iter_dir is the current invocation's files/ path.
        displays retains bars across steps. started_at obeys InitEnv's UTC
        contract. Missing ports are checked only when their accessor is used.
        """
        super().__init__(started_at, ports.context)
        self.iter_dir = iter_dir
        self._ports = ports
        self._displays = displays

    @property
    def soc(self) -> object:
        """Return the native SoC, or raise MissingCapability('soc').

        Use this for availability checks. Hardware operations should go through
        effects; direct access bypasses engine journaling and cancellation.
        """
        if self._ports.soc is None:
            raise MissingCapability("soc")
        return self._ports.soc

    @property
    def devices(self) -> DevicePort:
        """Return the injected adapter, or raise MissingCapability('devices').

        Use this for availability checks. Hardware operations should go through
        effects; direct access bypasses engine journaling and cancellation.
        """
        if self._ports.devices is None:
            raise MissingCapability("devices")
        return self._ports.devices

    def run[Cfg, Result](
        self,
        experiment: Callable[[Run[Cfg]], Result],
        cfg: Cfg,
        *,
        save: Callable[[Completed[Cfg, Result], Path], None],
        live: LiveUpdate | None = None,
    ) -> Generator[Effect, None, Completed[Cfg, Result] | Failed]:
        """Execute and save an experiment; return its typed Completed or Failed.

        experiment must have an ASCII letter/digit/underscore function name;
        anonymous or unnamed callables raise ValueError at effect entry.
        cfg must be a deepcopy-able dataclass or Pydantic BaseModel instance.
        save must write and close the exact temporary path; the engine publishes
        it before returning Completed.
        live optionally receives (axes, samples) synchronously on each update
        of the experiment's unique buffer: finite float64 axis vectors in buffer
        dimension order, and complex128 samples of the buffer shape, including
        partial NaNs. Do not mutate or retain these arrays; use Plots handles to
        copy/dispatch real projected values to the host owner. Callback failures
        fail the workflow even when Schedule catches them. The experiment owns
        device operations; the engine does not inspect cfg.dev or apply settings.
        Outer, live, saver, and deepcopy errors propagate; Schedule failures
        return Failed without data or a run file.
        Cancellation discards the step and never returns a cancellation value.
        """
        request = RunEffect(experiment, cfg, save, live)
        yield request
        if request.outcome is None:
            raise RuntimeError("Engine did not complete the run effect")
        return request.outcome

    def set_device(
        self, name: str, value: float, *, unit: Literal["A", "V", "native"]
    ) -> Generator[Effect, None, None]:
        """Set a finite absolute device value in the explicitly supplied unit.

        unit is A, V or native; no conversion or mode change occurs. Empty names,
        nonfinite values and unknown units raise ValueError. The adapter enforces
        connectivity and limits. Cancellation discards the entire step.
        """
        request = DeviceEffect(name, value, unit)
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

    @property
    def plots(self) -> Plots:
        """Return the injected Plots shared across steps and retries.

        Use typed handles or draw/save for synchronous host-owner operations.
        Do not mutate active artists on the workflow execution thread. The host
        owns finish/release after the producer stops; Pause does not finish plots.
        """
        return self._ports.plots

    def pbar(self, desc: str, total: int) -> BaseProgressBar:
        """Return a retained named bar with a consistent nonnegative integer total."""
        return self._displays.pbar(desc, total)
