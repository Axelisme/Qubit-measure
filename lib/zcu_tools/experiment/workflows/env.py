"""Async workflow operations over the shared cooperative script host."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Literal

from pydantic import BaseModel

from zcu_tools.core.script_protocol import Answer, Question
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.script_host import CallContext, ScriptHost
from zcu_tools.plotting.plots import Plots
from zcu_tools.progress_bar import BaseProgressBar

from .models import Completed, Failed, MissingCapability, ProgressSnapshot
from .ports import DevicePort, EnginePorts, ProgressFactory
from .run import LiveUpdate, Run


def _validate_model(value: object, model: type[BaseModel], label: str) -> None:
    if not isinstance(value, model):
        raise TypeError(f"{label} must be a {model.__name__} instance")


class Displays:
    """Retain named progress bars until the run's producer has drained.

    progress is a concrete runtime factory. Named bars survive operation retries
    and Pause. The engine closes them at terminal; Plots remains host-owned.
    """

    def __init__(self, progress: ProgressFactory) -> None:
        self._progress = progress
        self._bars: dict[str, BaseProgressBar] = {}
        self._lock = RLock()

    def pbar(self, desc: str, total: int) -> BaseProgressBar:
        """Return a named bar, requiring a consistent nonnegative integer total."""
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
        """Read display snapshots, not evidence of committed iteration records."""
        with self._lock:
            return tuple(
                ProgressSnapshot(name, bar.total, bar.n)
                for name, bar in self._bars.items()
            )

    def close(self) -> None:
        """Close all retained bars after authored and executor work has drained."""
        for bar in self._bars.values():
            bar.close()


@dataclass(frozen=True)
class EnvHooks:
    """Keep iteration storage and lifecycle policy in the execution owner.

    iteration_dir reads the current root. commit persists one model atomically
    with record publication. boundary acknowledges a reached suspension point.
    stopping reads explicit run cancellation. experiment_failed journals a
    callback name, reason and original acquisition cause.
    """

    new_commit_dir: Callable[[], Path]
    commit: Callable[[BaseModel, Path | None], None]
    boundary: Callable[[], None]
    stopping: Callable[[], bool]
    experiment_failed: Callable[[str, str, BaseException], None]
    question: Callable[[Literal["waiting", "answered"]], None]


class WorkflowEnv[C]:
    """One async author's capabilities, committed iteration root and operations.

    started_at is aware UTC. ports borrows native handles without connecting or
    copying them. host owns loop/executor and scope tokens. displays retains
    bars. hooks delegates persistence/control to the Engine execution owner.
    Authors use env.call for blocking work, env.ask for decisions, and explicit
    env.commit for records. Missing capabilities raise at use, not at Start.
    """

    def __init__(
        self,
        started_at: datetime,
        ports: EnginePorts[C],
        host: ScriptHost,
        displays: Displays,
        hooks: EnvHooks,
    ) -> None:
        if started_at.utcoffset() != timedelta(0):
            raise ValueError("Workflow start must be timezone-aware UTC")
        self._started_at = started_at
        self._ports = ports
        self._host = host
        self._displays = displays
        self._hooks = hooks

    @property
    def started_at(self) -> datetime:
        """Return the original aware UTC run start, unchanged by Resume."""
        return self._started_at

    @property
    def context(self) -> C:
        """Return host-owned read-only context, or raise MissingCapability."""
        if self._ports.context is None:
            raise MissingCapability("context")
        return self._ports.context

    def new_commit_dir(self) -> Path:
        """Allocate/create a new absolute commits/<six-digit number> directory.

        Numbers follow allocation order, independently of successful commits.
        Allocation does not commit a record or change the iteration token. Raw
        files in unused directories remain author-owned and are never parsed.
        """
        return self._hooks.new_commit_dir()

    def commit_dirs(self) -> Iterator[Path]:
        """Lazily allocate one new directory per next(), without an upper bound.

        Put authored values first in zip(values, env.commit_dirs()) so exhausting
        those values does not allocate an unused extra directory.
        """
        while True:
            yield self.new_commit_dir()

    @property
    def soc(self) -> object:
        """Return the native SoC handle, or raise MissingCapability('soc')."""
        if self._ports.soc is None:
            raise MissingCapability("soc")
        return self._ports.soc

    @property
    def devices(self) -> DevicePort:
        """Return connected-device access, or raise MissingCapability('devices')."""
        if self._ports.devices is None:
            raise MissingCapability("devices")
        return self._ports.devices

    async def call[T](self, fn: Callable[[CallContext], T]) -> Completed[T] | Failed:
        """Run fn(ctx) in the executor and deliver its value or acquire failure.

        ctx is the host CallContext: stop_signal and starting epoch only. Use
        closures over env for other capabilities. fn always starts, even with a
        stopped token, and must cooperate and return its partial/empty value.
        Arbitrary exceptions propagate into the author. A captured Schedule
        acquire failure returns Failed without partial data. Pause keeps this
        await pending: interrupted attempts restart with the same callable;
        naturally completed attempts wait for Resume without being repeated.
        Stop does not throw CancelledError, and does not suppress later calls.
        """

        def invoke(ctx: CallContext) -> T | Failed:
            value = fn(ctx)
            error = ctx.stop_signal.error
            if error is not None and error.status == "failed":
                return Failed(error.reason)
            return value

        while True:
            outcome = await self._host.call(invoke)
            self._hooks.boundary()
            await self._host.wait_until_resumed()
            if (
                outcome.epoch != self._host.epoch
                and outcome.partial_result
                and not self._hooks.stopping()
            ):
                continue
            if isinstance(outcome.result, Failed):
                return outcome.result
            return Completed(outcome.result, partial_result=outcome.partial_result)

    async def run[Cfg: ExpCfgModel, Result](
        self,
        experiment: Callable[[Run[Cfg]], Result],
        cfg: Cfg,
        *,
        live: LiveUpdate | None = None,
    ) -> Completed[RunRecord[Cfg, Result]] | Failed:
        """Run an experiment with a copied ExpCfgModel, without saving it.

        The callback receives the existing Run with its own cfg, operation
        token, captured connected devices and optional live buffer callback.
        Completed.value is a RunRecord: cfg reflects callback-return time and
        result is the experiment data. Saver callers pass that record directly.
        cfg must be an ExpCfgModel value, not a factory; dataclass/other BaseModel
        inputs instead use env.call and construct the general Run explicitly.
        Retry after Pause uses the same captured cfg. Device-capture, callback,
        live and copy errors propagate; captured acquire failure returns Failed
        and journals ExperimentFailed. Stop delivers partial/empty RunRecord.
        """
        _validate_model(cfg, ExpCfgModel, "env.run cfg")
        captured = deepcopy(cfg)

        def invoke(ctx: CallContext) -> RunRecord[Cfg, Result]:
            run = Run(
                captured,
                ctx.stop_signal,
                device_snapshot=self.devices.capture_snapshot(),
                soc=self._ports.soc,
                soccfg=self._ports.soccfg,
                devices=self._ports.devices,
            )
            if live is not None:
                run.bind_live(live)
            result = experiment(run)
            if run.live_error is not None:
                raise run.live_error
            outcome = run.outcome
            if outcome.status == "failed":
                cause = outcome.exception or RuntimeError(outcome.reason)
                self._hooks.experiment_failed(
                    getattr(experiment, "__qualname__", type(experiment).__name__),
                    outcome.reason or str(cause),
                    cause,
                )
            return RunRecord(run.cfg, result)

        return await self.call(invoke)

    def get_device_cfg(self, name: str) -> BaseDeviceInfo:
        """Live-read detached typed settings; name/I/O errors propagate."""
        return self.devices.get_device_cfg(name)

    async def setup_device(
        self, name: str, cfg: BaseDeviceInfo
    ) -> Completed[None] | Failed:
        """Apply a captured full settings model with cooperative operation stop.

        The device owner revalidates fields and enforces limits. Errors propagate
        unchanged. Pause retries the same captured settings; Stop still invokes
        the device with an already-stopped token for cooperative empty work.
        """
        captured = deepcopy(cfg)
        return await self.call(
            lambda ctx: self.devices.setup_device(name, captured, ctx.stop_signal)
        )

    async def wait_until(self, target: datetime) -> Completed[None] | Failed:
        """Wait for an aware deadline through call; Stop cooperatively returns.

        A naive deadline raises ValueError. A past target returns immediately.
        Pause retries the same deadline, not a new relative delay.
        """
        if target.tzinfo is None or target.utcoffset() is None:
            raise ValueError("Wait target must be timezone-aware")
        return await self.call(
            lambda ctx: self._ports.clock.wait_until(target, ctx.stop_signal)
        )

    async def ask[A: Answer](self, question: Question[A]) -> A | None:
        """Await one request-id-bound answer, or None after explicit Stop.

        Questions have no timeout/default and do not authorize writes. Stop
        releases an existing question and later asks return None immediately.
        No new worker segment is created when a caller delivers the answer.
        """
        self._hooks.boundary()
        self._hooks.question("waiting")
        try:
            return await self._host.ask(question)
        finally:
            self._hooks.question("answered")

    async def commit(self, record: BaseModel, d: Path | None = None) -> None:
        """Commit one typed record, then advance iteration storage and scope.

        record must be a BaseModel. d is a directory allocated by this run;
        None allocates a fresh one. Foreign or already committed d raises
        ValueError without publishing a record. Journal append/flush precedes
        publication; record.json is best effort, not a crash checkpoint. Commit
        waits at the pause gate; Stop permits authored cleanup commits. Files
        left in uncommitted allocated directories remain without a record.
        """
        _validate_model(record, BaseModel, "Workflow record")
        self._hooks.boundary()
        await self._host.wait_until_resumed()
        self._hooks.commit(record, d)
        await self._host.commit()

    @property
    def plots(self) -> Plots:
        """Return shared injected Plots; its host owns artist thread/lifetime."""
        return self._ports.plots

    def pbar(self, desc: str, total: int) -> BaseProgressBar:
        """Return a retained named bar with a consistent nonnegative total."""
        return self._displays.pbar(desc, total)
