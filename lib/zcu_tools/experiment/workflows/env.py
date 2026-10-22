"""Async workflow operations over the shared cooperative script host."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Literal, Protocol

from pydantic import BaseModel

from zcu_tools.core.script_protocol import Answer, Question
from zcu_tools.datafile import JsonObject
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.experiment.acquisition import capture_device_snapshot
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.script_host import CallContext, ScriptHost
from zcu_tools.progress_bar import BaseProgressBar

from .models import (
    Completed,
    Failed,
    MissingCapability,
    OperationFacts,
    OperationKind,
    OperationOutcome,
    ProgressSnapshot,
    RunIdentity,
)
from .plotting import WorkflowPlots
from .ports import DevicePort, EnginePorts, ProgressFactory

_logger = logging.getLogger(__name__)


def _validate_model(value: object, model: type[BaseModel], label: str) -> None:
    if not isinstance(value, model):
        raise TypeError(f"{label} must be a {model.__name__} instance")


class RunFunction[Cfg: ExpCfgModel, Result](Protocol):
    """Core experiment callable, including bound Experiment.run methods.

    cfg is an owned, mutable experiment configuration passed positionally; its
    parameter name is unrestricted. context provides borrowed hardware, plots,
    drivers and the cooperative operation cancellation token.
    """

    def __call__(self, cfg: Cfg, /, *, context: RunContext) -> Result: ...


@dataclass
class _Attempt:
    iteration: int
    number: int
    started_at: datetime
    finished_at: datetime
    device_snapshot: Mapping[str, JsonObject] | None = None
    stopped: bool = False


@dataclass(frozen=True)
class _Value[T]:
    # Distinguish arbitrary raw values (including Failed) from acquire failure.
    value: T


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

    new_commit_dir allocates a fresh author directory. commit persists a model
    before record publication. boundary acknowledges suspension or raises the
    original control I/O error. stopping reads explicit cancellation; question
    marks Ask suspension/return. next_attempt returns the one-based successful
    commit scope and actual executor attempt number within it, including retries;
    operation_finished persists their facts. experiment_failed retains acquire
    causes; device_setup retains the requested full typed settings.
    """

    new_commit_dir: Callable[[], Path]
    commit: Callable[[BaseModel, Path | None], None]
    boundary: Callable[[], None]
    stopping: Callable[[], bool]
    experiment_failed: Callable[[str, str, BaseException], None]
    question: Callable[[Literal["waiting", "answered"]], None]
    next_attempt: Callable[[], tuple[int, int]]
    operation_finished: Callable[[OperationFacts], None]
    device_setup: Callable[[str, BaseDeviceInfo], None]


class WorkflowEnv[C]:
    """One async author's capabilities, commit allocation and operations.

    started_at is aware UTC. identity is the existing host-supplied RunIdentity,
    detached on construction and access; no new provenance is captured. ports
    borrows native handles without connecting or copying them. host owns
    loop/executor and scope tokens. displays retains bars. hooks delegates
    persistence/control to the Engine execution owner.
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
        *,
        identity: RunIdentity,
    ) -> None:
        if started_at.utcoffset() != timedelta(0):
            raise ValueError("Workflow start must be timezone-aware UTC")
        self._started_at = started_at
        self._identity = deepcopy(identity)
        self._ports = ports
        self._host = host
        self._displays = displays
        self._hooks = hooks

    @property
    def started_at(self) -> datetime:
        """Return the original aware UTC run start, unchanged by Resume."""
        return self._started_at

    @property
    def identity(self) -> RunIdentity:
        """Return detached startup identity/provenance, unchanged across operations."""
        return deepcopy(self._identity)

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
        Completed.facts is the same completion-time OperationFacts written to
        journal and commit, including iteration, attempt and stopped. Directory
        allocation does not advance iteration; successful commit does.
        """

        return await self._call("call", lambda ctx, attempt: fn(ctx))

    async def _call[T](
        self, kind: OperationKind, fn: Callable[[CallContext, _Attempt], T]
    ) -> Completed[T] | Failed:
        while True:
            attempts: list[_Attempt] = []

            def invoke(
                ctx: CallContext, attempts: list[_Attempt] = attempts
            ) -> _Value[T] | Failed:
                return self._invoke(fn, attempts, ctx)

            try:
                result = await self._host.call(invoke)
            except BaseException as error:
                if attempts:  # Executor admission can fail before an attempt starts.
                    self._record_attempt(
                        kind, attempts[0], "raised", f"{type(error).__name__}: {error}"
                    )
                raise
            attempt = attempts[0]
            # Use the host's completion bit, never a later control-state read.
            attempt.stopped = result.partial_result
            discard = (
                result.epoch != self._host.epoch
                and result.partial_result
                and not self._hooks.stopping()
            )
            if isinstance(result.result, Failed):
                facts = self._record_attempt(
                    kind, attempt, "failed", result.result.reason
                )
            elif discard:
                facts = self._record_attempt(
                    kind, attempt, "discarded", "Pause interrupted this attempt"
                )
            elif result.partial_result:
                facts = self._record_attempt(
                    kind, attempt, "partial", "Operation stop token was set"
                )
            else:
                facts = self._record_attempt(kind, attempt, "completed")
            self._hooks.boundary()
            await self._host.wait_until_resumed()
            if discard:
                continue
            if isinstance(result.result, Failed):
                return result.result
            return Completed(result.result.value, result.partial_result, facts)

    def _invoke[T](
        self,
        fn: Callable[[CallContext, _Attempt], T],
        attempts: list[_Attempt],
        ctx: CallContext,
    ) -> _Value[T] | Failed:
        started = self._ports.clock.now()
        iteration, number = self._hooks.next_attempt()
        attempt = _Attempt(iteration, number, started, started)
        attempts.append(attempt)
        try:
            value = fn(ctx, attempt)
            error = ctx.stop_signal.error
            if error is not None and error.status == "failed":
                return Failed(error.reason)
            return _Value(value)
        finally:
            attempt.stopped = ctx.stop_signal.is_set()
            attempt.finished_at = self._ports.clock.now()

    def _record_attempt(
        self,
        kind: OperationKind,
        attempt: _Attempt,
        outcome: OperationOutcome,
        reason: str | None = None,
    ) -> OperationFacts:
        facts = OperationFacts(
            kind=kind,
            iteration=attempt.iteration,
            attempt=attempt.number,
            started_at=attempt.started_at,
            finished_at=attempt.finished_at,
            device_snapshot=attempt.device_snapshot,
            outcome=outcome,
            reason=reason,
            stopped=attempt.stopped,
        )
        self._hooks.operation_finished(facts)
        return facts

    async def run[Cfg: ExpCfgModel, Result](
        self, experiment: RunFunction[Cfg, Result], cfg: Cfg
    ) -> Completed[RunRecord[Cfg, Result]] | Failed:
        """Run a core callable with a copied configuration, without saving it.

        experiment accepts (cfg, *, context: RunContext), including a bound
        T1Exp().run. cfg must be an ExpCfgModel; other scans use raw env.call.
        Each attempt builds RunContext with this operation token and borrowed
        soc/soccfg/devices and fresh operation Plots on the workflow's same
        presentation host, captures device settings, then invokes the core
        callable with a fresh deepcopy of cfg. Completed.value is RunRecord with
        callback-return cfg and result. Completed.facts carries the same
        OperationFacts as journal/commit, including the pre-call device snapshot,
        one-based successful commit scope (iteration), attempt and stopped bit.
        Schedule acquire failure returns Failed; other exceptions propagate.
        Stop delivers a partial/empty record. Pause retries the captured cfg
        with a new plot collection. Each attempt finishes before delivery or
        discard; the next attempt releases the previous presentation. A core
        exception remains primary if finish also fails (cleanup adds a note).
        A finish-only failure propagates instead of delivering Completed.
        Framework journal/commit envelopes retain attempt facts and snapshots;
        provenance stays in workflow startup metadata, not a per-operation identity.
        """
        _validate_model(cfg, ExpCfgModel, "env.run cfg")
        captured = deepcopy(cfg)

        def invoke(ctx: CallContext, attempt: _Attempt) -> RunRecord[Cfg, Result]:
            plots = self.plots.start_operation()
            try:
                context = RunContext(
                    soc=self.soc,
                    soccfg=self._ports.soccfg,
                    plots=plots,
                    devices=self.devices.get_connected_devices(),
                    cancel_signal=ctx.stop_signal,
                )
                attempt.device_snapshot = capture_device_snapshot(context.devices)
                current = deepcopy(captured)
                result = experiment(current, context=context)
                error = context.cancel_signal.error
                if error is not None and error.status != "failed":
                    raise error from error.exception
                if error is not None:
                    self._hooks.experiment_failed(
                        getattr(experiment, "__qualname__", type(experiment).__name__),
                        error.reason,
                        error.exception or error,
                    )
            except BaseException as error:
                try:
                    plots.finish()
                except Exception as cleanup_error:
                    _logger.exception(
                        "Failed to finish operation plots after core failure"
                    )
                    error.add_note(
                        f"Operation plot finish failed: {type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise
            plots.finish()
            return RunRecord(current, result)

        return await self._call("run", invoke)

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
        Completed.facts records this setup attempt with device_snapshot=None;
        no snapshot read is required.
        """
        captured = deepcopy(cfg)

        def invoke(ctx: CallContext, _attempt: _Attempt) -> None:
            self.devices.setup_device(name, captured, ctx.stop_signal)
            self._hooks.device_setup(name, captured)

        return await self._call("setup_device", invoke)

    async def wait_until(self, target: datetime) -> Completed[None] | Failed:
        """Wait for an aware deadline through call; Stop cooperatively returns.

        A naive deadline raises ValueError. A past target returns immediately.
        Pause retries the same deadline, not a new relative delay. Completed.facts
        records this wait attempt with device_snapshot=None and its stop bit.
        """
        if target.tzinfo is None or target.utcoffset() is None:
            raise ValueError("Wait target must be timezone-aware")
        return await self._call(
            "wait_until",
            lambda ctx, attempt: self._ports.clock.wait_until(target, ctx.stop_signal),
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
    def plots(self) -> WorkflowPlots:
        """Return injected workflow trends, independent of each env.run attempt.

        This collection persists across commits. The operation/ name prefix is
        reserved for the pane/MCP projection and raises ValueError at registration.
        """
        return self._ports.plots

    def pbar(self, desc: str, total: int) -> BaseProgressBar:
        """Return a retained named bar with a consistent nonnegative total."""
        return self._displays.pbar(desc, total)
