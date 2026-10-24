"""Async workflow operations over the shared cooperative script host."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Literal

from pydantic import BaseModel

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.core.script_protocol import Answer, Question
from zcu_tools.datafile import JsonObject
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.experiment.acquisition import capture_device_snapshot
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.operations import RunFunction
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.scope import Scope, ScriptCancelled
from zcu_tools.experiment.script_host import CallContext, CallResult, ScriptHost
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
        """Attempt every close after producer drain, raising the first failure.

        Later close failures are logged and attached as notes to the first one.
        """
        errors: list[Exception] = []
        for bar in self._bars.values():
            try:
                bar.close()
            except Exception as error:
                _logger.exception("Failed to close workflow progress bar")
                errors.append(error)
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                primary.add_note(
                    f"Another progress close failed: {type(error).__name__}: {error}"
                )
            raise primary


@dataclass(frozen=True)
class EnvHooks:
    """Keep iteration storage and lifecycle policy in the execution owner.

    directory lazily mirrors the explicit receiver scope. commit snapshots a model
    before record publication. boundary acknowledges suspension or raises the
    original control I/O error. stopping reads explicit cancellation; question
    marks Ask suspension/return. next_attempt returns the nearest iteration's
    birth number (zero outside iterations) and its actual attempt number, including retries;
    operation_finished persists their facts. experiment_failed retains acquire
    causes; device_setup retains the requested full typed settings.
    """

    directory: Callable[[Scope], Path]
    commit: Callable[[Scope, BaseModel], None]
    boundary: Callable[[], None]
    stopping: Callable[[], bool]
    experiment_failed: Callable[[str, str, BaseException], None]
    question: Callable[[Literal["waiting", "answered"]], None]
    next_attempt: Callable[[Scope], tuple[int, int]]
    operation_finished: Callable[[Scope, OperationFacts], None]
    device_setup: Callable[[str, BaseDeviceInfo], None]


class WorkflowEnv[C](Scope):
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
        hooks: EnvHooks,
        *,
        identity: RunIdentity,
        name: str,
        host_factory: Callable[[Scope], ScriptHost] = ScriptHost,
    ) -> None:
        super().__init__("workflow", name=name, runtime=self)
        if started_at.utcoffset() != timedelta(0):
            raise ValueError("Workflow start must be timezone-aware UTC")
        self._started_at = started_at
        self._identity = deepcopy(identity)
        self._ports = ports
        self._host = host_factory(self)
        self.displays = Displays(ports.progress)
        self._hooks = hooks

    @property
    def host(self) -> ScriptHost:
        """Shared host whose root is this exact author environment."""
        return self._host

    def directory(self, scope: Scope) -> Path:
        """Create the receiver's directory through the persistence owner."""
        return self._hooks.directory(scope)

    @property
    def started_at(self) -> datetime:
        """Return the original aware UTC run start, unchanged by Resume."""
        return self._started_at

    def read_identity(self) -> RunIdentity:
        """Return detached startup identity/provenance, unchanged across operations."""
        return deepcopy(self._identity)

    @property
    def context(self) -> C:
        """Return host-owned read-only context, or raise MissingCapability."""
        if self._ports.context is None:
            raise MissingCapability("context")
        return self._ports.context

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

    async def execute_call[T](
        self, scope: Scope, fn: Callable[[CallContext], T], *, name: str | None = None
    ) -> Completed[T] | Failed:
        """Run fn(ctx) in the executor and deliver its value or acquire failure.

        ctx is the host CallContext: explicit operation scope and starting epoch.
        Use closures for other capabilities. Child-cancelled work starts with its
        stopped token and cooperatively returns a partial/empty value.
        Arbitrary exceptions propagate into the author. A captured Schedule
        acquire failure returns Failed without partial data. Pause keeps this
        await pending: interrupted attempts restart with the same callable;
        naturally completed attempts wait for Resume without being repeated.
        Root stop raises ordinary ScriptCancelled before new work or after drain.
        Completed.facts is the same completion-time OperationFacts written to
        journal and commit, including iteration birth, attempt and stopped.
        Neither directory allocation nor commit advances iteration.
        """

        return await self._call(scope, "call", lambda ctx, attempt: fn(ctx), name=name)

    async def _call[T](
        self,
        scope: Scope,
        kind: OperationKind,
        fn: Callable[[CallContext, _Attempt], T],
        *,
        name: str | None = None,
    ) -> Completed[T] | Failed:
        while True:
            attempts: list[_Attempt] = []

            def invoke(
                ctx: CallContext, attempts: list[_Attempt] = attempts
            ) -> _Value[T] | Failed:
                return self._invoke(scope, fn, attempts, ctx)

            completions: list[CallResult[_Value[T] | Failed]] = []
            cancelled: ScriptCancelled | None = None
            self._hooks.boundary()
            try:
                result = await self._host.call(
                    invoke, scope=scope, name=name, on_completed=completions.append
                )
            except ScriptCancelled as error:
                if not completions:
                    raise
                result = completions[0]
                cancelled = error
            except BaseException as error:
                if attempts:  # Executor admission can fail before an attempt starts.
                    self._record_attempt(
                        scope,
                        kind,
                        attempts[0],
                        "raised",
                        f"{type(error).__name__}: {error}",
                    )
                raise
            attempt = attempts[0]
            # Descendant partial completion does not set the operation stop bit.
            # Use the atomic host snapshot, never a later control-state read.
            attempt.stopped = result.stopped
            discard = (
                result.epoch != self._host.epoch
                and result.stopped
                and not self._hooks.stopping()
            )
            facts = self._complete_attempt(
                scope, kind, attempt, result, discard=discard
            )
            if cancelled is not None:
                raise cancelled
            self._hooks.boundary()
            await self._host.wait_until_resumed()
            if self.cancelled:
                raise ScriptCancelled()
            if discard:
                continue
            if isinstance(result.result, Failed):
                return result.result
            return Completed(result.result.value, result.partial_result, facts)

    def _complete_attempt[T](
        self,
        scope: Scope,
        kind: OperationKind,
        attempt: _Attempt,
        result: CallResult[_Value[T] | Failed],
        *,
        discard: bool,
    ) -> OperationFacts:
        """Classify the immutable completion snapshot before any cancel exception."""
        if isinstance(result.result, Failed):
            return self._record_attempt(
                scope, kind, attempt, "failed", result.result.reason
            )
        if discard:
            return self._record_attempt(
                scope, kind, attempt, "discarded", "Pause interrupted this attempt"
            )
        if result.partial_result:
            return self._record_attempt(
                scope, kind, attempt, "partial", "Operation or descendant was stopped"
            )
        return self._record_attempt(scope, kind, attempt, "completed")

    def _invoke[T](
        self,
        scope: Scope,
        fn: Callable[[CallContext, _Attempt], T],
        attempts: list[_Attempt],
        ctx: CallContext,
    ) -> _Value[T] | Failed:
        started = self._ports.clock.now()
        iteration, number = self._hooks.next_attempt(scope)
        attempt = _Attempt(iteration, number, started, started)
        attempts.append(attempt)
        try:
            value = fn(ctx, attempt)
            error = ctx.scope.stop.error
            if error is not None and error.status == "failed":
                return Failed(error.reason)
            return _Value(value)
        finally:
            attempt.stopped = ctx.scope.stop.is_set()
            attempt.finished_at = self._ports.clock.now()

    def _record_attempt(
        self,
        scope: Scope,
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
        self._hooks.operation_finished(scope, facts)
        return facts

    async def execute_run[Cfg: ConfigBase, Result](
        self,
        scope: Scope,
        experiment: RunFunction[Cfg, Result],
        cfg: Cfg,
        *,
        name: str | None = None,
    ) -> Completed[RunRecord[Cfg, Result]] | Failed:
        """Run a core callable with a copied configuration, without saving it.

        experiment accepts (cfg, *, context: RunContext), including a bound
        T1Exp().run. cfg must be a ConfigBase; other scans use raw env.call.
        Each attempt builds RunContext with this operation token and borrowed
        soc/soccfg/devices and fresh operation Plots on the workflow's same
        presentation host, captures device settings, then invokes the core
        callable with a fresh deepcopy of cfg. Completed.value is RunRecord with
        callback-return cfg and result. Completed.facts carries the same
        OperationFacts as journal/commit, including the pre-call device snapshot,
        nearest iteration birth number, attempt and stopped bit.
        Schedule acquire failure returns Failed; other exceptions propagate.
        Child cancellation delivers a partial/empty record. Root stop raises
        ScriptCancelled after completion recording. Pause retries the captured cfg
        with a new plot collection. Each attempt finishes before delivery or
        discard; the next attempt releases the previous presentation. A core
        exception remains primary if finish also fails (cleanup adds a note).
        A finish-only failure propagates instead of delivering Completed.
        Framework journal/commit envelopes retain attempt facts and snapshots;
        provenance stays in workflow startup metadata, not a per-operation identity.
        """
        _validate_model(cfg, ConfigBase, "env.run cfg")
        captured = deepcopy(cfg)

        def invoke(ctx: CallContext, attempt: _Attempt) -> RunRecord[Cfg, Result]:
            plots = self.plots.start_operation()
            try:
                context = RunContext(
                    soc=self.soc,
                    soccfg=self._ports.soccfg,
                    plots=plots,
                    devices=self.devices.get_connected_devices(),
                    scope=ctx.scope,
                )
                attempt.device_snapshot = capture_device_snapshot(context.devices)
                current = deepcopy(captured)
                result = experiment(current, context=context)
                error = context.scope.stop.error
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

        return await self._call(scope, "run", invoke, name=name)

    def read_device_cfg(self, name: str) -> BaseDeviceInfo:
        """Live-read detached typed settings; name/I/O errors propagate."""
        return self.devices.get_device_cfg(name)

    async def execute_setup(
        self, scope: Scope, name: str, cfg: BaseDeviceInfo
    ) -> Completed[None] | Failed:
        """Apply a captured full settings model with cooperative operation stop.

        The device owner revalidates fields and enforces limits. Errors propagate
        unchanged. Pause retries the same captured settings. Child cancellation
        still invokes the device with a stopped token; root stop rejects new work.
        Completed.facts records this setup attempt with device_snapshot=None;
        no snapshot read is required.
        """
        captured = deepcopy(cfg)

        def invoke(ctx: CallContext, _attempt: _Attempt) -> None:
            self.devices.setup_device(name, captured, ctx.scope.stop)
            self._hooks.device_setup(name, captured)

        return await self._call(scope, "setup_device", invoke)

    async def execute_wait(
        self, scope: Scope, target: datetime
    ) -> Completed[None] | Failed:
        """Wait for an aware deadline through call; Stop cooperatively returns.

        A naive deadline raises ValueError. A past target returns immediately.
        Pause retries the same deadline, not a new relative delay. Completed.facts
        records this wait attempt with device_snapshot=None and its stop bit.
        """
        if target.tzinfo is None or target.utcoffset() is None:
            raise ValueError("Wait target must be timezone-aware")
        return await self._call(
            scope,
            "wait_until",
            lambda ctx, attempt: self._ports.clock.wait_until(target, ctx.scope.stop),
        )

    async def ask[A: Answer](self, question: Question[A]) -> A:
        """Await one request-id-bound answer, raising ScriptCancelled on root Stop.

        Questions have no timeout/default and do not authorize writes. Root Stop
        releases an existing question and rejects later asks with ScriptCancelled.
        No new worker segment is created when a caller delivers the answer.
        """
        self._hooks.boundary()
        self._hooks.question("waiting")
        try:
            return await self._host.ask(question)
        finally:
            self._hooks.question("answered")

    def execute_commit(self, scope: Scope, record: BaseModel) -> None:
        """Publish this receiver's snapshot synchronously, without waiting for Resume."""
        self._hooks.commit(scope, record)

    @property
    def plots(self) -> WorkflowPlots:
        """Return injected workflow trends, independent of each env.run attempt.

        This collection persists across commits. The operation/ name prefix is
        reserved for the pane/MCP projection and raises ValueError at registration.
        """
        return self._ports.plots

    def pbar(self, desc: str, total: int) -> BaseProgressBar:
        """Return a retained named bar with a consistent nonnegative total."""
        return self.displays.pbar(desc, total)
