"""Async workflow operations over the shared cooperative script host."""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Literal

from pydantic import BaseModel

from zcu_tools.core.script_protocol import Answer, Question
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.experiment.scope import Scope, ScriptCancelled
from zcu_tools.experiment.script_host import CallContext, CallResult, ScriptHost
from zcu_tools.progress_bar import BaseProgressBar

from .models import (
    Completed,
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


@dataclass
class _Attempt:
    iteration: int
    number: int
    started_at: datetime
    finished_at: datetime
    stopped: bool = False
    name: str | None = None


class Displays:
    """Retain named progress bars until the run's producer has drained.

    progress is a concrete runtime factory. Named bars survive operations and
    Ask. The engine closes them at terminal; Plots remains host-owned.
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
    before record publication. boundary raises the original control I/O error;
    question marks Ask suspension/return. next_attempt returns the nearest
    iteration's birth number (zero outside iterations) and its call number;
    operation_finished persists their facts; device_setup retains the requested
    full typed settings.
    """

    directory: Callable[[Scope], Path]
    commit: Callable[[Scope, BaseModel], None]
    boundary: Callable[[], None]
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
        """Return the original aware UTC run start throughout this run."""
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
    ) -> Completed[T]:
        """Run fn(ctx) in the executor and deliver its unchanged value.

        ctx supplies the explicit operation scope, borrowed soc/soccfg and
        connected drivers (None when unavailable), and fresh operation plots.
        The framework finishes plots before returning or propagating errors. Child-cancelled work starts with its
        stopped token and cooperatively returns a partial/empty value.
        The callable runs once. Every exception propagates unchanged into the
        author; cooperative child cancellation delivers its returned value.
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
    ) -> Completed[T]:
        attempts: list[_Attempt] = []

        def invoke(ctx: CallContext) -> T:
            return self._invoke(scope, fn, attempts, ctx)

        completions: list[CallResult[T]] = []
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
        # Descendant cancellation does not set the operation stop bit.
        # Use the atomic host snapshot, never a later control-state read.
        attempt.stopped = result.stopped
        facts = self._record_attempt(scope, kind, attempt, "completed")
        if cancelled is not None:
            raise cancelled
        self._hooks.boundary()
        if self.cancelled:
            raise ScriptCancelled()
        return Completed(result.result, facts)

    def _invoke[T](
        self,
        scope: Scope,
        fn: Callable[[CallContext, _Attempt], T],
        attempts: list[_Attempt],
        ctx: CallContext,
    ) -> T:
        started = self._ports.clock.now()
        iteration, number = self._hooks.next_attempt(scope)
        attempt = _Attempt(iteration, number, started, started, name=ctx.scope.name)
        attempts.append(attempt)
        plots = self.plots.start_operation()
        try:
            try:
                context = replace(
                    ctx,
                    soc=self._ports.soc,
                    soccfg=self._ports.soccfg,
                    devices=(
                        None
                        if self._ports.devices is None
                        else self._ports.devices.get_connected_devices()
                    ),
                    plots=plots,
                )
                value = fn(context, attempt)
            except BaseException as error:
                try:
                    plots.finish()
                except Exception as cleanup_error:
                    _logger.exception(
                        "Failed to finish operation plots after callback failure"
                    )
                    error.add_note(
                        f"Operation plot finish failed: {type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise
            plots.finish()
            return value
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
            outcome=outcome,
            reason=reason,
            stopped=attempt.stopped,
            name=attempt.name,
        )
        self._hooks.operation_finished(scope, facts)
        return facts

    def read_device_cfg(self, name: str) -> BaseDeviceInfo:
        """Live-read detached typed settings; name/I/O errors propagate."""
        return self.devices.get_device_cfg(name)

    async def execute_setup(
        self, scope: Scope, name: str, cfg: BaseDeviceInfo
    ) -> Completed[None]:
        """Apply a captured full settings model with cooperative operation stop.

        The device owner revalidates fields and enforces limits. Errors propagate
        unchanged. The device is invoked once. Child cancellation
        still invokes the device with a stopped token; root stop rejects new work.
        Completed.facts records this setup attempt; no driver snapshot is read.
        """
        captured = deepcopy(cfg)

        def invoke(ctx: CallContext, _attempt: _Attempt) -> None:
            self.devices.setup_device(name, captured, ctx.scope.stop)
            self._hooks.device_setup(name, captured)

        return await self._call(scope, "setup_device", invoke)

    async def execute_wait(self, scope: Scope, target: datetime) -> Completed[None]:
        """Wait for an aware deadline through call; Stop cooperatively returns.

        A naive deadline raises ValueError. A past target returns immediately.
        The clock is invoked once. Completed.facts records this wait attempt
        with its stop bit.
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
        """Publish this receiver's snapshot synchronously, including during cleanup."""
        self._hooks.commit(scope, record)

    @property
    def plots(self) -> WorkflowPlots:
        """Return injected workflow trends, independent of each operation.

        This collection persists across commits. The operation/ name prefix is
        reserved for the pane/MCP projection and raises ValueError at registration.
        """
        return self._ports.plots

    def pbar(self, desc: str, total: int) -> BaseProgressBar:
        """Return a retained named bar with a consistent nonnegative total."""
        return self.displays.pbar(desc, total)
