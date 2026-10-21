"""One run owner for native workflow admission, continuation and terminal delivery."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, JsonValue

from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    Actor,
    Engine,
    EnginePorts,
    InvalidRunState,
    RunIdentity,
    RunMismatch,
    RunPaths,
    RunStatus,
    TunablesSnapshot,
)
from zcu_tools.experiment.workflows.declaration import WorkflowStep
from zcu_tools.gui.form import FormErrors, FormModel, FormWritePath
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    ExclusionRequest,
    OperationSpec,
    SettleFn,
)
from zcu_tools.gui.session.services.progress import BoundProgressFactory

from .tunables import TunablesHost

if TYPE_CHECKING:
    from zcu_tools.gui.session.operation_runner import OperationRunner

logger = logging.getLogger(__name__)


@dataclass
class _Run[F: FormModel, C]:
    """Keep binding, producer ownership and cleanup together for one native run."""

    binding: TunablesHost[F, C]
    phase: Literal[
        "starting", "producing", "paused", "resuming", "finishing", "terminal"
    ] = "starting"
    stop: StopSignal = field(default_factory=StopSignal)
    token: int | None = None
    engine: Engine[C] | None = None
    settle: SettleFn | None = None
    native_error: Exception | None = None


class WorkflowController[F: FormModel, C]:
    """Own one current run from Start admission through final owner delivery.

    A fixed controller accepts a replacement only after terminal cleanup. The
    current binding owns the unique form draft and fixed Start capture. Native
    Engine status remains authoritative; a token or binding is not acceptance.
    Pause retains the operation, local RUN lease and progress. Resume submits a
    producer through the same Runner handle, never another begin. This does not
    arbitrate other processes or restore hardware. Commands run on the owner
    thread; startup publication and cooperative controls share a binding lock.
    """

    def __init__(
        self,
        runner: OperationRunner,
        owner_id: str,
        *,
        on_finished: Callable[[RunStatus | None, Exception | None], None],
    ) -> None:
        """Borrow runner and report paused/terminal native delivery to the view.

        owner_id labels the run lease and progress. runner owns that progress
        service. on_finished receives authoritative native status (None before
        acceptance) and the original producer error, if any. Terminal cleanup
        fixes the native handle outcome and releases admission/lease/progress
        before notification. View errors are logged and do not alter outcome or
        binding. Paused notification retains ownership. Callback reentry may
        Stop, or Start a replacement after terminal cleanup; no old binding is
        written after notification. No instrument call occurs here.
        """
        self._runner = runner
        self._owner_id = owner_id
        self._on_finished = on_finished
        self._lock = RLock()
        self._run: _Run[F, C] | None = None

    @property
    def binding(self) -> TunablesHost[F, C] | None:
        """Return current pending/accepted form owner, or None before Start."""
        return None if self._run is None else self._run.binding

    @property
    def engine(self) -> Engine[C] | None:
        """Return current native Engine, or None before startup acceptance."""
        with self._lock:
            return None if self._run is None else self._run.engine

    @property
    def run_id(self) -> str | None:
        """Return current native target, including pending/terminal, or None.

        Start publishes it before admission notifications. Synchronous Start
        rejection clears the pending binding. It is not an operation token.
        """
        binding = self.binding
        return None if binding is None else binding.run_id

    @property
    def operation_id(self) -> int | None:
        """Return the one live run token, retained while paused; None at terminal."""
        run = self._run
        return None if run is None or run.phase == "terminal" else run.token

    @property
    def has_pending_segment(self) -> bool:
        """Read whether admission, producer or terminal cleanup is still pending.

        False after paused owner delivery, even though operation_id is retained.
        Stop/close must wait for pending work; paused Stop cleans up synchronously.
        """
        run = self._run
        return run is not None and run.phase not in {"paused", "terminal"}

    # tolerated: One attachment submitter pairs native plan/tunables; ports and pending binding have distinct owners.
    def start[P: BaseModel, T: BaseModel, S, R](  # noqa: PLR0913
        self,
        step: WorkflowStep[P, T, S, R, C],
        *,
        plan: P,
        tunables: T,
        paths: RunPaths,
        identity: RunIdentity,
        ports: EnginePorts[C],
        binding: TunablesHost[F, C],
    ) -> int:
        """Publish prepared binding and submit native startup once via Runner.

        ports contains this run's live resources and initial context; Runner's
        progress factory replaces ports.progress. binding must be pending, with
        run_id equal to identity.run_id. The caller converts binding input to
        native tunables through its adapter and is responsible for their match;
        this generic Controller does not validate that conversion. Mismatched
        identity raises RunMismatch, accepted bindings raise RuntimeError.
        Pending/paused/active/finishing replacement raises InvalidRunState before
        mutation. Engine validates declaration/values on the worker. Admission,
        factory and submit errors propagate, clear pending ownership and create
        no native files. Stop before worker startup does not create a native run.
        Startup errors retain partial files and deliver the original error.
        The returned token is a receipt, not native acceptance or completion.
        """
        with self._lock:
            if self._run is not None and self._run.phase != "terminal":
                status = self.status()
                raise InvalidRunState(
                    "start", "running" if status is None else status.lifecycle
                )
            if binding.run_id != identity.run_id:
                raise RunMismatch(identity.run_id, binding.run_id)
            if binding.engine is not None:
                raise RuntimeError("Start requires a pending tunables binding")
            run = _Run(binding)
            self._run = run

        def opened(token: int) -> None:
            run.token = token

        def work(factory: BoundProgressFactory) -> RunStatus | None:
            if run.stop.is_set():
                return None
            engine = Engine(replace(ports, progress=factory))
            status = engine.start(
                step, plan=plan, tunables=tunables, paths=paths, identity=identity
            )
            with self._lock:
                binding.accept(engine)
                run.engine = engine
                if run.stop.is_set():
                    engine.stop(status.run_id)
            return engine.execute(status.run_id)

        def cancel() -> None:
            self.stop(identity.run_id)

        try:
            token = self._runner.begin(
                OperationSpec(
                    exclusion=ExclusionRequest("run", self._owner_id, "workflow run"),
                    owner_id=self._owner_id,
                    wants_progress=True,
                    cancel_hook=cancel,
                    work=work,
                    run_in_pool=False,
                    on_terminal=lambda bg, settle: self._delivered(run, bg, settle),
                    on_opened=opened,
                )
            )
        except Exception:
            self._run = None
            raise
        if run.phase == "starting":
            run.phase = "producing"
        return token

    def _notify(self, status: RunStatus | None, error: Exception | None) -> None:
        try:
            self._on_finished(status, error)
        except Exception:
            logger.exception("Workflow view notification failed")

    def _record_native_failure(self, run: _Run[F, C], error: Exception) -> None:
        status = None if run.engine is None else run.engine.status()
        # Active native failures first request stopping; delivery still owns cleanup.
        if (
            status is not None
            and status.lifecycle in {"stopping", "failed"}
            and status.reason is not None
        ):
            if run.native_error is None:
                run.native_error = error
            if run.phase == "paused" and status.lifecycle == "failed":
                self._finish(run, error)

    def _finish(self, run: _Run[F, C], error: Exception | None) -> None:
        if run.phase in {"finishing", "terminal"}:
            return
        status = None if run.engine is None else run.engine.status()
        if run.native_error is not None:
            error = run.native_error
        if status is not None and status.lifecycle == "failed":
            outcome = OperationOutcome("failed", status.reason)
        elif error is not None:
            outcome = OperationOutcome("failed", str(error))
        elif status is None or status.lifecycle == "stopped":
            outcome = OperationOutcome("cancelled")
        else:
            outcome = OperationOutcome("finished")
        if run.settle is None:
            raise RuntimeError("Terminal delivery requires Runner ownership")
        run.phase = "finishing"
        run.settle(outcome)
        run.phase = "terminal"
        self._notify(status, error)

    def _delivered(self, run: _Run[F, C], bg: BgResult, settle: SettleFn) -> None:
        run.settle = settle
        status = None if run.engine is None else run.engine.status()
        if bg.ok and status is not None and status.lifecycle == "paused":
            run.phase = "paused"
            self._notify(status, None)
        else:
            self._finish(run, bg.error)

    def status(self) -> RunStatus | None:
        """Return current accepted native status; None means no acceptance."""
        engine = self.engine
        return None if engine is None else engine.status()

    def require_run_id(self, run_id: str) -> None:
        """Reject absent/stale target with RunMismatch before any side effect."""
        if run_id != self.run_id:
            raise RunMismatch(run_id, self.run_id)

    def pause(self, run_id: str) -> RunStatus:
        """Request cooperative Pause, retaining the run's operation and lease.

        Stale/absent targets raise RunMismatch. Pending acceptance raises
        RuntimeError. Only a producing running/pausing Engine accepts Pause;
        other phases raise InvalidRunState. Worker return alone is not delivery.
        """
        with self._lock:
            self.require_run_id(run_id)
            run = self._run
            if run is None or run.engine is None:
                raise RuntimeError("Pause requires an accepted workflow run")
            status = run.engine.status()
            if (
                run.phase != "producing"
                or status is None
                or status.lifecycle not in {"running", "pausing"}
            ):
                raise InvalidRunState(
                    "pause", "running" if status is None else status.lifecycle
                )
            try:
                return run.engine.pause(run_id)
            except Exception as error:
                self._record_native_failure(run, error)
                raise

    def resume(self, run_id: str) -> int:
        """Continue a delivered paused run with its retained token and RUN lease.

        Target/state and Runner ownership checks precede continuation submission.
        Submit failure preserves a healthy paused run and original ownership.
        Stop during admission with no producer finishes synchronously, then
        reraises any original error. Native resume/execute run off-main with
        unchanged context, cursor, progress and records; no repeated init.
        """
        with self._lock:
            self.require_run_id(run_id)
            run = self._run
            if run is None or run.engine is None:
                raise RuntimeError("Resume requires an accepted paused workflow run")
            status = run.engine.status()
            if run.phase != "paused" or status is None or status.lifecycle != "paused":
                raise InvalidRunState(
                    "resume", "running" if status is None else status.lifecycle
                )
            if run.token is None or run.settle is None:
                raise RuntimeError("Resume requires retained Runner ownership")
            engine = run.engine
            run.phase = "resuming"
        try:

            def work() -> RunStatus | None:
                with self._lock:
                    if run.stop.is_set():
                        # Delivery still owns cleanup after Stop during admission.
                        return engine.status()
                    engine.resume(run_id)
                return engine.execute(run_id)

            token = self._runner.continue_operation(
                run.token,
                work=work,
                run_in_pool=False,
                on_terminal=lambda bg, settle: self._delivered(run, bg, settle),
                settle=run.settle,
            )
        except Exception:
            status = engine.status()
            if run.stop.is_set() or (
                status is not None and status.lifecycle == "failed"
            ):
                self._finish(run, None)
            else:
                run.phase = "paused"
            raise
        if run.phase == "resuming":
            run.phase = "producing"
        return token

    def stop(self, run_id: str) -> RunStatus | None:
        """Request explicit-target Stop; finish idle paused/failed runs once.

        Stale/absent targets raise RunMismatch. Before acceptance, set Stop and
        return None. Pending work retains ownership until delivery/unwind. An
        idle paused run stops and cleans up synchronously. Terminal calls do not
        repeat notification. Native failures propagate; no hardware rollback.
        """
        with self._lock:
            self.require_run_id(run_id)
            run = self._run
            if run is None:
                raise RuntimeError("Stop requires a current run")
            run.stop.set()
            if run.engine is None:
                return None
            try:
                stopped = run.engine.stop(run_id)
            except Exception as error:
                self._record_native_failure(run, error)
                raise
            if run.phase == "paused":
                self._finish(run, None)
            return stopped

    def apply_tunables(
        self,
        run_id: str,
        batch: Mapping[FormWritePath, JsonValue] | None = None,
        *,
        actor: Actor,
        local_errors: FormErrors | None = None,
    ) -> TunablesSnapshot | FormErrors:
        """Submit GUI draft or external batch through the current unique Host.

        Target checks precede draft writes. None submits GUI input, with optional
        local_errors; a mapping writes external edits then submits complete input.
        Pending acceptance raises RuntimeError without writes. FormErrors retains
        invalid draft; native exceptions propagate unchanged. A native failure
        ends idle ownership immediately, or waits for active producer delivery.
        Success accepts expression input/base; no revision guard or retry.
        """
        with self._lock:
            self.require_run_id(run_id)
            run = self._run
            if run is None:
                raise RuntimeError("Apply requires a current workflow run")
            try:
                if batch is None:
                    return run.binding.apply(actor=actor, local_errors=local_errors)
                return run.binding.apply_external(batch, actor=actor)
            except Exception as error:
                self._record_native_failure(run, error)
                raise
