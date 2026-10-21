"""Typed run owner: step isolation, effects, source-aware failure, and control."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from threading import RLock
from types import GeneratorType
from typing import Literal

from pydantic import BaseModel, JsonValue, TypeAdapter

from ...core.script_protocol import Answer, Question, Signal, Wait
from ...device import BaseDeviceInfo
from ...progress_bar import use_pbar_factory
from ..script_driver import ScriptDriver
from ..stop_signal import ScheduleOutcomeError, StopSignal
from .artifacts import IterationArtifacts, RunArtifacts
from .declaration import WorkflowDefinition
from .effects import RunEffect, dispatch_effect
from .encoding import encode_record
from .env import Displays, InitEnv, WorkflowEnv
from .journal import (
    DeviceSetup,
    ExperimentFailed,
    IterationStarted,
    Paused,
    PauseRequested,
    Resumed,
    RunEnded,
    RunMetadata,
    StepCommitted,
    StepDiscarded,
    StepFailed,
    StepFinished,
    StopRequested,
    TunablesChanged,
    describe_error,
)
from .models import (
    Aborted,
    Actor,
    CommittedRecord,
    Completed,
    Done,
    Effect,
    Failed,
    InvalidRunState,
    Lifecycle,
    MissingCapability,
    Next,
    RunIdentity,
    RunPaths,
    RunStatus,
    TunablesSnapshot,
)
from .ports import EnginePorts
from .run import Run
from .tunables import TunableValues

_logger = logging.getLogger(__name__)
_TERMINAL = ("done", "aborted", "stopped", "failed")


class _DiscardStep(Exception):
    """Internal control transfer, never delivered into workflow code."""


class WorkflowRun[P: BaseModel, T: BaseModel, S, R, C]:
    """Retain a declaration's typed plan/tunables/state/record pairing.

    This package-internal owner implements Engine's erased control view and
    EffectExecutor. definition supplies the original step/init. ports retains
    native capabilities; lock is Engine's shared control/journal RLock.
    start must succeed before any other method. Control methods require the
    caller to hold lock; execute runs outside it after reserve_execution.
    No worker, connection, or implicit hardware cleanup is created here.
    """

    def __init__(
        self,
        definition: WorkflowDefinition[P, T, S, R, C],
        ports: EnginePorts[C],
        lock: RLock,
    ) -> None:
        self._definition = definition
        self._ports = ports
        self._lock = lock
        self._displays = Displays(ports.progress)
        self._displays_closed = False
        self._lifecycle: Lifecycle = "running"
        self._reason: str | None = None
        self._executing = False
        self._initialized = False
        self._call_seq: int | None = None
        self._commit_seq = 0
        self._call_revision = 0
        self._run_no = 0
        self._iteration: IterationArtifacts | None = None
        self._signal: StopSignal | None = None
        self._pending_error: Exception | None = None
        self._records: list[CommittedRecord] = []
        self._state: S
        self._driver: ScriptDriver[Next[R, S] | Done | Aborted] | None = None

    def start(
        self,
        plan: P,
        tunables: T,
        paths: RunPaths,
        identity: RunIdentity,
    ) -> None:
        """Validate detached inputs before creating startup files.

        Invalid model/path/provenance inputs propagate.
        init is not called. Partial I/O artifacts are retained on failure.
        """
        self._plan = deepcopy(
            self._definition.plan.model_validate(
                plan.model_dump(mode="python"), by_name=True, by_alias=False
            )
        )
        self._tunables = TunableValues(
            self._definition.tunables,
            tunables,
            self._lock,
            self._append_tunables,
        )
        self._identity = deepcopy(identity)
        self._started_at = self._ports.clock.now()
        metadata = RunMetadata(
            identity=self._identity,
            workflow=self._definition.name,
            plan=TypeAdapter(JsonValue).validate_python(
                self._plan.model_dump(mode="json", by_alias=False, warnings="error")
            ),
            tunables=self._tunables.snapshot().values,
            roots=paths,
            started_at=self._started_at,
        )
        self._artifacts = RunArtifacts(metadata, self._ports.clock, self._lock)

    @property
    def run_id(self) -> str:
        """Return the stable identity without querying display backends."""
        return self._identity.run_id

    def allow_replacement(self) -> None:
        """Reject replacement until terminal and execute has actually settled."""
        if self._lifecycle not in _TERMINAL or self._executing:
            raise InvalidRunState("start", self._lifecycle)

    def reserve_execution(self) -> None:
        """Reserve the only segment before Engine releases its control lock."""
        if self._executing or self._lifecycle not in ("running", "pausing", "stopping"):
            raise InvalidRunState("execute", self._lifecycle)
        self._executing = True

    def reserve_answer(self, answer: Answer) -> None:
        """Validate before journaling/reserving a settled Ask continuation.

        Caller holds the control lock. Wrong model leaves the run unchanged;
        journal failures fail the run just as resume failures do.
        """
        driver = self._driver
        if (
            self._lifecycle != "paused"
            or self._executing
            or driver is None
            or driver.pending_question is None
        ):
            raise InvalidRunState("answer", self._lifecycle)
        driver.validate_answer(answer)
        try:
            self._artifacts.append(Resumed(self._commit_seq, self._tunables.revision))
            self._artifacts.set_lifecycle("running")
            self._lifecycle = "running"
            self._executing = True
        except Exception as error:
            self._request_failure(error)
            raise

    def execute(self, answer: Answer | None = None) -> RunStatus:
        """Run a reserved segment, optionally delivering its validated Answer.

        Capture detached status before releasing the reservation. Answer code
        runs under the same progress context and failure boundary as execute.
        """
        try:
            with use_pbar_factory(self._ports.progress):
                self._produce(answer)
        except Exception as error:
            with self._lock:
                cause = (
                    self._pending_error if self._pending_error is not None else error
                )
                _logger.error("Workflow segment failed", exc_info=cause)
                self._fail(cause)
        finally:
            with self._lock:
                try:
                    status = self.status()
                finally:
                    self._signal = None
                    self._executing = False
        return status

    def _produce(self, answer: Answer | None = None) -> None:
        try:
            if answer is not None:
                self._check_cancel()
                driver = self._driver
                if driver is None:
                    raise RuntimeError("Answer segment has no suspended invocation")
                driver.answer(answer)
                self._call_step()
            if not self._settle_control():
                if not self._initialized:
                    self._state = deepcopy(
                        self._definition.init(
                            InitEnv(self._started_at, self._ports.context),
                            deepcopy(self._plan),
                        )
                    )
                    self._initialized = True
                while not self._settle_control():
                    self._call_step()
                    if self._lifecycle in _TERMINAL:
                        break
        except _DiscardStep:
            self._close_step()
            self._discard_iteration()
            self._settle_control()
        except KeyboardInterrupt:
            self._host_stop()
            self._close_step()
            self._discard_iteration()
            self._settle_control()

    def _start_step(self) -> None:
        with self._lock:
            captured = self._tunables.capture()
            self._call_seq = (self._call_seq or 0) + 1
            self._call_revision = captured.revision
            self._run_no = 0
            self._iteration = None
            self._iteration_inputs = None
            iteration = self._artifacts.new_iteration(self._call_seq)
            self._iteration = iteration
            started = IterationStarted(
                self._call_seq,
                self._commit_seq,
                captured.revision,
                iteration.iteration_dir,
            )
            self._artifacts.append(started)
            self._iteration_inputs = self._artifacts.capture_record(
                started, captured.model
            )
        env = WorkflowEnv(self._started_at, self._ports, iteration.root, self._displays)
        generator = self._definition.step(
            env, deepcopy(self._plan), captured.model, deepcopy(self._state)
        )
        if not isinstance(generator, GeneratorType):
            raise TypeError("Workflow step must return a generator")
        self._driver = ScriptDriver(
            generator, deliver_cancelled=self._deliver_cancelled
        )

    def _deliver_cancelled(self, operation: object) -> None:
        raise NotImplementedError("Workflow cancelled delivery is not connected yet")

    def _call_step(self) -> None:
        if self._driver is None:
            self._start_step()
        driver = self._driver
        if driver is None:
            raise RuntimeError("Workflow invocation has no driver")
        try:
            outcome = self._drive(driver)
        except _DiscardStep:
            # close may run user finally blocks; their failures are not cancellation.
            self._close_step()
            self._discard_iteration()
            return
        except BaseException:
            # Cleanup cannot replace the acquisition/analysis failure source.
            try:
                self._close_step()
            except (Exception, KeyboardInterrupt):
                _logger.exception("Workflow shutdown also failed")
            raise
        if outcome is None:
            return
        self._driver = None
        self._finish_step(outcome)

    def _finish_step(self, outcome: Next[R, S] | Done | Aborted) -> None:
        iteration, _ = self._current_iteration()
        if isinstance(outcome, Next):
            self._commit(outcome)
        else:
            with self._lock:
                if self._pending_error is not None:
                    raise self._pending_error
                lifecycle = "done" if isinstance(outcome, Done) else "aborted"
                reason = None if isinstance(outcome, Done) else outcome.reason
                finished = StepFinished(
                    self._current_iteration()[1],
                    self._call_revision,
                    iteration.iteration_dir,
                    lifecycle,
                    reason,
                )
                self._artifacts.append(finished)
                self._artifacts.save_record(iteration, self._iteration_inputs, finished)
                self._end(lifecycle, reason)

    def _close_step(self) -> None:
        """Close a retained producer; cleanup errors propagate unchanged."""
        driver = self._driver
        self._driver = None
        if driver is not None and driver.status not in (
            "completed",
            "cancelled",
            "failed",
        ):
            driver.signal(Signal("cancel"))
            driver.advance()

    def _drive(
        self, driver: ScriptDriver[Next[R, S] | Done | Aborted]
    ) -> Next[R, S] | Done | Aborted | None:
        # The shared driver alone resumes generators. This host loop executes
        # synchronous workflow operations and publishes unanswered questions.
        while True:
            driver.advance()
            if driver.status == "completed":
                outcome = driver.result
                if not isinstance(outcome, (Next, Done, Aborted)):
                    raise TypeError("Workflow step must return Next, Done, or Aborted")
                return outcome
            if driver.status == "cancelled":
                self._host_stop()
                raise _DiscardStep()
            self._check_cancel()
            if driver.status == "paused":
                with self._lock:
                    self._check_cancel()
                    self._artifacts.append(Paused(self._call_seq, self._commit_seq))
                    self._artifacts.set_lifecycle("paused")
                    self._lifecycle = "paused"
                return None
            request = driver.pending_request
            if not isinstance(request, Wait) or not isinstance(request.op, Effect):
                raise TypeError(
                    "Use yield from env.<effect>(...); unknown effect request"
                )
            # A genuine Schedule Failed reaches workflow code even when a control
            # request arrived with it; dispatch retains source-aware cancellation.
            dispatch_effect(request.op, self)

    def _commit(self, outcome: Next[R, S]) -> None:
        record = deepcopy(outcome.record)
        state = deepcopy(outcome.state)
        encoded = encode_record(record)
        with self._lock:
            if self._pending_error is not None:
                raise self._pending_error
            iteration, call_seq = self._current_iteration()
            committed = StepCommitted(
                self._commit_seq + 1,
                call_seq,
                self._call_revision,
                encoded,
                iteration.iteration_dir,
            )
            # Requests cannot interrupt this journal-before-state publication.
            self._artifacts.append(committed)
            self._state = state
            self._commit_seq += 1
            self._records.append(committed)
            self._artifacts.save_record(iteration, self._iteration_inputs, committed)
            self._iteration = None

    def _current_iteration(self) -> tuple[IterationArtifacts, int]:
        if self._iteration is None or self._call_seq is None:
            raise RuntimeError("Effect has no active workflow invocation")
        return self._iteration, self._call_seq

    def _discard_iteration(self) -> None:
        with self._lock:
            if self._pending_error is not None:
                raise self._pending_error
            if self._iteration is not None:
                iteration, call_seq = self._current_iteration()
                discarded = StepDiscarded(
                    call_seq,
                    self._call_revision,
                    iteration.iteration_dir,
                    "stop" if self._lifecycle == "stopping" else "pause",
                )
                self._artifacts.append(discarded)
                self._artifacts.save_record(
                    iteration, self._iteration_inputs, discarded
                )
                self._iteration = None

    def _check_cancel(self) -> None:
        with self._lock:
            if self._pending_error is not None:
                raise self._pending_error
            if self._lifecycle in ("pausing", "stopping"):
                raise _DiscardStep()

    def _settle_control(self) -> bool:
        with self._lock:
            if self._pending_error is not None:
                raise self._pending_error
            if self._lifecycle == "pausing":
                self._artifacts.append(Paused(self._call_seq, self._commit_seq))
                self._artifacts.set_lifecycle("paused")
                self._lifecycle = "paused"
                return True
            if self._lifecycle == "stopping":
                self._end("stopped")
                return True
            return self._lifecycle == "paused" or self._lifecycle in _TERMINAL

    def _new_signal(self) -> StopSignal:
        with self._lock:
            self._check_cancel()
            signal = StopSignal()
            self._signal = signal
            return signal

    def _clear_signal(self, signal: StopSignal) -> None:
        with self._lock:
            if self._signal is signal:
                self._signal = None

    def run[Cfg, Result](
        self, request: RunEffect[Cfg, Result]
    ) -> Completed[Cfg, Result] | Failed:
        """Execute one typed experiment with a fresh signal, without saving.

        Recover only this Run's Schedule failure. Live, external wrapper,
        device and cfg-copy errors propagate to segment isolation.
        Controls without a true failure discard the step, never return Failed.
        """
        signal = self._new_signal()
        self._run_no += 1
        try:
            devices = self._ports.devices
            if devices is None:
                raise MissingCapability("devices")
            device_snapshot = devices.capture_snapshot()
            run = Run(
                request.cfg,
                signal,
                soc=self._ports.soc,
                soccfg=self._ports.soccfg,
                devices=devices,
                device_snapshot=device_snapshot,
            )
            if request.live is not None:
                run.bind_live(request.live)
            try:
                result = request.experiment(run)
            except ScheduleOutcomeError as error:
                if run.live_error is not None:
                    raise run.live_error from error
                if error is not signal.error:
                    raise
                failure = self._inspect_run(run, request.name)
                if failure is None:
                    raise
                return failure
            failure = self._inspect_run(run, request.name)
            if failure is not None:
                return failure
            self._check_cancel()
            return Completed(deepcopy(run.cfg), result)
        finally:
            self._clear_signal(signal)

    def _inspect_run[Cfg](self, run: Run[Cfg], experiment: str) -> Failed | None:
        with self._lock:
            if self._pending_error is not None:
                raise self._pending_error
        if run.live_error is not None:
            raise run.live_error
        outcome = run.outcome
        error = run.cancel_signal.error
        if outcome.status == "interrupted" or (
            error is not None and error.status == "interrupted"
        ):
            self._host_stop()
            raise _DiscardStep()
        if outcome.status == "failed":
            if error is None:
                raise RuntimeError("Failed Schedule has no original signal error")
            reason = error.reason
            cause = error.exception if error.exception is not None else error
        elif outcome.status == "stopped" and outcome.exception is not None:
            # Native stop/retry policy is unchanged; an observed cause still wins.
            cause = outcome.exception
            reason = str(cause) or type(cause).__name__
        else:
            if outcome.status == "stopped" or run.cancel_signal.is_set():
                with self._lock:
                    if self._lifecycle not in ("pausing", "stopping"):
                        self._host_stop()
                self._check_cancel()
            return None
        _, call_seq = self._current_iteration()
        self._artifacts.append(
            ExperimentFailed(
                call_seq,
                self._run_no,
                experiment,
                reason,
                describe_error(cause),
            )
        )
        return Failed(reason)

    def setup_device(self, name: str, cfg: BaseDeviceInfo) -> None:
        """Forward complete settings with the active signal and journal the cfg."""
        signal = self._new_signal()
        try:
            if self._ports.devices is None:
                raise MissingCapability("devices")
            self._ports.devices.setup_device(name, cfg, signal)
            self._artifacts.append(DeviceSetup(name, cfg))
            self._check_cancel()
        finally:
            self._clear_signal(signal)

    def wait_until(self, target: datetime) -> None:
        """Use the injected cancellable clock; already-past targets need no wait."""
        signal = self._new_signal()
        try:
            if target > self._ports.clock.now():
                self._ports.clock.wait_until(target, signal)
            self._check_cancel()
        finally:
            self._clear_signal(signal)

    def pause(self) -> RunStatus:
        """Publish one Pause request, or return an existing pausing/paused state."""
        if self._lifecycle in ("pausing", "paused"):
            return self.status()
        if self._lifecycle != "running":
            raise InvalidRunState("pause", self._lifecycle)
        try:
            self._artifacts.append(
                PauseRequested(self._ports.clock.now(), self._call_seq)
            )
            self._artifacts.set_lifecycle("pausing")
            self._lifecycle = "pausing"
            if self._signal is not None:
                self._signal.set()
        except Exception as error:
            self._request_failure(error)
            raise
        return self.status()

    def resume(self) -> RunStatus:
        """Resume only after paused segment return under the retained lease."""
        if self._lifecycle != "paused" or self._executing or self._driver is not None:
            raise InvalidRunState("resume", self._lifecycle)
        try:
            self._artifacts.append(Resumed(self._commit_seq, self._tunables.revision))
            self._artifacts.set_lifecycle("running")
            self._lifecycle = "running"
        except Exception as error:
            self._request_failure(error)
            raise
        return self.status()

    def stop(self) -> RunStatus:
        """Publish one Stop request; paused runs can end without a producer."""
        if self._lifecycle in _TERMINAL or self._lifecycle == "stopping":
            return self.status()
        try:
            self._artifacts.append(
                StopRequested(self._ports.clock.now(), self._call_seq)
            )
            if self._lifecycle == "paused":
                self._lifecycle = "stopping"
                self._close_step()
                self._discard_iteration()
                self._end("stopped")
            else:
                self._artifacts.set_lifecycle("stopping")
                self._lifecycle = "stopping"
                if self._signal is not None:
                    self._signal.set()
        except Exception as error:
            self._request_failure(error)
            raise
        return self.status()

    def _host_stop(self) -> None:
        with self._lock:
            self.stop()

    def update_tunables(
        self, tunables: BaseModel, expected_revision: int, actor: Actor
    ) -> TunablesSnapshot:
        """Enforce lifecycle; TunableValues owns validation and atomic revision."""
        if self._lifecycle not in ("running", "pausing", "paused"):
            raise InvalidRunState("update_tunables", self._lifecycle)
        return self._tunables.update(
            tunables, expected_revision=expected_revision, actor=actor
        )

    def _append_tunables(self, event: TunablesChanged) -> None:
        try:
            self._artifacts.append(event)
        except Exception as error:
            self._request_failure(error)
            raise

    def _request_failure(self, error: Exception) -> None:
        _logger.error("Workflow control failed", exc_info=error)
        if self._pending_error is None:
            self._pending_error = error
        if self._executing:
            self._lifecycle = "stopping"
            self._reason = f"{type(error).__name__}: {error}"
            if self._signal is not None:
                self._signal.set()
        else:
            self._fail(self._pending_error)

    def _close_displays(self) -> None:
        if not self._displays_closed:
            self._displays_closed = True
            self._displays.close()

    def _end(
        self,
        lifecycle: Literal["done", "aborted", "stopped"],
        reason: str | None = None,
    ) -> None:
        self._close_displays()
        self._artifacts.append(
            RunEnded(lifecycle, reason, self._call_seq, self._commit_seq)
        )
        self._artifacts.set_lifecycle(lifecycle, reason)
        self._lifecycle = lifecycle
        self._reason = reason

    def _fail(self, error: Exception) -> None:
        with self._lock:
            if self._lifecycle == "failed":
                return
            # Diagnostic/cleanup failures never replace the primary run cause.
            self._lifecycle = "failed"
            self._reason = f"{type(error).__name__}: {error}"
            try:
                self._close_step()
            except (Exception, KeyboardInterrupt):
                _logger.exception("Workflow producer cleanup failed")
            try:
                self._close_displays()
            except Exception:
                _logger.exception("Workflow progress cleanup failed")
            try:
                if self._iteration is not None:
                    iteration, call_seq = self._current_iteration()
                    failed = StepFailed(
                        call_seq,
                        self._call_revision,
                        iteration.iteration_dir,
                        describe_error(error),
                    )
                    self._artifacts.append(failed)
                    self._artifacts.save_record(
                        iteration, self._iteration_inputs, failed
                    )
                self._artifacts.append(
                    RunEnded("failed", self._reason, self._call_seq, self._commit_seq)
                )
            except Exception:
                _logger.exception("Cannot append workflow failure diagnostics")
            try:
                self._artifacts.set_lifecycle("failed", self._reason)
            except Exception:
                _logger.exception("Cannot publish failed workflow manifest")

    def pending_question(self) -> Question[Answer] | None:
        """Return detached unanswered evidence after the paused worker settles."""
        if self._lifecycle != "paused" or self._executing or self._driver is None:
            return None
        return deepcopy(self._driver.pending_question)

    def status(self) -> RunStatus:
        """Return detached lifecycle/sequence/revision/display observations."""
        return RunStatus(
            self._identity.run_id,
            self._definition.name,
            self._lifecycle,
            self._call_seq,
            self._commit_seq,
            self._tunables.revision,
            self._reason,
            self._displays.snapshot(),
        )

    def tunables(self) -> TunablesSnapshot:
        """Return the current complete detached JSON projection and revision."""
        return self._tunables.snapshot()

    def records(self, since: int) -> tuple[CommittedRecord, ...]:
        """Return detached committed points after the nonnegative exclusive bound."""
        if type(since) is not int or since < 0:
            raise ValueError("since must be a nonnegative integer")
        return deepcopy(
            tuple(record for record in self._records if record.commit_seq > since)
        )
