"""Typed run owner: step isolation, effects, source-aware failure, and control."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from threading import RLock
from types import GeneratorType
from typing import Literal

from pydantic import BaseModel, JsonValue, TypeAdapter

from ...progress_bar import use_pbar_factory
from ..stop_signal import ScheduleOutcomeError, StopSignal
from .artifacts import IterationArtifacts, RunArtifacts
from .declaration import WorkflowDefinition
from .effects import RunEffect, dispatch_effect
from .encoding import encode_record
from .env import Displays, InitEnv, WorkflowEnv
from .journal import (
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
    Failed,
    InvalidRunState,
    Lifecycle,
    MissingCapability,
    Next,
    RunIdentity,
    RunPaths,
    RunStatus,
    Step,
    TunableChange,
    TunablesSnapshot,
)
from .ports import DeviceSnapshot, EnginePorts
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
        self._displays = Displays(ports.plots, ports.progress)
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

    def start(
        self, plan: P, tunables: T, paths: RunPaths, identity: RunIdentity
    ) -> None:
        """Validate detached inputs/capabilities before creating startup files.

        Missing requirements and invalid model/path/provenance inputs propagate.
        init is not called. Partial I/O artifacts are retained on failure.
        """
        for capability in self._definition.requires:
            if getattr(self._ports, capability) is None:
                raise MissingCapability(capability)
        self._plan = deepcopy(
            self._definition.plan.model_validate(
                plan.model_dump(mode="python"), by_name=True, by_alias=False
            )
        )
        self._tunables = TunableValues(
            self._definition.tunables, tunables, self._lock, self._append_tunables
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
            requires=self._definition.requires,
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

    def execute(self) -> RunStatus:
        """Run the reserved segment and return only after producer shutdown."""
        try:
            with use_pbar_factory(self._ports.progress):
                self._produce()
        except Exception as error:
            with self._lock:
                cause = (
                    self._pending_error if self._pending_error is not None else error
                )
                _logger.error("Workflow segment failed", exc_info=cause)
                self._fail(cause)
        finally:
            with self._lock:
                self._signal = None
                self._executing = False
        with self._lock:
            return self.status()

    def _produce(self) -> None:
        try:
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
        except KeyboardInterrupt:
            self._host_stop()
            self._discard_iteration()
            self._settle_control()

    def _call_step(self) -> None:
        with self._lock:
            captured = self._tunables.capture()
            self._call_seq = (self._call_seq or 0) + 1
            self._call_revision = captured.revision
            self._run_no = 0
            self._iteration = None
            iteration = self._artifacts.new_iteration(self._call_seq)
            self._iteration = iteration
            self._artifacts.append(
                IterationStarted(
                    self._call_seq,
                    self._commit_seq,
                    captured.revision,
                    iteration.iteration_dir,
                )
            )
        env = WorkflowEnv(
            self._started_at, self._ports.context, iteration.files_dir, self._displays
        )
        generator = self._definition.step(
            env, deepcopy(self._plan), captured.model, deepcopy(self._state)
        )
        if not isinstance(generator, GeneratorType):
            raise TypeError("Workflow step must return a generator")
        try:
            outcome = self._drive(generator)
        except _DiscardStep:
            # close may run user finally blocks; their failures are not cancellation.
            generator.close()
            self._discard_iteration()
            return
        except BaseException:
            # Cleanup cannot replace the acquisition/analysis failure source.
            try:
                generator.close()
            except (Exception, KeyboardInterrupt):
                _logger.exception("Workflow generator cleanup also failed")
            raise
        self._displays.refresh()
        if isinstance(outcome, Next):
            self._commit(outcome)
        else:
            with self._lock:
                if self._pending_error is not None:
                    raise self._pending_error
                lifecycle = "done" if isinstance(outcome, Done) else "aborted"
                reason = None if isinstance(outcome, Done) else outcome.reason
                self._artifacts.append(
                    StepFinished(
                        self._current_iteration()[1],
                        self._call_revision,
                        iteration.iteration_dir,
                        iteration.run_files,
                        lifecycle,
                        reason,
                    )
                )
                self._end(lifecycle, reason)

    def _drive(self, generator: Step[S, R]) -> Next[R, S] | Done | Aborted:
        while True:
            try:
                effect = next(generator)
            except StopIteration as finished:
                outcome: object = finished.value
                if not isinstance(outcome, (Next, Done, Aborted)):
                    raise TypeError(
                        "Workflow step must return Next, Done, or Aborted"
                    ) from None
                return outcome
            self._check_cancel()
            self._displays.refresh()
            # Effect exits handle pure cancellation. A genuine Schedule Failed
            # reaches workflow code even when a control request arrived with it.
            dispatch_effect(effect, self)

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
                iteration.run_files,
            )
            # Requests cannot interrupt this journal-before-state publication.
            self._artifacts.append(committed)
            self._state = state
            self._commit_seq += 1
            self._records.append(committed)
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
                self._artifacts.append(
                    StepDiscarded(
                        call_seq,
                        self._call_revision,
                        iteration.iteration_dir,
                        iteration.run_files,
                        "stop" if self._lifecycle == "stopping" else "pause",
                    )
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
            return self._lifecycle in _TERMINAL

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
        """Execute one typed experiment with a fresh signal and save before return.

        Recover only this Run's Schedule failure. Live, external wrapper,
        device, cfg-copy, and saver errors propagate to segment isolation.
        Controls without a true failure discard the step, never return Failed.
        """
        signal = self._new_signal()
        iteration, _ = self._current_iteration()
        self._run_no += 1
        try:
            run = Run(
                request.cfg,
                signal,
                soc=self._ports.soc,
                soccfg=self._ports.soccfg,
                devices=self._ports.devices,
            )
            if request.live is not None:
                run.bind_live(request.live, self._ports.plots)
            settings = run.device_setup
            if request.setup_devices and settings is not None:
                run.devices.setup(settings, signal)
                self._check_cancel()
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
            completed = Completed(deepcopy(run.cfg), result)
            iteration.save(self._run_no, request.name, completed, request.save)
            self._check_cancel()
            return completed
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

    def set_device(self, name: str, value: float) -> None:
        """Apply an absolute setpoint through the device port and current signal."""
        signal = self._new_signal()
        try:
            if self._ports.devices is None:
                raise MissingCapability("devices")
            self._ports.devices.set_value(name, value, signal)
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

    def resume(self, devices: DeviceSnapshot) -> RunStatus:
        """Resume only after paused segment return, with detached provenance."""
        if self._lifecycle != "paused" or self._executing:
            raise InvalidRunState("resume", self._lifecycle)
        try:
            self._artifacts.append(
                Resumed(self._commit_seq, self._tunables.revision, deepcopy(devices))
            )
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
        self, changes: tuple[TunableChange, ...], expected_revision: int, actor: Actor
    ) -> TunablesSnapshot:
        """Enforce lifecycle; TunableValues owns validation and atomic revision."""
        if self._lifecycle not in ("running", "pausing", "paused"):
            raise InvalidRunState("update_tunables", self._lifecycle)
        return self._tunables.update(
            changes, expected_revision=expected_revision, actor=actor
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
                self._close_displays()
            except Exception:
                _logger.exception("Workflow progress cleanup failed")
            try:
                if self._iteration is not None:
                    iteration, call_seq = self._current_iteration()
                    self._artifacts.append(
                        StepFailed(
                            call_seq,
                            self._call_revision,
                            iteration.iteration_dir,
                            iteration.run_files,
                            describe_error(error),
                        )
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
