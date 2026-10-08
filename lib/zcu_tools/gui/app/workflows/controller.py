"""Owner-thread admission and worker lifecycle for native workflow runs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from threading import RLock
from typing import TYPE_CHECKING

from pydantic import BaseModel

from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    DeviceSnapshot,
    Engine,
    EnginePorts,
    InvalidRunState,
    RunIdentity,
    RunPaths,
    RunStatus,
)
from zcu_tools.experiment.workflows.declaration import WorkflowStep
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    ExclusionRequest,
    OperationSpec,
    SettleFn,
)
from zcu_tools.gui.session.services.progress import (
    BoundProgressFactory,
    ProgressService,
)

if TYPE_CHECKING:
    from zcu_tools.gui.session.operation_runner import OperationRunner


logger = logging.getLogger(__name__)


class WorkflowController[C]:
    """Submit Start/Resume segments and cooperative Pause/Stop through the runner.

    All methods except Engine queries are called on the runner's owner thread.
    The controller owns admission and the pending operation token, not native
    lifecycle or a second device registry. ``engine`` is None until the worker
    accepts startup; its status and artifacts remain authoritative thereafter.
    An operation token is not proof of native acceptance. Each segment holds its
    own lease until owner delivery. Paused runs retain progress and plots; only
    whole-run completion, Stop or failure discards the Start-token progress.
    """

    def __init__(
        self,
        runner: OperationRunner,
        ports: EnginePorts[C],
        owner_id: str,
        progress: ProgressService,
        *,
        on_finished: Callable[[RunStatus | None, Exception | None], None],
    ) -> None:
        """Borrow runner/ports and report worker completion on the owner thread.

        owner_id labels the run lease and progress. progress must be the runner's
        owning ProgressService. Start creates one bound factory using its token;
        Resume reuses it, without consulting ambient progress or ports.progress.
        on_finished receives
        the authoritative status (None if startup was never accepted) and the
        original startup/worker error when present. Producer owner delivery runs
        the callback before handle settle and lease release; reentrant Start
        remains blocked until that segment cleanup. Synchronous Resume admission
        failure has no producer: begin has already unwound any segment resources.
        If Stop terminated the native run during admission, on_finished then
        receives that stopped status with error=None, retained progress is
        discarded, and the original admission error is reraised to the caller.
        The callback also reports each paused segment; the caller keeps plan,
        plots and run inputs alive and locked in that case. Terminal run callbacks
        may finish plots; release belongs to the caller. No rollback is promised.
        """
        self._runner = runner
        self._ports = ports
        self._owner_id = owner_id
        self._progress = progress
        self._on_finished = on_finished
        self._lock = RLock()
        self._stop_signal = StopSignal()
        self._started = False
        self._admitting = False
        self._operation_id: int | None = None
        self._progress_token: int | None = None
        self._factory: BoundProgressFactory | None = None
        self._engine: Engine[C] | None = None

    @property
    def engine(self) -> Engine[C] | None:
        """Return the accepted native engine, or None before accepted startup."""
        with self._lock:
            return self._engine

    @property
    def operation_id(self) -> int | None:
        """Return the pending runner token, or None before opening/after delivery.

        Registration can notify before its token is available. During that gap,
        has_pending_segment is True even though this property is None.
        """
        return self._operation_id

    @property
    def has_pending_segment(self) -> bool:
        """Read whether synchronous Start/Resume segment admission is in flight.

        True from immediately before Runner.begin until it returns or raises,
        including lease registration notifications before on_opened supplies a
        token. False afterward; operation_id separately identifies an opened
        segment until owner terminal delivery. Stop/Start/close must treat either
        this property or a non-None operation_id as pending segment work.
        This is observed state, not a configurable policy or an operation ID.
        """
        return self._admitting

    def start[P: BaseModel, T: BaseModel, S, R](
        self,
        step: WorkflowStep[P, T, S, R, C],
        *,
        plan: P,
        tunables: T,
        paths: RunPaths,
        identity: RunIdentity,
    ) -> int:
        """Admit and submit captured, validated startup, returning a runner token.

        One controller accepts exactly one Start; duplicate/pending Start raises
        InvalidRunState. Native declaration validation belongs to Engine and is
        performed on the worker while holding the run lease. Runner admission,
        registration/factory/submit failures propagate synchronously and leave
        no native run directories. Worker startup failures settle failed with
        their original error passed to on_finished; partial files are retained.
        Stop before worker startup creates no native run. Stop arriving during
        startup is delivered before execute calls init/step. Terminal is not
        reported until the worker has returned. No retry or hardware cleanup.
        """
        if self._started:
            status = self.status()
            raise InvalidRunState(
                "start", "running" if status is None else status.lifecycle
            )
        self._started = True

        def opened(token: int) -> None:
            self._operation_id = token
            # Retain the existing Start token, not a new run tracking identity.
            self._progress_token = token
            self._factory = self._progress.make_factory(token, self._owner_id)

        def work(_factory: None) -> RunStatus | None:
            if self._stop_signal.is_set():
                return None
            factory = self._factory
            if factory is None:
                raise RuntimeError("Start requires its bound progress factory")
            engine = Engine(replace(self._ports, progress=factory))
            status = engine.start(
                step, plan=plan, tunables=tunables, paths=paths, identity=identity
            )
            # Startup stays unpublished so Stop never waits for startup I/O.
            with self._lock:
                self._engine = engine
                if self._stop_signal.is_set():
                    engine.stop(status.run_id)
            return engine.execute(status.run_id)

        def cancel() -> None:
            self.stop()

        try:
            self._admitting = True
            try:
                return self._runner.begin(
                    OperationSpec(
                        exclusion=ExclusionRequest(
                            "run", self._owner_id, "workflow run"
                        ),
                        owner_id=self._owner_id,
                        wants_progress=False,
                        cancel_hook=cancel,
                        work=work,
                        run_in_pool=False,
                        on_terminal=self._segment_finished,
                        on_opened=opened,
                    )
                )
            finally:
                self._admitting = False
        except Exception:
            try:
                self._discard_progress()
            finally:
                self._operation_id = None
                self._started = False
            raise

    def _discard_progress(self) -> None:
        """Discard the retained Start-token container once, including on unwind."""
        token = self._progress_token
        self._progress_token = None
        self._factory = None
        if token is not None:
            try:
                self._progress.discard_operation(token)
            except Exception:
                logger.exception("Failed to discard workflow run progress")

    def _segment_finished(self, bg: BgResult, settle: SettleFn) -> None:
        """Report a returned producer before releasing its segment resources."""
        status = self.status()
        try:
            try:
                self._on_finished(status, bg.error)
            finally:
                # The callback may synchronously Stop a newly paused run.
                status = self.status()
                if not bg.ok or status is None or status.lifecycle != "paused":
                    self._discard_progress()
            if not bg.ok:
                outcome = OperationOutcome("failed", str(bg.error))
            elif status is None or status.lifecycle == "stopped":
                outcome = OperationOutcome("cancelled")
            elif status.lifecycle == "failed":
                outcome = OperationOutcome("failed", status.reason)
            else:
                outcome = OperationOutcome("finished")
            settle(outcome)
        finally:
            self._operation_id = None

    def status(self) -> RunStatus | None:
        """Read accepted native status; None distinguishes pending startup."""
        engine = self.engine
        return None if engine is None else engine.status()

    def pause(self) -> RunStatus:
        """Request cooperative Pause on an accepted active segment.

        No accepted run raises RuntimeError. Terminal runs and segments already
        awaiting owner delivery reject invalid controls with InvalidRunState. Repeated pausing requests
        follow Engine.pause. The lease stays held until producer return and owner
        delivery; Pause does not discard progress, plots or submitted inputs.
        """
        with self._lock:
            engine = self._engine
            status = None if engine is None else engine.status()
            if engine is None or status is None:
                raise RuntimeError("Pause requires an accepted workflow run")
            if self._operation_id is None or status.lifecycle not in {
                "running",
                "pausing",
            }:
                raise InvalidRunState("pause", status.lifecycle)
            return engine.pause(status.run_id)

    def resume(self, *, devices: Callable[[], DeviceSnapshot]) -> int:
        """Submit the paused run again, returning the new segment token.

        No accepted run raises RuntimeError. Only paused with previous owner
        delivery complete is admitted; other states raise InvalidRunState.
        After obtaining the RUN lease, invoke devices once
        on the owner thread to capture current cached native device provenance.
        Engine.resume and execute run on the worker with the original progress
        factory, state and inputs. No init or Start is repeated. Capture, lease or
        submission failure propagates. Without synchronous Stop it leaves the run
        paused and retains progress. If Stop already terminated the native run
        and no producer was submitted, synchronous unwind reports that terminal
        once and discards progress, then reraises the original failure. It does
        not invent a RunStatus or restore stopped to paused.
        Native failed status retains Engine's reason; exceptions escaping the
        worker reach on_finished unchanged. Either failure discards progress.
        Stop while this segment is pending prevents its step from restarting.
        """
        with self._lock:
            engine = self._engine
            status = None if engine is None else engine.status()
            if engine is None or status is None:
                raise RuntimeError("Resume requires an accepted paused workflow run")
            if (
                status.lifecycle != "paused"
                or self._admitting
                or self._operation_id is not None
            ):
                raise InvalidRunState("resume", status.lifecycle)

        captured: DeviceSnapshot | None = None

        def opened(token: int) -> None:
            nonlocal captured
            self._operation_id = token
            captured = devices()

        def work(_factory: None) -> RunStatus:
            with self._lock:
                if self._stop_signal.is_set():
                    stopped = engine.status()
                    if stopped is None:
                        raise RuntimeError("accepted engine has no run status")
                    return stopped
                if captured is None:
                    raise RuntimeError("Resume requires captured devices")
                engine.resume(status.run_id, devices=captured)
            return engine.execute(status.run_id)

        def cancel() -> None:
            self.stop()

        try:
            self._admitting = True
            try:
                return self._runner.begin(
                    OperationSpec(
                        exclusion=ExclusionRequest(
                            "run", self._owner_id, "workflow run"
                        ),
                        owner_id=self._owner_id,
                        wants_progress=False,
                        cancel_hook=cancel,
                        work=work,
                        run_in_pool=False,
                        on_terminal=self._segment_finished,
                        on_opened=opened,
                    )
                )
            finally:
                self._admitting = False
        except Exception:
            self._operation_id = None
            stopped = engine.status()
            if stopped is not None and stopped.lifecycle == "stopped":
                # begin unwound the lease; no producer can deliver this terminal.
                try:
                    self._on_finished(stopped, None)
                except Exception:
                    logger.exception("Workflow admission terminal callback failed")
                finally:
                    self._discard_progress()
            raise

    def stop(self) -> RunStatus | None:
        """Request cooperative Stop without waiting for worker terminal delivery.

        Before acceptance this cancels pending startup and returns None. After
        acceptance it delegates to Engine.stop and returns its native status.
        An active segment, including registration before its token is available,
        keeps its lease until worker terminal delivery or synchronous admission
        unwind when no producer exists. An already-delivered paused run with no
        admission in flight stops synchronously, reports on_finished and
        discards its retained progress. Native failures propagate; no hardware
        rollback or cleanup occurs.
        """
        with self._lock:
            self._stop_signal.set()
            if self._engine is None:
                return None
            status = self._engine.status()
            if status is None:
                raise RuntimeError("accepted engine has no run status")
            stopped = self._engine.stop(status.run_id)
        if (
            status.lifecycle == "paused"
            and not self._admitting
            and self._operation_id is None
        ):
            try:
                self._on_finished(stopped, None)
            finally:
                self._discard_progress()
        return stopped
