"""Owner-thread admission and worker lifecycle for native workflow runs."""

from __future__ import annotations

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
        original startup/worker error when present. It runs before handle settle
        and lease release; reentrant Start remains blocked until cleanup.
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
        self._operation_id: int | None = None
        self._engine: Engine[C] | None = None

    @property
    def engine(self) -> Engine[C] | None:
        """Return the accepted native engine, or None before accepted startup."""
        with self._lock:
            return self._engine

    @property
    def operation_id(self) -> int | None:
        """Return the pending runner token, or None after terminal delivery."""
        return self._operation_id

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

        def work(factory: BoundProgressFactory) -> RunStatus | None:
            if self._stop_signal.is_set():
                return None
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

        def terminal(bg: BgResult, settle: SettleFn) -> None:
            try:
                status = self.status()
                self._on_finished(status, bg.error)
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

        try:
            return self._runner.begin(
                OperationSpec(
                    exclusion=ExclusionRequest("run", self._owner_id, "workflow run"),
                    owner_id=self._owner_id,
                    wants_progress=True,
                    cancel_hook=cancel,
                    work=work,
                    run_in_pool=False,
                    on_terminal=terminal,
                    on_opened=opened,
                )
            )
        except Exception:
            self._operation_id = None
            self._started = False
            raise

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
        raise NotImplementedError("Pause segment control is not implemented")

    def resume(self, *, devices: Callable[[], DeviceSnapshot]) -> int:
        """Submit the paused run again, returning the new segment token.

        No accepted run raises RuntimeError. Only paused with previous owner
        delivery complete is admitted; other states raise InvalidRunState. After obtaining the RUN lease, invoke devices once
        on the owner thread to capture current cached native device provenance.
        Engine.resume and execute run on the worker with the original progress
        factory, state and inputs. No init or Start is repeated. Capture, lease or
        submission failure propagates, leaves the run paused and retains progress;
        native worker failure reports the original error and discards progress.
        Stop while this segment is pending prevents its step from restarting.
        """
        raise NotImplementedError("Resume segment control is not implemented")

    def stop(self) -> RunStatus | None:
        """Request cooperative Stop without waiting for worker terminal delivery.

        Before acceptance this cancels pending startup and returns None. After
        acceptance it delegates to Engine.stop and returns its native status.
        An active segment keeps its lease until worker terminal delivery. An
        already-delivered paused run stops synchronously, reports on_finished and
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
            return self._engine.stop(status.run_id)
