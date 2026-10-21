"""Host-facing synchronous workflow engine, without a worker or hardware owner."""

from __future__ import annotations

from threading import RLock
from typing import Protocol

from pydantic import BaseModel

from ...core.script_protocol import Answer, Question
from .declaration import WorkflowStep, workflow_definition
from .execution import WorkflowRun
from .models import (
    Actor,
    CommittedRecord,
    RunIdentity,
    RunMismatch,
    RunPaths,
    RunStatus,
    TunablesSnapshot,
)
from .ports import EnginePorts


class _RunView(Protocol):
    """Erase only the workflow's P/T/S/R types at Engine's storage boundary.

    WorkflowRun retains those types together while this view exposes only
    detached control/read results. No caller casts or heterogeneous state escape.
    The Engine lock surrounds every method except the reserved execute segment.
    """

    @property
    def run_id(self) -> str: ...

    def reserve_execution(self) -> None: ...
    def reserve_answer(self, answer: Answer) -> None: ...
    def execute(self, answer: Answer | None = None) -> RunStatus: ...
    def pending_question(self) -> Question[Answer] | None: ...
    def pause(self) -> RunStatus: ...
    def resume(self) -> RunStatus: ...
    def stop(self) -> RunStatus: ...
    def update_tunables(
        self, tunables: BaseModel, expected_revision: int, actor: Actor
    ) -> TunablesSnapshot: ...
    def status(self) -> RunStatus: ...
    def tunables(self) -> TunablesSnapshot: ...
    def records(self, since: int) -> tuple[CommittedRecord, ...]: ...
    def allow_replacement(self) -> None: ...


class Engine[C]:
    """Run one named workflow with host-owned ports and worker scheduling.

    start validates and persists startup, but never executes workflow code.
    execute synchronously runs a segment until paused or terminal. The host
    schedules it on a worker; Engine creates no threads and owns no connections.
    answer is also a synchronous worker segment, never a UI-thread control.
    Control/read methods are thread-safe and return detached values. A new run
    replaces only a terminal run whose execute segment has actually returned.
    Startup validation/I/O errors propagate without replacing the previous run.
    Execution failures return lifecycle=failed with diagnostics in the journal
    and logs; KeyboardInterrupt behaves as Stop. Files are never rolled back.
    """

    def __init__(self, ports: EnginePorts[C]) -> None:
        self._ports = ports
        self._lock = RLock()
        self._run: _RunView | None = None

    def start[P: BaseModel, T: BaseModel, S, R](
        self,
        step: WorkflowStep[P, T, S, R, C],
        *,
        plan: P,
        tunables: T,
        paths: RunPaths,
        identity: RunIdentity,
    ) -> RunStatus:
        """Validate a declared step and persist revision-zero startup.

        plan/tunables are revalidated against the declaration and copied. paths
        are distinct, non-overlapping, nonexistent directories; identity is
        host-supplied provenance. Port availability is not checked here;
        missing ports raise MissingCapability when first used during execute.
        An active run raises InvalidRunState. init waits for first execute.
        """
        with self._lock:
            if self._run is not None:
                self._run.allow_replacement()
            run = WorkflowRun(workflow_definition(step), self._ports, self._lock)
            run.start(plan, tunables, paths, identity)
            self._run = run
            return run.status()

    def execute(self, run_id: str) -> RunStatus:
        """Execute one segment on this thread, until paused or terminal.

        Only one execute may be in flight. paused/terminal calls raise
        InvalidRunState. A pausing/stopping request made before execute is
        settled without calling init or step. The returned status proves the
        producer has settled, unlike the status returned by pause/stop.
        """
        with self._lock:
            run = self._current(run_id)
            run.reserve_execution()
        return run.execute()

    def pause(self, run_id: str) -> RunStatus:
        """Request cooperative Pause without waiting for an effect to return.

        Running becomes pausing. Repeated pausing/paused requests are no-ops.
        Other states raise InvalidRunState; Stop cannot be downgraded to Pause.
        The current step is discarded unless it returns Next before a yield.
        """
        with self._lock:
            return self._current(run_id).pause()

    def resume(self, run_id: str) -> RunStatus:
        """Return paused to running while the host retains the same RUN lease.

        Only paused with no execute in flight is accepted. init/start time are
        unchanged. The next execute repeats the uncommitted step using the
        latest tunables. An unanswered Ask raises InvalidRunState; use answer
        on a worker instead. Devices are captured only when env.run executes.
        """
        with self._lock:
            return self._current(run_id).resume()

    def pending_question(self, run_id: str) -> Question[Answer] | None:
        """Read detached authored evidence while an Ask worker is settled.

        Return None outside an unanswered Ask pause. This never runs workflow
        code or consumes the question. A stale run_id raises RunMismatch.
        """
        with self._lock:
            return self._current(run_id).pending_question()

    def answer(self, run_id: str, answer: Answer) -> RunStatus:
        """Validate and execute an Ask continuation on this worker thread.

        The host schedules this synchronous segment with the same RUN lease and
        operation as execute; do not call it on the UI owner thread. Only a
        settled Ask pause accepts an answer. Reserve atomically, then run until
        the next paused or terminal state, retaining the suspended invocation
        and its original tunables capture. No resume call is needed.

        Wrong Answer model raises TypeError without delivery, journal changes or
        leaving paused. No pending Ask or an in-flight segment raises
        InvalidRunState; stale run_id raises RunMismatch. Workflow exceptions
        after valid delivery return failed, as execute does.
        """
        with self._lock:
            run = self._current(run_id)
            run.reserve_answer(answer)
        return run.execute(answer)

    def stop(self, run_id: str) -> RunStatus:
        """Request Stop, upgrade Pause, or end an already paused run.

        This does not wait for an in-flight effect. Stop wins over Pause;
        repeated stopping/stopped and other terminal calls do not add events
        or relabel done/aborted/failed. No implicit hardware cleanup occurs.
        """
        with self._lock:
            return self._current(run_id).stop()

    def update_tunables(
        self,
        run_id: str,
        tunables: BaseModel,
        *,
        expected_revision: int,
        actor: Actor,
    ) -> TunablesSnapshot:
        """Validate/journal a complete native model, then replace all tunables.

        tunables must be an instance of the workflow-declared model, not JSON
        or expressions. Fields are revalidated, including nested models/lists.
        Wrong model types raise TypeError, invalid values ValidationError.
        expected_revision must be nonnegative; stale versions raise
        RevisionConflict. actor attributes the full new JSON snapshot.
        Accepted only in running/pausing/paused; other lifecycle states raise
        InvalidRunState. The current step keeps its captured copy. Journal
        failure publishes nothing, requests run failure, and re-raises.
        Every accepted model creates a revision, including identical values.
        """
        with self._lock:
            return self._current(run_id).update_tunables(
                tunables, expected_revision, actor
            )

    def status(self) -> RunStatus | None:
        """Return detached current/last run status, or None before first start.

        Progress is a display observation, not evidence of committed records.
        """
        with self._lock:
            return None if self._run is None else self._run.status()

    def tunables(self, run_id: str) -> TunablesSnapshot:
        """Return detached complete JSON values and opaque text at one revision."""
        with self._lock:
            return self._current(run_id).tunables()

    def records(self, run_id: str, *, since: int = 0) -> tuple[CommittedRecord, ...]:
        """Return detached journal-committed records with commit_seq > since.

        since must be a nonnegative integer. No records returns an empty tuple.
        Terminal data remains readable until a new start replaces this run.
        """
        with self._lock:
            return self._current(run_id).records(since)

    def _current(self, run_id: str) -> _RunView:
        run = self._run
        actual = None if run is None else run.run_id
        if run is None or actual != run_id:
            raise RunMismatch(run_id, actual)
        return run
