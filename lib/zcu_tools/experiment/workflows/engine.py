"""Thread-safe controls over one caller-worker-owned async workflow run."""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from typing import Protocol

from pydantic import BaseModel

from zcu_tools.core.script_protocol import Answer
from zcu_tools.experiment.script_host import PendingQuestion
from zcu_tools.experiment.stop_signal import CancelResult

from .declaration import WorkflowScript, workflow_definition
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
    """Erase model parameters only at Engine storage, retaining typed execution."""

    @property
    def run_id(self) -> str: ...
    def allow_replacement(self) -> None: ...
    def reserve_execution(self) -> None: ...
    def execute(self) -> RunStatus: ...
    def pause(self) -> RunStatus: ...
    def resume(self) -> RunStatus: ...
    def stop(self) -> RunStatus: ...
    def cancel(self, scope: str, *, expected: int | None = None) -> CancelResult: ...
    def answer(self, answer: Answer, *, request_id: str) -> RunStatus: ...
    def pending_question(self) -> PendingQuestion | None: ...
    def parse_answer(
        self, mapping: Mapping[str, object], *, request_id: str
    ) -> Answer: ...
    def update_tunables(
        self, tunables: BaseModel, expected_revision: int, actor: Actor
    ) -> TunablesSnapshot: ...
    def status(self) -> RunStatus: ...
    def tunables(self) -> TunablesSnapshot: ...
    def records(self, since: int) -> tuple[CommittedRecord, ...]: ...


class Engine[C]:
    """Own one declared run, with native resources and worker borrowed from host.

    start validates/persists startup but executes no author code. execute drives
    the entire coroutine synchronously on the caller worker, including retained
    Ask/Pause waits. Control methods acknowledge loop delivery, not completion.
    They never hold the Engine lock while waiting on a cross-thread host ack.
    No hardware connections, worker threads or asyncio Tasks are created here.
    Terminal replacement requires actual worker/finally/executor drain.
    """

    def __init__(self, ports: EnginePorts[C]) -> None:
        self._ports = ports
        self._lock = RLock()
        self._run: _RunView | None = None

    def start[P: BaseModel, T: BaseModel, Summary: BaseModel | None](
        self,
        script: WorkflowScript[P, T, Summary, C],
        *,
        plan: P,
        tunables: T,
        paths: RunPaths,
        identity: RunIdentity,
    ) -> RunStatus:
        """Validate a declared coroutine and persist detached revision-zero input.

        paths are distinct nonexistent metadata/data roots. identity is supplied
        provenance. Missing ports fail only when used. Active replacement raises
        InvalidRunState; model/path/I/O errors propagate without replacing the
        previous run. The author starts only on execute.
        """
        with self._lock:
            if self._run is not None:
                self._run.allow_replacement()
            run = WorkflowRun(workflow_definition(script), self._ports, self._lock)
            run.start(plan, tunables, paths, identity)
            self._run = run
            return run.status()

    def _current(self, run_id: str) -> _RunView:
        with self._lock:
            run = self._run
            if run is None or run.run_id != run_id:
                raise RunMismatch(run_id, None if run is None else run.run_id)
            return run

    def execute(self, run_id: str) -> RunStatus:
        """Drive the single async author until terminal on this caller worker.

        Pause and Ask keep this call pending. Reentry/reexecution raises
        InvalidRunState. Exceptions return failed with journal diagnostics.
        """
        with self._lock:
            run = self._current(run_id)
            run.reserve_execution()
        return run.execute()

    def pause(self, run_id: str) -> RunStatus:
        """Request operation stop and close the pause gate without awaiting drain."""
        return self._current(run_id).pause()

    def resume(self, run_id: str) -> RunStatus:
        """Open the same worker's gate; a pending Ask requires answer instead."""
        return self._current(run_id).resume()

    def stop(self, run_id: str) -> RunStatus:
        """Stop cooperatively; release Ask with None and permit author cleanup."""
        return self._current(run_id).stop()

    def cancel(
        self, run_id: str, scope: str = "script", *, expected: int | None = None
    ) -> CancelResult:
        """Acknowledge a matching live scope; stale run IDs raise RunMismatch.

        scope/expected use ScriptHost's label or label#N contract. Script requests
        retain Stop lifecycle/journal semantics; narrower requests leave the
        workflow running. Acknowledgement does not wait for cleanup.
        """
        return self._current(run_id).cancel(scope, expected=expected)

    def pending_question(self, run_id: str) -> PendingQuestion | None:
        """Read the exact pending host request including id, evidence and schema type."""
        return self._current(run_id).pending_question()

    def pending_answer_type(self, run_id: str) -> type[Answer] | None:
        """Read the current declared Answer type without accepting an answer."""
        pending = self.pending_question(run_id)
        return None if pending is None else pending.answer_type

    def parse_answer(
        self, run_id: str, mapping: Mapping[str, object], *, request_id: str
    ) -> Answer:
        """Validate fields for this request without accepting or delivering them.

        Stale/absent IDs raise QuestionMismatch with the current pending ID before
        schema validation. Wrong fields raise ValidationError; neither failure
        changes the question, lifecycle or journal.
        """
        return self._current(run_id).parse_answer(mapping, request_id=request_id)

    def answer(self, run_id: str, answer: Answer, *, request_id: str) -> RunStatus:
        """Deliver to this exact question on the retained worker loop.

        Stale/absent ids raise QuestionMismatch with current pending_request_id.
        Invalid model/fields leave the question answerable and propagate their
        TypeError/ValidationError. Return acknowledges delivery, not completion.
        """
        return self._current(run_id).answer(answer, request_id=request_id)

    def update_tunables(
        self,
        run_id: str,
        tunables: BaseModel,
        *,
        expected_revision: int,
        actor: Actor,
    ) -> TunablesSnapshot:
        """Journal then replace a whole validated model, incrementing its revision.

        Stale revisions raise RevisionConflict. Invalid models/fields propagate.
        Every successful update increments even for equal values. tun.current
        sees the next published revision on its next read, including mid-loop.
        """
        return self._current(run_id).update_tunables(tunables, expected_revision, actor)

    def status(self) -> RunStatus | None:
        """Read detached lifecycle/progress/commit counters and pending question."""
        with self._lock:
            return None if self._run is None else self._run.status()

    def tunables(self, run_id: str) -> TunablesSnapshot:
        """Read the latest detached complete model JSON and revision."""
        return self._current(run_id).tunables()

    def records(self, run_id: str, *, since: int = 0) -> tuple[CommittedRecord, ...]:
        """Read journal-authoritative records after nonnegative commit sequence since."""
        return self._current(run_id).records(since)
