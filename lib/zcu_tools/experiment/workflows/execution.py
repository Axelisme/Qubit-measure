"""One async run's journal-authoritative records and cooperative lifecycle."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from threading import Lock, RLock
from typing import Literal

from pydantic import BaseModel, JsonValue, TypeAdapter

from zcu_tools.core.script_protocol import Answer
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.experiment.script_host import PendingQuestion, ScriptHost
from zcu_tools.progress_bar import use_pbar_factory

from .artifacts import RecordStarted, RunArtifacts
from .declaration import WorkflowDefinition
from .encoding import encode_record
from .env import Displays, EnvHooks, WorkflowEnv
from .journal import (
    DeviceSetup,
    ExperimentFailed,
    OperationFinished,
    Paused,
    PauseRequested,
    QuestionAnswered,
    RecordCommitted,
    Resumed,
    RunEnded,
    RunMetadata,
    StopRequested,
    TunablesChanged,
    describe_error,
)
from .models import (
    Actor,
    CommittedRecord,
    InvalidRunState,
    Lifecycle,
    OperationFacts,
    QuestionMismatch,
    RunIdentity,
    RunPaths,
    RunStatus,
    TunablesSnapshot,
)
from .ports import EnginePorts
from .tunables import Live, TunableValues

_logger = logging.getLogger(__name__)
_TERMINAL = ("done", "cancelled", "failed")


def _validate_summary(summary: object) -> None:
    if summary is not None and not isinstance(summary, BaseModel):
        raise TypeError("Workflow return must be a BaseModel or None")


class WorkflowRun[P: BaseModel, T: BaseModel, Summary: BaseModel | None, C]:
    """Retain one declaration's plan/tunables/summary pairing while driving it.

    definition supplies the original coroutine. ports borrows host resources.
    lock serializes short journal/publication sections, never cross-thread host
    acknowledgements or executor drain. start precedes all other operations;
    reserve_execution precedes the one synchronous execute on a caller worker.
    Controls keep that same worker alive through Ask and Pause.
    """

    def __init__(
        self,
        definition: WorkflowDefinition[P, T, Summary, C],
        ports: EnginePorts[C],
        lock: RLock,
    ) -> None:
        self._definition = definition
        self._ports = ports
        self._lock = lock
        self._host = ScriptHost()
        self._displays = Displays(ports.progress)
        self._lifecycle: Lifecycle = "running"
        self._reason: str | None = None
        self._executing = False
        self._stop_requested = False
        self._pending_error: Exception | None = None
        self._call_seq: int | None = None
        self._commit_seq = 0
        self._run_no = 0
        self._records: list[CommittedRecord] = []
        self._revisions: list[int] = []
        self._accepted_answers: list[tuple[PendingQuestion, Answer]] = []
        self._answer_lock = Lock()

    def start(
        self, plan: P, tunables: T, paths: RunPaths, identity: RunIdentity
    ) -> None:
        """Revalidate/copy models and persist startup without executing the author."""
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
        self._artifacts = RunArtifacts(
            RunMetadata(
                self._identity,
                self._definition.name,
                TypeAdapter(JsonValue).validate_python(
                    self._plan.model_dump(mode="json")
                ),
                self._tunables.snapshot().values,
                paths,
                self._started_at,
            ),
            self._ports.clock,
            self._lock,
        )

    @property
    def run_id(self) -> str:
        """Read stable host-supplied identity without querying display resources."""
        return self._identity.run_id

    def allow_replacement(self) -> None:
        """Reject replacement until terminal status and worker drain are both true."""
        if self._lifecycle not in _TERMINAL or self._executing:
            raise InvalidRunState("start", self._lifecycle)

    def reserve_execution(self) -> None:
        """Reserve the run's only worker entry while Engine holds the lock."""
        if (
            self._executing
            or self._call_seq is not None
            or self._lifecycle in _TERMINAL
        ):
            raise InvalidRunState("execute", self._lifecycle)
        self._executing = True

    def execute(self) -> RunStatus:
        """Drive the single author to terminal, retaining this worker while paused.

        Exceptions fail the run and retain diagnostics. All authored finally and
        executor drain finish before the reservation is released. No asyncio
        Task is cancelled and no connection is closed here.
        """
        try:
            with use_pbar_factory(self._ports.progress):
                summary = self._host.run(self._author())
            if self._pending_error is not None:
                raise self._pending_error
            self._finish(summary)
        except Exception as error:
            _logger.exception("Workflow execution failed")
            self._fail(self._pending_error or error)
        finally:
            self._displays.close()
            with self._lock:
                self._executing = False
        return self.status()

    async def _author(self) -> Summary:
        with self._lock:
            self._new_iteration()
        env = WorkflowEnv(
            self._started_at,
            self._ports,
            self._host,
            self._displays,
            EnvHooks(
                self._artifacts.new_commit_dir,
                self._commit,
                self._boundary,
                self._stopping,
                self._experiment_failed,
                self._question_boundary,
                self._next_attempt,
                self._operation_finished,
                self._device_setup,
            ),
            identity=self._identity,
        )
        return await self._definition.script(
            env, deepcopy(self._plan), Live(self._read_tunables)
        )

    def _new_iteration(self) -> None:
        self._call_seq = self._commit_seq + 1
        self._scope_revision = self._tunables.revision
        self._revisions = []
        self._run_no = 0
        self._attempt = 0
        self._operations: list[OperationFacts] = []

    def _next_attempt(self) -> tuple[int, int]:
        with self._lock:
            self._attempt += 1
            return self._commit_seq + 1, self._attempt

    def _operation_finished(self, operation: OperationFacts) -> None:
        with self._lock:
            self._artifacts.append(OperationFinished(self._commit_seq + 1, operation))
            self._operations.append(deepcopy(operation))

    def _device_setup(self, name: str, cfg: BaseDeviceInfo) -> None:
        self._artifacts.append(DeviceSetup(name, cfg.model_copy(deep=True)))

    def _read_tunables(self) -> T:
        with self._lock:
            captured = self._tunables.capture()
            if captured.revision not in self._revisions:
                self._revisions.append(captured.revision)
            return captured.model

    def _commit(self, record: BaseModel, directory: Path | None) -> None:
        with self._lock:
            directory = self._artifacts.prepare_commit(directory)
            record_type = self._artifacts.register_record(record)
            revisions = tuple(self._revisions)
            event = RecordCommitted(
                self._commit_seq + 1,
                self._commit_seq + 1,
                self._scope_revision,
                encode_record(record),
                directory.relative_to(self._artifacts.paths.data_root).as_posix(),
                self._ports.clock.now(),
                revisions,
                record_type,
                tuple(self._operations),
            )
            self._artifacts.publish_record(
                directory,
                event,
                RecordStarted(self._commit_seq + 1, self._commit_seq, revisions),
            )
            self._records.append(event)
            self._commit_seq += 1
            self._new_iteration()

    def _boundary(self) -> None:
        with self._lock:
            if self._pending_error is not None:
                raise self._pending_error
            if self._lifecycle == "pausing":
                self._artifacts.append(Paused(self._call_seq, self._commit_seq))
                self._artifacts.set_lifecycle("paused")
                self._lifecycle = "paused"

    def _question_boundary(self, phase: Literal["waiting", "answered"]) -> None:
        with self._lock:
            self._flush_answers()
            if self._pending_error is not None:
                raise self._pending_error
            if self._lifecycle in ("stopping", "failed"):
                return
            if phase == "waiting" and self._lifecycle != "paused":
                self._artifacts.append(Paused(self._call_seq, self._commit_seq))
                self._artifacts.set_lifecycle("paused")
                self._lifecycle = "paused"
            elif phase == "answered" and self._lifecycle == "paused":
                self._artifacts.append(
                    Resumed(self._commit_seq, self._tunables.revision)
                )
                self._artifacts.set_lifecycle("running")
                self._lifecycle = "running"

    def _stopping(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _experiment_failed(
        self, experiment: str, reason: str, cause: BaseException
    ) -> None:
        with self._lock:
            self._run_no += 1
            self._artifacts.append(
                ExperimentFailed(
                    self._commit_seq + 1,
                    self._run_no,
                    experiment,
                    reason,
                    describe_error(cause),
                )
            )

    def pause(self) -> RunStatus:
        """Record Pause intent, then acknowledge host control outside the lock."""
        try:
            with self._lock:
                if self._lifecycle == "pausing":
                    return self.status()
                if self._lifecycle != "running":
                    raise InvalidRunState("pause", self._lifecycle)
                self._artifacts.append(
                    PauseRequested(self._ports.clock.now(), self._call_seq)
                )
                self._artifacts.set_lifecycle("pausing")
                self._lifecycle = "pausing"
        except InvalidRunState:
            raise
        except Exception as error:
            self._control_failure(error)
            raise
        self._host.pause()
        return self.status()

    def resume(self) -> RunStatus:
        """Open the existing worker's pause gate; an Ask requires an answer instead."""
        try:
            with self._lock:
                if (
                    self._lifecycle != "paused"
                    or self._host.pending_question is not None
                ):
                    raise InvalidRunState("resume", self._lifecycle)
                self._artifacts.append(
                    Resumed(self._commit_seq, self._tunables.revision)
                )
                self._artifacts.set_lifecycle("running")
                self._lifecycle = "running"
        except InvalidRunState:
            raise
        except Exception as error:
            self._control_failure(error)
            raise
        self._host.resume()
        return self.status()

    def stop(self) -> RunStatus:
        """Latch explicit Stop and release an Ask/gate without waiting for drain."""
        try:
            with self._lock:
                if self._lifecycle in _TERMINAL or self._stop_requested:
                    return self.status()
                self._artifacts.append(
                    StopRequested(self._ports.clock.now(), self._call_seq)
                )
                self._artifacts.set_lifecycle("stopping")
                self._lifecycle = "stopping"
                self._stop_requested = True
        except Exception as error:
            self._control_failure(error)
            raise
        self._host.cancel()
        return self.status()

    def pending_question(self) -> PendingQuestion | None:
        """Read the host's exact current request, including its unique request id."""
        return deepcopy(self._host.pending_question)

    def _require_question(self, request_id: str) -> PendingQuestion:
        pending = self.pending_question()
        if pending is None or pending.request_id != request_id:
            raise QuestionMismatch(
                request_id, None if pending is None else pending.request_id
            )
        return pending

    def parse_answer(self, mapping: Mapping[str, object], *, request_id: str) -> Answer:
        """Match the exact request first, then validate fields without delivery."""
        pending = self._require_question(request_id)
        return pending.answer_type.model_validate(mapping)

    def answer(self, answer: Answer, *, request_id: str) -> RunStatus:
        """Acknowledge delivery to the exact pending request on the existing loop.

        Wrong ids raise QuestionMismatch carrying pending_request_id. Wrong model
        types/fields raise TypeError/ValidationError. Rejection publishes no
        answered fact or lifecycle change. No new worker segment is started.
        """

        self._require_question(request_id)
        accepted_answer = answer.model_copy(deep=True)

        def deliver() -> None:
            pending = self._require_question(request_id)
            self._host.answer(request_id, accepted_answer)
            # The host ack guarantees validation and Future delivery. Keep this
            # callback memory-only; the author boundary or caller flushes later.
            accepted = (pending, accepted_answer)
            with self._answer_lock:
                self._accepted_answers.append(accepted)

        try:
            self._host.control(deliver)
        except (ValueError, RuntimeError) as error:
            current = self.pending_question()
            if current is None or current.request_id != request_id:
                raise QuestionMismatch(
                    request_id, None if current is None else current.request_id
                ) from error
            raise
        self._flush_answers()
        return self.status()

    def _flush_answers(self) -> None:
        with self._lock:
            with self._answer_lock:
                accepted = self._accepted_answers
                self._accepted_answers = []
            for pending, answer in accepted:
                self._artifacts.append(
                    QuestionAnswered(
                        pending.request_id,
                        type(pending.question).__name__,
                        pending.question.model_dump(mode="json"),
                        answer.model_dump(mode="json"),
                    )
                )

    def _control_failure(self, error: Exception) -> None:
        with self._lock:
            if self._pending_error is None:
                self._pending_error = error
            self._reason = (
                f"{type(self._pending_error).__name__}: {self._pending_error}"
            )
            self._lifecycle = "stopping" if self._executing else "failed"
        # Acknowledgement may wait for the loop; never retain the Engine lock.
        self._host.cancel()

    def update_tunables(
        self, tunables: BaseModel, expected_revision: int, actor: Actor
    ) -> TunablesSnapshot:
        """Replace one whole model after its journal append succeeds."""
        try:
            with self._lock:
                if self._lifecycle not in ("running", "pausing", "paused"):
                    raise InvalidRunState("update_tunables", self._lifecycle)
                return self._tunables.update(
                    tunables, expected_revision=expected_revision, actor=actor
                )
        except Exception as error:
            if self._pending_error is error:
                self._control_failure(error)
            raise

    def _append_tunables(self, event: TunablesChanged) -> None:
        try:
            self._artifacts.append(event)
        except Exception as error:
            if self._pending_error is None:
                self._pending_error = error
            raise

    def _finish(self, summary: Summary) -> None:
        _validate_summary(summary)
        with self._lock:
            lifecycle = "cancelled" if self._stop_requested else "done"
            saved_summary = lifecycle == "done" and summary is not None
            if saved_summary:
                self._artifacts.save_summary(encode_record(summary))
            self._artifacts.append(
                RunEnded(
                    lifecycle,
                    None,
                    self._call_seq,
                    self._commit_seq,
                    summary=saved_summary,
                )
            )
            self._artifacts.set_lifecycle(lifecycle)
            self._lifecycle = lifecycle

    def _fail(self, error: Exception) -> None:
        with self._lock:
            self._lifecycle = "failed"
            self._reason = f"{type(error).__name__}: {error}"
            try:
                self._artifacts.append(
                    RunEnded("failed", self._reason, self._call_seq, self._commit_seq)
                )
                self._artifacts.set_lifecycle("failed", self._reason)
            except Exception:
                _logger.exception("Cannot persist workflow failure diagnostics")

    def status(self) -> RunStatus:
        """Read detached records/progress counters and the current pending request."""
        with self._lock:
            return RunStatus(
                self.run_id,
                self._definition.name,
                self._lifecycle,
                self._call_seq,
                self._commit_seq,
                self._tunables.revision,
                self._reason,
                self._displays.snapshot(),
                self.pending_question(),
            )

    def tunables(self) -> TunablesSnapshot:
        """Read the latest detached whole-model JSON and its revision."""
        return self._tunables.snapshot()

    def records(self, since: int) -> tuple[CommittedRecord, ...]:
        """Read committed records after the nonnegative commit sequence since."""
        if type(since) is not int or since < 0:
            raise ValueError("since must be a nonnegative integer")
        with self._lock:
            return deepcopy(
                tuple(record for record in self._records if record.commit_seq > since)
            )
