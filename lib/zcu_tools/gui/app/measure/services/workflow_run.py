"""One run owner for native workflow admission, continuation and terminal delivery."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

from pydantic import BaseModel, JsonValue

from zcu_tools.core.script_protocol import Answer
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
from zcu_tools.experiment.workflows.declaration import WorkflowScript
from zcu_tools.gui.form import FormErrors, FormModel, FormWritePath
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    ExclusionRequest,
    OperationSpec,
    SettleFn,
)
from zcu_tools.gui.session.services.progress import BoundProgressFactory

from .workflow_tunables import TunablesHost

if TYPE_CHECKING:
    from zcu_tools.gui.session.operation_runner import OperationRunner

logger = logging.getLogger(__name__)


class PendingQuestion(TypedDict):
    """Detached Ask envelope for GUI and wire consumers.

    request_id is the exact host question ID required by answer. question_type
    is the authored class name. question contains only authored fields.
    answer_schema is the declared Answer model JSON schema.
    None, rather than an empty envelope, denotes no currently answerable Ask.
    """

    request_id: str
    question_type: str
    question: dict[str, JsonValue]
    answer_schema: dict[str, JsonValue]


@dataclass(frozen=True)
class QuestionExchange:
    """Latest per-run Ask presentation shared by every driving adapter.

    question is the detached question/type/schema envelope. answer contains only
    an Engine-accepted model's JSON fields, or None before acceptance. state is
    waiting, submitted, answered or cancelled; submitted is not acceptance.
    Invalid native answers return to waiting without retaining their input.
    """

    question: PendingQuestion
    answer: dict[str, JsonValue] | None
    state: Literal["waiting", "submitted", "answered", "cancelled"]


def _answer_model[C](
    engine: Engine[C], run_id: str, value: Answer | JsonValue, *, request_id: str
) -> Answer:
    """Decode the carrier; Engine owns exact request matching and field validation."""
    if isinstance(value, Answer):
        return value
    if not isinstance(value, dict):
        raise TypeError("Workflow answer must be a JSON object or Answer model")
    return engine.parse_answer(run_id, value, request_id=request_id)


@dataclass
class _Run[F: FormModel, C]:
    """Keep binding, producer ownership and cleanup together for one native run."""

    binding: TunablesHost[F, C]
    phase: Literal["starting", "producing", "finishing", "terminal"] = "starting"
    stop: StopSignal = field(default_factory=StopSignal)
    token: int | None = None
    engine: Engine[C] | None = None
    settle: SettleFn | None = None
    native_error: Exception | None = None
    last_exchange: QuestionExchange | None = None


class WorkflowStartInputs[P: BaseModel, T: BaseModel, F: FormModel, C](TypedDict):
    """Required Start inputs kept paired with their native and form declarations.

    plan/tunables are detached validated native values. paths fixes metadata/data
    roots; identity fixes run and entry provenance. ports supplies borrowed run
    resources. binding owns the exact form input that produced tunables.
    """

    plan: P
    tunables: T
    paths: RunPaths
    identity: RunIdentity
    ports: EnginePorts[C]
    binding: TunablesHost[F, C]


class WorkflowController[F: FormModel, C]:
    """Own one current run from Start admission through final owner delivery.

    A fixed controller accepts a replacement only after terminal cleanup. The
    current binding owns the unique form draft and fixed Start capture. Native
    Engine status remains authoritative; a token or binding is not acceptance.
    Pause and Ask retain the original worker, operation, RUN lease and progress.
    Resume and Answer only signal that worker; they submit no producer. This does not
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
        """Borrow runner and report terminal native delivery to the view.

        owner_id labels the run lease and progress. runner owns that progress
        service. on_finished receives authoritative native status (None before
        acceptance) and the original producer error, if any. Terminal cleanup
        fixes the native handle outcome and releases admission/lease/progress
        before notification. View errors are logged and do not alter outcome or
        binding. Pause and Ask retain the original worker until terminal delivery. Callback reentry may
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
    def last_exchange(self) -> QuestionExchange | None:
        """Read detached latest question/accepted answer facts, or None before Ask."""
        with self._lock:
            if self._run is None:
                return None
            self.pending_question(self._run.binding.run_id)
            return deepcopy(self._run.last_exchange)

    @property
    def has_pending_segment(self) -> bool:
        """Read whether admission, producer or terminal cleanup is still pending.

        Pause and Ask leave work pending. Stop/close wait for the original worker
        to finish author cleanup and deliver its terminal result.
        """
        run = self._run
        return run is not None and run.phase != "terminal"

    def start[P: BaseModel, T: BaseModel, Summary: BaseModel | None](
        self,
        script: WorkflowScript[P, T, Summary, C],
        **inputs: Unpack[WorkflowStartInputs[P, T, F, C]],
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
        plan, tunables = inputs["plan"], inputs["tunables"]
        paths, identity = inputs["paths"], inputs["identity"]
        ports, binding = inputs["ports"], inputs["binding"]
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
                script, plan=plan, tunables=tunables, paths=paths, identity=identity
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
            and run.native_error is None
        ):
            run.native_error = error

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
        elif status is None or status.lifecycle == "cancelled":
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
        """Open the native pause gate on the retained worker; return its run token.

        Stale targets raise RunMismatch. Pending acceptance raises RuntimeError.
        Engine rejects a nonpaused run or pending Ask without changing ownership.
        No second worker or Runner continuation is submitted.
        """
        with self._lock:
            self.require_run_id(run_id)
            run = self._run
            if run is None or run.engine is None or run.token is None:
                raise RuntimeError("Resume requires an accepted workflow run")
            run.engine.resume(run_id)
            return run.token

    def pending_question(self, run_id: str) -> PendingQuestion | None:
        """Read the retained worker's current Ask, including its unique request ID.

        Reads update the detached presentation exchange, not execution. Pending
        acceptance or no native Ask returns None. Stale targets raise RunMismatch.
        Engine owns question evidence and answer schema inference.
        """
        with self._lock:
            self.require_run_id(run_id)
            run = self._run
            if run is None or run.engine is None:
                return None
            pending = run.engine.pending_question(run_id)
            if pending is None:
                return None
            question = PendingQuestion(
                request_id=pending.request_id,
                question_type=type(pending.question).__name__,
                question=pending.question.model_dump(mode="json"),
                answer_schema=pending.answer_type.model_json_schema(),
            )
            if (
                run.last_exchange is None
                or run.last_exchange.question["request_id"] != pending.request_id
            ):
                run.last_exchange = QuestionExchange(question, None, "waiting")
            return question

    def answer(
        self, run_id: str, answer: Answer | JsonValue, *, request_id: str
    ) -> int:
        """Deliver this question ID through Engine and return the original run token.

        Engine owns ID matching, model/schema validation and loop delivery.
        Rejected answers propagate without changing the waiting exchange, worker
        or RUN lease. Acceptance signals the retained worker; it is not terminal
        delivery. Author code never runs on this calling thread.
        """
        with self._lock:
            self.require_run_id(run_id)
            run = self._run
            if run is None or run.engine is None or run.token is None:
                raise RuntimeError("Answer requires an accepted workflow run")
            self.pending_question(run_id)
            validated = _answer_model(run.engine, run_id, answer, request_id=request_id)
            run.engine.answer(run_id, validated, request_id=request_id)
            if run.last_exchange is not None:
                run.last_exchange = replace(
                    run.last_exchange,
                    answer=validated.model_dump(mode="json"),
                    state="answered",
                )
            return run.token

    def stop(self, run_id: str) -> RunStatus | None:
        """Stop the original worker cooperatively without cancelling its author.

        Stale targets raise RunMismatch. Before native acceptance, record intent
        and return None. Ask resumes with None on its existing worker. The token,
        progress and RUN lease remain until terminal owner delivery, including
        after native control I/O failure. No hardware rollback occurs.
        """
        with self._lock:
            self.require_run_id(run_id)
            run = self._run
            if run is None:
                raise RuntimeError("Stop requires a current run")
            self.pending_question(run_id)
            run.stop.set()
            if run.last_exchange is not None and run.last_exchange.state in {
                "waiting",
                "submitted",
            }:
                run.last_exchange = replace(run.last_exchange, state="cancelled")
            if run.engine is None:
                return None
            try:
                stopped = run.engine.stop(run_id)
            except Exception as error:
                self._record_native_failure(run, error)
                raise
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
